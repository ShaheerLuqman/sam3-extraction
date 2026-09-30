"""SAM3 web app backend — FastAPI, single worker, one GPU (physical GPU 1).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/uvicorn webapp.backend.main:app --port 8000
"""
from __future__ import annotations

import asyncio
import os
import logging
import time
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sam3webapp")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from . import config  # noqa: E402
from . import fawadseg  # noqa: E402
from .engine import Engine  # noqa: E402
from .extraction import QwenWorker  # noqa: E402
from .clipprep import ClipPrepStore  # noqa: E402
from .jobs import JobRegistry  # noqa: E402
from .classes import ClassStore  # noqa: E402
from .routes import classes as classes_routes  # noqa: E402
from .routes import extract as extract_routes  # noqa: E402
from .routes import fawadseg as fawadseg_routes  # noqa: E402
from .routes import segx as segx_routes  # noqa: E402
from .routes import health, jobs_routes, runs as runs_routes, uploads  # noqa: E402
from .runs import RunStore  # noqa: E402
from .store import UploadStore  # noqa: E402


async def _sweeper(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(config.SWEEP_INTERVAL_SECONDS)
        try:
            app.state.jobs.gc(config.JOB_TTL_SECONDS, config.KEEP_LAST_RUNS)
            app.state.cpu_jobs.gc(config.JOB_TTL_SECONDS, config.KEEP_LAST_RUNS)
            app.state.qwen_jobs.gc(config.JOB_TTL_SECONDS, config.KEEP_LAST_RUNS)
            app.state.uploads.sweep(config.UPLOAD_TTL_SECONDS)
            _sweep_dir(config.RESULT_DIR, config.RESULT_TTL_SECONDS, config.KEEP_LAST_RUNS)
            _sweep_dir(config.TMP_DIR, 3600, 0)
            _sweep_dir(config.EMBED_DIR, config.UPLOAD_TTL_SECONDS, 0)  # dies with its upload
            _sweep_dir(config.PROXY_DIR, config.UPLOAD_TTL_SECONDS, 0)
            fawadseg.sweep(config.RESULT_TTL_SECONDS)
        except Exception:  # noqa: BLE001
            log.warning("sweeper pass failed", exc_info=True)


def _sweep_dir(path, ttl: float, keep_last_runs: int) -> None:
    """Delete files older than `ttl`, always sparing the newest N runs.

    A run writes `<id>.mp4` *and* `<id>.json`, so the spare list is counted by
    stem — counting files would have kept half as many runs as it claimed.
    """
    now = time.time()
    files = sorted((p for p in path.iterdir() if p.is_file() and p.name != ".gitkeep"),
                   key=lambda p: p.stat().st_mtime)
    keep: set[str] = set()
    if keep_last_runs:
        for p in reversed(files):          # newest first
            keep.add(p.stem)
            if len(keep) >= keep_last_runs:
                break
    for p in files:
        if p.stem not in keep and now - p.stat().st_mtime > ttl:
            p.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gpu_info = config.require_single_gpu()
    app.state.engine = Engine(config.CHECKPOINT)
    app.state.engine.set_gpu_info(app.state.gpu_info)
    app.state.jobs = JobRegistry()
    # clip preprocessing is CPU-only, so it gets its own worker: preparing a long
    # clip must never delay a click preview on the inference thread
    app.state.cpu_jobs = JobRegistry()
    # frame extraction's embedding runs, on Qwen's own GPU and its own thread
    app.state.qwen_jobs = JobRegistry()
    app.state.qwen = QwenWorker("embed")
    # segment extraction's VLM; only one of the two is ever loaded on the Qwen GPU
    app.state.qwen_vlm = QwenWorker("vlm")
    app.state.clips = ClipPrepStore()
    app.state.runs = RunStore()
    app.state.uploads = UploadStore()
    app.state.classes = ClassStore()

    log.info("queuing model load on the inference worker...")
    app.state.jobs.submit_bare(app.state.engine.load)

    sweep_task = asyncio.create_task(_sweeper(app))
    try:
        yield
    finally:
        sweep_task.cancel()
        app.state.jobs.shutdown()
        app.state.cpu_jobs.shutdown()
        app.state.qwen_jobs.shutdown()
        app.state.qwen.stop()
        app.state.qwen_vlm.stop()
        app.state.clips.clear()


app = FastAPI(title="SAM3 web app", lifespan=lifespan)

if config.cors_origins():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(health.router, prefix="/api")
app.include_router(uploads.router, prefix="/api")
app.include_router(classes_routes.router, prefix="/api")
app.include_router(jobs_routes.router, prefix="/api")
app.include_router(runs_routes.router, prefix="/api")
app.include_router(extract_routes.router, prefix="/api")
app.include_router(segx_routes.router, prefix="/api")
app.include_router(fawadseg_routes.router, prefix="/api")
app.mount("/api/files", StaticFiles(directory=config.RESULT_DIR), name="files")
app.mount("/api/proxies", StaticFiles(directory=config.PROXY_DIR), name="proxies")
app.mount("/api/fawadseg", StaticFiles(directory=config.FAWADSEG_RUNS), name="fawadseg")

# The built frontend is for single-process mode (`make serve`). Under run.sh Vite
# serves the live source on :5173, and a dist/ served here as well only goes stale
# unnoticed — so run.sh sets SAM3_SERVE_FRONTEND=0 and :8000 is the API alone.
if os.environ.get("SAM3_SERVE_FRONTEND", "1") != "0" and config.FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
    log.info("serving frontend from %s", config.FRONTEND_DIST)
