"""Frame extraction: find the segments of a video that match a few reference images."""
from __future__ import annotations

import threading
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .. import config, extraction
from ..schemas import (ExtractEmbedRequest, ExtractExportRequest, ExtractPlaybackRequest,
                       ExtractScoreRequest)
from .jobs_routes import _video_upload

router = APIRouter()


def _image_uploads(request: Request, ids: list[str]):
    ups = []
    for i in ids:
        up = request.app.state.uploads.get(i)
        if up is None:
            raise HTTPException(404, f"unknown reference image {i} — upload it again")
        if up.kind != "image":
            raise HTTPException(400, f"reference {i} is a {up.kind}, not an image")
        ups.append(up)
    return ups


@router.post("/jobs/extract-embed", status_code=202)
def embed(request: Request, body: ExtractEmbedRequest):
    """Embed the video and reference images with Qwen3-VL-Embedding.

    Runs on Qwen's own worker thread and GPU (GPU 0 by default), alongside
    anything SAM 3 is doing. When everything is already cached it finishes at
    once without touching the GPU.
    """
    up = _video_upload(request, body.upload_id)
    images = _image_uploads(request, body.image_ids)
    instruction = (body.instruction or "").strip() or config.QWEN_EMB_INSTRUCTION
    engine = request.app.state.engine
    # its own GPU -> its own worker thread, so tracking and embedding run side by
    # side; a shared GPU means taking turns with SAM 3 on the inference thread
    registry = request.app.state.qwen_jobs if config.qwen_dedicated() else request.app.state.jobs
    job = registry.create("extract-embed")

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        return extraction.ensure_embeddings(request.app.state.qwen, engine, up, images,
                                            body.stride, instruction, progress_cb, cancel)

    registry.submit(job, fn)
    return {"job_id": job.id, "key": extraction.cache_key(instruction, body.stride)}


@router.post("/jobs/extract-playback", status_code=202)
def playback(request: Request, body: ExtractPlaybackRequest):
    """A browser-playable copy of the video, for smooth segment playback.

    Played frame-by-frame as JPEGs, a segment crawls along at ~5 frames a second;
    this is ~8 s of ffmpeg per 5 min of 640x480, once per video, on the CPU worker.
    """
    up = _video_upload(request, body.upload_id)
    registry = request.app.state.cpu_jobs
    job = registry.create("extract-playback")

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        return extraction.playback_copy(up, progress_cb, body.format)

    registry.submit(job, fn)
    return {"job_id": job.id}


@router.post("/extract/score")
async def score(request: Request, body: ExtractScoreRequest) -> dict:
    """z-scored similarity rows, one per reference. CPU only, milliseconds."""
    up = _video_upload(request, body.upload_id)
    images = {u.id: u for u in _image_uploads(request, [r.id for r in body.refs if r.kind == "image"])}
    try:
        return await run_in_threadpool(extraction.score, up, body.key,
                                       [r.model_dump() for r in body.refs], images)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/jobs/extract-export", status_code=202)
def export(request: Request, body: ExtractExportRequest):
    """Write the selection out: JSON, a clip of the selected frames, a JPEG ZIP.

    CPU work, so it goes on the CPU worker and never waits behind the GPU.
    """
    up = _video_upload(request, body.upload_id)
    registry = request.app.state.cpu_jobs
    job = registry.create("extract-export")
    stem = str(config.RESULT_DIR / f"extract_{job.id}")
    payload = body.model_dump()

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        return extraction.export(up, payload, stem, progress_cb, cancel)

    registry.submit(job, fn)
    return {"job_id": job.id}
