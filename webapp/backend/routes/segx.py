"""Segment extraction: the step in some clips -> where it happens in another video."""
from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from .. import config, media, segx
from ..schemas import SegxCutRequest, SegxDescribeRequest, SegxSearchRequest
from .jobs_routes import _video_upload

router = APIRouter()


def _clips(request: Request, ids: list[str], what: str):
    ups = []
    for i in ids:
        up = request.app.state.uploads.get(i)
        if up is None:
            raise HTTPException(404, f"unknown {what} {i} — upload it again")
        if up.kind != "video":
            raise HTTPException(400, f"{what} {i} is a {up.kind}, not a video clip")
        ups.append(up)
    return ups


def _registry(request: Request):
    # Qwen has its own GPU and thread; on a shared GPU it takes turns with SAM 3
    return request.app.state.qwen_jobs if config.qwen_dedicated() else request.app.state.jobs


@router.post("/segx/cut")
async def cut(request: Request, body: SegxCutRequest) -> dict:
    """A segment marked on a reference video, as a clip upload the other calls take."""
    src = _video_upload(request, body.upload_id)
    last = (src.frames or 0) - 1
    if body.end < body.start:
        raise HTTPException(400, "the segment ends before it starts")
    if last >= 0 and body.start > last:
        raise HTTPException(400, f"frame {body.start} is past the end of the video (0..{last})")
    end = min(body.end, last) if last >= 0 else body.end
    fps = float(src.fps or 20.0)
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}.mp4"
    n = await run_in_threadpool(media.cut, src.path, body.start, end, fps, dest)
    if n <= 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"frames {body.start}-{end} could not be decoded")
    meta = await run_in_threadpool(media.probe, dest)
    stem = Path(src.name or "video").stem
    up = request.app.state.uploads.add(
        "video", dest, meta["width"], meta["height"], name=f"{stem} [{body.start}-{end}]",
        frames=meta["frames"], fps=meta["fps"] or fps, duration=meta["duration"])
    return {**up.public(), "source_id": src.id, "start": body.start, "end": end}


@router.post("/segx/clip-frames")
async def clip_frames(request: Request, frames: list[UploadFile] = File(...),
                      fps: float = Form(...), name: str = Form("clip")) -> dict:
    """A clip marked in the browser, sent as its frames (JPEGs, in order).

    The browser plays the user's own copy of the reference video, so the video
    itself never has to be uploaded: only the frames of the marked range come
    over, and they become a clip upload like /segx/cut makes."""
    if not 1 <= len(frames) <= 3000:
        raise HTTPException(400, f"{len(frames)} frames: send 1 to 3000")
    if not 1 <= fps <= 240:
        raise HTTPException(400, f"fps {fps} is out of range")
    work = config.TMP_DIR / f"clip_{uuid.uuid4().hex[:12]}"
    work.mkdir(parents=True)
    try:
        for i, f in enumerate(frames):
            (work / f"{i:06d}.jpg").write_bytes(await f.read())
        dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}.mp4"
        n = await run_in_threadpool(media.encode_frames, work, fps, dest)
    finally:
        for p in work.iterdir():
            p.unlink(missing_ok=True)
        work.rmdir()
    if n != len(frames):
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"encoded {n} of {len(frames)} frames")
    meta = await run_in_threadpool(media.probe, dest)
    up = request.app.state.uploads.add(
        "video", dest, meta["width"], meta["height"], name=name,
        frames=meta["frames"], fps=meta["fps"] or fps, duration=meta["duration"])
    return up.public()


@router.post("/jobs/segx-describe", status_code=202)
def describe(request: Request, body: SegxDescribeRequest):
    """The VLM watches up to three step clips and names and describes the step."""
    clips = _clips(request, body.clip_ids, "step clip")
    st = request.app.state
    registry = _registry(request)
    job = registry.create("segx-describe")

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        return segx.describe(st.qwen_vlm, st.engine, clips, job.id, progress_cb, cancel)

    registry.submit(job, fn)
    return {"job_id": job.id}


@router.post("/jobs/segx-search", status_code=202)
def search(request: Request, body: SegxSearchRequest):
    """Embed, pick candidates, and classify them with the VLM. Minutes, on GPU 0."""
    target = _video_upload(request, body.upload_id)
    steps = _clips(request, body.step_ids, "step clip")
    others = _clips(request, body.other_ids, "other clip")
    refs, knn_steps, knn_others = [], [], []
    if body.candidates == "knn":
        if not body.references and not body.knn_step_ids:
            raise HTTPException(400, "the kNN candidates need the reference video(s) the step was marked on")
        refs = [(_clips(request, [r.upload_id], "reference video")[0], r.steps, r.others)
                for r in body.references]
        knn_steps = _clips(request, body.knn_step_ids, "step clip")
        knn_others = _clips(request, body.knn_other_ids, "other clip")
    st = request.app.state
    registry = _registry(request)
    job = registry.create("segx-search")
    payload = body.model_dump()

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        return segx.search(st.qwen, st.qwen_vlm, st.engine, target, steps, others, payload,
                           job.id, progress_cb, cancel,
                           refs=refs, knn_steps=knn_steps, knn_others=knn_others)

    registry.submit(job, fn)
    return {"job_id": job.id}
