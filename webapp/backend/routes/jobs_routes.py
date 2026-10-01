from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import config, media
from ..clipprep import prepare_clip
from ..schemas import (
    ClickPreviewRequest,
    ExemplarRequest,
    MultiTrackRequest,
    PrepareRequest,
    PrepClipRequest,
    SegmentTrackRequest,
)

router = APIRouter()


def _video_upload(request: Request, upload_id: str):
    up = request.app.state.uploads.get(upload_id)
    if up is None:
        raise HTTPException(404, "unknown upload")
    if up.kind != "video":
        raise HTTPException(400, f"expected a video upload, got {up.kind}")
    return up


def _files_url(path: str | Path) -> str:
    return f"/api/files/{Path(path).name}"


@router.post("/jobs/prep-clip", status_code=202)
def prep_clip(request: Request, body: PrepClipRequest):
    """Trim and resize the tracking window ahead of time.

    Runs on the CPU registry, not the inference worker, so a long clip never
    blocks the click previews the user is making while it works.
    """
    up = _video_upload(request, body.upload_id)
    store = request.app.state.clips
    registry = request.app.state.cpu_jobs
    frames = max(config.MIN_FRAMES, min(body.frames, up.frames or config.UNKNOWN_LENGTH))

    job = registry.create("prep-clip")

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        if store.covers(body.upload_id, frames):
            return {"frames": frames, "reused": True, "message": f"{frames} frames already prepared"}
        prepped = prepare_clip(up.path, body.upload_id, frames,
                               config.TRACKER_IMAGE_SIZE, progress_cb, cancel)
        store.put(prepped)
        return {"frames": prepped.frames, "reused": False,
                "message": f"{prepped.frames} frames prepared — tracking can start on the GPU"}

    registry.submit(job, fn)
    return {"job_id": job.id}


@router.post("/jobs/prepare", status_code=202)
def prepare(request: Request, body: PrepareRequest):
    """Warm one frame's encodings so the first prompt on it doesn't pay for them.

    The frontend fires this the moment a video is uploaded, and again when the
    viewer settles on a new frame.
    """
    up = _video_upload(request, body.upload_id)
    engine = request.app.state.engine
    registry = request.app.state.jobs
    job = registry.create("prepare")
    frame = max(0, min(body.frame, (up.frames or 1) - 1))

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        progress_cb(0.1, "extracting frame")
        jpeg = media.extract_frame_jpeg(str(up.path), frame)
        img_path = config.TMP_DIR / f"{job.id}_prep.jpg"
        img_path.write_bytes(jpeg)
        try:
            return engine.prepare_frame(body.upload_id, frame, str(img_path), progress_cb)
        finally:
            img_path.unlink(missing_ok=True)

    registry.submit(job, fn)
    return {"job_id": job.id}


@router.post("/jobs/exemplar", status_code=202)
def exemplar(request: Request, body: ExemplarRequest):
    up = _video_upload(request, body.upload_id)

    engine = request.app.state.engine
    registry = request.app.state.jobs
    job = registry.create("exemplar")
    frame = max(0, min(body.frame, (up.frames or 1) - 1))

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        progress_cb(0.1, "extracting frame")
        jpeg = media.extract_frame_jpeg(str(up.path), frame)
        img_path = config.TMP_DIR / f"{job.id}_exemplar.jpg"
        img_path.write_bytes(jpeg)
        try:
            desc = f"searching '{body.text}'" if body.text else f"finding similar objects ({body.method})"
            progress_cb(0.4, desc)
            r = engine.find_similar(
                str(img_path),
                box_xyxy=body.box,
                neg_boxes=body.neg_boxes,
                method=body.method,
                text=body.text,
                polygons=body.polygons,
                cache_key=(body.upload_id, frame),
            )
        finally:
            img_path.unlink(missing_ok=True)
        r["frame"] = frame
        return r

    registry.submit(job, fn)
    return {"job_id": job.id}


@router.post("/jobs/click-preview", status_code=202)
def click_preview(request: Request, body: ClickPreviewRequest):
    """Clicks on one frame -> the mask outline the tracker would seed with."""
    up = _video_upload(request, body.upload_id)

    engine = request.app.state.engine
    registry = request.app.state.jobs
    job = registry.create("click-preview")
    frame = max(0, min(body.frame, (up.frames or 1) - 1))

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        progress_cb(0.1, "extracting frame")
        jpeg = media.extract_frame_jpeg(str(up.path), frame)
        img_path = config.TMP_DIR / f"{job.id}_click.jpg"
        img_path.write_bytes(jpeg)
        try:
            progress_cb(0.4, "segmenting from clicks")
            r = engine.click_preview(str(img_path), body.points, body.labels, body.box,
                                     cache_key=(body.upload_id, frame))
        finally:
            img_path.unlink(missing_ok=True)
        r["frame"] = frame
        return r

    registry.submit(job, fn)
    return {"job_id": job.id}


@router.post("/jobs/video-track", status_code=202)
def video_track(request: Request, body: MultiTrackRequest):
    up = _video_upload(request, body.upload_id)

    engine = request.app.state.engine
    registry = request.app.state.jobs
    job = registry.create("video-track")
    # the clip's own length is the ceiling; nothing shorter is imposed
    max_frames = max(config.MIN_FRAMES, min(body.max_frames, up.frames or config.UNKNOWN_LENGTH))
    result_stem = str(config.RESULT_DIR / job.id)
    req = body.model_dump()
    req["max_frames"] = max_frames

    # resolve class ids against the uploaded classes.txt here, so the engine stays
    # unaware of the label set and the result JSON reads without it
    store = request.app.state.classes
    used: dict[int, str] = {}
    for obj in req["objects"]:
        cid = obj.get("cls")
        name = store.name(cid)
        obj["class_name"] = name
        if cid is not None:
            if name is None:
                raise HTTPException(
                    400, f"class id {cid} is out of range for the current classes.txt "
                         f"({store.count()} classes) — re-upload it or clear the class")
            used[cid] = name
    req["class_names"] = used

    clips = request.app.state.clips
    runs = request.app.state.runs

    # one history record per job — pressing "detect more objects" and tracking
    # again is a new job, so it gets its own record rather than replacing this
    runs.start(
        job.id,
        upload_id=body.upload_id,
        source=up.name or Path(up.path).name,
        width=up.width, height=up.height,
        source_frames=up.frames, fps=up.fps,
        settings={"max_frames": max_frames, "threshold": body.threshold,
                  "bidirectional": body.bidirectional},
        class_names=used,
        objects=req["objects"],
    )

    def _track(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        prepped = clips.get(body.upload_id, max_frames)
        if prepped is not None and prepped.image_size == engine.tracker_image_size():
            # the CPU preamble already ran while the user was annotating
            progress_cb(0.02, "using prepared frames")
            r = engine.run_multitrack(str(prepped.clip_path), req, result_stem,
                                      progress_cb, cancel,
                                      images=prepped.images[:max_frames])
            r["tracked_video_url"] = _files_url(r.pop("served_path"))
            return r

        progress_cb(0.01, "trimming clip")
        trimmed = config.TMP_DIR / f"{job.id}_trim.mp4"
        n = media.trim(str(up.path), max_frames, trimmed)
        if n <= 0:
            raise RuntimeError("trim produced no frames")
        try:
            r = engine.run_multitrack(str(trimmed), req, result_stem, progress_cb, cancel)
        finally:
            trimmed.unlink(missing_ok=True)
        r["tracked_video_url"] = _files_url(r.pop("served_path"))
        return r

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        try:
            r = _track(progress_cb, cancel)
        except BaseException as exc:
            runs.fail(job.id, f"{type(exc).__name__}: {exc}")
            raise
        runs.finish(job.id, r)
        return r

    registry.submit(job, fn)
    return {"job_id": job.id}


@router.post("/jobs/segment-track", status_code=202)
def segment_track(request: Request, body: SegmentTrackRequest):
    """Track objects across frames start..end of a video from their labelled
    frames. Unlike video-track, it covers just that range (not 0..max_frames),
    renders nothing, and returns the mask outlines as well as the boxes; frame
    numbers in the result are clip-relative, so absolute = `start` + i."""
    up = _video_upload(request, body.upload_id)
    last = (up.frames or 0) - 1
    if last >= 0 and body.start > last:
        raise HTTPException(400, f"frame {body.start} is past the end of the video (0..{last})")
    end = min(body.end, last) if last >= 0 else body.end
    objects = [{"name": o.name,
                "seeds": [{**s.model_dump(), "frame": min(s.frame, end) - body.start} for s in o.seeds]}
               for o in body.objects]

    engine = request.app.state.engine
    registry = request.app.state.jobs
    job = registry.create("segment-track")

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        progress_cb(0.01, "cutting the segment")
        clip = config.TMP_DIR / f"{job.id}_segment.mp4"
        n = media.cut(up.path, body.start, end, float(up.fps or 20.0), clip)
        if n <= 0:
            raise RuntimeError(f"frames {body.start}-{end} could not be decoded")
        try:
            r = engine.track_segment(str(clip), objects, progress_cb, cancel)
        finally:
            clip.unlink(missing_ok=True)
        r["start"] = body.start
        return r

    registry.submit(job, fn)
    return {"job_id": job.id}


@router.get("/jobs/{job_id}")
def job_status(request: Request, job_id: str):
    for registry in (request.app.state.jobs, request.app.state.cpu_jobs, request.app.state.qwen_jobs):
        snap = registry.snapshot(job_id)
        if snap is not None:
            return snap
    raise HTTPException(404, "unknown job")


@router.delete("/jobs/{job_id}")
def cancel_job(request: Request, job_id: str):
    for registry in (request.app.state.jobs, request.app.state.cpu_jobs, request.app.state.qwen_jobs):
        if registry.request_cancel(job_id):
            return JSONResponse({"cancelled": True})
    raise HTTPException(404, "unknown or already-finished job")
