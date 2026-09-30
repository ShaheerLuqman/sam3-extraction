"""Multiple class segmentation: several steps marked on a reference video -> where each
happens in up to three other videos. Clips are cut with the segx routes."""
from __future__ import annotations

import threading
from typing import Callable

from fastapi import APIRouter, HTTPException, Request

from .. import mcseg
from ..schemas import McsegDescribeRequest, McsegSearchRequest
from .jobs_routes import _video_upload
from .segx import _clips, _registry

router = APIRouter()


@router.post("/jobs/mcseg-describe", status_code=202)
def describe(request: Request, body: McsegDescribeRequest):
    """The VLM names and describes each step, from up to three of its clips."""
    clips = [_clips(request, ids, "step clip") for ids in body.classes]
    if any(not c for c in clips):
        raise HTTPException(400, "every step to describe needs a clip")
    st = request.app.state
    registry = _registry(request)
    job = registry.create("mcseg-describe")

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        return mcseg.describe(st.qwen_vlm, st.engine, clips, job.id, progress_cb, cancel)

    registry.submit(job, fn)
    return {"job_id": job.id}


@router.post("/jobs/mcseg-search", status_code=202)
def search(request: Request, body: McsegSearchRequest):
    """Embed the reference and the targets, pick candidates per class, classify them."""
    ref = _video_upload(request, body.ref_upload_id)
    targets = [_video_upload(request, i) for i in body.target_ids]
    clips = _clips(request, [c.clip_id for c in body.classes], "step clip")
    classes = [{"name": c.name, "description": c.description, "range": (c.start, c.end), "clip": clip}
               for c, clip in zip(body.classes, clips)]
    st = request.app.state
    registry = _registry(request)
    job = registry.create("mcseg-search")
    payload = body.model_dump()

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        return mcseg.search(st.qwen, st.qwen_vlm, st.engine, ref, classes, targets, payload,
                            job.id, progress_cb, cancel)

    registry.submit(job, fn)
    return {"job_id": job.id}
