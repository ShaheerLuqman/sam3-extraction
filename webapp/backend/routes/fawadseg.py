"""Frame extraction fawad segment: the research's class_N_desc_vlm_hints pipeline, as is."""
from __future__ import annotations

import threading
from typing import Callable

from fastapi import APIRouter, HTTPException, Request

from .. import config, fawadseg
from ..schemas import FawadSegRequest
from .jobs_routes import _video_upload

router = APIRouter()


@router.post("/jobs/fawadseg-run", status_code=202)
def run(request: Request, body: FawadSegRequest):
    """Frames -> embeddings -> kNN candidates -> VLM, on the Qwen GPU. Minutes."""
    ref = _video_upload(request, body.ref_upload_id)
    target = _video_upload(request, body.upload_id)
    payload = body.model_dump()
    try:
        fawadseg.validate(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    st = request.app.state
    # Qwen has its own GPU and thread; on a shared GPU it takes turns with SAM 3
    registry = st.qwen_jobs if config.qwen_dedicated() else st.jobs
    job = registry.create("fawadseg-run")

    def fn(progress_cb: Callable[[float, str], None], cancel: threading.Event) -> dict:
        return fawadseg.run(st.engine, [st.qwen, st.qwen_vlm], ref, target, payload, job.id,
                            progress_cb, cancel)

    registry.submit(job, fn)
    return {"job_id": job.id}
