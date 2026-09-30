from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from .. import config, media

router = APIRouter()

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@router.post("/uploads")
async def create_upload(request: Request, file: UploadFile) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    ctype = (file.content_type or "").lower()
    if ext in _VIDEO_EXT or ctype.startswith("video/"):
        kind = "video"
    elif ext in _IMAGE_EXT or ctype.startswith("image/"):
        kind = "image"
    else:
        raise HTTPException(415, f"unsupported file type: {file.filename!r} ({ctype})")

    if not ext:
        ext = ".mp4" if kind == "video" else ".jpg"
    dest = config.UPLOAD_DIR / f"{_tmp_name()}{ext}"

    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > config.UPLOAD_MAX_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "file exceeds the 500 MB limit")
            out.write(chunk)

    store = request.app.state.uploads
    try:
        original = Path(file.filename or "").name or dest.name
        if kind == "video":
            meta = await run_in_threadpool(media.probe, dest)
            up = store.add("video", dest, meta["width"], meta["height"], name=original,
                           frames=meta["frames"], fps=meta["fps"], duration=meta["duration"])
        else:
            w, h = await run_in_threadpool(media.image_size, dest)
            up = store.add("image", dest, w, h, name=original)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    return up.public()


@router.get("/uploads/{upload_id}/frame/{idx}.jpg")
async def get_frame(request: Request, upload_id: str, idx: int) -> Response:
    up = request.app.state.uploads.get(upload_id)
    if up is None:
        raise HTTPException(404, "unknown upload")
    if up.kind != "video":
        raise HTTPException(400, "not a video")
    if idx < 0 or (up.frames and idx >= up.frames):
        raise HTTPException(400, f"frame {idx} out of range (0..{(up.frames or 1) - 1})")
    try:
        jpeg = await run_in_threadpool(media.extract_frame_jpeg, up.path, idx)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


def _tmp_name() -> str:
    import uuid
    return uuid.uuid4().hex[:12]
