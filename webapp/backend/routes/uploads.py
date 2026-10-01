from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from .. import config, media
from ..schemas import UploadFromUrlRequest

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

    return await _register(request, dest, kind, Path(file.filename or "").name or dest.name)


@router.post("/uploads/from-url")
async def create_upload_from_url(request: Request, body: UploadFromUrlRequest) -> dict:
    """Like /uploads, but the server fetches the video from `url` itself (an S3
    presigned URL, say), so it never has to pass through the caller's network.
    Videos only; the response is the same as /uploads'."""
    if urlparse(body.url).scheme not in ("http", "https"):
        raise HTTPException(400, "url must be http(s)")
    name = Path(body.name or urlparse(body.url).path).name
    ext = Path(name).suffix.lower() or ".mp4"
    if ext not in _VIDEO_EXT:
        raise HTTPException(415, f"unsupported video type: {name!r}")
    dest = config.UPLOAD_DIR / f"{_tmp_name()}{ext}"
    try:
        await run_in_threadpool(_download, body.url, dest)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except (httpx.HTTPError, OSError) as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(502, f"could not download the video: {exc}") from exc
    return await _register(request, dest, "video", name or dest.name)


def _download(url: str, dest: Path) -> None:
    size = 0
    with httpx.stream("GET", url, timeout=httpx.Timeout(30, read=300), follow_redirects=True) as res:
        if res.is_error:
            raise HTTPException(502, f"could not download the video ({res.status_code})")
        with dest.open("wb") as out:
            for chunk in res.iter_bytes(1 << 20):
                size += len(chunk)
                if size > config.UPLOAD_MAX_BYTES:
                    raise HTTPException(413, "file exceeds the 500 MB limit")
                out.write(chunk)


async def _register(request: Request, dest: Path, kind: str, original: str) -> dict:
    """Probe a file saved under UPLOAD_DIR and add it to the upload store."""
    store = request.app.state.uploads
    try:
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
