from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, UploadFile

from ..classes import parse

router = APIRouter()

MAX_FILE_BYTES = 256 * 1024


@router.get("/classes")
def get_classes(request: Request) -> dict:
    return request.app.state.classes.public()


@router.post("/classes")
async def upload_classes(request: Request, file: UploadFile) -> dict:
    """Replace the label set with an uploaded classes.txt (one name per line)."""
    raw = await file.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(413, "classes file is too large (256 KB limit)")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")

    names = parse(text)
    if not names:
        raise HTTPException(400, f"{file.filename or 'the file'} has no class names in it")
    dupes = {n for n in names if names.count(n) > 1}
    out = request.app.state.classes.replace(names, file.filename)
    if dupes:
        out["warning"] = ("duplicate class names: "
                          + ", ".join(sorted(dupes)[:5])
                          + (" …" if len(dupes) > 5 else ""))
    return out


@router.delete("/classes")
def clear_classes(request: Request) -> dict:
    return request.app.state.classes.clear()
