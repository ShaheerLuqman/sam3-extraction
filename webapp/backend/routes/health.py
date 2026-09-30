from __future__ import annotations

from fastapi import APIRouter, Request

from .. import config, media

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    engine = request.app.state.engine
    h = engine.health()
    return {
        **h,
        "ffmpeg": media.FFMPEG_AVAILABLE,
        # whichever Qwen model is loaded (they take turns on the Qwen GPU)
        "qwen": next((w.status() for w in (request.app.state.qwen, request.app.state.qwen_vlm)
                      if w.status()["state"] != "off"), request.app.state.qwen.status()),
        "config": config.public_config(),
    }
