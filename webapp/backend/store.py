"""In-memory registry of uploaded files. Not persisted; a restart drops it."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Upload:
    id: str
    kind: str  # "image" | "video"
    path: Path
    width: int
    height: int
    #: the name the user's file had, kept for the run history
    name: str = ""
    frames: Optional[int] = None
    fps: Optional[float] = None
    duration: Optional[float] = None
    created_at: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = {"upload_id": self.id, "kind": self.kind, "name": self.name,
             "width": self.width, "height": self.height}
        if self.kind == "video":
            d.update(frames=self.frames, fps=self.fps, duration=self.duration)
        return d


class UploadStore:
    def __init__(self) -> None:
        self._items: dict[str, Upload] = {}
        self._lock = threading.Lock()

    def add(self, kind: str, path: Path, width: int, height: int, **extra) -> Upload:
        up = Upload(id=uuid.uuid4().hex[:12], kind=kind, path=path,
                    width=width, height=height, **extra)
        with self._lock:
            self._items[up.id] = up
        return up

    def get(self, upload_id: str) -> Optional[Upload]:
        with self._lock:
            return self._items.get(upload_id)

    def sweep(self, ttl: float) -> None:
        now = time.time()
        with self._lock:
            for uid in list(self._items):
                up = self._items[uid]
                if now - up.created_at > ttl:
                    up.path.unlink(missing_ok=True)
                    self._items.pop(uid, None)
