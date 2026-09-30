"""History of tracking runs — what was asked for, and what came back.

One record per `video-track` job, written when the job starts and updated when
it finishes. Going back from the results screen to detect more objects and
tracking again is a *new* job, so it gets its own record rather than
overwriting the first.

Persisted to var/runs.json so it survives a restart. The records are small (the
prompts, not the pixels); the outputs they point at live in var/results and are
swept on the normal TTL, so a record can outlive its files — `outputs_present`
says whether they are still there.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import config

log = logging.getLogger("sam3webapp.runs")

#: keep this many records; the oldest fall off. Records are ~KBs each.
MAX_RUNS = 200


def _now() -> float:
    return time.time()


class RunStore:
    """Newest-first history of tracking runs, persisted as one JSON file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path or config.RUNS_PATH)
        self._lock = threading.Lock()
        self._runs: list[dict] = []
        self._load()

    # -- persistence -------------------------------------------------- #
    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text())
            if isinstance(raw, list):
                self._runs = [r for r in raw if isinstance(r, dict) and r.get("id")]
        except FileNotFoundError:
            self._runs = []
        except Exception:  # noqa: BLE001 - a corrupt history must not stop the app
            log.warning("could not read %s — starting an empty history", self._path,
                        exc_info=True)
            self._runs = []

    def _save_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._runs, indent=1))
            tmp.replace(self._path)
        except Exception:  # noqa: BLE001 - history is not worth failing a run over
            log.warning("could not write %s", self._path, exc_info=True)

    # -- writes ------------------------------------------------------- #
    def start(self, run_id: str, **fields: Any) -> dict:
        """Record a run as it is queued. `fields` carries inputs and settings."""
        run = {
            "id": run_id,
            "status": "running",
            "created_at": _now(),
            "finished_at": None,
            "error": None,
            **fields,
        }
        with self._lock:
            self._runs.insert(0, run)
            del self._runs[MAX_RUNS:]
            self._save_locked()
        return run

    def finish(self, run_id: str, result: dict) -> None:
        self._update(run_id, status="done", finished_at=_now(), result=result)

    def fail(self, run_id: str, error: str) -> None:
        self._update(run_id, status="error", finished_at=_now(), error=error)

    def _update(self, run_id: str, **fields: Any) -> None:
        with self._lock:
            for r in self._runs:
                if r["id"] == run_id:
                    r.update(fields)
                    self._save_locked()
                    return

    def delete(self, run_id: str, drop_files: bool = True) -> bool:
        with self._lock:
            before = len(self._runs)
            self._runs = [r for r in self._runs if r["id"] != run_id]
            gone = len(self._runs) < before
            if gone:
                self._save_locked()
        if gone and drop_files:
            for p in config.RESULT_DIR.glob(f"{run_id}.*"):
                p.unlink(missing_ok=True)
        return gone

    # -- reads -------------------------------------------------------- #
    @staticmethod
    def _outputs_present(run: dict) -> bool:
        res = run.get("result") or {}
        for key in ("tracked_video_url", "json_url"):
            url = res.get(key)
            if url and not (config.RESULT_DIR / Path(str(url)).name).exists():
                return False
        return bool(res.get("tracked_video_url"))

    def _public(self, run: dict, *, full: bool) -> dict:
        out = {k: v for k, v in run.items() if full or k != "objects"}
        out["outputs_present"] = self._outputs_present(run)
        if not full:
            out["object_count"] = len(run.get("objects") or [])
        return out

    def list(self) -> list[dict]:
        """Every run, newest first, without the (bulky) prompt payloads."""
        with self._lock:
            return [self._public(r, full=False) for r in self._runs]

    def get(self, run_id: str) -> Optional[dict]:
        with self._lock:
            for r in self._runs:
                if r["id"] == run_id:
                    return self._public(r, full=True)
        return None
