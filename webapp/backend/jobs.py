"""In-memory job registry backed by a single worker thread.

Every model call (image or video) runs on the one executor thread, which also
does the initial model load. That single thread IS the serialization the SAM3
predictor needs — its `hotstart_*` / `score_threshold_detection` attributes and
the CUDA context are shared global state with no internal locking.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import config

log = logging.getLogger("sam3webapp.jobs")

Status = str  # "queued" | "running" | "done" | "error"


@dataclass
class Job:
    id: str
    kind: str
    status: Status = "queued"
    progress: float = 0.0
    stage: str = "queued"
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        # max_workers=1 -> model load + all inference serialized on one thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3-infer")
        self._running: Optional[str] = None
        self._pending: list[str] = []

    # -- lifecycle ---------------------------------------------------------- #
    def create(self, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._pending.append(job.id)
        return job

    def submit(self, job: Job, fn: Callable[[Callable[[float, str], None], threading.Event], dict]) -> Future:
        """`fn(progress_cb, cancel_event) -> result dict`, run on the worker thread."""
        return self._executor.submit(self._run, job, fn)

    def submit_bare(self, fn: Callable[[], Any]) -> Future:
        """Run a no-arg callable on the worker thread (used for model load)."""
        return self._executor.submit(fn)

    def _run(self, job: Job, fn: Callable) -> dict:
        with self._lock:
            self._running = job.id
            if job.id in self._pending:
                self._pending.remove(job.id)
        if job.cancel.is_set():
            self.update(job.id, status="error", error="cancelled before start", stage="cancelled")
            with self._lock:
                self._running = None
            return {}
        self.update(job.id, status="running", stage="starting", progress=0.0)

        def progress_cb(frac: float, stage: str) -> None:
            self.update(job.id, progress=max(0.0, min(1.0, float(frac))), stage=stage)

        try:
            result = fn(progress_cb, job.cancel)
            self.update(job.id, status="done", progress=1.0, stage="done", result=result)
            return result
        except Exception as exc:  # noqa: BLE001 - surface everything to the client
            log.error("job %s (%s) failed:\n%s", job.id, job.kind, traceback.format_exc())
            self.update(job.id, status="error", error=f"{type(exc).__name__}: {exc}", stage="error")
            return {}
        finally:
            with self._lock:
                self._running = None

    # -- reads/writes ----------------------------------------------------- #
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            job.updated_at = time.time()

    def snapshot(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status == "queued":
                try:
                    queued_ahead = self._pending.index(job_id)
                except ValueError:
                    queued_ahead = 0
                if self._running is not None:
                    queued_ahead += 1
            else:
                queued_ahead = 0
            return {
                "id": job.id,
                "kind": job.kind,
                "status": job.status,
                "progress": round(job.progress, 4),
                "stage": job.stage,
                "queued_ahead": queued_ahead,
                "result": job.result,
                "error": job.error,
            }

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in ("done", "error"):
                return False
            job.cancel.set()
            if job.status == "queued":
                job.status = "error"
                job.error = "cancelled"
                job.stage = "cancelled"
                if job_id in self._pending:
                    self._pending.remove(job_id)
            return True

    # -- housekeeping --------------------------------------------------- #
    def gc(self, ttl: float, keep_last: int) -> None:
        now = time.time()
        with self._lock:
            finished = [jid for jid in self._order
                        if (j := self._jobs.get(jid)) and j.status in ("done", "error")]
            drop = set(finished[:-keep_last]) if len(finished) > keep_last else set()
            for jid in list(self._jobs):
                j = self._jobs[jid]
                if jid in drop or (j.status in ("done", "error") and now - j.updated_at > ttl):
                    self._jobs.pop(jid, None)
                    if jid in self._order:
                        self._order.remove(jid)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
