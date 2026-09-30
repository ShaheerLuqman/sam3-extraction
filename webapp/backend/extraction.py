"""Frame extraction: find the stretches of a video that look like a few reference images.

The method is the one that held up in qwen_vl/extraction_project (see the README
section): embed every `stride`-th frame and each reference with
Qwen3-VL-Embedding-8B, then score each frame by its best match to any reference,
where each reference's cosine similarities are z-scored against the video's own
distribution first. That normalisation is what lets references of different
"typicality" share one threshold, and it beat raw cosine on every step tested.

Three pieces, with different costs:

  * ensure_embeddings — GPU. Only for what is not cached yet. A request to the
    long-lived QwenWorker (qwen_embed_worker.py in the qwen venv), on its own GPU
    (GPU 0) and its own job thread, so it runs alongside SAM 3 on GPU 1.
  * score — CPU, milliseconds. One z-score row per reference against the cached
    video embedding. A reference picked from the video itself is just a row of
    that embedding, so refining never touches the GPU.
  * export — CPU. Writes the selection as JSON, a clip of the selected frames,
    and optionally a ZIP of the frames as JPEGs.

Thresholding, smoothing and segment cleanup happen in the browser, on the rows,
so the threshold slider is live.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import config, media

log = logging.getLogger("sam3webapp.extract")

ProgressCb = Callable[[float, str], None]

WORKER = Path(__file__).resolve().parent / "qwen_embed_worker.py"


class Cancelled(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def cache_key(instruction: str, stride: int) -> str:
    """Everything that changes an embedding. A different key is a re-embed."""
    raw = f"{config.QWEN_EMB_MODEL}|{config.QWEN_MAX_SIDE}|{stride}|{instruction.strip()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


_digests: dict[Path, tuple[float, str]] = {}
_digest_lock = threading.Lock()


def digest(path: Path) -> str:
    """Content hash of an uploaded file. Caches are keyed on this rather than the
    upload id, so the same video uploaded again (or after a restart, which forgets
    every upload) reuses its embedding instead of spending minutes on the GPU."""
    path = Path(path)
    mtime = path.stat().st_mtime
    with _digest_lock:
        hit = _digests.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    d = h.hexdigest()[:16]
    with _digest_lock:
        _digests[path] = (mtime, d)
    return d


def video_emb_path(up, key: str) -> Path:
    return config.EMBED_DIR / f"{digest(up.path)}__{key}.npz"


def image_emb_path(up, key: str) -> Path:
    # the stride is irrelevant to a single image, but keying on it keeps one
    # scheme; an image embeds in well under a second once the model is up
    return config.EMBED_DIR / f"img_{digest(up.path)}__{key}.npy"


class _VideoEmbCache:
    """The last couple of video embeddings, as float32, so scoring skips the load."""

    def __init__(self, size: int = 2) -> None:
        self._items: "OrderedDict[Path, tuple]" = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, path: Path) -> tuple[np.ndarray, np.ndarray, int]:
        with self._lock:
            hit = self._items.get(path)
            if hit is not None and hit[0] == path.stat().st_mtime:
                self._items.move_to_end(path)
                return hit[1:]
        z = np.load(path)
        item = (path.stat().st_mtime, z["frames"].astype(np.int64),
                z["emb"].astype(np.float32), int(z["decoded"]))
        with self._lock:
            self._items[path] = item
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return item[1:]


_cache = _VideoEmbCache()


# --------------------------------------------------------------------------- #
# embedding (GPU, via the qwen worker)
# --------------------------------------------------------------------------- #
def ensure_embeddings(worker: "QwenWorker", engine, video_up, image_ups: list, stride: int,
                      instruction: str, progress_cb: ProgressCb, cancel: threading.Event) -> dict:
    """Embed whatever of (video, reference images) is not cached for this key.

    Dedicated GPU (the default: Qwen on GPU 0, SAM 3 on GPU 1): just a request to
    the worker, which stays warm afterwards. Shared GPU: SAM 3 is parked in RAM
    for the run and the worker is stopped before SAM 3 moves back.
    """
    key = cache_key(instruction, stride)
    vpath = video_emb_path(video_up, key)
    todo_imgs = [u for u in image_ups if not image_emb_path(u, key).exists()]
    need_video = not vpath.exists()

    if need_video or todo_imgs:
        req = {
            "instruction": instruction,
            "video": ({"path": str(video_up.path), "stride": stride, "out": str(vpath)}
                      if need_video else None),
            "images": [{"path": str(u.path), "out": str(image_emb_path(u, key))}
                       for u in todo_imgs],
        }
        if config.qwen_dedicated():
            worker.request(req, progress_cb, cancel)
        else:
            progress_cb(0.01, "moving SAM 3 off the GPU")
            engine.offload()
            try:
                worker.request(req, progress_cb, cancel)
            finally:
                worker.stop()  # SAM 3 needs the memory back
                progress_cb(0.98, "moving SAM 3 back onto the GPU")
                engine.restore()

    frames, _, decoded = _cache.get(vpath)
    return {"key": key, "stride": stride, "decoded_frames": decoded,
            "embedded_frames": int(len(frames)),
            "embedded_now": {"video": need_video, "images": len(todo_imgs)}}


# per worker kind: (script, spare GB left outside vLLM's budget, least budget GB, cap)
#  embed: with only 0.7 GB spare it OOMed on the first batch (vLLM's budget misses the
#         vision encoder's activations on a big batch) — 2.5 GB, as the research had.
#  vlm:   16.3 GB of weights and a 16k-token prompt (2.25 GB of KV cache) need a
#         larger share: at 0.88 of the card it could not start; the research ran 0.92.
_KINDS = {
    "embed": (WORKER, 2.5, config.QWEN_MIN_GPU_GB, 0.9),
    "vlm": (Path(__file__).resolve().parent / "qwen_vlm_worker.py", 1.2, 21.0, 0.93),
}


def _gpu_util(kind: str = "embed") -> float:
    """vLLM's gpu_memory_utilization, sized to what is actually free right now."""
    _, spare, least, cap = _KINDS[kind]
    free, total = _qwen_gpu_mem()
    gib = 1024 ** 3
    budget = free - spare * gib
    if budget < least * gib:
        raise RuntimeError(
            f"only {free / gib:.1f} GB is free on GPU {config.QWEN_GPU}; the Qwen "
            f"{'VLM' if kind == 'vlm' else 'embedding model'} needs about {least:.0f} GB. "
            "Something else is using that GPU.")
    return round(min(cap, budget / total), 3)


def _qwen_gpu_mem() -> tuple[int, int]:
    """(free, total) bytes on Qwen's GPU. nvidia-smi, because the backend's own
    torch can only see GPU 1 (CUDA_VISIBLE_DEVICES), and its index order is the
    PCI order the worker is launched with."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits",
         "-i", config.QWEN_GPU], capture_output=True, text=True, timeout=15)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi could not read GPU {config.QWEN_GPU}: {out.stderr.strip()}")
    free, total = (int(v) * 1024 ** 2 for v in out.stdout.strip().split(","))
    return free, total


class QwenWorker:
    """A long-lived Qwen process: started on first use, kept warm for
    `QWEN_KEEP_WARM_S` after its last request, then stopped to free the GPU.

    Two kinds: "embed" (Qwen3-VL-Embedding-8B) and "vlm" (Qwen3-VL-8B-Instruct).
    They do not fit on one 24 GB card together, so starting one stops the other.
    One request at a time — callers are a single-thread JobRegistry.
    """

    _all: list["QwenWorker"] = []

    def __init__(self, kind: str = "embed") -> None:
        self.kind = kind
        QwenWorker._all.append(self)
        # reentrant: a cancel inside request() stops the worker on the same thread
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._log_path = config.TMP_DIR / f"qwen_{kind}_worker.log"
        self._timer: Optional[threading.Timer] = None
        self.state = "off"  # off | starting | ready | busy
        self.last_used = 0.0

    # -- lifecycle ----------------------------------------------------------- #
    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self, progress_cb: ProgressCb, cancel: threading.Event) -> None:
        if not config.QWEN_PYTHON.is_file():
            raise RuntimeError(f"the Qwen venv is missing: {config.QWEN_PYTHON} "
                               "(set SAM3_QWEN_PYTHON to its python)")
        for other in QwenWorker._all:  # one Qwen model on the card at a time
            if other is not self and other._alive():
                progress_cb(0.01, f"unloading the Qwen {other.kind} model")
                other.stop()
        label = "Qwen3-VL-8B (VLM)" if self.kind == "vlm" else "Qwen3-VL-Embedding"
        progress_cb(0.02, f"loading {label} on GPU {config.QWEN_GPU}")
        if self.kind == "vlm":
            cfg = {"model": config.QWEN_VLM_MODEL, "gpu_util": _gpu_util("vlm"),
                   "max_len": 16384, "max_videos": 12}
        else:
            cfg = {"model": config.QWEN_EMB_MODEL, "gpu_util": _gpu_util("embed"),
                   "max_side": config.QWEN_MAX_SIDE}
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "CUDA_VISIBLE_DEVICES": config.QWEN_GPU}
        log.info("starting qwen %s worker on GPU %s (util %s)", self.kind, config.QWEN_GPU,
                 cfg["gpu_util"])
        self.state = "starting"
        err = self._log_path.open("w")
        # its own process group: vLLM forks an EngineCore that holds the GPU
        # memory, and stopping has to take that down too, not just the parent
        self._proc = subprocess.Popen(
            [str(config.QWEN_PYTHON), str(_KINDS[self.kind][0]), json.dumps(cfg)], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
            start_new_session=True)
        err.close()
        lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._lines = lines
        proc = self._proc

        def pump() -> None:
            for line in proc.stdout:  # type: ignore[union-attr]
                lines.put(line.rstrip("\n"))
            lines.put(None)

        threading.Thread(target=pump, daemon=True, name="qwen-stdout").start()
        self._await(lambda line: line == "READY", None, cancel)
        self.state = "ready"
        log.info("qwen %s worker ready", self.kind)

    def _await(self, done, progress_cb: Optional[ProgressCb], cancel: threading.Event) -> None:
        """Read the worker's replies until `done(line)`; ERROR raises, exit raises."""
        while True:
            if cancel.is_set():
                self.stop()
                raise Cancelled("cancelled")
            try:
                line = self._lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                tail = self._log_path.read_text()[-1500:] if self._log_path.exists() else ""
                log.error("qwen worker exited; log %s:\n%s", self._log_path, tail)
                last = [l for l in tail.splitlines() if l.strip()]
                self.stop()
                raise RuntimeError(f"the Qwen worker exited: {last[-1] if last else 'no output'}")
            if line.startswith("PROGRESS ") and progress_cb:
                _, frac, stage = line.split(" ", 2)
                # the worker's own 0..1 maps into this job's 0.05..0.97
                progress_cb(0.05 + 0.92 * float(frac), stage)
            elif line.startswith("ERROR "):
                raise RuntimeError(f"Qwen embedding failed: {line[6:]}")
            elif done(line):
                return

    def request(self, req: dict, progress_cb: ProgressCb, cancel: threading.Event) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            try:
                if not self._alive():
                    self._start(progress_cb, cancel)
                self.state = "busy"
                self._proc.stdin.write(json.dumps(req) + "\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
                self._await(lambda line: line == "DONE", progress_cb, cancel)
            finally:
                if self._alive():
                    self.state = "ready"
                    self.last_used = time.time()
                    self._arm_idle_stop()

    def _arm_idle_stop(self) -> None:
        if config.QWEN_KEEP_WARM_S <= 0:
            self._stop_locked()
            return
        self._timer = threading.Timer(config.QWEN_KEEP_WARM_S, self._idle_stop)
        self._timer.daemon = True
        self._timer.start()

    def _idle_stop(self) -> None:
        with self._lock:  # a request in flight holds the lock, so this waits it out
            if time.time() - self.last_used >= config.QWEN_KEEP_WARM_S - 1:
                log.info("qwen %s worker idle for %ss — stopping to free GPU %s",
                         self.kind, config.QWEN_KEEP_WARM_S, config.QWEN_GPU)
                self._stop_locked()

    def stop(self) -> None:
        """Stop the worker. From another thread this waits for a running request."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._proc is not None:
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            _reap(self._proc)
            self._proc = None
        self.state = "off"

    def status(self) -> dict:
        return {"kind": self.kind, "gpu": config.QWEN_GPU, "dedicated": config.qwen_dedicated(),
                "state": self.state if self._alive() else "off",
                "keep_warm_s": config.QWEN_KEEP_WARM_S,
                "idle_s": round(time.time() - self.last_used) if self.state == "ready" else None}


def _reap(proc: subprocess.Popen) -> None:
    """Make sure the worker's whole process group is gone, so its GPU memory is
    free. SIGTERM first, so vLLM can shut down cleanly."""
    pgid = proc.pid
    for sig, wait in ((signal.SIGTERM, 20.0), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.time() + wait
        while time.time() < deadline:
            try:
                proc.poll()  # reap the direct child, or it stays a zombie in the group
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
    log.warning("qwen worker group %s still alive after SIGKILL", pgid)


# --------------------------------------------------------------------------- #
# playback (CPU)
# --------------------------------------------------------------------------- #
def playback_copy(video_up, progress_cb: ProgressCb, fmt: str = "mp4") -> dict:
    """The browser's copy of the video for smooth segment playback, cached by content."""
    fps = float(video_up.fps or 20.0)
    dest = config.PROXY_DIR / f"{digest(video_up.path)}_{fps:g}.{fmt}"
    if not dest.exists():
        progress_cb(0.1, "making a playback copy")
        media.playback_copy(video_up.path, fps, dest, fmt)
    return {"url": f"/api/proxies/{dest.name}", "fps": fps, "format": fmt}


# --------------------------------------------------------------------------- #
# scoring (CPU)
# --------------------------------------------------------------------------- #
def score(video_up, key: str, refs: list[dict], images: dict) -> dict:
    """One z-scored similarity row per reference, over the embedded frames.

    refs: [{"kind": "image", "id": <image upload id>} | {"kind": "frame", "frame": N}]
    images: image upload id -> Upload, for the image refs
    """
    vpath = video_emb_path(video_up, key)
    if not vpath.exists():
        raise FileNotFoundError("this video has not been embedded with these settings yet")
    frames, V, decoded = _cache.get(vpath)
    rows = []
    for r in refs:
        if r["kind"] == "image":
            p = image_emb_path(images[r["id"]], key)
            if not p.exists():
                raise FileNotFoundError(f"reference image {r['id']} is not embedded yet")
            q = np.load(p).astype(np.float32)
        else:  # a frame of this video: its nearest embedded frame
            q = V[int(np.abs(frames - int(r["frame"])).argmin())]
        s = V @ q
        z = (s - s.mean()) / max(float(s.std()), 1e-6)
        rows.append(np.round(z, 3).tolist())
    return {"frames": frames.tolist(), "decoded_frames": decoded, "rows": rows}


# --------------------------------------------------------------------------- #
# export (CPU)
# --------------------------------------------------------------------------- #
def export(video_up, body: dict, result_stem: str, progress_cb: ProgressCb,
           cancel: threading.Event) -> dict:
    """Write the selection: JSON always, a clip of the selected frames, a JPEG ZIP."""
    segs = sorted((int(s["start"]), int(s["end"])) for s in body["segments"])
    fps = float(video_up.fps or 20.0)
    last = segs[-1][1]
    want_video, zip_every = body.get("video", True), int(body.get("zip_every") or 0)

    progress_cb(0.02, "reading the video")
    cap = cv2.VideoCapture(str(video_up.path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_up.path}")
    vw = raw_path = zf = None
    zip_path = Path(f"{result_stem}_frames.zip")
    written, zipped, k = 0, 0, 0
    try:
        idx = 0
        while idx <= last:
            if cancel.is_set():
                raise Cancelled("cancelled")
            while k < len(segs) and idx > segs[k][1]:
                k += 1
            inside = k < len(segs) and segs[k][0] <= idx <= segs[k][1]
            if not inside:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            if zip_every and (idx - segs[k][0]) % zip_every == 0:
                if zf is None:
                    zf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED)
                ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                zf.writestr(f"segment_{k + 1:03d}/frame_{idx:06d}.jpg", buf.tobytes())
                zipped += 1
            if want_video:
                if vw is None:
                    h, w = bgr.shape[:2]
                    from .engine import _video_writer

                    vw, raw_path = _video_writer(str(config.TMP_DIR / Path(result_stem).name),
                                                 fps, (w, h))
                label = f"segment {k + 1}/{len(segs)}  frame {idx}  {idx / fps:6.1f}s"
                cv2.rectangle(bgr, (0, 0), (min(bgr.shape[1], 12 + 11 * len(label)), 30),
                              (0, 0, 0), -1)
                cv2.putText(bgr, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)
                vw.write(bgr)
            written += 1
            if written % 50 == 0:
                progress_cb(0.05 + 0.8 * idx / max(last, 1), f"writing frame {idx}")
            idx += 1
    finally:
        cap.release()
        if vw is not None:
            vw.release()
        if zf is not None:
            zf.close()

    out: dict = {"segments": len(segs), "frames": written, "message": ""}
    if raw_path:
        progress_cb(0.88, "encoding the clip")
        served, mime, warn = media.finalize_video(raw_path, result_stem)
        out.update(video_url=f"/api/files/{Path(served).name}", mime=mime, codec_warning=warn)
    if zipped:
        out["zip_url"] = f"/api/files/{zip_path.name}"
        out["zipped"] = zipped

    progress_cb(0.97, "writing JSON")
    selected = [f for a, b in segs for f in range(a, b + 1)]
    scores = {int(s["start"]): s for s in body["segments"]}
    doc = {
        "source": video_up.name or Path(video_up.path).name,
        "fps": fps, "width": video_up.width, "height": video_up.height,
        "decoded_frames": body.get("decoded_frames"),
        "created_at": time.time(),
        "method": {"model": config.QWEN_EMB_MODEL, "score": "max over references of the "
                   "z-scored cosine similarity", **(body.get("settings") or {})},
        "references": body.get("references") or [],
        "segments": [
            {"index": i + 1, "start_frame": a, "end_frame": b, "frames": b - a + 1,
             "start_s": round(a / fps, 3), "end_s": round((b + 1) / fps, 3),
             # the scores the browser computed for this segment
             **{k2: v for k2, v in scores.get(a, {}).items() if k2 in ("peak", "mean", "best_ref")}}
            for i, (a, b) in enumerate(segs)
        ],
        "selected_frames": selected,
    }
    json_path = Path(f"{result_stem}.json")
    json_path.write_text(json.dumps(doc, indent=1))
    out["json_url"] = f"/api/files/{json_path.name}"
    if written < sum(b - a + 1 for a, b in segs):
        out["message"] = "the video ended before the last segment did; the clip is shorter"
    return out
