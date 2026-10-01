"""cv2 / ffmpeg helpers: probe, frame extraction, trimming, browser-safe finalize.

These are CPU-only and run in Starlette's default threadpool, never on the
inference worker thread.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2

from . import config

log = logging.getLogger("sam3webapp.media")


def _pick_ffmpeg() -> tuple[str | None, bool]:
    """Return (ffmpeg_path, has_libx264). Prefer a build that can encode H.264
    for the browser; the conda ffmpeg on this box has a broken libopenh264 and
    no libx264, so an explicit /usr/bin/ffmpeg check comes first."""
    env = os.environ.get("SAM3_FFMPEG")
    candidates = [env] if env else []
    candidates += ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", shutil.which("ffmpeg")]
    seen = set()
    fallback = None
    for c in candidates:
        if not c or c in seen or not Path(c).exists():
            continue
        seen.add(c)
        fallback = fallback or c
        try:
            out = subprocess.run([c, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=10).stdout
            if "libx264" in out:
                return c, True
        except Exception:  # noqa: BLE001
            continue
    return fallback, False


FFMPEG, FFMPEG_HAS_X264 = _pick_ffmpeg()
FFMPEG_AVAILABLE = FFMPEG is not None
log.info("ffmpeg=%s libx264=%s", FFMPEG, FFMPEG_HAS_X264)


def probe(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"width": w, "height": h, "frames": frames, "fps": round(fps, 3),
            "duration": round(frames / fps, 2) if fps else 0.0}


def image_size(path: str | Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = img.shape[:2]
    return w, h


def extract_frame_jpeg(path: str | Path, idx: int, quality: int = 90) -> bytes:
    """Return frame `idx` as JPEG bytes. Sequential read for small indices
    (seeking is keyframe-inaccurate on many codecs)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        if idx > config.SEQUENTIAL_SEEK_LIMIT:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        else:
            for _ in range(idx):
                if not cap.grab():
                    break
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise ValueError(f"frame {idx} out of range")
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("jpeg encode failed")
        return buf.tobytes()
    finally:
        cap.release()


def trim(path: str | Path, max_frames: int, out_path: str | Path) -> int:
    """Write the first `max_frames` frames of `path` to `out_path` (mp4).
    Returns the number of frames actually written."""
    out_path = Path(out_path)
    if FFMPEG_AVAILABLE:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast"] if FFMPEG_HAS_X264 else ["-c:v", "mpeg4", "-q:v", "3"]
        cmd = [FFMPEG, "-y", "-i", str(path), "-frames:v", str(max_frames),
               "-an", *vcodec, "-pix_fmt", "yuv420p", str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and out_path.exists():
            return probe(out_path)["frames"]
        log.warning("ffmpeg trim failed (%s), falling back to cv2:\n%s", r.returncode, r.stderr[-800:])

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while n < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        vw.write(bgr)
        n += 1
    vw.release()
    cap.release()
    return n


def finalize_video(raw_path: str | Path, dest_stem: str | Path) -> tuple[str, str, str | None]:
    """Produce a browser-playable clip. Prefer H.264 mp4; otherwise the raw
    VP8/webm from cv2 already plays in Chrome/Firefox/Edge, so keep it.
    Returns (served_path, mime, codec_warning)."""
    raw_path = Path(raw_path)

    if FFMPEG_AVAILABLE and FFMPEG_HAS_X264:
        dest = Path(f"{dest_stem}.mp4")
        cmd = [FFMPEG, "-y", "-i", str(raw_path), "-c:v", "libx264",
               "-preset", "veryfast", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(dest)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and dest.exists():
            return str(dest), "video/mp4", None
        log.warning("ffmpeg finalize failed (%s):\n%s", r.returncode, r.stderr[-800:])

    if raw_path.suffix == ".webm":  # cv2 VP8 — plays in all modern browsers except Safari
        dest = Path(f"{dest_stem}.webm")
        shutil.move(str(raw_path), str(dest))
        return str(dest), "video/webm", None

    # last resort: mp4v in an .mp4 — often won't play in Chrome
    if FFMPEG_AVAILABLE:  # try VP9 webm
        dest = Path(f"{dest_stem}.webm")
        r = subprocess.run([FFMPEG, "-y", "-i", str(raw_path), "-c:v", "libvpx-vp9",
                            "-b:v", "2M", "-pix_fmt", "yuv420p", str(dest)],
                           capture_output=True, text=True)
        if r.returncode == 0 and dest.exists():
            return str(dest), "video/webm", None
    dest = Path(f"{dest_stem}{raw_path.suffix}")
    shutil.move(str(raw_path), str(dest))
    return str(dest), "video/mp4", "could not transcode to a web codec — playback may fail."


def playback_copy(path: str | Path, fps: float, dest: str | Path, fmt: str = "mp4") -> None:
    """An H.264 copy the browser can play smoothly, frame-exact with cv2.

    Re-timed to a constant `fps` so frame N (in cv2's sequential count) sits at
    exactly N / fps; the source's own timestamps drift, which would make
    `currentTime` land on the wrong frame. A keyframe every second keeps seeks
    quick. Truncated sources are fine: ffmpeg complains about the tail and
    stops where cv2 stops.

    fmt "mp4" is H.264 (quickest to make); "webm" is VP9, for browsers without an
    H.264 decoder (VS Code's built-in browser, Chromium builds without the
    proprietary codecs). VP9 realtime runs ~240 frames/s at 640x480.
    """
    if not FFMPEG_AVAILABLE or (fmt == "mp4" and not FFMPEG_HAS_X264):
        raise RuntimeError("smooth playback needs an ffmpeg with libx264")
    dest = Path(dest)
    tmp = dest.with_suffix(f".tmp.{fmt}")
    gop = max(1, round(fps))
    if fmt == "webm":
        codec = ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
                 "-threads", "8", "-b:v", "0", "-crf", "34", "-g", str(gop)]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-g", str(gop),
                 "-movflags", "+faststart"]
    cmd = [FFMPEG, "-y", "-v", "error", "-i", str(path), "-an",
           "-vf", f"setpts=N/({fps}*TB),scale='min(1280,iw)':-2",
           "-r", str(fps), "-vsync", "cfr", *codec, "-pix_fmt", "yuv420p", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not make a playback copy: {r.stderr[-400:]}")
    tmp.replace(dest)


def cut(path: str | Path, start: int, end: int, fps: float, dest: str | Path) -> int:
    """Frames `start`..`end` (inclusive, cv2's sequential count) of `path` as their
    own near-lossless mp4 at the source rate. Returns the frames written.

    Picked by decoded-frame number, not by time, for the reason playback_copy
    re-times: the sources' timestamps drift, so `-ss` would land off by frames.
    """
    dest = Path(dest)
    if FFMPEG_AVAILABLE and FFMPEG_HAS_X264:
        cmd = [FFMPEG, "-y", "-v", "error", "-i", str(path), "-an",
               "-vf", f"select='between(n\\,{start}\\,{end})',setpts=N/({fps}*TB)",
               "-r", str(fps), "-vsync", "cfr", "-frames:v", str(end - start + 1),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size:
            return probe(dest)["frames"]
        log.warning("ffmpeg cut failed (%s), falling back to cv2:\n%s", r.returncode, r.stderr[-800:])

    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    try:
        for i in range(end + 1):
            if i < start:
                if not cap.grab():
                    break
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            vw.write(bgr)
            n += 1
    finally:
        vw.release()
        cap.release()
    return n


def cut_as_page(path: str | Path, start: int, end: int, fps: float, dest: str | Path,
                max_side: int) -> int:
    """`cut`, but the clip made the way the page makes one from a local MP4
    (lib/localVideo grabFrames + /segx/clip-frames): each frame scaled so its long
    side is at most `max_side` (even dimensions), saved as JPEG at quality 92,
    and the JPEGs encoded with encode_frames. Frames are picked by decoded-frame
    number, which is what the page's sample-table seeking lands on."""
    work = Path(config.TMP_DIR) / f"cut_{Path(dest).stem}"
    work.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    n = 0
    try:
        for i in range(end + 1):
            if i < start:
                if not cap.grab():
                    break
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            scale = min(1.0, max_side / max(w, h))
            size = (round(w * scale / 2) * 2, round(h * scale / 2) * 2)
            if size != (w, h):
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(work / f"{n:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
        cap.release()
        return encode_frames(work, fps, dest) if n else 0
    finally:
        cap.release()
        shutil.rmtree(work, ignore_errors=True)


def encode_frames(folder: str | Path, fps: float, dest: str | Path) -> int:
    """folder/000000.jpg, 000001.jpg, ... -> a near-lossless mp4 at `fps`, one frame
    per image. Returns the frames written."""
    dest = Path(dest)
    if FFMPEG_AVAILABLE and FFMPEG_HAS_X264:
        cmd = [FFMPEG, "-y", "-v", "error", "-framerate", str(fps), "-i", str(Path(folder) / "%06d.jpg"),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(dest)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size:
            return probe(dest)["frames"]
        log.warning("ffmpeg encode_frames failed (%s):\n%s", r.returncode, r.stderr[-800:])
    files = sorted(Path(folder).glob("*.jpg"))
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in files:
        vw.write(cv2.imread(str(f)))
    vw.release()
    return len(files)
