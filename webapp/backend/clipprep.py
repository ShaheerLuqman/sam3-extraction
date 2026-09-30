"""Clip preprocessing — the CPU half of a tracking run, done while you annotate.

A tracking job is: trim the clip, decode it, resize every frame to the tracker's
input resolution, then propagate on the GPU. Only the last step needs the GPU.
The rest is ~30 ms/frame of pure CPU work (8.7 s for 300 frames at 1008²) and it
grows with the tracking window, so doing it the moment the upload lands means
the Track button starts at the GPU step instead of paying for it first.

Two things this deliberately does NOT do:

  * It does not precompute the per-frame vision features. The tracker's
    `cached_features` holds exactly one frame and is replaced wholesale on every
    miss (sam3_tracking_predictor._get_image_feature), so a whole-clip set would
    be discarded on first use — and at ~40 MB/frame it would not fit anyway.
  * It does not run on the inference worker. This has its own thread, so
    preparing a long clip never delays a click preview.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import torch

from . import config, media

log = logging.getLogger("sam3webapp.clipprep")


@dataclass
class PreppedClip:
    """Everything a tracking run needs before the GPU gets involved."""
    upload_id: str
    #: how many frames were prepared — a run may use fewer, never more
    frames: int
    #: the trimmed clip, read back for compositing
    clip_path: Path
    #: (frames, 3, image_size, image_size) float16 on CPU, as init_state builds it
    images: torch.Tensor
    video_h: int
    video_w: int
    image_size: int

    def release(self) -> None:
        self.images = torch.empty(0)
        self.clip_path.unlink(missing_ok=True)


def prepare_clip(
    src: Path,
    upload_id: str,
    frames: int,
    image_size: int,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> PreppedClip:
    """Trim `src` to `frames` and resize them to the tracker's input tensor."""
    from sam3.model.io_utils import load_video_frames

    note = progress_cb or (lambda _f, _s: None)
    clip = config.TMP_DIR / f"prep_{upload_id}_{frames}.mp4"

    note(0.05, "trimming clip")
    n = media.trim(str(src), frames, clip)
    if n <= 0:
        clip.unlink(missing_ok=True)
        raise RuntimeError("trim produced no frames")
    if cancel is not None and cancel.is_set():
        clip.unlink(missing_ok=True)
        raise RuntimeError("cancelled")

    note(0.2, f"decoding {n} frames")
    # Deliberately the JPEG-folder route, which is exactly what a non-prepared
    # run does — the tensor comes out bit-identical, so preparing ahead of time
    # cannot change a result.
    #
    # Do NOT be tempted to hand load_video_frames the .mp4 directly: its
    # video-file branch never scales pixels to [0, 1] before applying
    # img_mean/img_std, so it returns roughly [-1, 509] where the folder branch
    # returns [-1, 1]. It is faster and it is wrong.
    frame_dir = Path(tempfile.mkdtemp(prefix="sam3_prep_"))
    try:
        cap = cv2.VideoCapture(str(clip))
        i = 0
        while i < n:
            ok, bgr = cap.read()
            if not ok:
                break
            cv2.imwrite(str(frame_dir / f"{i:05d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            i += 1
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled")
            if i % 50 == 0:
                note(0.2 + 0.3 * i / n, f"decoding {n} frames")
        cap.release()

        note(0.55, f"preparing {n} frames")
        images, video_h, video_w = load_video_frames(
            video_path=str(frame_dir),
            image_size=image_size,
            offload_video_to_cpu=True,
        )
    except BaseException:
        clip.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
    note(0.95, f"{n} frames ready")
    log.info("prepped %s: %d frames, %.2f GB", upload_id, n, images.numel() * 2 / 1e9)
    return PreppedClip(
        upload_id=upload_id,
        frames=int(images.shape[0]),
        clip_path=clip,
        images=images,
        video_h=video_h,
        video_w=video_w,
        image_size=image_size,
    )


class ClipPrepStore:
    """Holds the prepared clip. Only the newest is kept — at 6 MB a frame, a
    second one would be a large amount of RAM for something nothing will read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Optional[PreppedClip] = None

    def get(self, upload_id: str, need_frames: int) -> Optional[PreppedClip]:
        """The prepared clip, if it covers at least `need_frames` of this upload."""
        with self._lock:
            it = self._item
            if it is None or it.upload_id != upload_id or it.frames < need_frames:
                return None
            return it

    def put(self, prepped: PreppedClip) -> None:
        with self._lock:
            old, self._item = self._item, prepped
        if old is not None and old is not prepped:
            old.release()

    def covers(self, upload_id: str, frames: int) -> bool:
        with self._lock:
            it = self._item
            return it is not None and it.upload_id == upload_id and it.frames >= frames

    def clear(self) -> None:
        with self._lock:
            old, self._item = self._item, None
        if old is not None:
            old.release()
