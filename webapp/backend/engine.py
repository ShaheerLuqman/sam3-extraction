"""SAM3 model + multi-object video tracking, fused into one annotated clip + JSON.

Only the jobs worker thread may call load() / run_multitrack(). Small helpers are
lifted from scripts/app.py and scripts/track_bbox.py (line numbers noted).
"""
from __future__ import annotations

import gc
import json
import logging
import shutil
import tempfile
import threading
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
from PIL import Image

from . import config

log = logging.getLogger("sam3webapp.engine")

_DEVICE = "cuda"
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _video_writer(path_no_ext, fps, size):  # scripts/app.py:59
    webm = path_no_ext + ".webm"
    vw = cv2.VideoWriter(webm, cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if vw.isOpened():
        return vw, webm
    mp4 = path_no_ext + ".mp4"
    return cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, size), mp4


def _read_frames(path, max_frames):  # scripts/app.py:192
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def _mask_to_xyxy(m):  # scripts/track_bbox.py:180
    ys, xs = np.where(m)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16) if _DEVICE == "cuda" else nullcontext()


def _sync() -> None:
    """Finish queued GPU work before propagation reads the seeds.

    `propagate_in_video` can otherwise start from state a prompt hasn't landed
    in yet and segment the *previous* prompt's object — reproducible with
    back-to-back single-frame click prompts.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_mem() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {"mem_free_mb": round(free / 1e6), "mem_total_mb": round(total / 1e6)}


def _clamp_box(b, W, H):
    x1, y1, x2, y2 = b
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    return max(0, x1), max(0, y1), min(W, x2), min(H, y2)


def _to_cxcywh(box_xyxy, W, H):  # scripts/segment_image.py:to_norm_cxcywh
    x1, y1, x2, y2 = box_xyxy
    return [((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, abs(x2 - x1) / W, abs(y2 - y1) / H]


def _click_prompt(points, labels, box, W, H) -> dict:
    """Clicks (+ at most one box) in image pixels -> `add_new_points_or_box` kwargs.

    The tracker takes relative coordinates (`rel_coordinates=True`, its default)
    and scales them by its own input resolution, so everything is divided by the
    frame size here rather than by the model's image size.
    """
    kw = {
        "points": torch.tensor([[x / W, y / H] for x, y in points], dtype=torch.float32),
        "labels": torch.tensor(list(labels), dtype=torch.int32),
        # a box prompt has to be the first thing the encoder sees, so any
        # previously stored clicks for this (object, frame) are replaced
        "clear_old_points": True,
    }
    if box is not None:
        x1, y1, x2, y2 = _clamp_box(box, W, H)
        kw["box"] = np.array([[x1 / W, y1 / H, x2 / W, y2 / H]], dtype=np.float32)
    return kw


def _shade_mask(canvas, polys, box, col) -> None:
    """Tint and outline one instance's mask on the composite frame.

    Filling happens inside the mask's own bounding box rather than over the
    whole frame — same result, a fraction of the work per instance per frame.
    """
    x1, y1, x2, y2 = box
    h, w = max(0, y2 - y1), max(0, x2 - x1)
    if h == 0 or w == 0:
        return
    shifted = [
        (np.array(p, dtype=np.int32) - (x1, y1)).reshape((-1, 1, 2))
        for p in polys if len(p) >= 3
    ]
    if not shifted:
        return
    stencil = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(stencil, shifted, 1)
    sel = stencil.astype(bool)
    roi = canvas[y1:y2, x1:x2]
    roi[sel] = 0.55 * roi[sel] + 0.45 * col
    cv2.polylines(canvas, [np.array(p, dtype=np.int32).reshape((-1, 1, 2))
                           for p in polys if len(p) >= 3],
                  True, col.tolist(), 2, cv2.LINE_AA)


def _mask_to_polygons(mask, max_contours: int = 24) -> list[list[list[int]]]:
    """Binary mask -> simplified outlines in image pixels, biggest first.

    Sent to the browser so a click preview can be stroked on the frame canvas
    without shipping a full-resolution mask image.
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_contours]
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < 16:
            continue
        approx = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append([[int(x), int(y)] for x, y in approx])
    return polys


def _polygons_to_xyxy(polygons: list) -> Optional[list[int]]:
    """Tight bounding box bounding a set of polygon contour points."""
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")
    found = False
    for poly in polygons:
        for pt in poly:
            found = True
            min_x = min(min_x, pt[0])
            min_y = min(min_y, pt[1])
            max_x = max(max_x, pt[0])
            max_y = max(max_y, pt[1])
    if not found:
        return None
    return [int(round(min_x)), int(round(min_y)), int(round(max_x)), int(round(max_y))]


# --------------------------------------------------------------------------- #
# Per-frame encoder cache
#
# Encoding a frame is most of the cost of a prompt: the image model's vision
# backbone for "find similar", and the tracker's one-frame inference state for
# the click preview. Both depend only on the pixels, so they are computed once
# per (upload, frame) and reused by every prompt on that frame. The frontend
# also asks for frame 0 to be encoded the moment a video is uploaded, so the
# first prompt is already paid for by the time anyone clicks.
#
# Only the jobs worker thread touches any of this.
# --------------------------------------------------------------------------- #
CACHED_FRAMES = 3


@dataclass
class _FrameEntry:
    width: int = 0
    height: int = 0
    #: pristine vision features from Sam3Processor.set_image — never prompted in
    #: place, each prompt gets a shallow copy so it cannot leave keys behind
    image_features: Optional[dict] = None
    #: a one-frame tracker inference state, cleared between previews
    track_state: Optional[dict] = None
    track_dir: Optional[Path] = None


# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, checkpoint: Path):
        self.checkpoint = Path(checkpoint)
        self.video_predictor = None
        self.image_model = None
        self.yoloe_model = None  # lazy — only built the first time "find similar" uses YOLOE
        self.loaded = False
        self.offloaded = False  # True while frame extraction borrows the GPU
        self.load_error: Optional[str] = None
        self._gpu_info: dict = {}
        self._frames: "OrderedDict[tuple[str, int], _FrameEntry]" = OrderedDict()
        self._image_proc = None  # Sam3Processor, built once the image model exists

    def load(self) -> None:
        ckpt = str(self.checkpoint) if self.checkpoint.is_file() else None
        if ckpt is None:
            log.warning("checkpoint %s missing — will try HF auto-download", self.checkpoint)
        try:
            from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor

            bpe = {"bpe_path": config.BPE_PATH} if config.BPE_PATH else {}
            log.info("building video predictor (single GPU)... bpe=%s", config.BPE_PATH)
            vkw = {"checkpoint_path": ckpt} if ckpt else {}
            self.video_predictor = build_sam3_video_predictor(gpus_to_use=[0], **vkw, **bpe)

            tracker = self.video_predictor.model.tracker
            if getattr(tracker, "backbone", None) is None:
                tracker.backbone = self.video_predictor.model.detector.backbone
                log.info("wired tracker.backbone from detector")

            log.info("building image model (for exemplar 'find similar')...")
            ikw = {"checkpoint_path": ckpt, "load_from_HF": False} if ckpt else {}
            self.image_model = build_sam3_image_model(device=_DEVICE, **ikw, **bpe)

            self.loaded = True
            log.info("models ready. gpu=%s", _gpu_mem())
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("model load failed")
            raise

    def health(self) -> dict:
        g = {}
        try:
            g = _gpu_mem()
        except Exception:  # noqa: BLE001
            pass
        if self.offloaded:
            status = "busy"  # SAM 3 is parked in RAM while frame extraction has the GPU
        else:
            status = "ready" if self.loaded else ("error" if self.load_error else "loading")
        return {
            "status": status,
            "models": {"video": self.video_predictor is not None,
                       "image": self.image_model is not None,
                       "yoloe": self.yoloe_model is not None},
            "gpu": {**self._gpu_info, **g},
            "load_error": self.load_error,
        }

    def set_gpu_info(self, info: dict) -> None:
        self._gpu_info = info

    # -- sharing the GPU with frame extraction ---------------------------- #
    def offload(self) -> dict:
        """Park both SAM 3 models in CPU RAM so another process can have the GPU.

        Worker thread only, like every other model call — that is what keeps a
        prompt from landing on a half-moved model. Returns the GPU's free memory
        afterwards. YOLOE is simply dropped; it reloads lazily.
        """
        self._require_loaded()
        self.clear_frames()
        self._image_proc = None
        self.yoloe_model = None
        self.offloaded = True
        self.video_predictor.model.to("cpu")
        self.image_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        mem = _gpu_mem()
        log.info("SAM 3 offloaded to CPU. gpu=%s", mem)
        return mem

    def restore(self) -> None:
        """Move SAM 3 back onto the GPU after offload()."""
        if not self.offloaded:
            return
        self.video_predictor.model.to(_DEVICE)
        self.image_model.to(_DEVICE)
        self.offloaded = False
        log.info("SAM 3 restored to GPU. gpu=%s", _gpu_mem())

    def tracker_image_size(self) -> int:
        """The resolution clip preprocessing must build its frame tensor at."""
        tracker = getattr(getattr(self.video_predictor, "model", None), "tracker", None)
        return int(getattr(tracker, "image_size", config.TRACKER_IMAGE_SIZE))

    # -- per-frame encoder cache ------------------------------------------ #
    def _require_loaded(self) -> None:
        if self.load_error:
            raise RuntimeError(f"the model failed to load: {self.load_error}")
        if not self.loaded:
            raise RuntimeError("the model is still loading — try again in a moment")

    def _entry(self, key: tuple) -> _FrameEntry:
        entry = self._frames.get(key)
        if entry is None:
            entry = _FrameEntry()
            self._frames[key] = entry
        self._frames.move_to_end(key)
        while len(self._frames) > CACHED_FRAMES:
            self._release(self._frames.popitem(last=False)[1])
        return entry

    def _release(self, entry: _FrameEntry) -> None:
        entry.image_features = None
        entry.track_state = None
        if entry.track_dir is not None:
            shutil.rmtree(entry.track_dir, ignore_errors=True)
            entry.track_dir = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _drop(self, key: tuple) -> None:
        entry = self._frames.pop(key, None)
        if entry is not None:
            self._release(entry)

    def clear_frames(self) -> None:
        """Drop every cached frame encoding, e.g. to free the GPU for a track run."""
        while self._frames:
            self._release(self._frames.popitem(last=False)[1])

    def _cached(self, cache_key: Optional[tuple], fn: Callable[[tuple], dict]) -> dict:
        """Run `fn(key)` against this frame's cached encoding.

        A cached state can go stale — a prompt that raises part-way leaves the
        tracker mid-flight — so a failure throws the frame away and re-encodes
        once before giving up. Without a key the encoding is scratch: built for
        this one call and released straight after.
        """
        key = cache_key or (uuid4().hex, -1)
        try:
            try:
                return fn(key)
            except Exception:  # noqa: BLE001 - one retry on a clean encoding
                if cache_key is None:
                    raise
                log.warning("prompt failed on the cached frame — re-encoding once",
                            exc_info=True)
                self._drop(key)
                return fn(key)
        finally:
            if cache_key is None:
                self._drop(key)

    def _processor(self):
        from sam3.model.sam3_image_processor import Sam3Processor

        if self._image_proc is None:
            self._image_proc = Sam3Processor(self.image_model, device=_DEVICE,
                                             confidence_threshold=0.05)
        return self._image_proc

    def _image_state(self, key: tuple, image_path: str):
        """(processor, fresh prompt state) for this frame, encoding it once."""
        proc = self._processor()
        entry = self._entry(key)
        if entry.image_features is None:
            with Image.open(image_path) as src:
                pil = src.convert("RGB")
            entry.width, entry.height = pil.size
            with _autocast():
                entry.image_features = proc.set_image(pil)["backbone_out"]
        # shallow copy: the grounding pass adds its own keys, and they must not
        # survive into the next prompt on this frame
        state = {
            "original_width": entry.width,
            "original_height": entry.height,
            "backbone_out": dict(entry.image_features),
        }
        return proc, state

    def _tracker_state(self, key: tuple, frame_jpeg_path: str):
        """(tracker, empty one-frame state) for this frame, encoding it once."""
        tracker = self.video_predictor.model.tracker
        entry = self._entry(key)
        if entry.track_state is None:
            frame_dir = Path(tempfile.mkdtemp(prefix="sam3_frame_"))
            shutil.copyfile(frame_jpeg_path, frame_dir / "00000.jpg")
            with _autocast():
                entry.track_state = tracker.init_state(
                    video_path=str(frame_dir),
                    offload_video_to_cpu=True, offload_state_to_cpu=True,
                )
            entry.track_dir = frame_dir
        tracker.clear_all_points_in_video(entry.track_state)
        return tracker, entry.track_state

    def prepare_frame(self, upload_id: str, frame: int, jpeg_path: str,
                      progress_cb: Optional[Callable[[float, str], None]] = None) -> dict:
        """Encode one frame up front, so the first prompt on it returns quickly."""
        self._require_loaded()
        key = (upload_id, int(frame))
        note = progress_cb or (lambda _f, _s: None)
        note(0.25, "computing image embedding")
        self._image_state(key, jpeg_path)
        note(0.7, "initialising tracker state")
        self._tracker_state(key, jpeg_path)
        entry = self._entry(key)
        return {"frame": int(frame), "width": entry.width, "height": entry.height,
                "ready": True, "message": f"frame {frame} embedded and ready"}

    # -- exemplar & text search: "find similar objects in this frame" ---- #
    def find_similar(
        self,
        image_path: str,
        box_xyxy: Optional[list] = None,
        neg_boxes: Optional[list] = None,
        method: str = "sam3",
        text: Optional[str] = None,
        polygons: Optional[list] = None,
        cache_key: Optional[tuple] = None,
    ) -> dict:
        """Find similar objects on a frame by text phrase or visual exemplar (SAM 3 / YOLOE).
        Returns candidates with box, score, and segmentation polygons.
        """
        self._require_loaded()
        if text and text.strip():
            phrase = text.strip()
            return self._cached(cache_key, lambda k: self._find_similar_text(k, image_path, phrase))

        if box_xyxy is None and polygons:
            box_xyxy = _polygons_to_xyxy(polygons)

        if box_xyxy is None:
            raise ValueError("No text query or valid exemplar box/mask provided")

        if method == "yoloe":
            return self._find_similar_yoloe(image_path, box_xyxy)
        return self._cached(
            cache_key, lambda k: self._find_similar_sam3(k, image_path, box_xyxy, neg_boxes)
        )

    @staticmethod
    def _candidates(state: dict, W: int, H: int) -> list[dict]:
        """A prompted image state -> the boxes, scores and outlines the UI draws."""
        boxes = state.get("boxes")
        scores = state.get("scores")
        masks = state.get("masks")
        cands: list[dict] = []
        if boxes is None or scores is None:
            return cands
        for i, (b, s) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = (b.cpu().tolist() if hasattr(b, "cpu") else list(b))
            bx = [max(0, int(round(x1))), max(0, int(round(y1))),
                  min(W, int(round(x2))), min(H, int(round(y2)))]
            poly: list = []
            if masks is not None and i < len(masks):
                m = (masks[i, 0] if masks.dim() == 4 else masks[i]).cpu().numpy().astype(bool)
                poly = _mask_to_polygons(m)
            cands.append({"box": bx, "score": round(float(s), 4), "polygons": poly})
        cands.sort(key=lambda c: -c["score"])
        return cands

    def _find_similar_text(self, key: tuple, image_path: str, text: str) -> dict:
        proc, st = self._image_state(key, image_path)
        W, H = st["original_width"], st["original_height"]
        with _autocast():
            st = proc.set_text_prompt(text, st)
        cands = self._candidates(st, W, H)
        return {
            "candidates": cands,
            "width": W,
            "height": H,
            "method": "text",
            "suggest_threshold": 0.25,
            "message": (f"SAM 3 text prompt: {len(cands)} detection(s) for "
                        f"\u2018{text}\u2019 \u2014 adjust the confidence threshold"),
        }

    def _find_similar_sam3(self, key: tuple, image_path: str, box_xyxy: list,
                           neg_boxes: Optional[list] = None) -> dict:
        proc, st = self._image_state(key, image_path)
        W, H = st["original_width"], st["original_height"]
        with _autocast():
            st = proc.add_geometric_prompt(_to_cxcywh(box_xyxy, W, H), True, st)
            for nb in neg_boxes or []:
                st = proc.add_geometric_prompt(_to_cxcywh(nb, W, H), False, st)
        cands = self._candidates(st, W, H)
        return {"candidates": cands, "width": W, "height": H, "method": "sam3",
                "suggest_threshold": 0.3,
                "message": (f"SAM 3 exemplar: {len(cands)} candidate(s) \u2014 adjust "
                            "the confidence threshold")}

    def _ensure_yoloe(self):
        if self.yoloe_model is not None:
            return self.yoloe_model
        try:
            from ultralytics import YOLOE
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "YOLOE needs the 'ultralytics' package. Install it into the backend venv:\n"
                "  VIRTUAL_ENV=./.venv uv pip install ultralytics\n"
                f"(import failed: {type(exc).__name__}: {exc})"
            ) from exc
        weights = config.yoloe_weights()
        log.info("loading YOLOE visual-prompt model: %s", weights)
        self.yoloe_model = YOLOE(weights)
        log.info("YOLOE ready. gpu=%s", _gpu_mem())
        return self.yoloe_model

    def _find_similar_yoloe(self, image_path: str, box_xyxy: list) -> dict:
        """YOLOE visual prompt: the one box is the exemplar, everything visually
        like it on the same image comes back. Confidences are low/uncalibrated on
        OOD data, so we return everything ranked and steer the slider default."""
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        model = self._ensure_yoloe()
        pil = Image.open(image_path).convert("RGB")
        W, H = pil.size
        x1, y1, x2, y2 = _clamp_box(box_xyxy, W, H)
        vp = dict(bboxes=np.array([[x1, y1, x2, y2]], dtype=np.float32), cls=np.array([0]))
        # ultralytics manages its own precision/device — no _autocast() here.
        res = model.predict(image_path, visual_prompts=vp,
                            predictor=YOLOEVPSegPredictor, conf=0.001,
                            imgsz=config.YOLOE_IMGSZ, device=0, verbose=False)[0]

        cands = []
        boxes_list = getattr(res, "boxes", []) or []
        for i, b in enumerate(boxes_list):
            bx = b.xyxy[0].tolist() if hasattr(b.xyxy[0], "tolist") else list(b.xyxy[0])
            cx1, cy1, cx2, cy2 = _clamp_box(bx, W, H)
            if cx2 > cx1 and cy2 > cy1:
                poly = []
                if hasattr(res, "masks") and res.masks is not None and i < len(res.masks.xy):
                    pts = res.masks.xy[i]
                    if len(pts) >= 3:
                        poly = [[[int(round(p[0])), int(round(p[1]))] for p in pts]]
                cands.append({
                    "box": [cx1, cy1, cx2, cy2],
                    "score": round(float(b.conf[0]), 4),
                    "polygons": poly,
                })
        cands.sort(key=lambda c: -c["score"])
        cands = cands[:40]
        top = cands[0]["score"] if cands else 0.0
        suggest = round(min(0.5, max(0.05, top * 0.5)), 2)
        return {"candidates": cands, "width": W, "height": H, "method": "yoloe",
                "suggest_threshold": suggest,
                "message": (f"YOLOE: {len(cands)} candidate(s). Scores run low — the "
                            f"slider starts at {suggest:.2f}; review and adjust.")}

    # -- click prompts: preview the mask a set of clicks selects ---------- #
    def click_preview(self, frame_jpeg_path: str, points: list, labels: list,
                      box: Optional[list] = None,
                      cache_key: Optional[tuple] = None) -> dict:
        """Run one frame through the *tracking* model with these clicks.

        Deliberately the same code path `run_multitrack` seeds with, so the
        outline the user sees is the mask that actually gets propagated — an
        image-model preview would be a near-miss.
        """
        self._require_loaded()
        return self._cached(
            cache_key, lambda k: self._click_preview(k, frame_jpeg_path, points, labels, box)
        )

    def _click_preview(self, key: tuple, frame_jpeg_path: str, points: list,
                       labels: list, box: Optional[list]) -> dict:
        with Image.open(frame_jpeg_path) as pil:
            W, H = pil.size

        tracker, state = self._tracker_state(key, frame_jpeg_path)
        mask = np.zeros((H, W), dtype=bool)
        with _autocast():
            tracker.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=0,
                **_click_prompt(points, labels, box, W, H),
            )
            _sync()
            # The mask `add_new_points_or_box` hands back lags one call behind,
            # so read the seed the way tracking does — one propagation step
            # over the single frame.
            for _f, _ids, _low, vres, _sc in tracker.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=1,
                reverse=False, propagate_preflight=True, tqdm_disable=True,
            ):
                mask = (vres > 0.0).squeeze(1).cpu().numpy()[0].astype(bool)
                break

        area = int(mask.sum())
        pos = sum(1 for l in labels if l == 1)
        return {
            "width": W, "height": H,
            "box": _mask_to_xyxy(mask),
            "polygons": _mask_to_polygons(mask),
            "area": area,
            "coverage": round(area / float(W * H), 5),
            "message": (f"{pos} positive / {len(labels) - pos} negative click(s)"
                        + (" + box" if box else "")
                        + (f" -> {area} px mask" if area else
                           " -> empty mask, try another click")),
        }

    # -- multi-object track ------------------------------------------------ #
    def run_multitrack(self, clip_path: str, req: dict, result_stem: str,
                       progress_cb: Callable[[float, str], None],
                       cancel: threading.Event,
                       images: Optional["torch.Tensor"] = None) -> dict:
        """`images`, when given, is the frame tensor clip preprocessing already
        built (see clipprep.py) — it replaces the in-job resize."""
        max_frames = int(req["max_frames"])
        threshold = float(req["threshold"])
        bidir = bool(req["bidirectional"])
        objects = req["objects"]

        # tracking a whole clip wants every spare byte of GPU — the marking-time
        # frame encodings are rebuilt on demand afterwards
        self.clear_frames()

        frames, fps = _read_frames(clip_path, max_frames)
        if not frames:
            raise RuntimeError("could not read frames from the trimmed clip")
        H, W = frames[0].shape[:2]
        n = len(frames)

        box_objs = [o for o in objects if o["kind"] == "box"]
        text_objs = [o for o in objects if o["kind"] == "text"]
        order = {id(o): i for i, o in enumerate(objects)}
        results: list[dict] = []

        # ---- box objects: one tracker state, obj_idx per object -------- #
        if box_objs:
            progress_cb(0.03, "seeding box objects")
            results += self._track_boxes(frames, W, H, n, box_objs, bidir,
                                         progress_cb, cancel, images)

        # ---- text objects: one concept session each ------------------- #
        for oi, obj in enumerate(text_objs):
            progress_cb(0.05, f'concept "{obj["phrase"]}" ({oi + 1}/{len(text_objs)})')
            results.append(self._track_text(clip_path, n, max_frames, threshold, obj, progress_cb, cancel))

        results.sort(key=lambda r: order.get(r["_src_id"], 0))
        for r in results:
            r.pop("_src_id", None)

        # ---- composite ---------------------------------------------- #
        progress_cb(0.95, "compositing")
        raw_stem = str(Path(config.TMP_DIR) / (Path(result_stem).name + "_raw"))
        vw, raw_path = _video_writer(raw_stem, fps, (W, H))
        for f in range(n):
            canvas = frames[f].astype(np.float32)
            for obj in results:
                col = np.array(obj["color"], dtype=np.float32)
                sets = obj.get("_polys") or []
                shapes = sets[f] if f < len(sets) else []
                for i, b in enumerate(obj["per_frame"][f]):
                    x1, y1, x2, y2 = b
                    polys = shapes[i] if i < len(shapes) else []
                    if polys:
                        _shade_mask(canvas, polys, b, col)
                    elif x2 > x1 and y2 > y1:
                        # no outline came back — fall back to the box
                        sub = canvas[y1:y2, x1:x2]
                        canvas[y1:y2, x1:x2] = 0.65 * sub + 0.35 * col
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), col.tolist(), 2)
                    cv2.putText(canvas, obj["name"], (x1, max(y1 - 5, 12)),
                                _FONT, 0.5, col.tolist(), 2, cv2.LINE_AA)
            vw.write(cv2.cvtColor(canvas.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        vw.release()

        from . import media
        served, mime, codec_warning = media.finalize_video(raw_path, result_stem)
        Path(raw_path).unlink(missing_ok=True)

        # ---- JSON ------------------------------------------------- #
        # outlines were for the rendered video; in the file they would be
        # enormous, so they go before the doc is built rather than after
        for r in results:
            r.pop("_polys", None)
        doc = {
            "video": Path(clip_path).name, "fps": round(fps, 3), "frames": n,
            "width": W, "height": H, "bidirectional": bidir, "threshold": threshold,
            "box_format": "[x1, y1, x2, y2] pixels; per_frame[i] is the list of "
                          "instance boxes on frame i (empty = not visible)",
            "classes": req.get("class_names") or {},
            "objects": results,
        }
        json_path = f"{result_stem}.json"
        Path(json_path).write_text(json.dumps(doc, indent=1))

        counts = {o["name"]: sum(1 for fr in o["per_frame"] if fr) for o in results}
        msg = f"tracked {len(results)} object(s) across {n} frames · " + \
              ", ".join(f"{k}: {v}f" for k, v in counts.items())
        return {
            "served_path": served, "mime": mime, "codec_warning": codec_warning,
            "json_url": f"/api/files/{Path(json_path).name}",
            "objects": len(results), "frames": n, "width": W, "height": H, "message": msg,
        }

    # -- box tracking via model.tracker --------------------------------- #
    def _track_boxes(self, frames, W, H, n, box_objs, bidir, progress_cb, cancel,
                     images=None) -> list[dict]:
        tracker = self.video_predictor.model.tracker
        # with a prepared tensor there is nothing to spill to disk; without one
        # the frames go out as JPEGs for init_state to read back
        frame_dir = None if images is not None else Path(tempfile.mkdtemp(prefix="sam3_frames_"))
        try:
            if frame_dir is not None:
                for i, rgb in enumerate(frames):
                    cv2.imwrite(str(frame_dir / f"{i:05d}.jpg"),
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])

            per = {oi: [[] for _ in range(n)] for oi in range(len(box_objs))}
            # the mask is what the tracker actually produces; the box is derived
            # from it. Keep both — as polygons, since full-res masks for every
            # frame of every instance would be gigabytes.
            polys = {oi: [[] for _ in range(n)] for oi in range(len(box_objs))}
            seeded: dict[int, list[int]] = {oi: [] for oi in range(len(box_objs))}
            with _autocast():
                if images is not None:
                    # same keys init_state's own loader would set, just already built
                    state = tracker.init_state(video_height=H, video_width=W, num_frames=n,
                                               offload_video_to_cpu=True,
                                               offload_state_to_cpu=True)
                    state["images"] = images
                else:
                    state = tracker.init_state(video_path=str(frame_dir),
                                               offload_video_to_cpu=True,
                                               offload_state_to_cpu=True)
                tracker.clear_all_points_in_video(state)
                for oi, obj in enumerate(box_objs):
                    for seed in obj["seeds"]:
                        f = max(0, min(int(seed["frame"]), n - 1))
                        box = seed.get("box")
                        if box is not None:
                            box = _clamp_box(box, W, H)
                            if box[2] <= box[0] or box[3] <= box[1]:
                                box = None
                        points = seed.get("points") or []
                        polygons = seed.get("polygons") or []
                        if points:
                            # a click segment, optionally refining the drawn box
                            tracker.add_new_points_or_box(
                                inference_state=state, frame_idx=f, obj_id=oi,
                                **_click_prompt(points, seed["labels"], box, W, H),
                            )
                        elif polygons:
                            m_np = np.zeros((H, W), dtype=np.uint8)
                            for poly in polygons:
                                if len(poly) >= 3:
                                    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                                    cv2.fillPoly(m_np, [pts], 1)
                            m = torch.from_numpy(m_np.astype(bool))
                            tracker.add_new_mask(state, frame_idx=f, obj_id=oi, mask=m)
                        elif box is not None:
                            m = torch.zeros(H, W, dtype=torch.bool)
                            m[box[1]:box[3], box[0]:box[2]] = True
                            tracker.add_new_mask(state, frame_idx=f, obj_id=oi, mask=m)
                        else:
                            continue
                        seeded[oi].append(f)
                if not any(seeded.values()):
                    raise RuntimeError("no instance had a usable seed box or click segment")
                seed_frames = [f for fs in seeded.values() for f in fs]
                _sync()

                def collect(start, reverse):
                    span = (n - start) if not reverse else (start + 1)
                    done = 0
                    for f_idx, obj_ids, _low, vres, _sc in tracker.propagate_in_video(
                        state, start_frame_idx=start, max_frame_num_to_track=n,
                        reverse=reverse, propagate_preflight=True, tqdm_disable=True,
                    ):
                        if cancel.is_set():
                            raise RuntimeError("cancelled")
                        done += 1
                        if f_idx >= n:
                            continue
                        masks = (vres > 0.0).squeeze(1).cpu().numpy()
                        for i, oid in enumerate(obj_ids):
                            m = masks[i].astype(bool)
                            bx = _mask_to_xyxy(m)
                            per[int(oid)][f_idx] = [bx] if bx else []
                            polys[int(oid)][f_idx] = [_mask_to_polygons(m)] if bx else []
                        progress_cb(min(0.1 + 0.7 * done / max(span, 1), 0.9),
                                    f"tracking boxes {done}/{span}"
                                    + (" (reverse)" if reverse else ""))

                collect(min(seed_frames), reverse=False)
                if bidir and max(seed_frames) > 0:
                    collect(max(seed_frames), reverse=True)

            out = []
            for oi, obj in enumerate(box_objs):
                pf, pp = per[oi], polys[oi]
                if not bidir and seeded[oi]:
                    for f in range(min(seeded[oi])):
                        pf[f] = []
                        pp[f] = []
                out.append({"_src_id": id(obj), "name": obj["name"], "color": obj["color"],
                            "kind": "box", "cls": obj.get("cls"),
                            "class_name": obj.get("class_name"), "per_frame": pf,
                            "_polys": pp})
            del state
            return out
        finally:
            if frame_dir is not None:
                shutil.rmtree(frame_dir, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -- text tracking via the concept/detector session ---------------- #
    def _track_text(self, clip_path, n, max_frames, threshold, obj, progress_cb, cancel) -> dict:
        p = self.video_predictor
        try:
            p.model.score_threshold_detection = float(threshold)
        except Exception:  # noqa: BLE001
            pass
        for attr, on in zip(config.HOTSTART_ATTRS, config.HOTSTART_ON):
            if hasattr(p.model, attr):
                setattr(p.model, attr, on)

        sid = p.handle_request(request=dict(
            type="start_session", resource_path=clip_path,
            offload_video_to_cpu=True, offload_state_to_cpu=True,
        ))["session_id"]
        pf = [[] for _ in range(n)]
        pp = [[] for _ in range(n)]   # polygons, index-aligned with pf
        try:
            frame_index = max(0, min(int(obj.get("prompt_frame", 0)), n - 1))
            with _autocast():
                p.handle_request(request=dict(type="add_prompt", session_id=sid,
                                              frame_index=frame_index, text=obj["phrase"]))
                direction = "forward" if frame_index == 0 else "both"
                for resp in p.handle_stream_request(request=dict(
                    type="propagate_in_video", session_id=sid,
                    propagation_direction=direction, max_frame_num_to_track=max_frames,
                )):
                    if cancel.is_set():
                        raise RuntimeError("cancelled")
                    fi = resp["frame_index"]
                    if fi >= n:
                        continue
                    out = resp["outputs"]
                    if out is not None:
                        for k in range(len(out["out_obj_ids"])):
                            m = out["out_binary_masks"][k]
                            m = np.squeeze(m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)).astype(bool)
                            bx = _mask_to_xyxy(m)
                            if bx:
                                pf[fi].append(bx)
                                pp[fi].append(_mask_to_polygons(m))
                    progress_cb(min(0.1 + 0.8 * (fi + 1) / max(n, 1), 0.9),
                                f'"{obj["phrase"]}" frame {fi + 1}/{n}')
        finally:
            try:
                p.handle_request(dict(type="close_session", session_id=sid))
            except Exception:  # noqa: BLE001
                log.warning("close_session failed", exc_info=True)
        return {"_src_id": id(obj), "name": obj["name"], "color": obj["color"],
                "kind": "text", "cls": obj.get("cls"), "class_name": obj.get("class_name"),
                "phrase": obj["phrase"], "per_frame": pf, "_polys": pp}
