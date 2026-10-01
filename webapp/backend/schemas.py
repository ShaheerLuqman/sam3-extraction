"""Request models for multi-object video tracking.

Coordinates are pixels in the uploaded video's native resolution; the server
normalizes. Each object is tracked as one ID and drawn in its own colour, fused
into a single annotated clip + a combined per-frame JSON.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Box = list[float]  # [x1, y1, x2, y2] pixels
Point = list[float]  # [x, y] pixels
Color = list[int]  # [r, g, b] 0-255


def _check_clicks(points: list[Point], labels: list[int], has_box: bool, where: str) -> None:
    """Shared validation for a click prompt: points + labels, optionally on a box."""
    for p in points:
        if len(p) != 2:
            raise ValueError(f"{where}: each point must be [x, y]")
    if len(labels) != len(points):
        raise ValueError(f"{where}: points and labels must be the same length")
    if any(l not in (0, 1) for l in labels):
        raise ValueError(f"{where}: each label must be 1 (positive) or 0 (negative)")
    if points and not has_box and 1 not in labels:
        raise ValueError(f"{where}: needs at least one positive click to start from")


class Seed(BaseModel):
    """One instance's prompt on one frame — a box, click segment, or mask polygons.

      * `box` alone            — a drawn rectangle, seeded as a rectangular mask
      * `points` (+ `labels`)  — a click segment, optionally refining `box`
      * `polygons` alone       — a mask outline directly (e.g. from "find similar")

    One prompt per frame, because an instance is a single tracked identity and
    SAM 3 takes one prompt per (object, frame). Several segments on a frame are
    several instances.
    """
    frame: int = Field(0, ge=0)
    box: Optional[Box] = None
    points: list[Point] = Field(default_factory=list)  # click prompts
    labels: list[int] = Field(default_factory=list)  # 1 = positive, 0 = negative
    polygons: list[list[Point]] = Field(default_factory=list)  # mask outlines

    @model_validator(mode="after")
    def _shape(self):
        if self.box is not None and len(self.box) != 4:
            raise ValueError("box must be [x1, y1, x2, y2]")
        if self.box is None and not self.points and not self.polygons:
            raise ValueError(f"frame {self.frame}: seed has neither a box, click points, nor mask polygons")
        if self.points:
            _check_clicks(self.points, self.labels, self.box is not None, f"frame {self.frame}")
        return self


class TrackObject(BaseModel):
    """One tracked instance: a class, a colour, and how it was prompted."""
    name: str = Field(min_length=1, max_length=60)
    color: Color = [99, 102, 241]
    kind: Literal["box", "text"]
    cls: Optional[int] = Field(None, ge=0)  # index into the uploaded classes.txt
    # box kind:
    seeds: list[Seed] = Field(default_factory=list)
    # text kind:
    phrase: Optional[str] = None
    prompt_frame: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _kind_fields(self):
        if len(self.color) != 3:
            raise ValueError("color must be [r, g, b]")
        if self.kind == "box":
            if not self.seeds:
                raise ValueError(f"instance {self.name!r} has no seed box or click segment")
            if len({s.frame for s in self.seeds}) != len(self.seeds):
                raise ValueError(f"instance {self.name!r} has two prompts on one frame")
        else:
            if not (self.phrase and self.phrase.strip()):
                raise ValueError(f"text object {self.name!r} has no phrase")
        return self


class MultiTrackRequest(BaseModel):
    upload_id: str
    # no upper bound here — the route clamps to the uploaded clip's own length
    max_frames: int = Field(120, ge=10)
    threshold: float = Field(0.5, ge=0.05, le=0.95)
    bidirectional: bool = False
    objects: list[TrackObject] = Field(min_length=1, max_length=20)


class SegmentTrackObject(BaseModel):
    """One tracked identity: its labelled frames (absolute video frame numbers)."""
    name: str = Field(min_length=1, max_length=60)
    seeds: list[Seed] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_per_frame(self):
        if len({s.frame for s in self.seeds}) != len(self.seeds):
            raise ValueError(f"object {self.name!r} has two prompts on one frame")
        return self


class SegmentTrackRequest(BaseModel):
    """Track objects over frames start..end (inclusive) of a video, from the frames
    they were labelled on. JSON only: no rendered video."""
    upload_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    objects: list[SegmentTrackObject] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def _range(self):
        if self.end < self.start:
            raise ValueError("the segment ends before it starts")
        # every frame is held in memory at full resolution while tracking
        if self.end - self.start + 1 > 1800:
            raise ValueError("segments over 1800 frames are too long to track in one go")
        for o in self.objects:
            for s in o.seeds:
                if not self.start <= s.frame <= self.end:
                    raise ValueError(f"object {o.name!r}: frame {s.frame} is outside {self.start}-{self.end}")
        return self


class UploadFromUrlRequest(BaseModel):
    """A video the server downloads itself (e.g. an S3 presigned URL), instead of
    the caller sending the file."""
    url: str = Field(min_length=1, max_length=4096)
    name: str = Field("", max_length=255)  # original file name; else taken from the URL


class ClickPreviewRequest(BaseModel):
    """Clicks (+ optional box) on one frame -> the mask the tracker would seed with."""
    upload_id: str
    frame: int = Field(0, ge=0)
    points: list[Point] = Field(min_length=1, max_length=40)
    labels: list[int] = Field(min_length=1, max_length=40)
    box: Optional[Box] = None

    @model_validator(mode="after")
    def _shape(self):
        if self.box is not None and len(self.box) != 4:
            raise ValueError("box must be [x1, y1, x2, y2]")
        _check_clicks(self.points, self.labels, self.box is not None, "preview")
        return self


class PrepClipRequest(BaseModel):
    """Do a tracking run's CPU preamble up front, while the user is annotating."""
    upload_id: str
    frames: int = Field(120, ge=10)


class PrepareRequest(BaseModel):
    """Encode one frame ahead of a prompt, so the first click feels instant."""
    upload_id: str
    frame: int = Field(0, ge=0)


class ExemplarRequest(BaseModel):
    """Find similar objects on a frame by text phrase or exemplar box/mask."""
    upload_id: str
    frame: int = Field(0, ge=0)
    box: Optional[Box] = None
    neg_boxes: list[Box] = Field(default_factory=list)
    # "sam3" = single-image visual exemplar (PCS); "yoloe" = YOLOE visual prompt.
    # neg_boxes only apply to "sam3".
    method: Literal["sam3", "yoloe"] = "sam3"
    text: Optional[str] = None
    points: list[Point] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    polygons: list[list[Point]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self):
        if self.box is not None and len(self.box) != 4:
            raise ValueError("box must be [x1, y1, x2, y2]")
        has_text = bool(self.text and self.text.strip())
        has_box = self.box is not None
        has_mask = bool(self.polygons or self.points)
        if not (has_text or has_box or has_mask):
            raise ValueError("needs at least a text phrase, exemplar box, or mask")
        return self


# --------------------------------------------------------------------------- #
# frame extraction
# --------------------------------------------------------------------------- #
class ExtractEmbedRequest(BaseModel):
    """Embed the video and the reference images (whatever is not cached yet)."""
    upload_id: str
    image_ids: list[str] = Field(default_factory=list, max_length=64)
    stride: int = Field(5, ge=1, le=60)
    instruction: Optional[str] = Field(None, max_length=2000)


class ExtractPlaybackRequest(BaseModel):
    upload_id: str
    #: "webm" (VP9) for browsers that cannot decode H.264
    format: Literal["mp4", "webm"] = "mp4"


class ExtractRef(BaseModel):
    """A reference: an uploaded image, or a frame of the video being searched."""
    kind: Literal["image", "frame"]
    id: Optional[str] = None
    frame: Optional[int] = Field(None, ge=0)

    @model_validator(mode="after")
    def _shape(self):
        if self.kind == "image" and not self.id:
            raise ValueError("an image reference needs its upload id")
        if self.kind == "frame" and self.frame is None:
            raise ValueError("a frame reference needs a frame index")
        return self


class ExtractScoreRequest(BaseModel):
    upload_id: str
    key: str = Field(min_length=1, max_length=40)
    refs: list[ExtractRef] = Field(min_length=1, max_length=200)


class ExtractSegment(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    peak: Optional[float] = None
    mean: Optional[float] = None
    best_ref: Optional[int] = None

    @model_validator(mode="after")
    def _order(self):
        if self.end < self.start:
            raise ValueError("a segment must end at or after its start")
        return self


class ExtractExportRequest(BaseModel):
    upload_id: str
    segments: list[ExtractSegment] = Field(min_length=1, max_length=5000)
    video: bool = True
    zip_every: int = Field(0, ge=0, le=1000)   # 0 = no ZIP; N = every Nth frame of each segment
    decoded_frames: Optional[int] = None
    settings: dict = Field(default_factory=dict)
    references: list[dict] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# segment extraction
# --------------------------------------------------------------------------- #
class SegxCutRequest(BaseModel):
    """Cut frames `start`..`end` (inclusive) of a video upload into an upload of its own."""
    upload_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class SegxDescribeRequest(BaseModel):
    """Have the VLM name and describe the step shown in these clips."""
    clip_ids: list[str] = Field(min_length=1, max_length=20)


class SegxReference(BaseModel):
    """A reference video and what was marked on it (frame ranges, inclusive). For the
    kNN candidates: the `steps` ranges are the step, everything else is not."""
    upload_id: str
    steps: list[tuple[int, int]] = Field(default_factory=list, max_length=200)
    others: list[tuple[int, int]] = Field(default_factory=list, max_length=200)


class SegxSearchRequest(BaseModel):
    """Find where the step of `step_ids` happens in `upload_id`."""
    #: how the candidate stretches for the VLM are chosen: "similarity" (top `coverage`
    #: by similarity to the step clips) or "knn" (the research's match_frames.py
    #: --balanced: a kNN vote of step vs not-step frames of the reference videos)
    candidates: Literal["similarity", "knn"] = "similarity"
    references: list[SegxReference] = Field(default_factory=list, max_length=20)
    #: ready-cut clips not taken from a reference, added to the kNN pool as step / not step
    knn_step_ids: list[str] = Field(default_factory=list, max_length=20)
    knn_other_ids: list[str] = Field(default_factory=list, max_length=20)
    knn_k: int = Field(15, ge=1, le=200)
    knn_threshold: float = Field(0.5, gt=0.0, lt=1.0)
    upload_id: str
    step_ids: list[str] = Field(min_length=1, max_length=20)
    other_ids: list[str] = Field(default_factory=list, max_length=20)
    name: str = Field("", max_length=200)
    description: str = Field("", max_length=2000)
    use_description: bool = True      # put the description in the embedding instruction
    coverage: float = Field(0.25, gt=0.0, le=1.0)   # share of the video the VLM checks
    stride: int = Field(5, ge=1, le=30)


# --------------------------------------------------------------------------- #
# multiple class segmentation
# --------------------------------------------------------------------------- #
class McsegDescribeRequest(BaseModel):
    """Have the VLM name and describe each of these steps: one list of clips per step."""
    classes: list[list[str]] = Field(min_length=1, max_length=8)


class McsegClass(BaseModel):
    """One step marked on the reference: its range (inclusive) and the clip cut from it."""
    name: str = Field("", max_length=200)
    description: str = Field("", max_length=2000)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    clip_id: str

    @model_validator(mode="after")
    def _order(self):
        if self.end < self.start:
            raise ValueError("a marked range must end at or after its start")
        return self


class McsegSearchRequest(BaseModel):
    """Find where each of `classes` (marked on `ref_upload_id`) happens in `target_ids`."""
    ref_upload_id: str
    classes: list[McsegClass] = Field(min_length=1, max_length=8)
    target_ids: list[str] = Field(min_length=1, max_length=3)
    #: "similarity": per class, the top `coverage` by similarity to its marked frames;
    #: "knn": a balanced kNN vote over the classes plus the rest of the reference
    candidates: Literal["similarity", "knn"] = "similarity"
    knn_k: int = Field(15, ge=1, le=200)
    knn_threshold: float = Field(0.25, gt=0.0, lt=1.0)
    coverage: float = Field(0.25, gt=0.0, le=1.0)
    use_description: bool = True
    stride: int = Field(5, ge=1, le=30)

    @model_validator(mode="after")
    def _disjoint(self):
        r = sorted((c.start, c.end) for c in self.classes)
        if any(b0 <= a1 for (_, a1), (b0, _) in zip(r, r[1:])):
            raise ValueError("the marked ranges overlap: each frame can belong to one step only")
        return self


class FawadSegRequest(BaseModel):
    """The research's class_N_desc_vlm_hints pipeline on a target video (fawadseg.py).

    The JSON files go in as their text, verbatim: a preds.json is a few MB of
    per-frame entries, and the scripts read the files as written."""
    ref_upload_id: str               # the labelled reference video
    upload_id: str                   # the video to search
    ref_preds: str = Field(min_length=2)   # the reference's preds.json
    detector: str = Field(min_length=2)    # detector.json (cycle_steps: step names)
    tgt_preds: str = ""              # optional: the target's preds.json, for metrics only
    cls: int = Field(ge=0)           # the step to find
    confusers: list[int] = Field(min_length=1, max_length=12)   # --confusers
    description: str = Field("", max_length=5000)   # --describe (embedding instruction)
    hints: dict[str, str] = Field(default_factory=dict)   # --hints: step id / "other" -> text
