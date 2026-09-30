"""Paths, tunables, and startup guards for the SAM3 web app backend."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("sam3webapp")

# --- paths ------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parent
VAR_DIR = BACKEND_DIR / "var"
UPLOAD_DIR = VAR_DIR / "uploads"
FRAME_DIR = VAR_DIR / "frames"
RESULT_DIR = VAR_DIR / "results"
TMP_DIR = VAR_DIR / "tmp"
CLASSES_PATH = VAR_DIR / "classes.json"   # label set from an uploaded classes.txt
RUNS_PATH = VAR_DIR / "runs.json"         # history of tracking runs
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"

CHECKPOINT = Path(os.environ.get("SAM3_CHECKPOINT", REPO_ROOT / "checkpoints" / "sam3.pt"))

# The sam3 package is an editable/namespace install whose __file__ is None, so
# pkg_resources.resource_filename("sam3", ...) inside the model builders blows up.
# Pass the BPE vocab path explicitly instead.
def _find_bpe() -> Optional[str]:
    candidates = [
        REPO_ROOT / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    try:
        import importlib.util
        spec = importlib.util.find_spec("sam3")
        for loc in getattr(spec, "submodule_search_locations", None) or []:
            candidates.append(Path(loc) / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    except Exception:  # noqa: BLE001
        pass
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


BPE_PATH = _find_bpe()

for _d in (UPLOAD_DIR, FRAME_DIR, RESULT_DIR, TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- tunables --------------------------------------------------------------- #
MIN_FRAMES = 10
DEFAULT_MAX_FRAMES = 120
# There is no policy ceiling on how much of a clip can be tracked — the clip's
# own length is the limit, enforced per-upload in routes/jobs_routes.py. This
# only stands in when probing could not report a frame count, where a huge
# number correctly means "all of it": media.trim stops at the last frame anyway.
UNKNOWN_LENGTH = 1_000_000

# The tracker's input resolution. Clip preprocessing builds the frame tensor at
# this size ahead of a run; a mismatch against the loaded model means the
# prepared tensor is discarded rather than silently fed in at the wrong scale.
TRACKER_IMAGE_SIZE = 1008

# box-only (visual) prompt vs text prompt: hotstart heuristic must be off for the
# former or every tracked object is dropped (see scripts/track_video.py).
HOTSTART_ATTRS = ("hotstart_delay", "hotstart_unmatch_thresh", "hotstart_dup_thresh")
HOTSTART_ON = (15, 8, 8)
HOTSTART_OFF = (0, 0, 0)

# "find similar" — YOLOE visual-prompt fallback to SAM 3's exemplar PCS.
# YOLOE localises the sibling objects with far less cross-frame clutter than SAM 3
# but its confidences run low and uncalibrated on out-of-distribution data, so the
# frontend picks a per-method default for the confidence slider (see engine).
YOLOE_MODEL = os.environ.get("SAM3_YOLOE_MODEL", "yoloe-v8l-seg.pt")
YOLOE_IMGSZ = int(os.environ.get("SAM3_YOLOE_IMGSZ", "1280"))


def yoloe_weights() -> str:
    """A local checkpoint if present, else the bare name (ultralytics downloads)."""
    local = REPO_ROOT / "checkpoints" / YOLOE_MODEL
    return str(local) if local.is_file() else YOLOE_MODEL


# --- frame extraction (Qwen3-VL-Embedding, run as a subprocess) ------------- #
# vLLM needs its own torch build, so the embedder runs in the qwen_vl venv as a
# long-lived child process (qwen_embed_worker.py, extraction.QwenWorker).
QWEN_PYTHON = Path(os.environ.get(
    "SAM3_QWEN_PYTHON", Path.home() / "Documents" / "qwen_vl" / ".venv-vllm" / "bin" / "python"))
QWEN_EMB_MODEL = os.environ.get("SAM3_QWEN_EMB_MODEL", "Qwen/Qwen3-VL-Embedding-8B")
# segment extraction's VLM: describes the step, classifies clips to polish boundaries
QWEN_VLM_MODEL = os.environ.get("SAM3_QWEN_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
# Which physical GPU (nvidia-smi / PCI order) Qwen runs on. Default GPU 0, so each
# model has its own card: SAM 3 stays on GPU 1 and tracking keeps working while
# frames are embedded. Set it to the backend's own GPU ("1") to share instead:
# SAM 3 is then parked in RAM for each run (engine.offload / restore).
QWEN_GPU = os.environ.get("SAM3_QWEN_GPU", "0").strip()
# Keep the worker loaded this long after its last request (~45 s to start it
# again). On a shared GPU it always stops at once, so SAM 3 can move back.
QWEN_KEEP_WARM_S = int(os.environ.get("SAM3_QWEN_KEEP_WARM_S", "600"))


def qwen_dedicated() -> bool:
    """Does Qwen have a GPU of its own, rather than the one SAM 3 is on?"""
    return QWEN_GPU != os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
# The instruction the retrieval numbers were measured with
# (qwen_vl/extraction_project/scripts/embed_frames.py). Editable per run.
QWEN_EMB_INSTRUCTION = (
    "Represent the manual assembly step the worker is performing in this overhead "
    "workstation image: the panel's orientation and state, the tool in use, and the "
    "hand actions."
)
QWEN_MAX_SIDE = 768           # downscale frames/images past this; 640x480 was the tested size
QWEN_MIN_GPU_GB = 18.0        # 15.6 GB of weights + activations + a little KV cache
EXTRACT_DEFAULT_STRIDE = 5    # embed every 5th frame, as tested
EMBED_DIR = VAR_DIR / "embeds"
EMBED_DIR.mkdir(parents=True, exist_ok=True)
# frame extraction fawad segment (fawadseg.py): per-run folders, served under
# /api/fawadseg, and a cache of decoded frames and embeddings by file content
FAWADSEG_DIR = VAR_DIR / "fawadseg"
FAWADSEG_RUNS = FAWADSEG_DIR / "runs"
FAWADSEG_WORK = FAWADSEG_DIR / "work"
FAWADSEG_RUNS.mkdir(parents=True, exist_ok=True)
# browser playback copies of uploaded videos (H.264, constant frame rate)
PROXY_DIR = VAR_DIR / "proxies"
PROXY_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_MAX_BYTES = 500 * 1024 * 1024
SEQUENTIAL_SEEK_LIMIT = 400          # read frames sequentially below this index

# Retention. The run history keeps a record long after its files are gone
# (`outputs_present` reports that), but a history whose videos vanished the same
# afternoon is not worth much — hence 30 days rather than the old 6 hours.
# Results are tiny (~1 MB a run); uploads are the bulk.
RESULT_TTL_SECONDS = 30 * 24 * 3600
UPLOAD_TTL_SECONDS = 30 * 24 * 3600
#: newest N *runs* whose output files survive regardless of age
KEEP_LAST_RUNS = 50
#: in-memory job records; the durable copy is the run history, so this is short
JOB_TTL_SECONDS = 6 * 3600
SWEEP_INTERVAL_SECONDS = 30 * 60

IMAGE_POLL_MS = 250
VIDEO_POLL_MS = 750


def require_single_gpu() -> dict:
    """Refuse to start unless exactly one CUDA device is visible.

    The backend is launched with CUDA_VISIBLE_DEVICES=1 so SAM 3 only ever sees
    physical GPU 1. Frame extraction's Qwen worker is a separate process with its
    own CUDA_VISIBLE_DEVICES (QWEN_GPU, GPU 0 by default).
    """
    import torch

    n = torch.cuda.device_count()
    if not torch.cuda.is_available() or n == 0:
        raise RuntimeError("No CUDA device visible — SAM3 needs a GPU.")
    if n != 1 and os.environ.get("SAM3_ALLOW_MULTI_GPU") != "1":
        raise RuntimeError(
            f"{n} CUDA devices visible. Launch with CUDA_VISIBLE_DEVICES=1 so only "
            "GPU 1 is used (GPU 0 is training). Set SAM3_ALLOW_MULTI_GPU=1 to override."
        )
    props = torch.cuda.get_device_properties(0)
    uuid = getattr(props, "uuid", "n/a")
    info = {"name": props.name, "uuid": str(uuid),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")}
    log.info("CUDA ok: %s uuid=%s CUDA_VISIBLE_DEVICES=%s",
             info["name"], info["uuid"], info["visible_devices"])
    return info


def cors_origins() -> list[str]:
    raw = os.environ.get("SAM3_CORS_ORIGINS", "").strip()
    return [o.strip() for o in raw.split(",") if o.strip()]


def public_config() -> dict:
    return {
        "min_frames": MIN_FRAMES,
        "default_max_frames": DEFAULT_MAX_FRAMES,
        "image_poll_ms": IMAGE_POLL_MS,
        "video_poll_ms": VIDEO_POLL_MS,
        "upload_max_bytes": UPLOAD_MAX_BYTES,
        "extract": {
            "available": QWEN_PYTHON.is_file(),
            "default_stride": EXTRACT_DEFAULT_STRIDE,
            "default_instruction": QWEN_EMB_INSTRUCTION,
            "model": QWEN_EMB_MODEL,
            "gpu": QWEN_GPU,
            "dedicated_gpu": qwen_dedicated(),
            "keep_warm_s": QWEN_KEEP_WARM_S,
        },
    }
