# sam3-extraction

Identify a sample object (by text phrase or an exemplar crop/box) and segment /
track every instance of it in an image or video, using Meta's **SAM 3**.

## Layout

```
.venv/              Python 3.12 virtual env (uv-managed)
sam3/               cloned facebookresearch/sam3 (native repo, needed for video)
patches/            local fixes applied on top of the sam3 clone (see Setup)
checkpoints/        put sam3.pt here (from https://huggingface.co/facebook/sam3)
inputs/             your images / videos
outputs/            annotated results
scripts/
  check_env.py      verify torch + CUDA + model imports
  segment_image.py  single-image concept segmentation (HF transformers path)
  track_video.py    video concept detection + tracking (native sam3 path)
```

## Hardware

RTX 3080 mobile (16 GB) — enough for image inference and short/low-res video.
No `flash-attn-3` (Hopper-only); the model falls back to standard attention.

Measured on this laptop:
- image (1008px): ~a few seconds after model load
- video: model load ~1 min, then **~2.5–4 s / frame** at 960x540
  (200-frame clip ≈ 9 min). Downscale / trim clips for quicker iteration.

Tuning: raise `--threshold` (default 0.5 -> 0.6/0.7) to drop weak/duplicate
detections.

## Setup

```powershell
# 1. venv (already created with: uv venv --python 3.12 .venv)
.venv\Scripts\activate

# 2. torch (CUDA 12.8 wheels)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. the native SAM 3 repo (pinned, plus our local patch) + deps missing from its pyproject
git clone https://github.com/facebookresearch/sam3
git -C sam3 checkout 660a5e9
git -C sam3 apply ../patches/sam3-video-base-guards.patch
uv pip install -e ./sam3
uv pip install opencv-python matplotlib einops triton-windows pycocotools scikit-image psutil

# 4. (optional) HF auth — only if you skip the manual sam3.pt download
hf auth login
```

## Checkpoint

`sam3.pt` (3.45 GB) from https://huggingface.co/facebook/sam3 is all you need
for BOTH scripts — the BPE tokenizer is bundled inside the `sam3` package.

Put it at `checkpoints/sam3.pt` and pass `--ckpt checkpoints/sam3.pt`.
Omit `--ckpt` to let it auto-download from HF instead (needs `hf auth login`).

## Usage

Image — find every "yellow forklift":
```powershell
python scripts/segment_image.py --image inputs/frame.jpg --text "yellow forklift" --ckpt checkpoints/sam3.pt
```

Image — exemplar box instead of text:
```powershell
python scripts/segment_image.py --image inputs/frame.jpg --box 120 340 260 520 --ckpt checkpoints/sam3.pt
```

Video — track a concept through a clip:
```powershell
python scripts/track_video.py --video inputs/clip.mp4 --text "yellow forklift" --ckpt checkpoints/sam3.pt
python scripts/track_video.py --video inputs/clip.mp4 --box 120 340 260 520 --ckpt checkpoints/sam3.pt
```

Both scripts use the native `sam3` package and the same `sam3.pt`.

## Notes on "exemplar" prompts

The released inference APIs take an exemplar as a **box on the target image /
prompt frame**, not as a separate cropped image file. If your sample lives in a
different image, either locate the object in the target frame and box it, or use
a text phrase. Text prompts are currently the most reliable path for video.

## Interactive UI

`scripts/app.py` — a local Gradio app (our own, built on the native `sam3`
package + `sam3.pt`; Meta ships no UI). Three tabs:

1. **Image · concept** — type a phrase -> every matching instance
2. **Image · interactive** — click two corners to draw include / exclude boxes,
   optional text, live threshold — the exemplar-box refinement workflow
3. **Video · track** — text and/or a box on a chosen frame -> tracked, annotated MP4

```powershell
uv pip install gradio          # one-time
.venv\Scripts\python.exe scripts\app.py           # http://127.0.0.1:7860
.venv\Scripts\python.exe scripts\app.py --share   # public link
```

Models load lazily on first use of each tab (image ~1 model, video separate).
Point-click (SAM-1 style) masking isn't wired up — it needs tracker weights the
image checkpoint doesn't carry; box + text prompting covers the same intent.
