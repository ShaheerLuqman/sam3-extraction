"""Qwen3-VL-Embedding worker for frame extraction — runs in the *qwen* venv.

    <qwen venv>/bin/python qwen_embed_worker.py '{"model": ..., "gpu_util": 0.85, ...}'

A separate process on purpose: vLLM needs its own torch build (the
qwen_vl/.venv-vllm one), and when it exits every byte of GPU memory it held goes
with it. It imports nothing from the backend.

It loads the model once, prints `READY`, then serves requests — one JSON object
per stdin line — until stdin closes. That lets the backend keep it warm between
runs: the model takes ~45 s to come up and embeds ~10 frames/s after, so a new
reference image on a warm worker is well under a second instead of a minute.

Request:
    {"instruction": "...",              # system prompt; same for frames and images
     "video": {"path": "...", "stride": 5, "out": ".../x.npz"} | null,
     "videos": [{"path", "stride", "out", "label"?}, ...],   # optional, several
     "images": [{"path": "...", "out": ".../y.npy"}, ...]}

Replies on stdout: any number of `PROGRESS <0..1> <stage>`, then `DONE` or
`ERROR <message>`. A request that fails does not end the worker.

Writes, for the video, an npz with `frames` (int32, the embedded frame indices),
`emb` (float16 [N, D], L2-normalised) and `decoded` (how many frames actually
decoded — truncated MP4s list more in their header than they hold); for each
image, a float16 [D] .npy.

The prompt mirrors extraction_project/scripts/embed_frames.py, which is where the
retrieval numbers this feature relies on were measured.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # its JIT needs nvcc, absent here

import cv2
import numpy as np
from PIL import Image


def say(line: str) -> None:
    print(line, flush=True)


def fit(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    s = max_side / max(w, h)
    if s >= 1:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def video_batches(path: str, stride: int, batch: int, max_side: int):
    """Yield (frame_indices, PIL images) every `stride`-th frame, `batch` at a time.

    Sequential decode (grab() for the frames in between), so indices match what
    the frame endpoint and every other cv2 reader in the app call frame N.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    idx, fr, ims = 0, [], []
    try:
        while True:
            if idx % stride == 0:
                ok, bgr = cap.read()
                if not ok:
                    break
                fr.append(idx)
                ims.append(fit(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), max_side))
                if len(fr) == batch:
                    yield fr, ims
                    fr, ims = [], []
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
    if fr:
        yield fr, ims
    yield None, idx  # sentinel: how many frames decoded


class Embedder:
    def __init__(self, cfg: dict) -> None:
        from vllm import LLM

        self.max_side = int(cfg.get("max_side", 768))
        self.batch = int(cfg.get("batch", 64))
        t0 = time.time()
        self.llm = LLM(model=cfg["model"], runner="pooling", dtype="bfloat16", max_model_len=4096,
                       gpu_memory_utilization=float(cfg["gpu_util"]), disable_log_stats=True,
                       # CUDA-graph capture was most of a 90 s startup and buys little
                       # here: the vision encoder, which graphs don't cover, is the bulk
                       enforce_eager=bool(cfg.get("eager", True)),
                       limit_mm_per_prompt={"image": 1, "video": 0})
        self.tok = self.llm.get_tokenizer()
        print(f"model up in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    def embed(self, prompt: str, ims: list) -> np.ndarray:
        outs = self.llm.embed([{"prompt": prompt, "multi_modal_data": {"image": im}} for im in ims],
                              use_tqdm=False)
        e = np.asarray([o.outputs.embedding for o in outs], dtype=np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def handle(self, req: dict) -> None:
        # "video" (one) and "videos" (several: a target plus reference clips) both work
        videos = ([req["video"]] if req.get("video") else []) + list(req.get("videos") or [])
        images = req.get("images") or []
        # The chat template renders the same text whatever the image is, so build it once.
        conv = [{"role": "system", "content": [{"type": "text", "text": req["instruction"]}]},
                {"role": "user", "content": [{"type": "image"}]}]
        prompt = self.tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

        total = len(images)
        for v in videos:
            cap = cv2.VideoCapture(v["path"])
            header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            total += max(1, math.ceil(header / v["stride"]))
        done = 0

        for item in images:
            img = fit(Image.open(item["path"]).convert("RGB"), self.max_side)
            np.save(item["out"], self.embed(prompt, [img])[0].astype(np.float16))
            done += 1
            say(f"PROGRESS {done / max(total, 1):.4f} embedding reference images {done}/{len(images)}")

        for k, video in enumerate(videos):
            what = video.get("label") or ("video frames" if len(videos) == 1 else f"video {k + 1}/{len(videos)}")
            frames, embs, decoded = [], [], 0
            t1 = time.time()
            for fr, ims in video_batches(video["path"], video["stride"], self.batch, self.max_side):
                if fr is None:
                    decoded = ims
                    break
                embs.append(self.embed(prompt, ims))
                frames.extend(fr)
                done += len(fr)
                rate = len(frames) / max(time.time() - t1, 1e-6)
                say(f"PROGRESS {min(done / max(total, 1), 0.999):.4f} embedding {what}: "
                    f"{len(frames)} frames ({rate:.1f}/s)")
            if not frames:
                raise RuntimeError(f"{what} decoded to zero frames")
            emb = np.concatenate(embs).astype(np.float16)
            tmp = video["out"] + ".tmp.npz"  # np.savez appends .npz unless it is already there
            np.savez(tmp, frames=np.asarray(frames, np.int32), emb=emb, decoded=np.int32(decoded))
            os.replace(tmp, video["out"])


def main(cfg: dict) -> None:
    emb = Embedder(cfg)
    say("READY")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            emb.handle(json.loads(line))
            say("DONE")
        except Exception as exc:  # noqa: BLE001 - report it, stay up for the next request
            traceback.print_exc(file=sys.stderr)
            say(f"ERROR {type(exc).__name__}: {exc}".replace("\n", " "))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(json.loads(sys.argv[1]))
