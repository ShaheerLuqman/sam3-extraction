"""Embed extracted frames with Qwen3-VL-Embedding (vLLM pooling runner).

Usage:
    python embed_frames.py <frames_dir> [<frames_dir> ...] --stride 1

For each <frames_dir> (written by extract_frames.py) this writes
<frames_dir>/../../embeddings/<video_name>.npz with:
    frames: int32 [N]       frame indices that were embedded
    emb:    float16 [N, D]  L2-normalised embeddings
"""
import argparse
import json
import os
import time

# Pin to GPU 0; GPU 1 is shared with sam3-extraction.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
from PIL import Image
from vllm import LLM

MODEL_ID = os.environ.get("QWEN_EMB_MODEL", "Qwen/Qwen3-VL-Embedding-8B")
INSTRUCTION = (
    "Represent the manual assembly step the worker is performing in this overhead "
    "workstation image: the panel's orientation and state, the tool in use, and the "
    "hand actions."
)
BATCH = 256


def build_prompt(llm: LLM, image_path: str, instruction: str = INSTRUCTION) -> dict:
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": [{"type": "image", "image": "file://" + image_path}]},
    ]
    text = llm.get_tokenizer().apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": text, "multi_modal_data": {"image": Image.open(image_path).convert("RGB")}}


def embed_dir(llm: LLM, frames_dir: str, stride: int, instruction: str = INSTRUCTION,
              suffix: str = "", per_frame=None) -> None:
    """per_frame: optional callable frame_idx -> instruction (overrides `instruction`)."""
    frames_dir = os.path.abspath(frames_dir)
    meta = json.load(open(os.path.join(frames_dir, "meta.json")))
    frames = list(range(0, meta["decoded_frames"], stride))
    out_dir = os.path.join(os.path.dirname(os.path.dirname(frames_dir)), "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(frames_dir) + suffix + ".npz")

    t0 = time.time()
    embs = []
    for i in range(0, len(frames), BATCH):
        chunk = frames[i : i + BATCH]
        prompts = [build_prompt(llm, os.path.join(frames_dir, f"{f:06d}.jpg"),
                                per_frame(f) if per_frame else instruction) for f in chunk]
        outs = llm.embed(prompts, use_tqdm=False)
        embs.extend(o.outputs.embedding for o in outs)
        done = i + len(chunk)
        print(f"{os.path.basename(frames_dir)}: {done}/{len(frames)} "
              f"({done / (time.time() - t0):.1f} img/s)", flush=True)

    emb = np.asarray(embs, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(out_path, frames=np.asarray(frames, dtype=np.int32), emb=emb.astype(np.float16))
    print(f"saved {out_path} {emb.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dirs", nargs="+")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--describe", help="text file with a step description; it is added to the "
                    "embedding instruction so the embeddings focus on that step")
    ap.add_argument("--suffix", default="", help="appended to the output name, e.g. __class5desc")
    ap.add_argument("--class-descriptions", help="JSON {class: description}; with --preds, each "
                    "frame's instruction describes its own labelled class (reference videos only)")
    ap.add_argument("--preds", help="preds.json giving each frame's class for --class-descriptions")
    args = ap.parse_args()

    per_frame = None
    if args.class_descriptions:
        descs = {k: " ".join(v.split()) for k, v in json.load(open(args.class_descriptions)).items()
                 if k.isdigit()}
        labels = json.load(open(args.preds))["preds"]
        prefix = ("Represent this overhead workstation image for recognising the following "
                  "assembly step: ")
        per_frame = lambda f: prefix + " ".join(descs.get(str(c), "") for c in labels[str(f)]["pred"])

    instruction = INSTRUCTION
    if args.describe:
        desc = " ".join(open(args.describe).read().split())
        instruction = ("Represent this overhead workstation image for recognising whether the "
                       "worker is performing the following assembly step: " + desc)
    print("instruction:", instruction, flush=True)

    llm = LLM(
        model=MODEL_ID,
        runner="pooling",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    for d in args.frames_dirs:
        embed_dir(llm, d, args.stride, instruction, args.suffix, per_frame)


if __name__ == "__main__":
    main()
