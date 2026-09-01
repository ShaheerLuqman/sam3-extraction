"""
SAM 3 single-image concept segmentation (native facebookresearch/sam3).

Finds/segments every instance of a concept in ONE image, prompted by:
  - a text phrase ("red backpack"), and/or
  - a bounding-box exemplar (a box around one example of the object),
    plus optional negative boxes to exclude look-alikes.

Examples:
  python scripts/segment_image.py --image inputs/frame.jpg --text "yellow forklift"
  python scripts/segment_image.py --image inputs/frame.jpg --box 120 340 260 520
  python scripts/segment_image.py --image inputs/frame.jpg --text "box" --neg-box 10 10 90 90

Box format: x1 y1 x2 y2 in pixels (top-left, bottom-right).
Checkpoint: --ckpt checkpoints/sam3.pt  (or omit to auto-download from HF).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--text", default=None, help="concept phrase")
    p.add_argument("--box", nargs=4, type=float, action="append", default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="positive exemplar box (pixels); repeatable")
    p.add_argument("--neg-box", nargs=4, type=float, action="append", default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="negative box to exclude a region/object; repeatable")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/segmented.png"))
    return p.parse_args()


def to_norm_cxcywh(box_xyxy, w, h):
    x1, y1, x2, y2 = box_xyxy
    return [((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, abs(x2 - x1) / w, abs(y2 - y1) / h]


def main():
    args = parse_args()
    if not args.text and not args.box:
        raise SystemExit("Provide --text and/or --box")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {}
    if args.ckpt:
        kw.update(checkpoint_path=str(args.ckpt), load_from_HF=False)
    print("loading model ...")
    model = build_sam3_image_model(device=device, **kw)
    processor = Sam3Processor(model, device=device, confidence_threshold=args.threshold)

    image = Image.open(args.image).convert("RGB")
    W, H = image.size

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda" else torch.inference_mode())
    with autocast:
        state = processor.set_image(image)
        if args.text:
            state = processor.set_text_prompt(args.text, state)
        for b in (args.box or []):
            state = processor.add_geometric_prompt(to_norm_cxcywh(b, W, H), True, state)
        for b in (args.neg_box or []):
            state = processor.add_geometric_prompt(to_norm_cxcywh(b, W, H), False, state)

    masks = state.get("masks")
    n = 0 if masks is None else len(masks)
    print(f"found {n} instance(s)")

    arr = np.array(image).astype(np.float32)
    rng = np.random.default_rng(0)
    for i in range(n):
        m = np.squeeze(masks[i].cpu().numpy()).astype(bool)
        color = rng.integers(60, 256, size=3).astype(np.float32)
        arr[m] = 0.45 * arr[m] + 0.55 * color
    overlay = Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(overlay)
    for i in range(n):
        x1, y1, x2, y2 = state["boxes"][i].cpu().tolist()
        s = float(state["scores"][i])
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)
        draw.text((x1 + 2, y1 + 2), f"{s:.2f}", fill=(255, 255, 0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
