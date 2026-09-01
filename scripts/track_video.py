"""
SAM 3 video concept tracking via the native facebookresearch/sam3 predictor.

Give it a concept on frame 0 (text phrase and/or an exemplar box) and it
detects + tracks every matching instance through the whole clip, writing an
annotated MP4.

Examples:
  python scripts/track_video.py --video inputs/clip.mp4 --text "yellow forklift"
  python scripts/track_video.py --video inputs/clip.mp4 --box 120 340 260 520
  python scripts/track_video.py --video inputs/frames_dir --text "person" --frame 0

Box format: x1 y1 x2 y2 in pixels on the prompt frame.
Checkpoint: pass --ckpt path/to/sam3.pt, or rely on HF auto-download (hf auth login).
"""
import argparse
import glob
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True,
                   help="path to .mp4 or a folder of numbered .jpg frames")
    p.add_argument("--text", default=None, help="concept phrase")
    p.add_argument("--box", nargs=4, type=float, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="exemplar box on the prompt frame (pixels)")
    p.add_argument("--frame", type=int, default=0, help="prompt frame index")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="path to sam3.pt (else auto-download from HF)")
    p.add_argument("--out", type=Path, default=Path("outputs/tracked.webm"),
                   help=".webm (VP8, browser-playable) or .mp4 (mp4v)")
    p.add_argument("--fps", type=float, default=24.0)
    return p.parse_args()


def open_writer(out_path, fps, size):
    """.webm -> VP8 (plays in browsers/Gradio); else mp4v. Falls back if unavailable."""
    ext = out_path.suffix.lower()
    fourcc = "VP80" if ext == ".webm" else "mp4v"
    w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if w.isOpened():
        return w, out_path
    alt = out_path.with_suffix(".mp4")
    print(f"[warn] {fourcc} unavailable — writing {alt}")
    return cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*"mp4v"), fps, size), alt


def load_frames(video):
    if str(video).lower().endswith(".mp4"):
        cap = cv2.VideoCapture(str(video))
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    files = sorted(glob.glob(os.path.join(str(video), "*.jpg")))
    return [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]


def color_for(obj_id):
    rng = np.random.default_rng(int(obj_id) + 12345)
    return rng.integers(60, 256, size=3)


def propagate(predictor, session_id):
    outputs_per_frame = {}
    for resp in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[resp["frame_index"]] = resp["outputs"]
    return outputs_per_frame


def main():
    args = parse_args()
    if not args.text and not args.box:
        raise SystemExit("Provide --text and/or --box")

    from sam3.model_builder import build_sam3_video_predictor

    frames = load_frames(args.video)
    if not frames:
        raise SystemExit(f"no frames read from {args.video}")
    H, W = frames[0].shape[:2]
    print(f"{len(frames)} frames @ {W}x{H}")

    gpus = list(range(torch.cuda.device_count())) or None
    kw = {}
    if args.ckpt:
        kw["checkpoint_path"] = str(args.ckpt)
    predictor = build_sam3_video_predictor(gpus_to_use=gpus, **kw)

    resp = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(args.video))
    )
    session_id = resp["session_id"]

    req = dict(type="add_prompt", session_id=session_id, frame_index=args.frame)
    if args.text:
        req["text"] = args.text
    if args.box:
        x1, y1, x2, y2 = args.box
        # predictor wants [xmin, ymin, w, h] normalised to [0, 1] w.r.t. the video
        nb = [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]
        if min(nb) < 0 or nb[0] + nb[2] > 1.001 or nb[1] + nb[3] > 1.001:
            raise SystemExit(
                f"--box {int(x1)} {int(y1)} {int(x2)} {int(y2)} is outside the "
                f"{W}x{H} frame; give coords in this video's pixel space.")
        req["bounding_boxes"] = [[max(0.0, min(1.0, v)) for v in nb]]
        req["bounding_box_labels"] = [1]

    ac = (torch.autocast("cuda", dtype=torch.bfloat16)
          if torch.cuda.is_available() else nullcontext())
    with ac:  # box/visual-prompt path isn't autocast-wrapped upstream
        predictor.handle_request(request=req)
        outputs_per_frame = propagate(predictor, session_id)

    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_ids = set()
    for idx, rgb in enumerate(frames):
        canvas = rgb.astype(np.float32)
        out = outputs_per_frame.get(idx)
        if out is not None:
            obj_ids = out["out_obj_ids"].tolist()
            masks = out["out_binary_masks"]
            for i, oid in enumerate(obj_ids):
                m = masks[i]
                m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                m = np.squeeze(m).astype(bool)
                if not m.any():
                    continue
                total_ids.add(oid)
                canvas[m] = 0.5 * canvas[m] + 0.5 * color_for(oid).astype(np.float32)
                ys, xs = np.where(m)
                cv2.putText(canvas, f"#{oid}", (int(xs.min()), max(int(ys.min()) - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(cv2.cvtColor(canvas.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()

    predictor.handle_request(dict(type="close_session", session_id=session_id))
    predictor.shutdown()
    print(f"tracked {len(total_ids)} instance(s) -> {args.out}")


if __name__ == "__main__":
    main()
