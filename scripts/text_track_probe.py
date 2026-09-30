"""
Probe SAM 3 text/concept tracking on ONE video, one class at a time.

For every class we turn the class name into a natural-language phrase, run SAM 3
video concept detection + tracking over the whole clip with that phrase, and keep
ONE reference instance: the single (frame, object) with the highest detection
score. Classes SAM 3 never finds are skipped in the JSON.

Outputs (into --out-dir, default outputs/textprobe/<video stem>/):
  * <class>.mp4          annotated track for that phrase (masks + #id)
  * <video stem>_textrefs.json
        [{"name", "id", "prompt", "frame", "box": [x1,y1,x2,y2],
          "score", "peak_instances", "frames_present"}]

  python scripts/text_track_probe.py \
      --video "inputs/datasets/Stellantis Station 140 (3.3)/Videos/2026-08-03 06_51_58.mp4" \
      --names-json "<same>_dets.json" --phrases scripts/phrases_stellantis.json
"""
import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--names-json", type=Path, default=None,
                   help="a *_dets.json (uses meta.names) or a plain {id: name} map; "
                        "default: <video>_dets.json next to the video")
    p.add_argument("--phrases", type=Path, default=None,
                   help="optional {class_name: phrase} JSON; names without an "
                        "entry fall back to the class name with '_'->' '")
    p.add_argument("--only", nargs="*", default=None,
                   help="only probe these class names")
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sam3.pt"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--det-thr", type=float, default=0.5,
                   help="detection score threshold (lower = more, weaker hits)")
    p.add_argument("--single-gpu", action="store_true")
    p.add_argument("--offload", action="store_true",
                   help="keep decoded frames + tracker state on CPU RAM "
                        "(slower, but avoids GPU OOM on long / busy clips)")
    return p.parse_args()


def load_names(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d.get("meta", {}).get("names") or d.get("names") or d


def read_frames(path, max_frames):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def color_for(oid):
    rng = np.random.default_rng(int(oid) + 12345)
    return rng.integers(60, 256, size=3).astype(np.float32)


def mask_to_xyxy(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def main():
    args = parse_args()
    names_json = args.names_json or args.video.with_name(args.video.name + "_dets.json")
    names = load_names(names_json)                     # {id_str: name}
    phrases = json.loads(args.phrases.read_text()) if args.phrases else {}

    out_dir = args.out_dir or (Path("outputs/textprobe") / args.video.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, fps = read_frames(args.video, args.max_frames)
    if not frames:
        raise SystemExit(f"no frames read from {args.video}")
    H, W = frames[0].shape[:2]
    n = len(frames)
    print(f"{n} frames @ {W}x{H}")

    items = [(int(i), nm) for i, nm in names.items()]
    items.sort()
    if args.only:
        items = [(i, nm) for i, nm in items if nm in args.only]

    from sam3.model_builder import build_sam3_video_predictor
    gpus = [torch.cuda.current_device()] if (args.single_gpu and torch.cuda.is_available()) \
        else (list(range(torch.cuda.device_count())) or None)
    kw = {"checkpoint_path": str(args.ckpt)} if args.ckpt.exists() else {}
    predictor = build_sam3_video_predictor(gpus_to_use=gpus, **kw)
    try:
        predictor.model.score_threshold_detection = float(args.det_thr)
    except Exception:
        pass

    # SAM 3's _recondition_masklets can crash (IndexError) when a frame has a
    # tracker<->detection match but an empty detection-mask tensor. It's a
    # best-effort quality heuristic whose return value is ignored at the call
    # site, so guard it: on failure, skip reconditioning for that frame.
    # (Only patches this process's model -> pair with --single-gpu.)
    _cls = type(predictor.model)
    if not getattr(_cls, "_recondition_guarded", False):
        _orig_recond = _cls._recondition_masklets

        def _safe_recond(self, *a, **kw):
            try:
                return _orig_recond(self, *a, **kw)
            except (IndexError, RuntimeError) as e:
                print(f"  [warn] skipped reconditioning: {type(e).__name__}: {e}")
                return a[2] if len(a) > 2 else None

        _cls._recondition_masklets = _safe_recond
        _cls._recondition_guarded = True

    # `_cache_frame_outputs` stashes per-frame mask tensors for EVERY frame to
    # support interactive refinement, which this probe never does. On a long /
    # busy clip that cache grows until the GPU OOMs (offload_state_to_cpu doesn't
    # cover it). Move the stashed masks to CPU as they're cached -> plain
    # propagation still works, GPU stops filling up.
    if args.offload and not getattr(_cls, "_cache_on_cpu", False):
        _orig_cache = _cls._cache_frame_outputs

        def _cache_cpu(self, inference_state, frame_idx, obj_id_to_mask, *a, **kw):
            moved = {k: (v.detach().cpu() if hasattr(v, "detach") else v)
                     for k, v in obj_id_to_mask.items()}
            return _orig_cache(self, inference_state, frame_idx, moved, *a, **kw)

        _cls._cache_frame_outputs = _cache_cpu
        _cls._cache_on_cpu = True

    sid = predictor.handle_request(dict(
        type="start_session", resource_path=str(args.video),
        offload_video_to_cpu=args.offload,
        offload_state_to_cpu=args.offload,
    ))["session_id"]
    ac = (torch.autocast("cuda", dtype=torch.bfloat16)
          if torch.cuda.is_available() else nullcontext())

    results = []
    per_frame_cache = {}                     # phrase -> per_frame dict (dedup)
    for cid, name in items:
        phrase = phrases.get(name) or name.replace("_", " ")
        print(f"\n[{name}]  prompt = {phrase!r}")
        per_frame = per_frame_cache.get(phrase)
        if per_frame is None:
            predictor.handle_request(dict(type="reset_session", session_id=sid))
            per_frame = {}
            try:
                with ac:
                    predictor.handle_request(dict(
                        type="add_prompt", session_id=sid, frame_index=0, text=phrase))
                    for resp in predictor.handle_stream_request(
                            dict(type="propagate_in_video", session_id=sid)):
                        per_frame[resp["frame_index"]] = resp["outputs"]
            except torch.cuda.OutOfMemoryError:
                print(f"  [OOM] '{phrase}' ran out of GPU memory - skipped")
                per_frame = None
                torch.cuda.empty_cache()
                try:
                    predictor.handle_request(dict(type="reset_session", session_id=sid))
                except Exception:
                    pass
                results.append({"name": name, "id": cid, "prompt": phrase,
                                "status": "oom", "det_thr": args.det_thr})
                continue
            per_frame_cache[phrase] = per_frame
        else:
            print(f"  (reusing propagation for prompt {phrase!r})")

        best = None                 # (score, frame_idx, box)
        ids_seen, peak, frames_present = set(), 0, 0
        for fi, out in per_frame.items():
            oids = out["out_obj_ids"].tolist()
            probs = out.get("out_probs")
            probs = probs.tolist() if hasattr(probs, "tolist") else [None] * len(oids)
            here = 0
            for k, oid in enumerate(oids):
                m = np.squeeze(np.asarray(
                    out["out_binary_masks"][k].cpu().numpy()
                    if hasattr(out["out_binary_masks"][k], "cpu")
                    else out["out_binary_masks"][k])).astype(bool)
                if not m.any():
                    continue
                here += 1
                ids_seen.add(oid)
                sc = probs[k] if probs[k] is not None else 0.0
                if best is None or sc > best[0]:
                    box = mask_to_xyxy(m)
                    if box is not None:
                        best = (float(sc), fi, box)
            peak = max(peak, here)
            frames_present += here > 0
        n_sess = (max(per_frame) + 1) if per_frame else n

        # annotated video for this phrase
        vp = out_dir / f"{name}.mp4"
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for fi, rgb in enumerate(frames):
            canvas = rgb.astype(np.float32)
            out = per_frame.get(fi)
            if out is not None:
                for k, oid in enumerate(out["out_obj_ids"].tolist()):
                    m = np.squeeze(np.asarray(
                        out["out_binary_masks"][k].cpu().numpy()
                        if hasattr(out["out_binary_masks"][k], "cpu")
                        else out["out_binary_masks"][k])).astype(bool)
                    if not m.any():
                        continue
                    canvas[m] = 0.5 * canvas[m] + 0.5 * color_for(oid)
                    ys, xs = np.where(m)
                    cv2.putText(canvas, f"#{oid}", (int(xs.min()), max(int(ys.min()) - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, phrase, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
            vw.write(cv2.cvtColor(canvas.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        vw.release()

        if best is None:
            print(f"  no detections -> {vp.name} (empty), skipped in JSON")
            continue
        score, fi, box = best
        print(f"  ref: frame {fi}  box {box}  score {score:.3f}  | "
              f"peak {peak} inst, present on {frames_present}/{n_sess} frames -> {vp.name}")
        results.append({
            "name": name, "id": cid, "prompt": phrase,
            "frame": fi, "box": box, "score": round(score, 4),
            "peak_instances": peak, "frames_present": frames_present,
            "total_frames": n_sess,
            "det_thr": args.det_thr,
        })

    predictor.handle_request(dict(type="close_session", session_id=sid))
    try:
        predictor.shutdown()
    except Exception:
        pass

    out_json = out_dir / f"{args.video.stem}_textrefs.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out_json}  ({len(results)}/{len(items)} classes found)")


if __name__ == "__main__":
    main()
