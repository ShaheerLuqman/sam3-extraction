"""
Track bounding boxes through a video with SAM 3 (SAM 2-style interactive tracker).

Give it one or more boxes, each pinned to a frame where that object is clearly
visible. SAM 3 segments what is inside each box on its prompt frame and follows
that exact object through the rest of the clip. Different objects can be prompted
on different frames (useful when an object only enters partway through).

Outputs:
  * a JSON file  - per-frame [x1, y1, x2, y2] for each tracked target
                   (null on frames before its prompt frame, or where it is lost)
  * (optional)   - an annotated .mp4 with the tracked boxes drawn on

Uses SAM 3's instance tracker (`build_sam3_video_model().tracker`), NOT the
text/concept detector - one input box tracks exactly one object.

Boxes are x1 y1 x2 y2 in PIXELS.

Examples
--------
  # one box on frame 0
  python scripts/track_bbox.py --video clip.mp4 --box 856 173 1095 412

  # three boxes, all on frame 0
  python scripts/track_bbox.py --video clip.mp4 \
      --box 856 173 1095 412 --box 300 1 531 141 --box 174 219 295 353 \
      --out-video outputs/clip_tracked.mp4

  # boxes pinned to different frames (object 1 enters at frame 90)
  python scripts/track_bbox.py --video clip.mp4 \
      --box 856 173 1095 412 \
      --box-at 90 40 200 260 470

  # everything from a JSON file - items may be
  #   [x1,y1,x2,y2]              -> prompt frame = --frame (default 0)
  #   [frame, x1,y1,x2,y2]
  #   {"frame": N, "box": [x1,y1,x2,y2]}
  python scripts/track_bbox.py --video clip.mp4 --boxes-json boxes.json

  # also track each object backwards from its prompt frame
  python scripts/track_bbox.py --video clip.mp4 --box-at 120 ... --bidirectional

Backbone feature cache
----------------------
The vision backbone is the bulk of the per-frame cost and only depends on the
pixels. The FIRST run for a video computes it and stores it under
`features/<video filename>/`; every later run for the same video reuses that
cache and skips the backbone (~5x faster propagation, and the full frame tensor
never has to fit in RAM). `--stride` only matters on that first (build) run.
  --rebuild-cache   recompute even if a cache exists
  --no-cache        don't build or use a cache (pure live run)
  --features-dir D  use / build the cache at D instead of the default location

Checkpoint: --ckpt checkpoints/sam3.pt  (or omit to auto-download from HF).
"""
import argparse
import gc
import json
import math
import os
import shutil
import tempfile
import zlib
from pathlib import Path

# reduce CUDA fragmentation OOMs on long multi-object propagations (must be set
# before torch initialises the allocator). max_split_size_mb is cross-platform;
# override the whole var from the environment if you want expandable_segments etc.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import cv2
import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path,
                   help="path to an .mp4 (or a folder of numbered .jpg frames)")
    p.add_argument("--box", nargs=4, type=float, action="append", default=[],
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="a target box on the --frame frame, pixels; repeatable")
    p.add_argument("--box-at", nargs=5, type=float, action="append", default=[],
                   metavar=("FRAME", "X1", "Y1", "X2", "Y2"),
                   help="a target box pinned to a specific frame; repeatable")
    p.add_argument("--boxes-json", type=Path, default=None,
                   help="JSON list of boxes (see module docstring for accepted item shapes)")
    p.add_argument("--features-dir", type=Path, default=None,
                   help="build/use the backbone cache at this dir "
                        "(default: features/<video filename>)")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="recompute the backbone cache even if one already exists")
    p.add_argument("--no-cache", action="store_true",
                   help="neither build nor use a cache - run the backbone live")
    p.add_argument("--frame", type=int, default=0,
                   help="default prompt frame for --box / bare-4 JSON items (default 0)")
    p.add_argument("--bidirectional", action="store_true",
                   help="also track each object backwards from its prompt frame "
                        "(default: forward only)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="only process the first N ORIGINAL video frames "
                        "(caps memory/time; cache is not affected)")
    p.add_argument("--stride", type=int, default=1,
                   help="when BUILDING the cache: keep every Nth frame (default 1 = all). "
                        "Long clips (7000+ frames) can't be cached whole - stride them. "
                        "Ignored when an existing cache is reused.")
    p.add_argument("--offload", action="store_true",
                   help="keep frames + tracker state on CPU RAM (slower, but needed "
                        "for long clips / many objects on a small GPU)")
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sam3.pt"),
                   help="path to sam3.pt (omit / pass a missing path to use HF)")
    p.add_argument("--names", type=Path, default=None,
                   help="JSON with a class-id -> name map (a *_dets.json with "
                        "meta.names, or a plain {\"0\": \"person\", ...}); output is "
                        "then keyed by name instead of id")
    p.add_argument("--out-json", type=Path, default=None,
                   help="output tracks JSON (default: outputs/<video>_track.json)")
    p.add_argument("--out-video", type=Path, default=None,
                   help="optional annotated .mp4 to write")
    p.add_argument("--fps", type=float, default=24.0,
                   help="fps for --out-video (default 24)")
    return p.parse_args()


def collect_targets(args):
    """-> list of (obj_id, frame_idx, [x1,y1,x2,y2] floats), in the order given.

    obj_id is the caller's label for the object (defaults to its position); it is
    what the output JSON is keyed by.
    """
    targets = []
    if args.boxes_json:
        for pos, item in enumerate(json.loads(args.boxes_json.read_text())):
            if isinstance(item, dict):
                targets.append((item.get("name", item.get("id", item.get("cls", pos))),
                                int(item.get("frame", args.frame)),
                                [float(v) for v in item["box"]]))
            elif len(item) == 5:
                targets.append((pos, int(item[0]), [float(v) for v in item[1:]]))
            elif len(item) == 4:
                targets.append((pos, args.frame, [float(v) for v in item]))
            else:
                raise SystemExit(f"bad box item in {args.boxes_json}: {item!r}")
    for b in args.box:
        targets.append((len(targets), args.frame, [float(v) for v in b]))
    for b in args.box_at:
        targets.append((len(targets), int(b[0]), [float(v) for v in b[1:]]))
    return targets


def load_frames(video: Path, stride: int = 1):
    """-> (list of RGB frames, list of their original frame indices)."""
    if video.is_dir():
        files = sorted(video.glob("*.jpg"))[::stride]
        idxs = list(range(0, len(sorted(video.glob("*.jpg"))), stride))
        return [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files], idxs
    cap = cv2.VideoCapture(str(video))
    frames, idxs, i = [], [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            idxs.append(i)
        i += 1
    cap.release()
    return frames, idxs


def mask_to_xyxy(m):
    """bool mask -> [x1, y1, x2, y2] (ints) or None if empty."""
    if not m.any():
        return None
    ys, xs = np.where(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def color_for(key):
    rng = np.random.default_rng(zlib.crc32(str(key).encode()))
    return tuple(int(v) for v in rng.integers(60, 256, size=3))


class LazyFeatureCache(dict):
    """Feeds `inference_state["cached_features"]` from a precompute_features.py dir.

    `_get_image_feature` only ever reads the *current* frame's entry, so we load
    each `NNNNN.pt` on demand and keep a tiny LRU. The shared positional-encoding
    maps and a zero image tensor (unused by this model's SimpleMaskEncoder memory
    path) are held once.
    """

    def __init__(self, feat_dir: Path, n: int, image_size: int):
        super().__init__()
        self._dir = feat_dir
        self._n = n
        self._pos = [t.cuda() for t in torch.load(feat_dir / "pos_enc.pt",
                                                  map_location="cpu")]
        self._img = torch.zeros(1, 3, image_size, image_size, device="cuda")
        self._lru = {}
        self._order = []

    def get(self, frame_idx, default=None):
        if not 0 <= frame_idx < self._n:
            return default
        hit = self._lru.get(frame_idx)
        if hit is None:
            fpn = [t.cuda() for t in torch.load(self._dir / f"{frame_idx:05d}.pt",
                                                map_location="cpu")]
            hit = (self._img, {"backbone_fpn": fpn, "vision_pos_enc": self._pos})
            self._lru[frame_idx] = hit
            self._order.append(frame_idx)
            if len(self._order) > 6:
                self._lru.pop(self._order.pop(0), None)
        return hit

    def __getitem__(self, k):
        v = self.get(k)
        if v is None:
            raise KeyError(k)
        return v

    def __contains__(self, k):
        return 0 <= k < self._n

    def __len__(self):
        return self._n

    def __bool__(self):
        return True


def load_valid_cache_meta(cache_dir: Path, video: Path):
    """Return the cache meta dict if `cache_dir` holds a complete cache for
    `video` (matched by byte size), else None."""
    mp = cache_dir / "meta.json"
    if not mp.exists() or not (cache_dir / "pos_enc.pt").exists():
        return None
    try:
        meta = json.loads(mp.read_text())
    except (ValueError, OSError):
        return None
    if meta.get("video_bytes") != video.stat().st_size:
        return None
    n = meta.get("num_cached", 0)
    if n <= 0 or not (cache_dir / f"{n - 1:05d}.pt").exists() \
            or not (cache_dir / "00000.pt").exists():
        return None
    return meta


def build_feature_cache(video: Path, out_dir: Path, stride: int, ckpt: Path):
    """Run the vision backbone over the video and store per-frame features.

    Writes: pos_enc.pt (once), NNNNN.pt per frame (3 backbone_fpn maps, bf16),
    meta.json. Uses its own model instance, freed before returning.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, orig_idxs = load_frames(video, stride)
    if not frames:
        raise SystemExit(f"no frames read from {video}")
    H, W = frames[0].shape[:2]
    n = len(frames)
    print(f"[cache] computing backbone features: {n} frames (stride {stride}) "
          f"@ {W}x{H} -> {out_dir}")

    from sam3.model.utils.sam2_utils import _load_img_as_tensor
    from sam3.model_builder import build_sam3_video_model

    use_ckpt = ckpt and ckpt.exists()
    model = build_sam3_video_model(
        checkpoint_path=str(ckpt) if use_ckpt else None, load_from_HF=not use_ckpt,
    )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    image_size = predictor.image_size

    tmp = Path(tempfile.mkdtemp(prefix="sam3_pf_"))
    mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    std = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    pos_saved = False
    try:
        for k, rgb in enumerate(frames):
            jp = tmp / "f.jpg"
            Image.fromarray(rgb).save(jp, quality=95)
            img, _, _ = _load_img_as_tensor(jp, image_size)
            img = ((img - mean) / std).cuda().float().unsqueeze(0)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                bo = predictor.forward_image(img)
            torch.save([t[0].to(torch.bfloat16).cpu().clone()
                        for t in bo["backbone_fpn"]], out_dir / f"{k:05d}.pt")
            if not pos_saved:
                torch.save([t[0].to(torch.bfloat16).cpu().clone()
                            for t in bo["vision_pos_enc"]], out_dir / "pos_enc.pt")
                pos_saved = True
            if (k + 1) % 100 == 0 or k + 1 == n:
                print(f"[cache]   {k + 1}/{n}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    (out_dir / "meta.json").write_text(json.dumps({
        "video": str(video),
        "video_bytes": video.stat().st_size,
        "image_size": image_size,
        "resolution": [W, H],
        "frame_stride": stride,
        "num_cached": n,
        "sampled_frame_indices": orig_idxs,
    }, indent=2))

    del predictor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.pt")) / 1e6
    print(f"[cache] done: {n} frames, {total_mb:.0f} MB in {out_dir}")


def main():
    args = parse_args()

    targets = collect_targets(args)
    if not targets:
        raise SystemExit("give at least one --box / --box-at / --boxes-json")

    # Decide on the feature cache: default location is features/<video filename>.
    # First run for a video builds it; later runs reuse it. --no-cache opts out.
    if not args.no_cache:
        cache_dir = args.features_dir or (Path("features") / args.video.name)
        meta = None if args.rebuild_cache else load_valid_cache_meta(cache_dir, args.video)
        if meta is None:
            if not args.video.exists():
                raise SystemExit(f"video not found: {args.video}")
            build_feature_cache(args.video, cache_dir, args.stride, args.ckpt)
        else:
            print(f"[cache] reusing {cache_dir} "
                  f"({meta['num_cached']} frames, stride {meta['frame_stride']})")
            if args.stride != 1 and args.stride != meta["frame_stride"]:
                print(f"[cache] note: --stride {args.stride} ignored; cache is "
                      f"stride {meta['frame_stride']} (use --rebuild-cache to change)")
        args.features_dir = cache_dir
    else:
        args.features_dir = None  # --no-cache: pure live run

    if args.features_dir:
        meta = json.loads((args.features_dir / "meta.json").read_text())
        W, H = meta["resolution"]
        orig_idxs = meta["sampled_frame_indices"]
        stride = meta["frame_stride"]
        n = meta["num_cached"]
        image_size = meta["image_size"]
        frames = None
        if args.out_video:
            frames, _ = load_frames(args.video, stride)
            if len(frames) != n:
                print(f"[warn] decoded {len(frames)} frames, cache has {n} - video mismatch?")
        print(f"{n} cached frames (stride {stride}) @ {W}x{H}; {len(targets)} target box(es)")
    else:
        frames, orig_idxs = load_frames(args.video, args.stride)
        if not frames:
            raise SystemExit(f"no frames read from {args.video}")
        H, W = frames[0].shape[:2]
        n = len(frames)
        stride = args.stride
        print(f"{n} sampled frames (stride {stride}) @ {W}x{H}; "
              f"{len(targets)} target box(es)")
    n_orig = orig_idxs[-1] + 1 if orig_idxs else 0

    # optional cap on how many frames to actually track (memory / time)
    n_track = n
    if args.max_frames is not None:
        n_track = max(1, min(n, math.ceil(args.max_frames / stride)))
        orig_idxs = orig_idxs[:n_track]
        if frames is not None:
            frames = frames[:n_track]
        print(f"limiting tracking to the first {n_track} sampled frames "
              f"(~{args.max_frames} original)")

    # remap each target's prompt frame (original index) to the nearest sampled index;
    # drop targets whose prompt frame is past the tracked range.
    remapped = []
    for obj_id, f_orig, b in targets:
        if f_orig >= min(n_orig, orig_idxs[-1] + 1):
            print(f"  obj {obj_id}: SKIP - prompt frame {f_orig} is beyond the "
                  f"tracked range ({orig_idxs[-1] + 1})")
            continue
        s_idx = min(range(n_track), key=lambda j: abs(orig_idxs[j] - f_orig))
        if not (0 <= b[0] < b[2] <= W + 1 and 0 <= b[1] < b[3] <= H + 1):
            raise SystemExit(f"box {b} (obj {obj_id}) is outside {W}x{H} / not x1<x2,y1<y2")
        remapped.append((obj_id, s_idx, b))
        print(f"  obj {obj_id}: orig frame {f_orig} -> sample {s_idx} "
              f"(frame {orig_idxs[s_idx]})  {[int(v) for v in b]}")
    if not remapped:
        raise SystemExit("every target's prompt frame is beyond the video")
    targets = [(s_idx, b) for _, s_idx, b in remapped]
    orig_obj_ids = [k for k, _, _ in remapped]

    # optional id -> name relabelling for the output keys
    if args.names:
        nd = json.loads(args.names.read_text())
        name_map = nd.get("meta", {}).get("names") or nd.get("names") or nd
        seen = {}
        labelled = []
        for k in orig_obj_ids:
            lbl = str(name_map.get(str(k), k))
            if lbl in seen:
                seen[lbl] += 1
                lbl = f"{lbl}#{seen[lbl]}"
            else:
                seen[lbl] = 0
            labelled.append(lbl)
        orig_obj_ids = labelled

    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    from sam3.model_builder import build_sam3_video_model

    use_ckpt = args.ckpt and args.ckpt.exists()
    model = build_sam3_video_model(
        checkpoint_path=str(args.ckpt) if use_ckpt else None,
        load_from_HF=not use_ckpt,
    )
    predictor = model.tracker

    if args.features_dir:
        # backbone is precomputed - drop it so any cache miss fails loudly, and
        # feed the tracker its features straight from disk.
        predictor.backbone = None
        state = predictor.init_state(
            video_height=H, video_width=W, num_frames=n,
            offload_state_to_cpu=args.offload,
        )
        state["cached_features"] = LazyFeatureCache(args.features_dir, n, image_size)
    else:
        predictor.backbone = model.detector.backbone
        # SAM3's tracker loads frames from an .mp4 via `decord` (not installed here)
        # or a folder of "<idx>.jpg" files - so dump our decoded frames to a temp
        # JPEG dir and point it at that (same pixels the model would have seen).
        frame_dir = Path(tempfile.mkdtemp(prefix="sam3_frames_"))
        try:
            for i, rgb in enumerate(frames):
                cv2.imwrite(str(frame_dir / f"{i:05d}.jpg"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            state = predictor.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=args.offload,
                offload_state_to_cpu=args.offload,
            )
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)
    predictor.clear_all_points_in_video(state)

    # register each target box on its own prompt frame (obj_id = its index)
    for k, (f_idx, (x1, y1, x2, y2)) in enumerate(targets):
        norm_box = torch.tensor([x1 / W, y1 / H, x2 / W, y2 / H], dtype=torch.float32)
        predictor.add_new_points_or_box(
            state, frame_idx=f_idx, obj_id=k, box=norm_box, clear_old_points=True,
        )

    tracks = {str(k): [None] * n_track for k in range(len(targets))}

    def collect(start_frame_idx, reverse):
        for f_idx, obj_ids, _low, video_res_masks, _scores in predictor.propagate_in_video(
            state, start_frame_idx=start_frame_idx, max_frame_num_to_track=n_track,
            reverse=reverse, propagate_preflight=True,
        ):
            if f_idx >= n_track:
                continue
            masks = (video_res_masks > 0.0).squeeze(1).cpu().numpy()  # [obj, H, W]
            for i, oid in enumerate(obj_ids):
                box = mask_to_xyxy(masks[i].astype(bool))
                if box is not None:
                    tracks[str(oid)][f_idx] = box
            if torch.cuda.is_available() and f_idx % 40 == 0:
                torch.cuda.empty_cache()

    prompt_frames = [f for f, _ in targets]
    # forward pass from the earliest prompt frame covers each object from its own
    # prompt frame onward; optionally a reverse pass from the latest prompt frame.
    collect(min(prompt_frames), reverse=False)
    if args.bidirectional and max(prompt_frames) > 0:
        collect(max(prompt_frames), reverse=True)

    # Adding object B's prompt on frame f makes f a "conditioning frame"; every
    # OTHER object then gets a placeholder (empty) output on exactly frame f and
    # is skipped during propagation there. Patch that single-frame hole by
    # interpolating from the neighbours when both are present.
    foreign = set(prompt_frames)
    for k, (own_f, _) in enumerate(targets):
        t = tracks[str(k)]
        if not args.bidirectional:
            # forward-only: don't report an object before its own prompt frame
            # (SAM3's memory is not strictly causal and can leak a few frames back)
            for f in range(own_f):
                t[f] = None
        for f in foreign:
            if f == own_f or not 0 < f < n_track - 1 or t[f] is not None:
                continue
            if t[f - 1] is not None and t[f + 1] is not None:
                t[f] = [int(round((a + b) / 2)) for a, b in zip(t[f - 1], t[f + 1])]

    # re-key everything by the caller's original object id (class index)
    tracks_out = {str(orig_obj_ids[k]): tracks[str(k)] for k in range(len(targets))}
    for k in range(len(targets)):
        hits = sum(v is not None for v in tracks[str(k)])
        print(f"  obj {orig_obj_ids[k]}: tracked on {hits}/{n_track} sampled frames")

    out_json = args.out_json or Path("outputs") / f"{args.video.stem}_track.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "meta": {
            "video": str(args.video),
            "frame_count": n_track,
            "frame_stride": stride,
            "sampled_frame_indices": orig_idxs,
            "resolution": [W, H],
            "bidirectional": args.bidirectional,
            "box_format": "[x1, y1, x2, y2]  (pixels; null = not visible)",
            "note": "tracks[obj][i] is the box on sampled_frame_indices[i]",
            "prompt_boxes": {
                str(orig_obj_ids[k]): {
                    "sample_idx": f, "orig_frame": orig_idxs[f],
                    "box": [int(v) for v in b],
                }
                for k, (f, b) in enumerate(targets)
            },
        },
        "tracks": tracks_out,
    }, indent=2))
    print(f"wrote {out_json}")

    if args.out_video:
        args.out_video.parent.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(args.out_video),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        for i, rgb in enumerate(frames):
            canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            for k, t in tracks_out.items():
                box = t[i]
                if box is None:
                    continue
                c = color_for(k)
                cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), c, 2)
                cv2.putText(canvas, str(k), (box[0], max(box[1] - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
            vw.write(canvas)
        vw.release()
        print(f"wrote {args.out_video}")


if __name__ == "__main__":
    main()
