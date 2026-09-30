"""
Segmented box-as-mask tracking.

Same "box as mask" strategy as `build_perclass_boxes.py` + `track_bbox.py
--box-as-mask` + `fuse_perclass.py`, with ONE change to what gets tracked:

  * GLOBAL classes  - a class that is present on >= --global-cover of all frames
                      AND whose single longest continuous run covers >= 50% of
                      the video. Seeded once on its first GT frame (ref frame =
                      earliest appearance, every box of the class on that frame)
                      and tracked across ALL frames, exactly as before.

  * LOCAL classes   - everything else. Instead of one run over the whole video,
                      the class's GT frames are split into SEGMENTS (maximal runs
                      with gaps <= --gap frames). Each segment is tracked on its
                      own: ref frame = the first GT frame inside that segment,
                      every box of the class on it, box-as-mask seed, propagation
                      limited to [segment_start, segment_end]. Frames outside all
                      of a class's segments stay null.

The model is loaded once and every class / segment run reuses the pre-built
backbone feature cache (features/<video name>/), same as track_bbox.py.

Outputs (identical layout to outputs/box_as_mask_stellantis/):
  <out>/boxes/<cls>.json          seed boxes per class (with per-segment windows)
  <out>/boxes/_perclass_meta.json
  <out>/json/<cls>_track.json     per-class dense tracks (track_bbox.py format)
Then run scripts/fuse_perclass.py on <out> for fused_tracks.json + _fused.mp4 +
report.md.

  python scripts/track_segmented_box_as_mask.py \
      --video ".../2026-08-03 06_51_58.mp4" \
      --dets  ".../2026-08-03 06_51_58.mp4_dets.json" \
      --out-dir outputs/stellantis_fawad
"""
import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_bbox import (  # noqa: E402
    LazyFeatureCache,
    build_feature_cache,
    load_valid_cache_meta,
    mask_to_xyxy,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--dets", required=True, type=Path,
                   help="the *_dets.json ground-truth (per-frame [x1,y1,x2,y2,conf,cls])")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/sam3.pt"))
    p.add_argument("--gap", type=int, default=10,
                   help="merge GT frames into one segment across gaps <= this many "
                        "frames (default 10, same as global_vs_local.md)")
    p.add_argument("--global-cover", type=float, default=0.60,
                   help="min frame coverage for a class to be considered global "
                        "(also needs its longest run >= 50%% of the video)")
    p.add_argument("--segment-mode",
                   choices=["window", "multi", "window_gapmask"], default="multi",
                   help="local class: 'window' = one run over [first GT frame, last "
                        "GT frame]; 'multi' = one run per contiguous GT segment "
                        "(gap-merged) (default); 'window_gapmask' = one 'window' run "
                        "but blank the output on frames that fall in the gaps between "
                        "GT bursts (SAM still propagates through them, we just don't "
                        "emit a box there)")
    p.add_argument("--local-set", choices=["auto", "transient"], default="auto",
                   help="'auto' = classify by coverage/longest-run; 'transient' = "
                        "only main_cap, removing_main_cap, using_vaccum, "
                        "bearing_surface, empty_region are local")
    p.add_argument("--min-refs", type=int, default=1,
                   help="if > 1: seed EVERY class from at least this many reference "
                        "frames spread across its GT bursts (each ref re-seeds all "
                        "instances present; its track covers up to the next ref, or "
                        "to the midpoint with its neighbours if --bidirectional). "
                        "Overrides --segment-mode. Default 1 = original behaviour.")
    p.add_argument("--bidirectional", action="store_true",
                   help="propagate each ref both forward AND backward instead of "
                        "forward-only. With --min-refs>1 this also re-centers each "
                        "ref's window on the midpoints with its neighbouring refs "
                        "(so a mid-burst ref covers frames on both sides of it "
                        "instead of only from itself to the next ref); for global "
                        "classes it backward-fills frames before the first GT frame.")
    p.add_argument("--class-groups", type=str, default=None,
                   help="treat sets of classes as mutually-exclusive states of the "
                        "same physical slot (e.g. main_cap/empty_main_cap/"
                        "removing_main_cap/using_vaccum never co-occur). Segments, "
                        "reference-frame placement and windows are planned on the "
                        "GROUP's combined GT timeline (so a ref lands near a state "
                        "transition instead of each state reseeding in isolation); "
                        "each ref is still seeded+labeled with whichever single "
                        "class is actually present on it, one run per class per "
                        "ref, output stays one track file per original class. "
                        "Format: semicolon-separated groups of comma-separated "
                        "class ids, e.g. '1,6;2,3,4,5;7,8;9,10'. Requires "
                        "--min-refs > 1. Classes not listed keep their own "
                        "independent per-class planning.")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="drop GT boxes below this confidence before anything else")
    p.add_argument("--features-dir", type=Path, default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--offload", action="store_true")
    p.add_argument("--plan-only", action="store_true",
                   help="print the global/local plan and exit (no model, no cache)")
    return p.parse_args()


def segments(frames, gap):
    """sorted frame list -> [(start, end), ...] maximal runs with gaps <= gap."""
    frames = sorted(frames)
    out, s, prev = [], frames[0], frames[0]
    for x in frames[1:]:
        if x - prev <= gap:
            prev = x
        else:
            out.append((s, prev))
            s = prev = x
    out.append((s, prev))
    return out


def _plan_only(by_cls_frame, id2n, N, a):
    print(f"N={N}  gap={a.gap}  global_cover={a.global_cover}\n")
    print(f"{'class':20s} {'kind':7s} {'cov':>6s} {'longest':>7s} {'#seg':>4s}  segments")
    for cls, frames_map in sorted(by_cls_frame.items(), key=lambda kv: id2n.get(kv[0], str(kv[0]))):
        nm = id2n.get(cls, str(cls))
        frames = sorted(f for f in frames_map if f < N)
        if not frames:
            continue
        segs = segments(frames, a.gap)
        longest = max(e - s + 1 for s, e in segs)
        cover = len(frames) / N
        if a.local_set == "transient":
            is_global = nm not in {"main_cap", "removing_main_cap", "using_vaccum",
                                   "bearing_surface", "empty_region"}
        else:
            is_global = cover >= a.global_cover and longest >= 0.5 * N
        lens = [e - s + 1 for s, e in segs]
        print(f"{nm:20s} {'global' if is_global else 'local':7s} {cover:6.1%} "
              f"{longest:7d} {len(segs):4d}  {list(zip(segs, lens))}")


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "boxes").mkdir(exist_ok=True)
    (a.out_dir / "json").mkdir(exist_ok=True)

    d = json.loads(a.dets.read_text())
    id2n = {int(k): v for k, v in d["meta"]["names"].items()}
    det = d["detections"]
    N = int(d["meta"]["frame_count"])

    # cls -> frame -> [box]
    by_cls_frame = defaultdict(lambda: defaultdict(list))
    for k in sorted(det, key=int):
        fi = int(k)
        for b in det[k]:
            if b[4] < a.min_conf:
                continue
            by_cls_frame[int(b[5])][fi].append([int(round(v)) for v in b[:4]])

    if a.plan_only:
        _plan_only(by_cls_frame, id2n, N, a)
        return

    # ---- feature cache (same logic as track_bbox.py) --------------------------
    cache_dir = a.features_dir or (Path("features") / a.video.name)
    meta = None if a.rebuild_cache else load_valid_cache_meta(cache_dir, a.video)
    if meta is None:
        if not a.video.exists():
            raise SystemExit(f"video not found: {a.video}")
        build_feature_cache(a.video, cache_dir, 1, a.ckpt)
        meta = json.loads((cache_dir / "meta.json").read_text())
    else:
        print(f"[cache] reusing {cache_dir} ({meta['num_cached']} frames)")
    W, H = meta["resolution"]
    n = meta["num_cached"]
    image_size = meta["image_size"]
    orig_idxs = meta["sampled_frame_indices"]
    if n != N:
        print(f"[warn] cache has {n} frames, dets says {N}")
    N = min(N, n)

    TRANSIENT = {"main_cap", "removing_main_cap", "using_vaccum",
                 "bearing_surface", "empty_region"}

    # ---- class groups: plan refs/windows on the combined group timeline -----
    class_groups = []
    if a.class_groups:
        if a.min_refs <= 1:
            raise SystemExit("--class-groups requires --min-refs > 1")
        for spec in a.class_groups.split(";"):
            ids = [int(x) for x in spec.split(",") if x.strip() != ""]
            if ids:
                class_groups.append(ids)
    group_of = {}
    for gi, ids in enumerate(class_groups):
        for cid in ids:
            if cid in group_of:
                raise SystemExit(f"class {cid} listed in more than one --class-groups group")
            group_of[cid] = gi

    group_runs = defaultdict(list)  # class name -> [run, ...] from group-level planning
    for gi, ids in enumerate(class_groups):
        combined = defaultdict(list)  # frame -> [(cls_id, box), ...]
        for cid in ids:
            for f, boxes in by_cls_frame.get(cid, {}).items():
                if f < N:
                    for b in boxes:
                        combined[f].append((cid, b))
        frames = sorted(combined)
        names = [id2n.get(c, str(c)) for c in ids]
        if not frames:
            print(f"[group {gi}] {names}: no GT frames, skipping")
            continue
        segs = segments(frames, a.gap)
        tot = len(frames)
        per_burst = []
        for s, e in segs:
            bf = [f for f in frames if s <= f <= e]
            k = max(1, round(a.min_refs * len(bf) / tot))
            per_burst.append((bf, k))
        while sum(k for _, k in per_burst) < a.min_refs:
            i = max(range(len(per_burst)),
                    key=lambda j: len(per_burst[j][0]) / per_burst[j][1])
            per_burst[i] = (per_burst[i][0], per_burst[i][1] + 1)
        print(f"[group {gi}] {names}: {len(frames)} combined frames, "
              f"{len(segs)} segment(s)")
        for bf, k in per_burst:
            # guarantee >=1 ref on every distinct class's first appearance in this
            # burst, so a short-lived state never silently gets zero runs, on top
            # of the usual proportional-to-length spread.
            guaranteed, seen = [], set()
            for f in bf:
                for cid, _ in combined[f]:
                    if cid not in seen:
                        seen.add(cid)
                        guaranteed.append(f)
            k = max(k, len(guaranteed))
            idx = sorted(set(np.linspace(0, len(bf) - 1, k).round().astype(int)))
            refs = sorted(set(guaranteed) | {bf[i] for i in idx})
            for j, rf in enumerate(refs):
                if a.bidirectional:
                    w0 = bf[0] if j == 0 else (refs[j - 1] + rf) // 2 + 1
                    w1 = bf[-1] if j + 1 == len(refs) else (rf + refs[j + 1]) // 2
                else:
                    w0 = rf
                    w1 = refs[j + 1] - 1 if j + 1 < len(refs) else bf[-1]
                by_cls_here = defaultdict(list)
                for cid, box in combined[rf]:
                    by_cls_here[cid].append(box)
                for cid, boxes in by_cls_here.items():
                    nm = id2n.get(cid, str(cid))
                    group_runs[nm].append({"ref": rf, "window": [w0, w1],
                                           "boxes": boxes, "group": gi})

    # ---- classify + build runs ----------------------------------------------
    plan = {}          # name -> dict(kind, cover, longest, segments, runs)
    for cls, frames_map in sorted(by_cls_frame.items()):
        nm = id2n.get(cls, str(cls))
        frames = sorted(f for f in frames_map if f < N)
        if not frames:
            continue
        segs = segments(frames, a.gap)
        longest = max(e - s + 1 for s, e in segs)
        cover = len(frames) / N
        if a.local_set == "transient":
            is_global = nm not in TRANSIENT
        else:
            is_global = cover >= a.global_cover and longest >= 0.5 * N
        runs = []
        if cls in group_of:
            # Planned above on the group's combined timeline; just pick up this
            # class's own runs (sorted for readable logging/meta output).
            runs = sorted(group_runs.get(nm, []), key=lambda r: r["ref"])
            if not runs:
                print(f"[warn] {nm}: in class-group {group_of[cls]} but got no "
                      f"runs (unexpected)")
        elif a.min_refs > 1:
            # Re-seed strategy: spread >= --min-refs reference frames across the
            # class's GT bursts (>=1 per burst, ~proportional to burst length).
            # Each ref seeds EVERY GT instance of the class on that frame and its
            # track covers [ref, next_ref-1] inside the burst (last -> burst end).
            # Global and local classes both get burst-clipped output.
            tot = len(frames)
            per_burst = []
            for s, e in segs:
                bf = [f for f in frames if s <= f <= e]
                k = max(1, round(a.min_refs * len(bf) / tot))
                per_burst.append((bf, k))
            while sum(k for _, k in per_burst) < a.min_refs:
                i = max(range(len(per_burst)),
                        key=lambda j: len(per_burst[j][0]) / per_burst[j][1])
                per_burst[i] = (per_burst[i][0], per_burst[i][1] + 1)
            for bf, k in per_burst:
                idx = sorted(set(np.linspace(0, len(bf) - 1, k).round().astype(int)))
                refs = [bf[i] for i in idx]
                for j, rf in enumerate(refs):
                    if a.bidirectional:
                        # center each ref's window on the midpoints with its
                        # neighbours, so it covers frames on both sides of it
                        # instead of only forward to the next ref.
                        w0 = bf[0] if j == 0 else (refs[j - 1] + rf) // 2 + 1
                        w1 = bf[-1] if j + 1 == len(refs) else (rf + refs[j + 1]) // 2
                    else:
                        w0 = rf
                        w1 = refs[j + 1] - 1 if j + 1 < len(refs) else bf[-1]
                    runs.append({"ref": rf, "window": [w0, w1],
                                 "boxes": frames_map[rf]})
        elif is_global:
            ref = frames[0]
            runs.append({"ref": ref, "window": [0, N - 1],
                         "boxes": frames_map[ref]})
        elif a.segment_mode == "window":
            s, e = frames[0], frames[-1]
            runs.append({"ref": s, "window": [s, e], "boxes": frames_map[s]})
        elif a.segment_mode == "window_gapmask":
            s, e = frames[0], frames[-1]
            runs.append({"ref": s, "window": [s, e], "boxes": frames_map[s],
                         "keep_segs": [list(x) for x in segs]})
        else:
            for s, e in segs:
                ref = next(f for f in frames if s <= f <= e)
                runs.append({"ref": ref, "window": [s, e],
                             "boxes": frames_map[ref]})
        kind = f"group{group_of[cls]}" if cls in group_of else ("global" if is_global else "local")
        plan[nm] = {"cls_id": cls, "kind": kind,
                    "cover": cover, "longest": longest, "segments": segs,
                    "appears_on": len(frames), "range": [frames[0], frames[-1]],
                    "runs": runs}

    print(f"\n{'class':20s} {'kind':7s} {'cov':>6s} {'#seg':>4s} {'#runs':>5s} {'#seed boxes':>11s}")
    for nm, info in plan.items():
        nb = sum(len(r["boxes"]) for r in info["runs"])
        print(f"{nm:20s} {info['kind']:7s} {info['cover']:6.1%} "
              f"{len(info['segments']):4d} {len(info['runs']):5d} {nb:11d}")

    # ---- load model once ---------------------------------------------------
    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    from sam3.model_builder import build_sam3_video_model
    use_ckpt = a.ckpt and a.ckpt.exists()
    model = build_sam3_video_model(
        checkpoint_path=str(a.ckpt) if use_ckpt else None, load_from_HF=not use_ckpt,
    )
    predictor = model.tracker
    predictor.backbone = None
    feat_cache = LazyFeatureCache(cache_dir, n, image_size)

    def run_one(ref, window, boxes, bidirectional=False):
        """Seed each box (box-as-mask) on `ref`, propagate within `window`.
        -> list (one per box) of [box|null] length N, null outside `window`.
        Forward-only by default; if `bidirectional`, also runs a reverse pass
        from `ref` back to `window[0]` so a mid-window ref covers both sides."""
        w0, w1 = window
        state = predictor.init_state(video_height=H, video_width=W, num_frames=n,
                                     offload_state_to_cpu=a.offload)
        state["cached_features"] = feat_cache
        predictor.clear_all_points_in_video(state)
        for k, (x1, y1, x2, y2) in enumerate(boxes):
            m = torch.zeros(H, W, dtype=torch.bool)
            m[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))] = True
            predictor.add_new_mask(state, frame_idx=ref, obj_id=k, mask=m)

        seqs = [[None] * N for _ in boxes]

        def collect(start, reverse, max_track):
            for f_idx, obj_ids, _low, video_res_masks, _scores in predictor.propagate_in_video(
                state, start_frame_idx=start, max_frame_num_to_track=max_track,
                reverse=reverse, propagate_preflight=True,
            ):
                if not (w0 <= f_idx <= w1) or f_idx >= N:
                    continue
                masks = (video_res_masks > 0.0).squeeze(1).cpu().numpy()
                for i, oid in enumerate(obj_ids):
                    bx = mask_to_xyxy(masks[i].astype(bool))
                    if bx is not None:
                        seqs[int(oid)][f_idx] = bx
                if torch.cuda.is_available() and f_idx % 40 == 0:
                    torch.cuda.empty_cache()

        collect(ref, False, (w1 - ref) + 1)
        if bidirectional and w0 < ref:
            collect(ref, True, (ref - w0) + 1)
        if not bidirectional:
            # forward-only: never report before the ref (prompt) frame
            for s in seqs:
                for f in range(ref):
                    s[f] = None
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return seqs

    sampled = list(range(N))
    box_format = "[x1, y1, x2, y2]  (pixels; null = not visible)"

    for nm, info in plan.items():
        tracks, prompt_boxes, windows = {}, {}, {}
        krun = 0
        for r in info["runs"]:
            print(f"[{nm}] run {krun}: ref={r['ref']} window={r['window']} "
                  f"{len(r['boxes'])} box(es)")
            seqs = run_one(r["ref"], r["window"], r["boxes"], bidirectional=a.bidirectional)
            keep = r.get("keep_segs")
            if keep:
                keepset = set()
                for ks, ke in keep:
                    keepset.update(range(ks, ke + 1))
                for seq in seqs:
                    for f in range(N):
                        if f not in keepset:
                            seq[f] = None
            for j, seq in enumerate(seqs):
                name = f"{nm}#{krun}"
                krun += 1
                tracks[name] = seq
                prompt_boxes[name] = {"sample_idx": r["ref"], "orig_frame": r["ref"],
                                      "box": [int(v) for v in r["boxes"][j]]}
                windows[name] = list(r["window"])
                hits = sum(v is not None for v in seq)
                print(f"    {name}: {hits} frames")

        (a.out_dir / "json" / f"{nm}_track.json").write_text(json.dumps({
            "meta": {
                "video": str(a.video), "frame_count": N, "frame_stride": 1,
                "sampled_frame_indices": sampled, "resolution": [W, H],
                "bidirectional": a.bidirectional, "box_format": box_format,
                "note": "tracks[obj][i] is the box on sampled_frame_indices[i]",
                "class_kind": info["kind"],
                "prompt_boxes": prompt_boxes, "windows": windows,
            },
            "tracks": tracks,
        }, indent=2))

        seed_entries = []
        kk = 0
        for r in info["runs"]:
            for box in r["boxes"]:
                seed_entries.append({"name": f"{nm}#{kk}", "cls": nm,
                                     "id": info["cls_id"], "frame": r["ref"],
                                     "window": list(r["window"]), "box": box})
                kk += 1
        (a.out_dir / "boxes" / f"{nm}.json").write_text(json.dumps(seed_entries, indent=1) + "\n")

    perclass_meta = {
        "video": d["meta"].get("video"), "frame_count": N,
        "frames_per_class": 1, "gap": a.gap, "global_cover": a.global_cover,
        "segment_mode": a.segment_mode, "local_set": a.local_set,
        "min_refs": a.min_refs, "bidirectional": a.bidirectional,
        "class_groups": [[id2n.get(c, str(c)) for c in ids] for ids in class_groups],
        "classes": {
            nm: {
                "kind": info["kind"], "coverage": round(info["cover"], 4),
                "longest_run": info["longest"], "appears_on": info["appears_on"],
                "range": info["range"],
                "segments": [list(s) for s in info["segments"]],
                "picked_frames": [r["ref"] for r in info["runs"]],
                "targets": sum(len(r["boxes"]) for r in info["runs"]),
            }
            for nm, info in plan.items()
        },
    }
    (a.out_dir / "boxes" / "_perclass_meta.json").write_text(
        json.dumps(perclass_meta, indent=1) + "\n")

    del predictor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nwrote per-class tracks to {a.out_dir/'json'}")
    print(f"now run: python scripts/fuse_perclass.py --json-dir {a.out_dir/'json'} "
          f"--video '{a.video}' --meta {a.out_dir/'boxes'/'_perclass_meta.json'} "
          f"--out-dir {a.out_dir}")


if __name__ == "__main__":
    main()
