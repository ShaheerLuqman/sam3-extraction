"""
Fuse per-class track_bbox runs into one annotated video + combined JSON + report.

  python scripts/fuse_perclass.py \
      --json-dir "outputs/perclass_track/json" \
      --video   ".../2026-08-03 06_51_58.mp4" \
      --meta    "outputs/perclass_track/boxes/_perclass_meta.json" \
      --out-dir "outputs/perclass_track"
"""
import argparse
import colorsys
import json
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def cls_color(name):
    h = (zlib.crc32(name.encode()) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)   # BGR


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-dir", required=True, type=Path)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--fps", type=float, default=15.0)
    a = p.parse_args()

    meta = json.loads(a.meta.read_text())
    jfiles = sorted(a.json_dir.glob("*_track.json"))
    if not jfiles:
        raise SystemExit(f"no *_track.json in {a.json_dir}")

    # per class: {obj_name: [box|null]}, and the sampled frame indices
    per_class = {}
    n_frames = 0
    sampled = None
    for jf in jfiles:
        cls = jf.stem.replace("_track", "")
        d = json.loads(jf.read_text())
        per_class[cls] = d["tracks"]
        s = d["meta"].get("sampled_frame_indices")
        if s:
            sampled = s
        n_frames = max(n_frames, d["meta"]["frame_count"])
    sampled = sampled or list(range(n_frames))
    fidx_pos = {f: i for i, f in enumerate(sampled)}

    a.out_dir.mkdir(parents=True, exist_ok=True)

    # combined per-frame boxes: frame -> list of (cls, name, box)
    combined = defaultdict(list)
    held = defaultdict(int)                    # cls -> total object-frames
    n_obj = {}
    for cls, tracks in per_class.items():
        n_obj[cls] = len(tracks)
        for name, seq in tracks.items():
            for i, box in enumerate(seq):
                if box is None:
                    continue
                combined[sampled[i]].append([cls, name, box])
                held[cls] += 1

    (a.out_dir / "fused_tracks.json").write_text(json.dumps({
        "meta": {"video": str(a.video), "frame_count": n_frames,
                 "classes": sorted(per_class), "objects_per_class": n_obj},
        "frames": {str(f): combined.get(f, []) for f in sampled},
    }, indent=1) + "\n")

    # render
    cap = cv2.VideoCapture(str(a.video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_mp4 = a.out_dir / f"{a.video.stem}_fused.mp4"
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for cls, name, (x1, y1, x2, y2) in combined.get(fi, []):
            c = cls_color(cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, name, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        vw.write(frame)
        fi += 1
    cap.release()
    vw.release()

    # report
    L = [f"# Per-class fused box-tracking — `{a.video.stem}`\n",
         f"Video: **{meta['video']}** · {meta['frame_count']} frames · "
         f"{meta['frames_per_class']} prompt frames per class.\n",
         "Each class tracked in its **own** SAM 3 session (all its instances "
         "together), then all runs overlaid. Separating classes avoids the "
         "cross-class non-overlap suppression that killed boxes in the single "
         "combined run.\n",
         "| class | prompt frames | targets seeded | objects out | object-frames held | avg held/obj |",
         "|---|---|--:|--:|--:|--:|"]
    for cls in sorted(per_class):
        s = meta["classes"].get(cls, {})
        pf = s.get("picked_frames", [])
        seeded = s.get("targets", "?")
        no = n_obj[cls]
        h = held[cls]
        L.append(f"| `{cls}` | {pf} | {seeded} | {no} | {h} | {h // max(no,1)} |")
    L.append("")
    L.append(f"Fused video: `{out_mp4.name}` · combined boxes: `fused_tracks.json`")
    (a.out_dir / "report.md").write_text("\n".join(L))

    print(f"wrote {out_mp4}")
    print(f"wrote {a.out_dir / 'fused_tracks.json'}")
    print(f"wrote {a.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
