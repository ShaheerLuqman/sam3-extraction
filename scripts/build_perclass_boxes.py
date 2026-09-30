"""
Per-class prompt sets for the "track each class on its own" experiment.

For every class in a video's *_dets.json: take the frames where that class
appears, pick N of them evenly spaced, and emit every box of that class on
those frames as its own tracking target. One JSON file per class, so each is
tracked in its own SAM 3 session (no cross-class non-overlap suppression).

  python scripts/build_perclass_boxes.py \
      --dets ".../2026-08-03 06_51_58.mp4_dets.json" \
      --out-dir "outputs/perclass_track/boxes" --n 6
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dets", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n", type=int, default=6, help="frames per class (default 6)")
    p.add_argument("--min-conf", type=float, default=0.0)
    a = p.parse_args()

    d = json.loads(a.dets.read_text())
    id2n = {int(k): v for k, v in d["meta"]["names"].items()}
    det = d["detections"]
    fc = int(d["meta"]["frame_count"])

    by_cls_frame = defaultdict(lambda: defaultdict(list))   # cls -> frame -> [box]
    for k in sorted(det, key=int):
        fi = int(k)
        for b in det[k]:
            if b[4] < a.min_conf:
                continue
            by_cls_frame[int(b[5])][fi].append([int(round(v)) for v in b[:4]])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cls, frames_map in sorted(by_cls_frame.items()):
        nm = id2n.get(cls, str(cls))
        frames = sorted(frames_map)
        idx = sorted(set(np.linspace(0, len(frames) - 1, a.n).round().astype(int)))
        picks = [frames[i] for i in idx]

        entries, k = [], 0
        for f in picks:
            for box in frames_map[f]:
                entries.append({"name": f"{nm}#{k}", "cls": nm, "id": cls,
                                "frame": f, "box": box})
                k += 1
        (a.out_dir / f"{nm}.json").write_text(json.dumps(entries, indent=1) + "\n")
        summary[nm] = {"picked_frames": picks, "targets": len(entries),
                       "appears_on": len(frames), "range": [frames[0], frames[-1]]}

    meta = {"video": d["meta"].get("video"), "frame_count": fc,
            "frames_per_class": a.n, "classes": summary}
    (a.out_dir / "_perclass_meta.json").write_text(json.dumps(meta, indent=1) + "\n")

    print(f"{len(summary)} class prompt files -> {a.out_dir}")
    for nm, s in summary.items():
        print(f"  {nm:20s} {s['targets']:3d} targets on {s['picked_frames']}")


if __name__ == "__main__":
    main()
