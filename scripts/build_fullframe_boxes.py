"""
Build a "fully-labelled frames" prompt set for track_bbox.py.

By default: take the FIRST frame each class appears in a video's *_dets.json,
collect the unique set of those frames, and for every such frame emit EVERY
detection box on it (all classes) as its own tracking target.

`--frames a,b,...` overrides that with an explicit frame list; the tokens
`first` and `middle` resolve to 0 and frame_count//2.

Each box becomes a distinct target named "<class>#<n>" (n = per-class counter),
so track_bbox.py tracks it as its own object. The output also records, in a
leading "_meta" entry, which classes never appear on any labelled frame.

  python scripts/build_fullframe_boxes.py --dets ".../X.mp4_dets.json"
  python scripts/build_fullframe_boxes.py --dets ".../X.mp4_dets.json" --frames first,middle
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dets", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="default: <dets with _dets.json -> _fullframe_boxes.json>")
    p.add_argument("--frames", default=None,
                   help="explicit comma-separated frame list; 'first'->0, "
                        "'middle'->frame_count//2 (default: first frame per class)")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="drop detection boxes below this confidence")
    args = p.parse_args()

    d = json.loads(args.dets.read_text())
    names = d["meta"]["names"]
    id2n = {int(k): v for k, v in names.items()}
    det = d["detections"]
    fc = int(d["meta"]["frame_count"])

    if args.frames:
        toks = [t.strip() for t in args.frames.split(",") if t.strip()]
        frames = sorted({0 if t == "first" else fc // 2 if t == "middle" else int(t)
                         for t in toks})
    else:
        first = {}
        for k in sorted(det, key=int):
            for b in det[k]:
                first.setdefault(int(b[5]), int(k))
        frames = sorted(set(first.values()))

    present = {int(b[5]) for k in det for b in det[k]}
    on_labelled = {int(b[5]) for f in frames for b in det.get(str(f), [])}
    missing = sorted(id2n.get(c, str(c)) for c in present - on_labelled)

    entries, per_class = [], Counter()
    for f in frames:
        for b in det[str(f)]:
            if b[4] < args.min_conf:
                continue
            cls = int(b[5])
            nm = id2n.get(cls, str(cls))
            entries.append({
                "name": f"{nm}#{per_class[cls]}",
                "cls": nm, "id": cls, "frame": f,
                "box": [int(round(v)) for v in b[:4]],
                "conf": round(float(b[4]), 4),
            })
            per_class[cls] += 1

    out = args.out or args.dets.with_name(
        args.dets.name.replace("_dets.json", "_fullframe_boxes.json"))
    out.write_text(json.dumps(entries, indent=1) + "\n")

    meta = {
        "video": d["meta"].get("video"),
        "frame_count": fc,
        "labelled_frames": frames,
        "boxes_per_frame": {str(f): len(det.get(str(f), [])) for f in frames},
        "classes_in_video": sorted(id2n.get(c, str(c)) for c in present),
        "classes_not_on_any_labelled_frame": missing,
        "targets_per_class": {id2n[c]: n for c, n in sorted(per_class.items())},
    }
    meta_path = Path(str(out).replace("_boxes.json", "_meta.json"))
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")

    print(f"{len(frames)} labelled frame(s): {frames}")
    print(f"{len(entries)} target boxes over {len(per_class)} classes -> {out}")
    for cls, n in sorted(per_class.items()):
        print(f"   {id2n[cls]:20s} x{n}")
    print(f"classes never on a labelled frame ({len(missing)}): {missing}")
    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
