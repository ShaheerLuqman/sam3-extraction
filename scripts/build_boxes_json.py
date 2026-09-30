"""
Build a `<video>.mp4_boxes.json` next to every `<video>.mp4_dets.json` in the
datasets tree, in the same shape as `inputs/2026-08-20 09_02_58_seg1_boxes.json`:

    [{"name": "person", "id": 0, "frame": 0, "box": [x1, y1, x2, y2]}, ...]

For each class that the detector ever reports in a video we emit:
  * the FIRST detection of that class (any frame), and
  * an ADDITIONAL reference from the SECOND HALF of the video (frame >= N/2):
    the highest-confidence detection there, so it is a usable exemplar rather
    than a near-duplicate of the first. Skipped when the class never appears in
    the second half (or only on the same frame as the first detection).

The box for a chosen frame is that frame's highest-confidence box for the class.

    python scripts/build_boxes_json.py                       # whole datasets tree
    python scripts/build_boxes_json.py --root inputs/datasets/marmon_station1
    python scripts/build_boxes_json.py --dets "path/to/one.mp4_dets.json"
"""
import argparse
import json
from pathlib import Path

DATASETS_ROOT = Path("inputs/datasets")


def salvage_json(text: str):
    """Parse a dets.json, tolerating a truncated file by trimming to the last
    complete `"<frame>": [ ... ]` entry and closing the braces."""
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass
    # cut back to the last "]]," (end of a frame's box list, more entries follow)
    cut = text.rfind("]],")
    if cut == -1:
        cut = text.rfind("]]")  # maybe the last frame has a single box: "]]"
    if cut == -1:
        raise ValueError("dets.json is unparseable and cannot be salvaged")
    salvaged = text[: cut + 2] + "}}"
    return json.loads(salvaged), True


def first_and_second_half(dets: dict, frame_count: int):
    """-> {cls_id: {"first": (frame, box), "second": (frame, box) | None}}."""
    half = frame_count // 2
    # frame index (int) -> list of [x1,y1,x2,y2,conf,cls]
    frames = sorted((int(k), v) for k, v in dets.items() if v)

    def best_in_frame(boxes, cls):
        """(conf, [x1,y1,x2,y2]) of the top box for `cls` in one frame, or None."""
        cand = [b for b in boxes if int(b[5]) == cls]
        if not cand:
            return None
        b = max(cand, key=lambda x: x[4])
        return b[4], [int(round(v)) for v in b[:4]]

    classes = sorted({int(b[5]) for _, boxes in frames for b in boxes})
    out = {}
    for cls in classes:
        first = None
        second_best = None  # (conf, frame, box) with the max conf in the 2nd half
        for fi, boxes in frames:
            hit = best_in_frame(boxes, cls)
            if hit is None:
                continue
            conf, box = hit
            if first is None:
                first = (fi, box)
            if fi >= half and fi != first[0]:
                if second_best is None or conf > second_best[0]:
                    second_best = (conf, fi, box)
        second = (second_best[1], second_best[2]) if second_best else None
        out[cls] = {"first": first, "second": second}
    return out


def build_for_dets(dets_path: Path) -> Path | None:
    text = dets_path.read_text()
    data, salvaged = salvage_json(text)
    meta = data["meta"]
    names = meta.get("names", {})
    frame_count = int(meta["frame_count"])
    dets = data["detections"]
    if salvaged:
        have = max((int(k) for k in dets), default=-1) + 1
        print(f"  ! {dets_path.name} is truncated - only frames 0..{have - 1} of "
              f"{frame_count} recovered; second-half refs unavailable")

    picks = first_and_second_half(dets, frame_count)
    entries = []
    for cls, pv in picks.items():
        name = names.get(str(cls), str(cls))
        if pv["first"]:
            entries.append({"name": name, "id": cls,
                            "frame": pv["first"][0], "box": pv["first"][1]})
        if pv["second"]:
            entries.append({"name": name, "id": cls,
                            "frame": pv["second"][0], "box": pv["second"][1]})

    out_path = dets_path.with_name(dets_path.name.replace("_dets.json", "_boxes.json"))
    _write_aligned(out_path, entries)
    n_cls = len(picks)
    n_second = sum(1 for p in picks.values() if p["second"])
    print(f"  {out_path.name}: {n_cls} classes, {n_second} with a 2nd-half ref "
          f"-> {len(entries)} entries")
    return out_path


def _write_aligned(path: Path, entries: list):
    """One entry per line, name column padded, like the seg1 reference file."""
    if not entries:
        path.write_text("[]\n")
        return
    w = max(len(json.dumps(e["name"])) for e in entries)
    lines = []
    for e in entries:
        nm = (json.dumps(e["name"]) + ",").ljust(w + 1)
        lines.append(
            f'  {{"name": {nm} "id": {e["id"]:>2}, '
            f'"frame": {e["frame"]:>6}, "box": {json.dumps(e["box"])}}}'
        )
    path.write_text("[\n" + ",\n".join(lines) + "\n]\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DATASETS_ROOT,
                    help="datasets dir (or a single dataset dir) to scan")
    ap.add_argument("--dets", type=Path, default=None,
                    help="build for just this one *_dets.json")
    args = ap.parse_args()

    if args.dets:
        build_for_dets(args.dets)
        return

    dets_files = sorted(args.root.rglob("*_dets.json"))
    if not dets_files:
        raise SystemExit(f"no *_dets.json under {args.root}")
    cur = None
    for dp in dets_files:
        ds = dp.parent.parent.name
        if ds != cur:
            print(f"\n== {ds} ==")
            cur = ds
        try:
            build_for_dets(dp)
        except Exception as e:
            print(f"  !! {dp.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
