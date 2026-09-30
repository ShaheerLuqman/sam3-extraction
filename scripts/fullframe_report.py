"""
Write report.md for a build_fullframe_boxes.py + track_bbox.py experiment.

  python scripts/fullframe_report.py \
      --meta  ".../X.mp4_2frame_meta.json" \
      --track "outputs/fullframe_track_2f/X_track.json" \
      --out   "outputs/fullframe_track_2f/report.md"
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def base_cls(name):
    return name.rsplit("#", 1)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--track", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    meta = json.loads(a.meta.read_text())
    trk = json.loads(a.track.read_text())
    tmeta = trk["meta"]
    tracks = trk["tracks"]
    prompt_boxes = tmeta.get("prompt_boxes", {})
    n = tmeta["frame_count"]
    sampled = tmeta.get("sampled_frame_indices") or list(range(n))

    L = []
    L.append(f"# Full-frame box-tracking — `{Path(meta['video']).stem}`\n")
    L.append(f"Video: **{meta['video']}** · {meta['frame_count']} frames.\n")
    L.append("Method: on each labelled frame we take **every** detector box "
             "(all classes) from `*_dets.json` and hand it to SAM 3's box "
             "tracker as its own object, then propagate forward over the clip. "
             "No text, no re-detection — pure mask propagation from the prompt "
             "boxes.\n")

    L.append("## Labelled frames\n")
    L.append("| frame | boxes | position |")
    L.append("|--:|--:|---|")
    for f in meta["labelled_frames"]:
        pos = ("first" if f == 0 else
               "middle" if f == meta["frame_count"] // 2 else
               "last" if f >= meta["frame_count"] - 2 else f"{100*f/meta['frame_count']:.0f}%")
        L.append(f"| {f} | {meta['boxes_per_frame'][str(f)]} | {pos} |")
    L.append("")

    L.append("## Classes\n")
    L.append(f"- **in the video** ({len(meta['classes_in_video'])}): "
             + ", ".join(f"`{c}`" for c in meta["classes_in_video"]))
    miss = meta["classes_not_on_any_labelled_frame"]
    if miss:
        L.append(f"- **not on any labelled frame — excluded** ({len(miss)}): "
                 + ", ".join(f"`{c}`" for c in miss)
                 + "  \n  (these classes never appear on the labelled frame(s), "
                 "so no box exists to seed them and they cannot be tracked)")
    else:
        L.append("- every class in the video appears on a labelled frame")
    L.append(f"- **targets seeded** ({sum(meta['targets_per_class'].values())} "
             f"boxes over {len(meta['targets_per_class'])} classes): "
             + ", ".join(f"`{c}`×{k}" for c, k in meta["targets_per_class"].items()))
    L.append("")

    # per-object tracking coverage
    L.append("## Per-object tracking\n")
    L.append("`held` = frames with a non-null box after the prompt frame · "
             "`span` = first→last frame it was held.\n")
    L.append("| object | prompt frame | prompt box | held | span | % of clip after prompt |")
    L.append("|---|--:|---|--:|---|--:|")
    per_class_held = defaultdict(int)
    per_class_targets = defaultdict(int)
    rows = []
    for name, seq in tracks.items():
        pb = prompt_boxes.get(name, {})
        pf = pb.get("orig_frame", pb.get("sample_idx", 0))
        held_idx = [sampled[i] for i, b in enumerate(seq) if b is not None]
        held = len(held_idx)
        span = f"{held_idx[0]}–{held_idx[-1]}" if held_idx else "–"
        after = sum(1 for f in sampled if f >= pf)
        pct = f"{100*held/after:.0f}%" if after else "–"
        per_class_held[base_cls(name)] += held
        per_class_targets[base_cls(name)] += 1
        rows.append((base_cls(name), pf, name, pb.get("box"), held, span, pct))
    for _, pf, name, box, held, span, pct in sorted(rows, key=lambda r: (r[0], r[1])):
        bx = "`" + ",".join(map(str, box)) + "`" if box else "–"
        L.append(f"| `{name}` | {pf} | {bx} | {held} | {span} | {pct} |")
    L.append("")

    L.append("## Per-class summary\n")
    L.append("| class | targets | total frames held | avg held / target |")
    L.append("|---|--:|--:|--:|")
    for c in sorted(per_class_targets):
        t = per_class_targets[c]
        h = per_class_held[c]
        L.append(f"| `{c}` | {t} | {h} | {h // t} |")
    L.append("")

    out = a.out or a.track.with_name("report.md")
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
