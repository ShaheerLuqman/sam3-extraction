"""Compare several fused_tracks.json runs against one *_dets.json ground truth.

Writes a single stats.md with an overall matrix, global/local micro breakdown,
and a per-class F1 table across all runs.

  python scripts/fawad_compare.py --dets ".../..._dets.json" \
      --out outputs/stellantis_fawad/stats.md \
      --run "baseline (track all, all frames)=outputs/box_as_mask_stellantis/fused_tracks.json" \
      --run "multi / auto=outputs/stellantis_fawad/multi_auto/fused_tracks.json" \
      ...
"""
import argparse
import collections
import json
from pathlib import Path


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    acc = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return p, r, f, acc


def score(fused_path, gt_by_fc, all_frames, thr):
    pred = json.loads(Path(fused_path).read_text())
    pred_by_fc = collections.defaultdict(list)
    for f, items in pred["frames"].items():
        f = int(f)
        for cls, _n, box in items:
            pred_by_fc[(f, cls)].append(box)
    classes = sorted(set(c for _, c in gt_by_fc) | set(c for _, c in pred_by_fc))
    st = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    for f in all_frames:
        for c in classes:
            gts = gt_by_fc.get((f, c), [])
            preds = pred_by_fc.get((f, c), [])
            matched = [False] * len(gts)
            for pb in preds:
                best, bi = thr, -1
                for i, gb in enumerate(gts):
                    if matched[i]:
                        continue
                    v = iou(pb, gb)
                    if v >= best:
                        best, bi = v, i
                if bi >= 0:
                    matched[bi] = True
                    st[c]["tp"] += 1
                else:
                    st[c]["fp"] += 1
            st[c]["fn"] += matched.count(False)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run", action="append", default=[],
                    help="LABEL=path/to/fused_tracks.json ; repeatable, order kept")
    ap.add_argument("--meta", type=Path, default=None,
                    help="a _perclass_meta.json for the global/local class split")
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    gt = json.loads(a.dets.read_text())
    names = gt["meta"]["names"]
    det = gt["detections"]
    gt_by_fc = collections.defaultdict(list)
    for f, boxes in det.items():
        f = int(f)
        for b in boxes:
            gt_by_fc[(f, names[str(b[5])])].append(b[:4])
    all_frames = sorted(set(int(k) for k in det))

    kind = {}
    if a.meta and a.meta.exists():
        for c, info in json.loads(a.meta.read_text()).get("classes", {}).items():
            kind[c] = info.get("kind")

    runs = []
    for spec in a.run:
        label, path = spec.split("=", 1)
        runs.append((label, path, score(path, gt_by_fc, all_frames, a.iou)))

    classes = sorted({c for _, _, st in runs for c in st})

    def gcount(c):
        return sum(len(v) for (ff, cc), v in gt_by_fc.items() if cc == c)

    def totals(st, subset):
        tp = sum(st[c]["tp"] for c in subset if c in st)
        fp = sum(st[c]["fp"] for c in subset if c in st)
        fn = sum(st[c]["fn"] for c in subset if c in st)
        return tp, fp, fn

    glob = [c for c in classes if kind.get(c) == "global"]
    loc = [c for c in classes if kind.get(c) == "local"]

    def hl(st):
        return prf(*totals(st, classes))

    first_lbl, _, first_st = runs[0]
    last_lbl, _, last_st = runs[-1]
    fp0, rp0 = hl(first_st), hl(last_st)

    L = ["# Segmented box-as-mask — variant comparison", "",
         f"**Video:** `{gt['meta'].get('video')}`  ",
         f"**Ground truth:** `{a.dets.name}` (per-frame detector output, conf 0.4; "
         f"{len(all_frames)} frames)  ",
         f"**Matching:** per-frame per-class greedy, IoU ≥ {a.iou:.2f}, id suffix ignored", "",
         "## Headline", "",
         f"`{last_lbl}` vs `{first_lbl}`:", "",
         f"| | {first_lbl} | {last_lbl} |",
         "|---|--:|--:|",
         f"| Precision | {fp0[0]:.3f} | **{rp0[0]:.3f}** |",
         f"| Recall | {fp0[1]:.3f} | **{rp0[1]:.3f}** |",
         f"| F1 | {fp0[2]:.3f} | **{rp0[2]:.3f}** |",
         f"| Accuracy | {fp0[3]:.3f} | **{rp0[3]:.3f}** |", "",
         "## Variants", "",
         "| Label | What differs |",
         "|---|---|",
         "| box_as_mask / baseline | 1 seed box on each class's first GT frame, tracked "
         "across all frames |",
         "| multi / auto | local classes re-seeded once per contiguous GT burst "
         "(gap ≤ 10), 1 seed frame per burst; local = coverage < 60% or fragmented |",
         "| window / auto | local classes tracked once over `[first GT frame, last GT "
         "frame]`, 1 seed; same local set |",
         "| window+gapmask / auto | as `window / auto`, output blanked on the gap "
         "frames between GT bursts |",
         "| multi / transient | multi segments; local set is only the clearly transient "
         "classes |",
         "| window / transient | single window; same transient local set |",
         "| **reseed5 / auto** | **every class** seeded from **≥ 5 reference frames** "
         "spread across its GT bursts; each ref re-seeds *every* instance present and "
         "its track only runs to the next ref. Output burst-clipped. |", "",
         "## Overall (micro)", "",
         "| Run | TP | FP | FN | Precision | Recall | F1 | Accuracy | Macro-F1 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for label, _p, st in runs:
        tp, fp, fn = totals(st, classes)
        p, r, f, acc = prf(tp, fp, fn)
        macro = [prf(st[c]["tp"], st[c]["fp"], st[c]["fn"]) for c in st]
        mf = sum(x[2] for x in macro) / len(macro)
        L.append(f"| {label} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {f:.3f} | "
                 f"{acc:.3f} | {mf:.3f} |")
    L.append("")

    if glob:
        L += ["## Global classes only (micro)  —  identical seeding across all runs", "",
              "| Run | Precision | Recall | F1 | Accuracy |", "|---|--:|--:|--:|--:|"]
        for label, _p, st in runs:
            p, r, f, acc = prf(*totals(st, glob))
            L.append(f"| {label} | {p:.3f} | {r:.3f} | {f:.3f} | {acc:.3f} |")
        L.append("")
    if loc:
        L += ["## Local classes only (micro)", "",
              "| Run | Precision | Recall | F1 | Accuracy |", "|---|--:|--:|--:|--:|"]
        for label, _p, st in runs:
            p, r, f, acc = prf(*totals(st, loc))
            L.append(f"| {label} | {p:.3f} | {r:.3f} | {f:.3f} | {acc:.3f} |")
        L.append("")
        L.append("*(“local” here = the multi/auto split; under the transient set "
                 "`empty_cap` and `empty_surface` are tracked globally instead.)*")
        L.append("")

    L += ["## Per-class F1", "",
          "| Class | GT boxes | " + " | ".join(label for label, _p, _s in runs) + " |",
          "|---|--:|" + "|".join("--:" for _ in runs) + "|"]
    for c in classes:
        row = [f"`{c}`", str(gcount(c))]
        for _label, _p, st in runs:
            s = st.get(c)
            row.append(f"{prf(s['tp'], s['fp'], s['fn'])[2]:.3f}" if s else "—")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    L += ["## Per-class precision / recall (P / R)", "",
          "| Class | " + " | ".join(label for label, _p, _s in runs) + " |",
          "|---|" + "|".join("---" for _ in runs) + "|"]
    for c in classes:
        row = [f"`{c}`"]
        for _label, _p, st in runs:
            s = st.get(c)
            if s:
                p, r, _f, _a = prf(s["tp"], s["fp"], s["fn"])
                row.append(f"{p:.2f} / {r:.2f}")
            else:
                row.append("—")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")
    for label, _p, st in runs:
        p, r, f, acc = prf(*totals(st, classes))
        print(f"  {label:24s} P {p:.3f} R {r:.3f} F1 {f:.3f} Acc {acc:.3f}")


if __name__ == "__main__":
    main()
