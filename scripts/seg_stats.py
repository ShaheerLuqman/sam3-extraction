"""Accuracy report for a fused_tracks.json against a *_dets.json ground truth.

Per-frame, per-class greedy box matching at IoU >= --iou (default 0.5); the
tracking-id suffix (`#3`) is ignored, only the class name is matched. Writes
<out-dir>/stats.md with overall + per-class precision / recall / F1 / accuracy
and a global-vs-local breakdown taken from boxes/_perclass_meta.json.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True, type=Path)
    ap.add_argument("--dets", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--meta", type=Path, default=None,
                    help="boxes/_perclass_meta.json for the global/local breakdown")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--strategy", default="segmented box-as-mask")
    a = ap.parse_args()

    gt = json.loads(a.dets.read_text())
    names = gt["meta"]["names"]
    det = gt["detections"]
    pred = json.loads(a.fused.read_text())
    pmeta = json.loads(a.meta.read_text()) if a.meta and a.meta.exists() else None

    gt_by_fc = collections.defaultdict(list)
    for f, boxes in det.items():
        f = int(f)
        for b in boxes:
            gt_by_fc[(f, names[str(b[5])])].append(b[:4])
    pred_by_fc = collections.defaultdict(list)
    for f, items in pred["frames"].items():
        f = int(f)
        for cls, _name, box in items:
            pred_by_fc[(f, cls)].append(box)

    frames = sorted(set(int(k) for k in det) | set(int(k) for k in pred["frames"]))
    classes = sorted(set(c for _, c in gt_by_fc) | set(c for _, c in pred_by_fc))
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    for f in frames:
        for c in classes:
            gts = gt_by_fc.get((f, c), [])
            preds = pred_by_fc.get((f, c), [])
            matched = [False] * len(gts)
            for pb in preds:
                best, bi = a.iou, -1
                for i, gb in enumerate(gts):
                    if matched[i]:
                        continue
                    v = iou(pb, gb)
                    if v >= best:
                        best, bi = v, i
                if bi >= 0:
                    matched[bi] = True
                    stats[c]["tp"] += 1
                else:
                    stats[c]["fp"] += 1
            stats[c]["fn"] += matched.count(False)

    def gcount(c):
        return sum(len(v) for (ff, cc), v in gt_by_fc.items() if cc == c)

    def pcount(c):
        return sum(len(v) for (ff, cc), v in pred_by_fc.items() if cc == c)

    TP = sum(s["tp"] for s in stats.values())
    FP = sum(s["fp"] for s in stats.values())
    FN = sum(s["fn"] for s in stats.values())
    p, r, f, acc = prf(TP, FP, FN)
    macro = [prf(s["tp"], s["fp"], s["fn"]) for s in stats.values()]
    mp, mr, mf, ma = [sum(x[i] for x in macro) / len(macro) for i in range(4)]
    tot_gt = sum(gcount(c) for c in classes) or 1
    wp = sum(prf(stats[c]["tp"], stats[c]["fp"], stats[c]["fn"])[0] * gcount(c) for c in classes) / tot_gt
    wr = sum(prf(stats[c]["tp"], stats[c]["fp"], stats[c]["fn"])[1] * gcount(c) for c in classes) / tot_gt
    wf = sum(prf(stats[c]["tp"], stats[c]["fp"], stats[c]["fn"])[2] * gcount(c) for c in classes) / tot_gt

    kind = {}
    if pmeta:
        for c, info in pmeta.get("classes", {}).items():
            kind[c] = info.get("kind", "?")

    vid = gt["meta"].get("video", str(a.dets))
    L = [f"# Prediction Accuracy Report — {a.strategy}", "",
         f"**Video:** `{vid}`  ",
         f"**Prediction:** `{a.fused}`  ",
         f"**Ground truth:** `{a.dets.name}` (per-frame detector labels, conf 0.4)  ",
         f"**Frames evaluated:** {len(frames)}", "",
         "## Method", "",
         "- SAM 3 box-as-mask tracking. **Global** classes (present on a large",
         "  majority of frames with one continuous run) are seeded on their first GT",
         "  frame and tracked across every frame. **Local** classes are seeded once",
         "  per GT segment (maximal run of GT frames, ≤10-frame gaps) and tracked",
         "  only inside that segment; frames outside a local class's segments are",
         "  left empty.",
         f"- Matching: per-frame, per-class greedy assignment, IoU ≥ {a.iou:.2f}.",
         "- The tracking-id suffix (`#3`) is ignored; only the class name is matched.", "",
         "- **Precision** = TP/(TP+FP) · **Recall** = TP/(TP+FN) · "
         "**F1** = 2PR/(P+R) · **Accuracy** = TP/(TP+FP+FN)", "",
         "## Overall", "",
         "| Metric | Value |", "|---|---|",
         f"| True Positives | {TP} |",
         f"| False Positives | {FP} |",
         f"| False Negatives | {FN} |",
         f"| Precision | {p:.3f} |",
         f"| Recall | {r:.3f} |",
         f"| F1 | {f:.3f} |",
         f"| Accuracy | {acc:.3f} |", "",
         f"**Macro-average ({len(classes)} classes):** Precision {mp:.3f} · "
         f"Recall {mr:.3f} · F1 {mf:.3f} · Accuracy {ma:.3f}", "",
         f"**GT-weighted average:** Precision {wp:.3f} · Recall {wr:.3f} · F1 {wf:.3f}", ""]

    for grp, title in [("global", "Global classes"), ("local", "Local (segment) classes")]:
        sub = [c for c in classes if kind.get(c) == grp]
        if not sub:
            continue
        gTP = sum(stats[c]["tp"] for c in sub)
        gFP = sum(stats[c]["fp"] for c in sub)
        gFN = sum(stats[c]["fn"] for c in sub)
        gp, gr, gf, ga = prf(gTP, gFP, gFN)
        L += [f"## {title}", "",
              f"Micro P {gp:.3f} · R {gr:.3f} · F1 {gf:.3f} · Acc {ga:.3f}", "",
              "| Class | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Accuracy |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in sub:
            s = stats[c]
            cp, cr, cf, ca = prf(s["tp"], s["fp"], s["fn"])
            L.append(f"| `{c}` | {gcount(c)} | {pcount(c)} | {s['tp']} | {s['fp']} | "
                     f"{s['fn']} | {cp:.3f} | {cr:.3f} | {cf:.3f} | {ca:.3f} |")
        L.append("")

    other = [c for c in classes if c not in kind]
    if other:
        L += ["## Classes not in the plan", "",
              "| Class | GT | Pred | TP | FP | FN | Precision | Recall | F1 | Accuracy |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in other:
            s = stats[c]
            cp, cr, cf, ca = prf(s["tp"], s["fp"], s["fn"])
            L.append(f"| `{c}` | {gcount(c)} | {pcount(c)} | {s['tp']} | {s['fp']} | "
                     f"{s['fn']} | {cp:.3f} | {cr:.3f} | {cf:.3f} | {ca:.3f} |")
        L.append("")

    if pmeta:
        L += ["## Segment plan (from ground truth)", "",
              "| Class | Kind | GT coverage | # segments | Seed frames |",
              "|---|---|---|---|---|"]
        for c, info in sorted(pmeta["classes"].items(),
                              key=lambda kv: (kv[1]["kind"], -kv[1].get("coverage", 0))):
            L.append(f"| `{c}` | {info['kind']} | {info.get('coverage', 0):.1%} | "
                     f"{len(info.get('segments', []))} | {info.get('picked_frames', [])} |")
        L.append("")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "stats.md").write_text("\n".join(L) + "\n")
    print(f"wrote {a.out_dir/'stats.md'}")
    print(f"overall P {p:.3f} R {r:.3f} F1 {f:.3f} Acc {acc:.3f}")


if __name__ == "__main__":
    main()
