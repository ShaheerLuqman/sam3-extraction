"""Headless check of multiple class segmentation (webapp/backend/mcseg.py) on a labelled
dataset: the backend's own pipeline and Qwen workers, without the web server.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/mcseg_eval.py \
        --dataset "datasets/Stihl SH86 Packing (2.6)" --ref "2026-08-07 09_37_41" \
        --targets "2026-08-10 06_31_39" "2026-08-06 11_18_23" "2026-08-06 15_35_44" \
        --mark 1:0-18 3:19-40 6:92-120 --candidates knn similarity

The target preds.json files are read for scoring only. Their step labels are short
pulses (a few frames where the detector saw the step completed), so a class counts as
found in a target when one of its segments lands within --tol seconds of its pulse.
Selection mirrors the browser (lib/mcseg.ts): P per class smoothed within runs of
consecutive checked clips, each frame to its most likely letter, that letter's P >=
threshold, then gaps filled and short runs dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from webapp.backend import config, mcseg, media, segx  # noqa: E402
from webapp.backend.extraction import QwenWorker  # noqa: E402
from webapp.backend.store import Upload  # noqa: E402


def upload(path: Path, name: str) -> Upload:
    m = media.probe(path)
    return Upload(id=name, kind="video", path=path, width=m["width"], height=m["height"],
                  name=name, frames=m["frames"], fps=m["fps"], duration=m["duration"])


def pulses(preds_path: Path, cls: int) -> list[tuple[int, int]]:
    p = json.load(open(preds_path))["preds"]
    lab = np.array([(p[i]["pred"] or [-1])[0] for i in sorted(p, key=int)])
    return segx._runs(lab == cls)


def select(res: dict, n_cls: int, thr: float, smooth_s: float, min_s: float, gap_s: float):
    """lib/mcseg.ts's selection, per class: [[start, end], ...]."""
    n, fps, st = res["decoded_frames"], res["fps"], res["stride"]
    centers = res["centers"]
    if not centers:
        return [[] for _ in range(n_cls)]
    P = np.array([[0.0 if v is None else v for v in row] for row in res["p"]])
    win = max(1, round(smooth_s * fps / st))
    sm = np.zeros_like(P)
    a = 0
    while a < len(centers):
        b = a
        while b + 1 < len(centers) and centers[b + 1] - centers[b] == st:
            b += 1
        for c in range(P.shape[1]):
            sm[a:b + 1, c] = segx._smooth(P[a:b + 1, c], win)
        a = b + 1
    per = np.zeros((n, P.shape[1]))
    inr = np.zeros(n, bool)
    for s, e in res["region"]:
        inr[s:e + 1] = True
    cs = np.asarray(centers)
    near = np.abs(np.arange(n)[:, None] - cs[None, :]).argmin(1)
    per[inr] = sm[near[inr]]
    best = per.argmax(1)
    out = []
    for c in range(n_cls):
        mask = inr & (best == c) & (per[:, c] >= thr)
        out.append(segx._runs(segx._clean(mask, max(1, round(min_s * fps)), round(gap_s * fps))))
    return out


def overlaps(segs, a, b) -> bool:
    return any(s <= b and e >= a for s, e in segs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--mark", nargs="+", required=True, help="cls:start-end on the reference")
    ap.add_argument("--candidates", nargs="+", default=["knn", "similarity"])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--knn-threshold", type=float, default=0.25)
    ap.add_argument("--coverage", type=float, default=0.25)
    ap.add_argument("--no-description", action="store_true")
    ap.add_argument("--describe", action="store_true", help="VLM-named steps instead of detector.json's")
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "mcseg_eval")
    args = ap.parse_args()

    vids = args.dataset / "Videos"
    names = {int(v): k for k, v in json.load(open(args.dataset / "Detector" / "detector.json"))["cycle_steps"].items()}
    ref = upload(vids / f"{args.ref}_raw.mp4", args.ref)
    targets = [upload(vids / f"{t}_raw.mp4", t) for t in args.targets]
    marks = []
    for m in args.mark:
        c, r = m.split(":")
        a, b = map(int, r.split("-"))
        marks.append((int(c), a, b))
    args.out.mkdir(parents=True, exist_ok=True)

    classes = []
    for c, a, b in marks:
        dest = config.TMP_DIR / f"mcseg_eval_{args.ref}_{a}_{b}.mp4"
        if not dest.exists():
            media.cut(ref.path, a, b, float(ref.fps or 20), dest)
        classes.append({"cls": c, "name": names[c], "description": "", "range": (a, b),
                        "clip": upload(dest, f"{names[c]} [{a}-{b}]")})

    emb, vlm = QwenWorker("embed"), QwenWorker("vlm")
    cancel = threading.Event()
    last = [0.0]

    def progress(f: float, stage: str) -> None:
        if time.time() - last[0] > 5:
            last[0] = time.time()
            print(f"  {f:5.1%} {stage}", flush=True)

    try:
        if args.describe:
            d = mcseg.describe(vlm, None, [[cl["clip"]] for cl in classes], "eval", progress, cancel)
            for cl, got in zip(classes, d["classes"]):
                print(f"  class {cl['cls']} ({cl['name']}) -> {got['name']!r}: {got['description']}")
                cl["name"], cl["description"] = got["name"], got["description"]

        summary = {}
        for mode in args.candidates:
            print(f"\n=== candidates: {mode} ===", flush=True)
            body = {"candidates": mode, "stride": args.stride, "knn_k": args.knn_k,
                    "knn_threshold": args.knn_threshold, "coverage": args.coverage,
                    "use_description": not args.no_description}
            t0 = time.time()
            res = mcseg.search(emb, vlm, None, ref, classes, targets, body, f"eval_{mode}",
                               progress, cancel)
            print(f"  {time.time() - t0:.0f} s; reference frames per class {res['reference_frames']}")
            json.dump(res, open(args.out / f"{mode}.json", "w"))
            rows = []
            for t, tr in zip(args.targets, res["targets"]):
                fps = tr["fps"]
                tol = round(args.tol * fps)
                segs = select(tr, len(classes), 0.5, 0.95, 1.0, 2.0)   # the browser's defaults
                print(f"  {t}: VLM checked {tr['coverage']:.0%} ({len(tr['centers'])} clips)")
                for c, cl in enumerate(classes):
                    gt = pulses(vids / f"{t}.mp4_preds.json", cl["cls"])
                    win = [(max(0, a - tol), b + tol) for a, b in gt]
                    cand = any(overlaps(tr["regions"][c], a, b) for a, b in win)
                    hit = any(overlaps(segs[c], a, b) for a, b in win)
                    fp = [s for s in segs[c] if not any(s[0] <= b and s[1] >= a for a, b in win)]
                    rows.append((t, cl["cls"], bool(gt), cand, hit, len(fp)))
                    print(f"    class {cl['cls']} {cl['name']:24} pulse {gt} | candidate {'yes' if cand else 'NO'}"
                          f" | found {'yes' if hit else 'NO'} | segments {segs[c]} | false {len(fp)}")
            present = [r for r in rows if r[2]]
            summary[mode] = {
                "candidate_recall": f"{sum(r[3] for r in present)}/{len(present)}",
                "found": f"{sum(r[4] for r in present)}/{len(present)}",
                "false_segments": sum(r[5] for r in rows),
                "vlm_share": [tr["coverage"] for tr in res["targets"]],
            }
        print("\n=== summary ===")
        print(json.dumps(summary, indent=1))
    finally:
        emb.stop()
        vlm.stop()


if __name__ == "__main__":
    main()
