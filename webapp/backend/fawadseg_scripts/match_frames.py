"""Find target-video frames similar to a reference step class using Qwen3-VL embeddings.

Usage:
    python match_frames.py --ref-emb R.npz --ref-preds R_preds.json \
        --tgt-emb T.npz --tgt-frames T_frames_dir --tgt-preds T_preds.json \
        --detector detector.json --cls 7 --out OUT_DIR

Selection uses only the reference video's labels:
  * kNN vote: each target frame's k nearest reference frames (cosine) vote
    "class" vs "not class"; score = similarity-weighted fraction of class votes.
  * Temporal smoothing: moving average over the score, threshold 0.5, then drop
    segments shorter than --min-seg frames and fill gaps shorter than --max-gap.
The target preds.json (ground truth) is read only after selection, to compute metrics.
"""
import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np


# ---------------------------------------------------------------- helpers
def load_emb(path):
    z = np.load(path)
    return z["frames"].astype(int), z["emb"].astype(np.float32)


def load_labels(preds_path, frames, cls):
    p = json.load(open(preds_path))["preds"]
    labels = [tuple(p[str(f)]["pred"]) for f in frames]
    return np.array([cls in l for l in labels]), labels


def knn_scores(q, ref, ref_pos, k, exclude=None, balanced=False):
    """Similarity-weighted fraction of positive neighbours among the top-k.

    exclude: optional bool matrix [len(q), len(ref)] of pairs to ignore
    (used for leave-temporal-window-out scoring of the reference itself).
    balanced: divide each side's vote by its share of the reference, so a class
    that is rare in the reference is not outvoted just for being rare.
    """
    sims = q @ ref.T
    if exclude is not None:
        sims = np.where(exclude, -np.inf, sims)
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    s = np.take_along_axis(sims, idx, axis=1)
    w = np.exp((s - s.max(axis=1, keepdims=True)) / 0.02)  # softmax over neighbours
    pos = (w * ref_pos[idx]).sum(1)
    neg = (w * ~ref_pos[idx]).sum(1)
    if balanced:
        pos, neg = pos / ref_pos.mean(), neg / (1 - ref_pos.mean())
    return pos / (pos + neg), sims


def smooth(score, win):
    if win <= 1:
        return score
    k = np.ones(win) / win
    pad = win // 2
    return np.convolve(np.pad(score, pad, mode="edge"), k, mode="valid")[: len(score)]


def segments(mask):
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def clean(mask, min_seg, max_gap):
    mask = mask.copy()
    segs = segments(mask)
    for (a0, a1), (b0, b1) in zip(segs, segs[1:]):  # fill short gaps
        if b0 - a1 - 1 <= max_gap:
            mask[a1 + 1 : b0] = True
    for a, b in segments(mask):  # drop short segments
        if b - a + 1 < min_seg:
            mask[a : b + 1] = False
    return mask


def average_precision(score, gt):
    order = np.argsort(-score)
    g = gt[order]
    tp = np.cumsum(g)
    prec = tp / np.arange(1, len(g) + 1)
    return float((prec * g).sum() / max(g.sum(), 1))


def prf(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=r, f1=f,
                iou=tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                accuracy=(tp + tn) / len(gt))


# ---------------------------------------------------------------- outputs
def timeline_png(path, frames, score, pred, gt, cls_name):
    W, H, pad = 1600, 260, 40
    img = np.full((H, W, 3), 255, np.uint8)
    n = len(frames)
    x = lambda i: pad + int(i / max(n - 1, 1) * (W - 2 * pad))
    for row, (mask, color, label) in enumerate(
        [(gt, (60, 160, 60), "GT (eval only)"), (pred, (200, 90, 30), "Qwen selected")]
    ):
        y0 = 30 + row * 40
        cv2.putText(img, label, (pad, y0 - 6), 0, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        for a, b in segments(mask):
            cv2.rectangle(img, (x(a), y0), (max(x(b), x(a) + 1), y0 + 22), color, -1)
    y_top, y_bot = 130, 230
    cv2.putText(img, "kNN score (smoothed); dashed = 0.5", (pad, y_top - 8), 0, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (pad, y_top), (W - pad, y_bot), (200, 200, 200), 1)
    ymid = int(y_bot - 0.5 * (y_bot - y_top))
    for xx in range(pad, W - pad, 12):
        cv2.line(img, (xx, ymid), (xx + 6, ymid), (150, 150, 150), 1)
    pts = np.array([[x(i), int(y_bot - s * (y_bot - y_top))] for i, s in enumerate(score)], np.int32)
    cv2.polylines(img, [pts], False, (30, 30, 200), 1, cv2.LINE_AA)
    for f in range(0, int(frames[-1]) + 1, 500):
        i = int(np.searchsorted(frames, f))
        cv2.putText(img, str(f), (x(i) - 10, H - 10), 0, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"target class: {cls_name}", (W - 700, 20), 0, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def wrap(text, width):
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    return lines + [cur] if cur else lines


def write_video(path, frames_dir, chosen, score_by_frame, gt_by_frame, label_by_frame, fps, id2step):
    first = cv2.imread(os.path.join(frames_dir, f"{chosen[0]:06d}.jpg"))
    h, w = first.shape[:2]
    tmp = path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in chosen:
        im = cv2.imread(os.path.join(frames_dir, f"{f:06d}.jpg"))
        ok = gt_by_frame[f]
        tag = "TP" if ok else "FP"
        col = (0, 200, 0) if ok else (0, 0, 255)
        gt_lines = []
        for c in label_by_frame[f]:
            gt_lines += wrap(f"GT {c}: {' '.join(id2step.get(c, '?').split())}", 70)
        cv2.rectangle(im, (0, 0), (w, 28 + 20 * len(gt_lines)), (0, 0, 0), -1)
        cv2.putText(im, f"frame {f}  score {score_by_frame[f]:.2f}  GT {list(label_by_frame[f])}  {tag}",
                    (6, 20), 0, 0.55, col, 1, cv2.LINE_AA)
        for i, line in enumerate(gt_lines):
            cv2.putText(im, line, (6, 42 + 20 * i), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(im)
    vw.release()
    # re-encode to H.264 so it plays in browsers / VS Code
    rc = os.system(f'/usr/bin/ffmpeg -y -v error -i "{tmp}" -c:v libx264 -pix_fmt yuv420p -crf 23 "{path}"')
    if rc == 0:
        os.remove(tmp)
    else:
        os.replace(tmp, path)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-emb", required=True)
    ap.add_argument("--ref-preds", required=True)
    ap.add_argument("--tgt-emb", required=True)
    ap.add_argument("--tgt-frames", required=True)
    ap.add_argument("--tgt-preds", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--cls", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--balanced", action="store_true",
                    help="class-balanced kNN vote (corrects for class rarity in the reference)")
    ap.add_argument("--smooth", type=int, default=17, help="moving-average window (frames), ~1 s")
    ap.add_argument("--min-seg", type=int, default=18, help="drop selected runs shorter than this")
    ap.add_argument("--max-gap", type=int, default=35, help="fill unselected gaps shorter than this")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--excl", type=int, default=90,
                    help="reference self-check: ignore neighbours within +-excl frames")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    det = json.load(open(args.detector))
    id2step = {int(v): k for k, v in det["cycle_steps"].items()}
    cls_name = id2step.get(args.cls, str(args.cls))

    ref_f, ref_e = load_emb(args.ref_emb)
    ref_pos, _ = load_labels(args.ref_preds, ref_f, args.cls)
    tgt_f, tgt_e = load_emb(args.tgt_emb)
    tgt_meta = json.load(open(os.path.join(args.tgt_frames, "meta.json")))

    # ---- selection (reference labels only) ----
    # Scores are computed on the embedded (possibly strided) frames, then every
    # decoded frame takes the score of its nearest embedded frame, so selection
    # and metrics are per video frame. Window sizes are given in frames.
    stride = int(np.median(np.diff(tgt_f))) if len(tgt_f) > 1 else 1
    win = lambda frames: max(1, round(frames / stride))
    emb_f = tgt_f
    all_f = np.arange(tgt_meta["decoded_frames"])
    near = np.clip(np.round(all_f / stride).astype(int), 0, len(emb_f) - 1)

    raw_s, sims = knn_scores(tgt_e, ref_e, ref_pos, args.k, balanced=args.balanced)
    sm_s = smooth(raw_s, win(args.smooth))
    pred_s = clean(sm_s >= args.thr, win(args.min_seg), win(args.max_gap))

    proto = ref_e[ref_pos].mean(0)
    proto /= np.linalg.norm(proto)
    proto_sim = (tgt_e @ proto)[near]
    max_pos_sim = sims[:, ref_pos].max(1)[near]
    raw, sm, pred = raw_s[near], sm_s[near], pred_s[near]
    tgt_f = all_f

    # reference self-consistency (no target info): score each reference frame
    # against reference frames outside a +-excl window around it.
    excl = np.abs(ref_f[:, None] - ref_f[None, :]) <= args.excl
    ref_raw, _ = knn_scores(ref_e, ref_e, ref_pos, args.k, exclude=excl, balanced=args.balanced)
    ref_pred = clean(smooth(ref_raw, win(args.smooth)) >= args.thr, win(args.min_seg), win(args.max_gap))

    # ---- evaluation (target GT, used only here) ----
    gt, tgt_labels = load_labels(args.tgt_preds, tgt_f, args.cls)
    m_raw = prf(raw >= args.thr, gt)
    m_final = prf(pred, gt)
    ap_knn = average_precision(sm, gt)
    ap_proto = average_precision(proto_sim, gt)
    ap_maxsim = average_precision(max_pos_sim, gt)
    m_ref = prf(ref_pred, ref_pos)

    chosen = tgt_f[pred].tolist()
    gt_by_frame = dict(zip(tgt_f.tolist(), gt.tolist()))
    score_by_frame = dict(zip(tgt_f.tolist(), sm.tolist()))
    label_by_frame = dict(zip(tgt_f.tolist(), tgt_labels))

    pred_segs = [(int(tgt_f[a]), int(tgt_f[b])) for a, b in segments(pred)]
    gt_segs = [(int(tgt_f[a]), int(tgt_f[b])) for a, b in segments(gt)]
    fp_classes = Counter(tgt_labels[i] for i in np.where(pred & ~gt)[0])

    seg_rows = []
    for a, b in pred_segs:
        idx = (tgt_f >= a) & (tgt_f <= b)
        cover = gt[idx].mean()
        dom = Counter(tgt_labels[i] for i in np.where(idx)[0]).most_common(1)[0][0]
        seg_rows.append((a, b, int(idx.sum()), cover, dom, sm[idx].mean()))
    gt_rows = []
    for a, b in gt_segs:
        idx = (tgt_f >= a) & (tgt_f <= b)
        gt_rows.append((a, b, int(idx.sum()), pred[idx].mean()))

    # ---- write outputs ----
    tgt_name = os.path.basename(os.path.normpath(args.tgt_frames))
    video_path = os.path.join(args.out, f"{tgt_name}_class{args.cls}_selected.mp4")
    if chosen:
        write_video(video_path, args.tgt_frames, chosen, score_by_frame, gt_by_frame,
                    label_by_frame, tgt_meta["fps"], id2step)
    timeline_png(os.path.join(args.out, "timeline.png"), tgt_f, sm, pred, gt, cls_name)

    with open(os.path.join(args.out, "selected_frames.json"), "w") as f:
        json.dump({
            "class_id": args.cls, "class_name": cls_name,
            "selected_frames": chosen, "segments": pred_segs,
        }, f, indent=1)
    with open(os.path.join(args.out, "frame_scores.csv"), "w") as f:
        f.write("frame,knn_raw,knn_smoothed,proto_sim,max_pos_sim,selected,gt_is_class,gt_pred\n")
        for i, fr in enumerate(tgt_f):
            f.write(f"{fr},{raw[i]:.4f},{sm[i]:.4f},{proto_sim[i]:.4f},{max_pos_sim[i]:.4f},"
                    f"{int(pred[i])},{int(gt[i])},\"{list(tgt_labels[i])}\"\n")

    metrics = dict(final=m_final, raw_knn=m_raw, ap_knn=ap_knn, ap_proto=ap_proto,
                   ap_maxsim=ap_maxsim, ref_selfcheck=m_ref)
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)

    # ---- report ----
    ref_meta = json.load(open(os.path.join(os.path.dirname(args.tgt_frames), os.path.basename(
        os.path.splitext(args.ref_emb)[0]).split("__")[0], "meta.json")))
    fmt = lambda m: (f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['iou']:.3f} "
                     f"| {m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} |")
    L = []
    L.append(f"# Step {args.cls} frame retrieval: {tgt_name}\n")
    L.append(f"**Target step class {args.cls}:** {cls_name}\n")
    L.append(f"- Reference video: `{os.path.basename(ref_meta['video'])}` "
             f"({int(ref_pos.sum())} class-{args.cls} frames used as exemplars, "
             f"{int((~ref_pos).sum())} other frames as counter-examples)")
    L.append(f"- Target video: `{os.path.basename(tgt_meta['video'])}`")
    L.append(f"- Model: `Qwen/Qwen3-VL-Embedding-8B` (vLLM pooling runner, bf16), one "
             f"{tgt_e.shape[1]}-d embedding every {stride} frames in both videos; each target "
             f"frame takes the score of its nearest embedded frame, so selection and metrics "
             f"are per video frame")
    L.append(f"- Selected: **{len(chosen)} frames** in {len(pred_segs)} segment(s). "
             f"Video: `{os.path.basename(video_path)}` (every selected frame in order, "
             f"labelled with its score, GT step id and name, and TP/FP).\n")

    L.append("## Data caveat: truncated videos\n")
    L.append(f"Both MP4 files are truncated (the moov header lists more frames than the file "
             f"contains). Only the decodable frames were used:\n")
    L.append("| video | frames in header | decodable frames |")
    L.append("|---|---|---|")
    L.append(f"| reference | {ref_meta['header_frames']} | {ref_meta['decoded_frames']} (0–{ref_meta['decoded_frames']-1}) |")
    L.append(f"| target | {tgt_meta['header_frames']} | {tgt_meta['decoded_frames']} (0–{tgt_meta['decoded_frames']-1}) |")
    full_gt = json.load(open(args.tgt_preds))["preds"]
    full_gt_n = sum(args.cls in v["pred"] for v in full_gt.values())
    full_ref = json.load(open(args.ref_preds))["preds"]
    ref_total = sum(args.cls in v["pred"] for v in full_ref.values())
    ref_dec = sum(args.cls in v["pred"] for k, v in full_ref.items() if int(k) < ref_meta["decoded_frames"])
    L.append(f"\nThe reference has {ref_total} class-{args.cls} frames, of which {ref_dec} are decodable. "
             f"The target GT has "
             f"{full_gt_n} class-{args.cls} frames in total; **{int(gt.sum())}** of them are decodable "
             f"and are the ones evaluated below. All metrics are over target frames "
             f"0–{int(tgt_f[-1])}.\n")

    L.append("## Accuracy vs target ground truth\n")
    L.append("Per-frame, class " + str(args.cls) + " vs rest.\n")
    L.append("| method | precision | recall | F1 | IoU | accuracy | TP | FP | FN |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    L.append("| **final (kNN + temporal smoothing)** " + fmt(m_final))
    L.append("| kNN only (no smoothing) " + fmt(m_raw))
    L.append("")
    L.append("Threshold-free ranking quality (average precision of the per-frame score):\n")
    L.append("| score | AP |")
    L.append("|---|---|")
    L.append(f"| kNN vote (smoothed), used for selection | {ap_knn:.3f} |")
    L.append(f"| cosine to class-{args.cls} prototype (positives only) | {ap_proto:.3f} |")
    L.append(f"| max cosine to any class-{args.cls} exemplar (positives only) | {ap_maxsim:.3f} |")
    L.append(f"\nChance level (positive rate) = {gt.mean():.3f}.\n")

    L.append("### Ground-truth segments\n")
    L.append("| GT segment (frames) | length | fraction selected |")
    L.append("|---|---|---|")
    for a, b, n, c in gt_rows:
        L.append(f"| {a}–{b} | {n} | {c:.1%} |")
    L.append("\n### Selected segments\n")
    L.append("| selected segment (frames) | length | fraction truly class " + str(args.cls) +
             " | dominant GT label | mean score |")
    L.append("|---|---|---|---|---|")
    for a, b, n, c, dom, s in seg_rows:
        L.append(f"| {a}–{b} | {n} | {c:.1%} | {list(dom)} | {s:.2f} |")
    if fp_classes:
        L.append("\n### False positives by GT label\n")
        L.append("| GT label | frames | step name |")
        L.append("|---|---|---|")
        for lab, n in fp_classes.most_common():
            L.append(f"| {list(lab)} | {n} | {'; '.join(id2step.get(c, '?') for c in lab)} |")

    L.append("\n## Reference self-check (no target information)\n")
    L.append(f"The same kNN + smoothing applied to the reference video, where each frame may only "
             f"use neighbours more than {args.excl} frames away from it: "
             f"precision {m_ref['precision']:.3f}, recall {m_ref['recall']:.3f}, "
             f"F1 {m_ref['f1']:.3f}. This sanity-checks the settings without touching the target GT.\n")

    L.append("## Method\n")
    L.append("1. Decode every readable frame of both videos (`scripts/extract_frames.py`).")
    L.append("2. Embed every --stride-th frame (here 5) with Qwen3-VL-Embedding-8B using a task instruction "
             "(\"represent the assembly step … panel orientation, tool, hand actions\") "
             "(`scripts/embed_frames.py`).")
    L.append(f"3. Reference frames are labelled from the reference `preds.json`: class {args.cls} "
             f"(exemplars) vs everything else (counter-examples).")
    L.append(f"4. For each target frame, take its k={args.k} most similar reference frames "
             f"(cosine) and compute a similarity-weighted class-{args.cls} vote in [0, 1].")
    L.append(f"5. Smooth the vote with a {args.smooth}-frame moving average, threshold at "
             f"{args.thr}, fill gaps < {args.max_gap} frames, drop runs < {args.min_seg} frames.")
    L.append("6. Only then read the target `preds.json` to compute the metrics above.")
    L.append("\nThe parameters are fixed defaults and were not tuned on the target ground truth.\n")

    L.append("## Files\n")
    L.append(f"- `{os.path.basename(video_path)}`: the selected frames as a video")
    L.append("- `selected_frames.json`: selected frame indices and segments")
    L.append("- `frame_scores.csv`: per-frame scores, selection and GT")
    L.append("- `timeline.png`: GT vs selection vs score over the target video")
    L.append("- `metrics.json`: all numbers in this report")
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(L) + "\n")

    print(json.dumps(metrics, indent=1))
    print("pred segs:", pred_segs)
    print("gt segs:", gt_segs)


if __name__ == "__main__":
    main()
