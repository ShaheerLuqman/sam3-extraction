"""Classify candidate clips with Qwen3-VL-8B-Instruct using contrastive few-shot video examples.

Stage 2 of the pipeline: match_frames.py (with --balanced) proposes candidate segments with
high recall; this script asks the VLM, for each candidate, which of three options a short
clip around the frame shows: one letter per named step (the target --cls plus the
confusable steps in --confusers, in step order) and a last letter for anything else.
The options are shown as labelled example clips taken from the reference video. They are
picked automatically from the reference labels and embeddings (k-means medoids), so they
cover different sub-phases of each option. P(A/B/C) comes from the first-token logprobs.

Modes:
    refcheck  score reference clips (frames --ref-range, away from the examples) against
              the reference labels. No target data is touched.
    target    score the target candidates, select frames, evaluate against the target GT
              and write the outputs.
"""
import argparse
import json
import math
import os
import re

# Pin to GPU 0; GPU 1 is shared with sam3-extraction.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# FlashInfer's sampler JIT-compiles with nvcc, which is not installed here.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import cv2
import numpy as np

from match_frames import (average_precision, clean, load_emb, load_labels, prf, segments,
                          smooth, timeline_png, write_video)

MODEL_ID = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct")


# ---------------------------------------------------------------- clips
def load_clip(frames_dir, center, n_frames, step, total):
    """n_frames frames spaced `step` apart, centred on `center` (clamped to the video)."""
    half = (n_frames - 1) * step // 2
    start = min(max(0, center - half), max(0, total - 1 - (n_frames - 1) * step))
    idx = [start + i * step for i in range(n_frames)]
    frames = [cv2.cvtColor(cv2.imread(os.path.join(frames_dir, f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)
              for i in idx]
    return np.stack(frames), idx


def crop_clip(arr, box, margin=0.3, long_side=640):
    """Crop frames to box (pixels) grown by `margin` on each side, then upscale."""
    h, w = arr.shape[1:3]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return arr
    scale = long_side / max(x2 - x1, y2 - y1)
    size = (round((x2 - x1) * scale), round((y2 - y1) * scale))
    return np.stack([cv2.resize(fr[y1:y2, x1:x2], size, interpolation=cv2.INTER_CUBIC) for fr in arr])


def locate_panels(llm, jobs):
    """Ask the VLM for the panel's bounding box in each (frames_dir, frame) middle frame.

    Qwen3-VL returns boxes in 0-1000 relative coordinates. Returns pixel boxes or None.
    """
    from PIL import Image
    from vllm import SamplingParams

    q = ("Locate the rectangular metal panel lying on the workbench. Output its bounding box "
         "as JSON: {\"bbox_2d\": [x1, y1, x2, y2]}.")
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
        tokenize=False, add_generation_prompt=True)
    imgs = [Image.open(os.path.join(d, f"{f:06d}.jpg")).convert("RGB") for d, f in jobs]
    outs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": im}} for im in imgs],
                        SamplingParams(temperature=0.0, max_tokens=64))
    boxes = []
    for im, o in zip(imgs, outs):
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", o.outputs[0].text)
        if not m:
            boxes.append(None)
            continue
        x1, y1, x2, y2 = (int(v) / 1000 for v in m.groups())
        boxes.append((x1 * im.width, y1 * im.height, x2 * im.width, y2 * im.height))
    return boxes


def video_item(frames_dir, center, args, total, fps, box=None):
    arr, idx = load_clip(frames_dir, center, args.clip_frames, args.clip_step, total)
    if box is not None:
        arr = crop_clip(arr, box)
    rel = [i - idx[0] for i in idx]
    meta = {
        "fps": fps,
        "duration": (rel[-1] + 1) / fps,
        "total_num_frames": rel[-1] + 1,
        "frames_indices": rel,
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    return arr, meta


# ---------------------------------------------------------------- examples
def kmeans_medoids(emb, frames, k, seed=0, iters=50):
    """Frames closest to the centres of a small cosine k-means (no training, just grouping)."""
    rng = np.random.default_rng(seed)
    cent = emb[rng.choice(len(emb), k, replace=False)]
    for _ in range(iters):
        assign = np.argmax(emb @ cent.T, 1)
        new = np.stack([emb[assign == j].mean(0) if (assign == j).any() else cent[j] for j in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True)
        if np.allclose(new, cent):
            break
        cent = new
    return sorted(int(frames[np.argmax(emb @ c)]) for c in cent)


def pick_examples(ref_f, ref_e, ref_labels, ref_all, letters, step_ids, target, n_per, clip_half):
    """Example centre frames per option letter, using reference labels only.

    A clip is eligible only if every frame in its window has the same label, so the
    examples are clean. "Other" examples are hard negatives: frames of none of the named
    steps that are most similar to the target step's centroid.
    """
    def pure(f, want):  # ref_all: label of every reference frame
        return all(ref_all.get(g, ()) == want for g in range(f - clip_half, f + clip_half + 1))

    out = {}
    for letter, c in zip(letters, step_ids):
        m = np.array([l == (c,) and pure(int(f), (c,)) for f, l in zip(ref_f, ref_labels)])
        out[letter] = kmeans_medoids(ref_e[m], ref_f[m], n_per)
    t_mask = np.array([l == (target,) for l in ref_labels])
    centroid = ref_e[t_mask].mean(0)
    other = np.array([not set(l) & set(step_ids) and pure(int(f), l)
                      for f, l in zip(ref_f, ref_labels)])
    sim = ref_e[other] @ centroid
    top = np.argsort(-sim)[:60]
    out[letters[-1]] = kmeans_medoids(ref_e[other][top], ref_f[other][top], n_per)
    return out


# ---------------------------------------------------------------- prompt
def option_text(id2step, letters, step_ids, hints=None):
    hints = hints or {}
    opts = {}
    for l, c in zip(letters, step_ids):
        opts[l] = f"step {c}: \"{' '.join(id2step[c].split())}\""
        if str(c) in hints:
            opts[l] += " " + hints[str(c)]
    opts[letters[-1]] = hints.get("other", "anything else: idle, positioning or handling the panel, "
                                  "using the rivnut gun to crimp, fetching parts, or any other step")
    return opts


def build_messages(opts, examples, letters):
    content = [{"type": "text", "text":
                "These clips come from a fixed overhead camera at a manual assembly workstation "
                "where a worker installs rivnuts into a metal panel. Each clip is about two "
                "seconds long. The options are:\n"
                + "\n".join(f"{k} = {v}" for k, v in opts.items())
                + "\nHere are labelled example clips from another recording of the same "
                  "station.\n"}]
    for letter in letters:
        for _ in examples[letter]:
            content.append({"type": "text", "text": f"Example of {letter}:"})
            content.append({"type": "video"})
    content.append({"type": "text", "text":
                    "Now the clip to classify:"})
    content.append({"type": "video"})
    content.append({"type": "text", "text":
                    "Which option does this last clip show? First look at the state and "
                    "orientation of the panel (flat, uprights bent up, or turned), then at where "
                    "on the panel the hands are working. Crimping with the rivnut gun belongs to "
                    "the install step it follows. Answer with a single letter: "
                    + ", ".join(letters[:-1]) + " or " + letters[-1] + "."})
    return [{"role": "user", "content": content}]


def letter_probs(logprobs, letters):
    p = {k: 0.0 for k in letters}
    for lp in logprobs.values():
        t = lp.decoded_token.strip().upper()
        if t in p:
            p[t] += math.exp(lp.logprob)
    s = sum(p.values())
    return {k: (v / s if s else 1 / 3) for k, v in p.items()}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["refcheck", "target"])
    ap.add_argument("--ref-emb", required=True)
    ap.add_argument("--ref-frames", required=True)
    ap.add_argument("--ref-preds", required=True)
    ap.add_argument("--tgt-frames", required=True)
    ap.add_argument("--tgt-preds", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--cls", type=int, required=True, help="target step")
    ap.add_argument("--confusers", default="3,7", help="comma-separated confusable steps")
    ap.add_argument("--candidates", help="selected_frames.json from match_frames.py --balanced")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-range", default="1400,3400", help="refcheck frame range")
    ap.add_argument("--val-frames", help="refcheck: score clips of this held-out video instead")
    ap.add_argument("--val-preds", help="refcheck: labels of the held-out video")
    ap.add_argument("--excl", type=int, default=90, help="refcheck: skip frames near examples")
    ap.add_argument("--n-per", type=int, default=2, help="example clips per option")
    ap.add_argument("--hints", help="JSON of per-step visual descriptions for the options")
    ap.add_argument("--crop", action="store_true",
                    help="crop every clip to the panel, located by the VLM on the clip's middle frame")
    ap.add_argument("--clip-frames", type=int, default=8)
    ap.add_argument("--clip-step", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--smooth", type=int, default=17)
    ap.add_argument("--min-seg", type=int, default=18)
    ap.add_argument("--max-gap", type=int, default=35)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    det = json.load(open(args.detector))
    id2step = {int(v): k for k, v in det["cycle_steps"].items()}
    step_ids = sorted({args.cls, *map(int, args.confusers.split(","))})
    letters = [chr(ord("A") + i) for i in range(len(step_ids) + 1)]
    tl = letters[step_ids.index(args.cls)]  # target letter
    hints = json.load(open(args.hints)) if args.hints else None
    opts = option_text(id2step, letters, step_ids, hints)
    print("options:", opts, flush=True)
    clip_half = (args.clip_frames - 1) * args.clip_step // 2

    ref_f, ref_e = load_emb(args.ref_emb)
    _, ref_labels = load_labels(args.ref_preds, ref_f, args.cls)
    ref_meta = json.load(open(os.path.join(args.ref_frames, "meta.json")))
    ref_all = {int(k): tuple(v["pred"]) for k, v in json.load(open(args.ref_preds))["preds"].items()
               if int(k) < ref_meta["decoded_frames"]}
    examples = pick_examples(ref_f, ref_e, ref_labels, ref_all, letters, step_ids, args.cls,
                             args.n_per, clip_half)
    print("examples:", examples, flush=True)

    # ---- queries ----
    if args.mode == "refcheck":
        lo, hi = map(int, args.ref_range.split(","))
        if args.val_frames:  # held-out video: no overlap with the examples possible
            q_dir, check_preds = args.val_frames, args.val_preds
            q_meta = json.load(open(os.path.join(q_dir, "meta.json")))
            q_frames = list(range(lo, min(hi, q_meta["decoded_frames"] - 1) + 1, args.stride))
        else:
            q_dir, q_meta, check_preds = args.ref_frames, ref_meta, args.ref_preds
            centres = [c for L in examples.values() for c in L]
            q_frames = [f for f in range(lo, hi + 1, args.stride)
                        if all(abs(f - c) > args.excl for c in centres)]
    else:
        sel = json.load(open(args.candidates))
        n = json.load(open(os.path.join(args.tgt_frames, "meta.json")))["decoded_frames"]
        region = np.zeros(n, bool)
        for a, b in sel["segments"]:
            region[a:b + 1] = True
        q_frames = [f for f in range(0, n, args.stride) if region[f]]
        q_dir = args.tgt_frames
        q_meta = json.load(open(os.path.join(args.tgt_frames, "meta.json")))
    print(f"{len(q_frames)} query clips", flush=True)
    if not q_frames and args.mode == "target":  # no candidates: nothing to ask, empty selection
        gt, _ = load_labels(args.tgt_preds, np.arange(n), args.cls)
        empty = np.zeros(n, bool)
        json.dump({"candidates": prf(empty, gt), "final": prf(empty, gt), "n_queries": 0},
                  open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
        json.dump({"class_id": args.cls, "class_name": id2step[args.cls], "selected_frames": [],
                   "segments": []}, open(os.path.join(args.out, "selected_frames.json"), "w"), indent=1)
        print("no candidates; wrote empty selection")
        return

    # ---- VLM ----
    from vllm import LLM, SamplingParams

    n_videos = len(letters) * args.n_per + 1
    llm = LLM(model=MODEL_ID, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.92,
              max_num_seqs=8, limit_mm_per_prompt={"image": 1, "video": n_videos},
              enable_prefix_caching=True)
    ex_centres = [c for letter in letters for c in examples[letter]]
    boxes = {}
    if args.crop:
        jobs = [(args.ref_frames, c) for c in ex_centres] + [(q_dir, f) for f in q_frames]
        found = locate_panels(llm, jobs)
        boxes = dict(zip(jobs, found))
        print(f"panel located in {sum(b is not None for b in found)}/{len(found)} frames", flush=True)
    prompt = llm.get_tokenizer().apply_chat_template(
        build_messages(opts, examples, letters), tokenize=False, add_generation_prompt=True)
    ex_items = [video_item(args.ref_frames, c, args, ref_meta["decoded_frames"], ref_meta["fps"],
                           boxes.get((args.ref_frames, c))) for c in ex_centres]
    inputs = [{"prompt": prompt, "multi_modal_data": {"video": ex_items + [
        video_item(q_dir, f, args, q_meta["decoded_frames"], q_meta["fps"], boxes.get((q_dir, f)))]}}
        for f in q_frames]
    outs = llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    probs = [letter_probs(o.outputs[0].logprobs[0], letters) for o in outs]
    pb = np.array([p[tl] for p in probs])
    print("prompt tokens:", len(outs[0].prompt_token_ids), flush=True)

    # smooth P(target) within contiguous runs of query frames
    qf = np.array(q_frames)
    win = max(1, round(args.smooth / args.stride))
    pb_s = np.zeros_like(pb)
    runs = [[0]]
    for i in range(1, len(qf)):
        if qf[i] - qf[i - 1] != args.stride:
            runs.append([])
        runs[-1].append(i)
    for r in runs:
        pb_s[r] = smooth(pb[r], win)

    # save exemplar strip (middle frame of each example clip)
    tiles = []
    for letter in letters:
        for c in examples[letter]:
            im = cv2.resize(cv2.imread(os.path.join(args.ref_frames, f"{c:06d}.jpg")), (320, 240))
            cv2.rectangle(im, (0, 0), (320, 26), (0, 0, 0), -1)
            cv2.putText(im, f"{letter}: ref frame {c}", (6, 18), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(im)
    cv2.imwrite(os.path.join(args.out, "vlm_examples.jpg"),
                np.vstack([np.hstack(tiles[i:i + args.n_per]) for i in range(0, len(tiles), args.n_per)]))

    rows_csv = "frame," + ",".join(f"p_{l}" for l in letters) + ",p_target_smoothed,gt_pred\n"
    row = lambda i: ",".join(f"{probs[i][l]:.4f}" for l in letters) + f",{pb_s[i]:.4f}"

    if args.mode == "refcheck":
        gt, labels = load_labels(check_preds, qf, args.cls)
        m_raw, m_s = prf(pb >= 0.5, gt), prf(pb_s >= 0.5, gt)
        res = {"options": opts, "examples": examples, "n_queries": len(q_frames), "raw": m_raw, "smoothed": m_s,
               "ap": average_precision(pb_s, gt), "positive_rate": float(gt.mean())}
        # confusion of argmax letter vs GT
        conf = {}
        for p, l in zip(probs, labels):
            key = f"{list(l)}->{max(p, key=p.get)}"
            conf[key] = conf.get(key, 0) + 1
        res["confusion"] = dict(sorted(conf.items()))
        json.dump(res, open(os.path.join(args.out, "refcheck_metrics.json"), "w"), indent=2)
        with open(os.path.join(args.out, "refcheck_scores.csv"), "w") as f:
            f.write(rows_csv)
            for i, fr in enumerate(qf):
                f.write(f"{fr},{row(i)},\"{list(labels[i])}\"\n")
        print(json.dumps(res, indent=1))
        return

    # ---- target: expand to frames, clean, evaluate ----
    n = q_meta["decoded_frames"]
    sel_mask = np.zeros(n, bool)
    for i, f in enumerate(qf):
        sel_mask[max(0, f - args.stride // 2): min(n, f + args.stride - args.stride // 2)] = pb_s[i] >= 0.5
    sel_mask &= region
    sel_mask = clean(sel_mask, args.min_seg, args.max_gap)

    all_f = np.arange(n)
    gt, labels = load_labels(args.tgt_preds, all_f, args.cls)
    cand_mask = region
    m_cand, m_final = prf(cand_mask, gt), prf(sel_mask, gt)
    qgt = gt[qf]
    m_raw = prf(pb >= 0.5, qgt)
    near = np.clip(np.searchsorted(qf, all_f), 0, len(qf) - 1)
    score_all = np.where(region, pb_s[near], 0.0)

    tgt_name = os.path.basename(os.path.normpath(args.tgt_frames))
    video = f"{tgt_name}_class{args.cls}_selected.mp4"
    chosen = all_f[sel_mask].tolist()
    if chosen:
        write_video(os.path.join(args.out, video), args.tgt_frames, chosen,
                    dict(enumerate(score_all.tolist())), dict(enumerate(gt.tolist())),
                    dict(enumerate(labels)), q_meta["fps"], id2step)
    timeline_png(os.path.join(args.out, "timeline.png"), all_f, score_all, sel_mask, gt,
                 " ".join(id2step[args.cls].split()))
    with open(os.path.join(args.out, "vlm_scores.csv"), "w") as f:
        f.write(rows_csv)
        for i, fr in enumerate(qf):
            f.write(f"{fr},{row(i)},\"{list(labels[fr])}\"\n")
    segs = [(int(a), int(b)) for a, b in segments(sel_mask)]
    json.dump({"class_id": args.cls, "class_name": id2step[args.cls], "selected_frames": chosen,
               "segments": segs, "options": opts, "examples": examples},
              open(os.path.join(args.out, "selected_frames.json"), "w"), indent=1)
    metrics = {"candidates": m_cand, "final": m_final, "vlm_raw_on_queries": m_raw,
               "ap_vlm": average_precision(score_all, gt), "n_queries": len(q_frames)}
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=1))
    print("segments:", segs)
    print("gt segs:", [(int(a), int(b)) for a, b in segments(gt)])


if __name__ == "__main__":
    main()
