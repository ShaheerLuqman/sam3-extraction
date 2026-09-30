"""Multiple class segmentation: several steps marked on one reference video (one range
per class) -> where each of them happens in up to three other videos.

segx.py generalised from one step to C steps, with the rest of the reference as a
class of its own, "background" (idle, fetching parts, the steps nobody marked):

  1. describe  — per class, the VLM names and describes the step from its clip
     (only for the classes the user did not name themselves).
  2. embed     — Qwen3-VL-Embedding-8B embeds every `stride`-th frame of the reference
     and of every target, and every frame of each step's clip (a step of a second or
     two has too few stride-th frames to vote with), with all the class descriptions
     in one instruction.
  3. candidates — per class, which stretches of each target the VLM should check:
       "similarity": segx's mechanism, per class — each frame's best z-scored
         similarity to that class's frames, the top `coverage` kept (+ margin);
       "knn": a balanced kNN vote over C + 1 groups — each step's frames are a group,
         every reference frame outside the marked ranges is the background group,
         which pulls idle and unmarked frames away from the steps. A class's
         stretches are where its smoothed share of the vote reaches the threshold.
     The VLM checks the union over classes.
  4. classify  — for every `stride`-th candidate frame, a clip goes to the VLM as a
     multiple choice: one letter per class plus one for none of them, after example
     clips of each from the reference (k-means medoids of the step's frames; of the
     background frames for the last letter). P per letter from the answer's
     logprobs. The clip is segx's 8 frames, spaced to fit the shortest marked range.

The browser smooths, thresholds, assigns and cleans the per-class P, so it is live.
"""
from __future__ import annotations

import string
import threading

import numpy as np

from . import config, segx
from .extraction import Cancelled, _cache, cache_key, video_emb_path

MAX_CLASSES = 8
MAX_TARGETS = 3
MAX_VIDEOS = 12                        # the VLM worker's limit_mm_per_prompt
# segx's "anything else" names reaching for parts and moving the workpiece, which is
# what most placing steps look like; with the steps listed as options, it only has
# to be "none of them"
OTHER_TEXT = ("none of the steps above: idle, waiting, or any other activity that is not one "
              "of the listed steps")


def instruction_for(classes: list[dict], use_description: bool) -> str:
    """The embedding instruction: every class's name and description in one (a
    per-class instruction would embed each video once per class)."""
    parts = []
    for i, c in enumerate(classes):
        text = " ".join(f"{c['name'].strip().rstrip('.')}. {c.get('description', '').strip()}"
                        .split()).strip(". ")
        if text:
            parts.append(f"({i + 1}) {text}")
    if not use_description or not parts:
        return config.QWEN_EMB_INSTRUCTION
    return ("Represent this workstation image for recognising which of the following steps the "
            "worker is performing, if any: " + "; ".join(parts))


# --------------------------------------------------------------------------- #
# 1. describe
# --------------------------------------------------------------------------- #
def describe(vlm, engine, clips: list[list], job_id: str, progress_cb, cancel: threading.Event) -> dict:
    """clips: per class to describe, its clip uploads. One VLM request per class."""
    out = []
    for i, ups in enumerate(clips):
        if cancel.is_set():
            raise Cancelled("cancelled")
        sub = segx._sub(progress_cb, i / len(clips), (i + 1) / len(clips))
        d = segx.describe(vlm, engine, ups, f"{job_id}_{i}",
                          lambda f, stage, i=i, sub=sub: sub(f, f"step {i + 1}/{len(clips)}: {stage}"),
                          cancel)
        out.append(d)
    return {"classes": out}


# --------------------------------------------------------------------------- #
# 2-4. search
# --------------------------------------------------------------------------- #
def search(emb_worker, vlm, engine, ref, classes: list[dict], targets: list, body: dict,
           job_id: str, progress_cb, cancel: threading.Event) -> dict:
    """classes: [{"name", "description", "range": (start, end), "clip": upload}]."""
    C = len(classes)
    use_knn = body.get("candidates") == "knn"
    k, thr = int(body.get("knn_k") or 15), float(body.get("knn_threshold") or 0.25)
    stride = int(body.get("stride") or 5)
    instruction = instruction_for(classes, bool(body.get("use_description", True)))
    key = cache_key(instruction, stride)
    letters = list(string.ascii_uppercase[:C + 1])
    options = {letters[c]: f'the step "{cl["name"].strip() or f"step {c + 1}"}"'
               + (f": {cl['description'].strip()}" if cl.get("description", "").strip() else "")
               for c, cl in enumerate(classes)}
    options[letters[C]] = OTHER_TEXT

    # a step is often shorter than segx's ~2 s clip: space the clip's frames so that it
    # fits in the shortest marked range, else every clip mixes neighbouring steps
    shortest = min(cl["range"][1] - cl["range"][0] + 1 for cl in classes)
    clip_step = max(1, min(segx.CLIP_STEP, (shortest - 1) // (segx.CLIP_FRAMES - 1)))
    # the clips are short, and a handful of stride-th frames is too few to vote or rank
    # with: every frame of them is embedded
    key1 = cache_key(instruction, 1)

    with segx._qwen_gpu(engine, [emb_worker, vlm]):
        # -- 2. embed ------------------------------------------------------- #
        todo = [(u, lab, stride, key) for u, lab in [(ref, "the reference video")]
                + [(u, f"target video {i + 1}") for i, u in enumerate(targets)]
                if not video_emb_path(u, key).exists()]
        todo += [(cl["clip"], f"step {c + 1}, every frame", 1, key1) for c, cl in enumerate(classes)
                 if not video_emb_path(cl["clip"], key1).exists()]
        if todo:
            emb_worker.request({"instruction": instruction, "images": [], "videos": [
                {"path": str(u.path), "stride": s, "out": str(video_emb_path(u, k_)),
                 "label": lab} for u, lab, s, k_ in todo]}, segx._sub(progress_cb, 0.0, 0.40), cancel)
        if cancel.is_set():
            raise Cancelled("cancelled")

        # the labelled frames: every frame of each step's clip (0..C-1, by the reference
        # frame it was cut from), and the reference's embedded frames outside every
        # marked range as the background (C)
        z = np.load(video_emb_path(ref, key))
        rf = z["frames"].astype(np.int64)
        inside = np.zeros(len(rf), bool)
        for cl in classes:
            inside |= (rf >= cl["range"][0]) & (rf <= cl["range"][1])
        E, lab, frm = [z["emb"].astype(np.float32)[~inside]], [np.full(int((~inside).sum()), C)], [rf[~inside]]
        for c, cl in enumerate(classes):
            zc = np.load(video_emb_path(cl["clip"], key1))
            E.append(zc["emb"].astype(np.float32))
            lab.append(np.full(len(zc["frames"]), c))
            frm.append(cl["range"][0] + zc["frames"].astype(np.int64))
        E, lab, frm = np.concatenate(E), np.concatenate(lab), np.concatenate(frm)
        counts = np.bincount(lab, minlength=C + 1)
        for c, cl in enumerate(classes):
            if not counts[c]:
                raise RuntimeError(f'the clip of "{cl["name"] or f"step {c + 1}"}" has no frames')
        if use_knn and not counts[C]:
            raise RuntimeError("the marked ranges cover the whole reference, so there is no "
                               "background for the kNN vote")

        examples = _examples(classes, ref, int(z["decoded"]), frm, E, lab, letters, clip_step)

        # -- 3 + 4, per target ---------------------------------------------- #
        results = []
        span = 0.40 + 0.03
        per = (0.99 - span) / len(targets)
        for t, target in enumerate(targets):
            lo = span + t * per
            progress_cb(lo, f"target {t + 1}/{len(targets)}: choosing the candidate stretches")
            tf, T, decoded = _cache.get(video_emb_path(target, key))
            fps = float(target.fps or 20.0)
            if use_knn:
                scores = _smooth_cols(knn_scores_multi(T, E, lab, C + 1, k),
                                      max(1, round(0.95 * fps / stride)))
                regions = [_knn_region(scores[:, c], decoded, fps, stride, thr) for c in range(C)]
            else:
                scores = np.stack([_sim_score(T, E[lab == c]) for c in range(C)], 1)
                cov = float(body.get("coverage") or 0.25)
                regions = [segx._candidate_region(scores[:, c], tf, decoded, fps, stride, cov)
                           for c in range(C)]
            region = np.any(regions, 0)
            centers = [f for f in range(0, decoded, stride) if region[f]]

            out = config.TMP_DIR / f"{job_id}_{t}_p.npy"
            if centers:
                vlm.request({"op": "classify_multi", "letters": letters, "options": options,
                             "examples": examples, "clip_frames": segx.CLIP_FRAMES,
                             "clip_step": clip_step, "out": str(out),
                             "target": {"path": str(target.path), "fps": fps, "total": decoded,
                                        "centers": centers}},
                            _stage(segx._sub(progress_cb, lo + 0.01, lo + per),
                                   f"target {t + 1}/{len(targets)}: "), cancel)
                try:
                    p = np.load(out)
                finally:
                    out.unlink(missing_ok=True)
            else:
                p = np.zeros((0, C + 1))
            results.append({
                "upload_id": target.id, "name": target.name, "decoded_frames": decoded,
                "fps": fps, "stride": stride,
                "emb_frames_stride": int(np.median(np.diff(tf))) if len(tf) > 1 else stride,
                "scores": np.round(scores, 3).T.tolist(),        # per class, per embedded frame
                "regions": [segx._runs(r) for r in regions],     # per class
                "region": segx._runs(region),
                "coverage": round(float(region.mean()), 3),
                "centers": centers,
                "p": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in p],
            })

    return {
        "key": key, "instruction": instruction, "letters": letters, "options": options,
        "classes": [cl["name"] for cl in classes],
        "candidates": "knn" if use_knn else "similarity",
        "knn": {"k": k, "threshold": thr} if use_knn else None,
        "coverage": None if use_knn else float(body.get("coverage") or 0.25),
        "clip": {"frames": segx.CLIP_FRAMES, "step": clip_step},
        "reference_frames": {"per_class": counts[:C].tolist(), "background": int(counts[C])},
        "examples": {l: [{"frame": e["ref_frame"]} for e in v] for l, v in examples.items()},
        "targets": results,
    }


def _stage(cb, prefix: str):
    return lambda f, stage: cb(f, prefix + stage)


def _sim_score(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """segx's candidate score: best z-scored similarity to any of the class's frames."""
    S = T @ P.T
    sd = S.std(0)
    sd[sd < 1e-6] = 1e-6
    return ((S - S.mean(0)) / sd).max(1)


def knn_scores_multi(q: np.ndarray, ref: np.ndarray, lab: np.ndarray, n_cls: int, k: int,
                     balanced: bool = True) -> np.ndarray:
    """segx.knn_scores for n_cls groups: each query frame's k nearest reference frames
    vote with softmax weights (temperature 0.02); balanced divides each group's vote by
    its share of the reference, so a short step is not outvoted by the background just
    for being short. Rows sum to 1: (n_query, n_cls)."""
    sims = q @ ref.T
    k = min(k, ref.shape[0] - 1) if ref.shape[0] > 1 else 1
    idx = np.argpartition(-sims, k, axis=1)[:, :k] if ref.shape[0] > k else np.tile(
        np.arange(ref.shape[0]), (len(q), 1))
    s = np.take_along_axis(sims, idx, axis=1)
    w = np.exp((s - s.max(axis=1, keepdims=True)) / 0.02)
    onehot = np.eye(n_cls)[lab[idx]]                        # (n, k, n_cls)
    votes = (w[..., None] * onehot).sum(1)
    if balanced:
        prior = np.bincount(lab, minlength=n_cls) / len(lab)
        votes = votes / np.where(prior > 0, prior, 1)
    return votes / votes.sum(1, keepdims=True)


def _smooth_cols(x: np.ndarray, win: int) -> np.ndarray:
    return np.stack([segx._smooth(x[:, c], win) for c in range(x.shape[1])], 1)


def _knn_region(sm: np.ndarray, decoded: int, fps: float, stride: int, thr: float) -> np.ndarray:
    """segx._knn_region's cleanup on one class's smoothed vote: >= thr, gaps under 2 s
    filled, runs under 1 s dropped, then every decoded frame takes its nearest
    embedded frame's decision."""
    win = lambda frames: max(1, round(frames / stride))
    pred = segx._clean(sm >= thr, win(1.0 * fps), win(2.0 * fps))
    near = np.clip(np.round(np.arange(decoded) / stride).astype(int), 0, len(pred) - 1)
    return pred[near]


def _medoids(E: np.ndarray, k: int) -> list[int]:
    """segx._examples' k-means medoids: indices of up to k rows covering E's spread."""
    k = min(k, len(E))
    rng = np.random.default_rng(0)
    cent = E[rng.choice(len(E), k, replace=False)]
    for _ in range(50):
        assign = np.argmax(E @ cent.T, 1)
        new = np.stack([E[assign == j].mean(0) if (assign == j).any() else cent[j] for j in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True)
        if np.allclose(new, cent):
            break
        cent = new
    return sorted({int(np.argmax(E @ c)) for c in cent})


def _examples(classes: list[dict], ref, ref_decoded: int, frm: np.ndarray, E: np.ndarray,
              lab: np.ndarray, letters: list[str], clip_step: int) -> dict:
    """Example clips per letter, within the VLM's video limit, all cut from the reference
    as the target clips are cut from theirs (same length, real motion). A class's are
    centred where the whole clip lies inside its range; the background's at least half
    a clip away from every marked range."""
    C = len(classes)
    n_each = max(1, min(segx.N_EXAMPLES, (MAX_VIDEOS - 1) // (C + 1)))
    half = (segx.CLIP_FRAMES - 1) * clip_step // 2
    ex = lambda c, f: {"clip": c, "ref_frame": int(f), "center": int(f), "total": ref_decoded,
                       "path": str(ref.path), "fps": float(ref.fps or 20.0)}
    out: dict[str, list] = {}
    for c, cl in enumerate(classes):
        a, b = cl["range"]
        rows = np.flatnonzero(lab == c)
        whole = rows[(frm[rows] - half >= a) & (frm[rows] + half <= b)]
        rows = whole if len(whole) else rows
        out[letters[c]] = [ex(c, frm[rows[j]]) for j in _medoids(E[rows], n_each)]
    bg = np.flatnonzero(lab == C)
    far = [r for r in bg if all(frm[r] + half < cl["range"][0] or frm[r] - half > cl["range"][1]
                                for cl in classes)]
    rows = np.asarray(far if far else bg, np.int64)
    out[letters[C]] = [ex(-1, frm[rows[j]]) for j in (_medoids(E[rows], n_each) if len(rows) else [])]
    return out
