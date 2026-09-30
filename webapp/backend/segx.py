"""Segment extraction: find where a step happens in a video, given clips of that step
from other recordings.

The pipeline of qwen_vl/extraction_project (report.md, "class_N_desc_vlm_hints"),
adapted to uploaded clips instead of a labelled reference video:

  1. describe  — Qwen3-VL-8B-Instruct watches the step clips and names the step and
     describes it in natural language (the research's hand-written step hints).
  2. embed     — Qwen3-VL-Embedding-8B embeds every `stride`-th frame of the target
     and of every clip, with the description in the instruction
     (embed_frames.py --describe).
  3. candidates — each target frame's best z-scored similarity to any step-clip
     frame. Clips give no negatives, so there is no kNN vote; instead the top
     `coverage` of the video (+ a margin) goes on to the VLM. Measured on the
     research data: top 25% kept 95-100% of every step's frames.
  4. classify  — for every `stride`-th candidate frame, a ~2 s clip goes to the VLM
     after labelled example clips (k-means medoids of the step clips; "other" clips
     when the user gave some) and the step description; P(step) from the answer's
     logprobs (vlm_classify.py).

The browser smooths, thresholds and cleans P(step), so the threshold is live.
"""
from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from typing import Callable

import numpy as np

from . import config
from .extraction import Cancelled, _cache, cache_key, video_emb_path

log = logging.getLogger("sam3webapp.segx")

ProgressCb = Callable[[float, str], None]

CLIP_FRAMES, CLIP_STEP = 8, 5          # a ~2.3 s clip at ~17.5 fps, as in vlm_classify.py
N_EXAMPLES = 4                         # example clips per option
OTHER_TEXT = ("anything else: another step, idle, reaching for or fetching parts, "
              "positioning or moving the workpiece")


def instruction_for(name: str, description: str, use_description: bool) -> str:
    """The embedding instruction: the step description in it (embed_frames.py --describe)."""
    text = " ".join(f"{name.strip().rstrip('.')}. {description.strip()}".split()).strip(". ")
    if not use_description or not text:
        return config.QWEN_EMB_INSTRUCTION
    return ("Represent this workstation image for recognising whether the worker is "
            "performing the following step: " + text)


@contextmanager
def _qwen_gpu(engine, workers):
    """Shared-GPU fallback: park SAM 3 around the run and stop Qwen after it."""
    if config.qwen_dedicated():
        yield
        return
    engine.offload()
    try:
        yield
    finally:
        for w in workers:
            w.stop()
        engine.restore()


def _sub(progress_cb: ProgressCb, lo: float, hi: float) -> ProgressCb:
    """A progress callback mapped into [lo, hi] of the job's bar."""
    return lambda f, stage: progress_cb(lo + (hi - lo) * max(0.0, min(1.0, f)), stage)


# --------------------------------------------------------------------------- #
# 1. describe
# --------------------------------------------------------------------------- #
def describe(vlm, engine, clip_ups: list, job_id: str, progress_cb: ProgressCb,
             cancel: threading.Event) -> dict:
    out = config.TMP_DIR / f"{job_id}_describe.json"
    req = {"op": "describe", "out": str(out),
           "clips": [{"path": str(u.path), "fps": u.fps} for u in clip_ups[:3]]}
    with _qwen_gpu(engine, [vlm]):
        vlm.request(req, progress_cb, cancel)
    try:
        d = json.loads(out.read_text())
    finally:
        out.unlink(missing_ok=True)
    return {"name": d.get("name", ""), "description": d.get("description", ""),
            "clips_watched": min(3, len(clip_ups))}


# --------------------------------------------------------------------------- #
# 2-4. search
# --------------------------------------------------------------------------- #
def search(emb_worker, vlm, engine, target, steps: list, others: list, body: dict,
           job_id: str, progress_cb: ProgressCb, cancel: threading.Event,
           refs: list | None = None, knn_steps: list | None = None,
           knn_others: list | None = None) -> dict:
    """refs: [(reference upload, step ranges, other ranges)] for the kNN candidates."""
    refs, knn_steps, knn_others = refs or [], knn_steps or [], knn_others or []
    use_knn = body.get("candidates") == "knn"
    stride = int(body.get("stride") or 5)
    instruction = instruction_for(body.get("name") or "", body.get("description") or "",
                                  bool(body.get("use_description", True)))
    key = cache_key(instruction, stride)

    with _qwen_gpu(engine, [emb_worker, vlm]):
        # -- 2. embed ------------------------------------------------------- #
        todo = [(u, lab) for u, lab in
                [(target, "the target video")]
                + [(u, f"step clip {i + 1}") for i, u in enumerate(steps)]
                + [(u, f"other clip {i + 1}") for i, u in enumerate(others)]
                + [(u, f"reference video {i + 1}") for i, (u, _, _) in enumerate(refs)]
                + [(u, f"kNN clip {i + 1}") for i, u in enumerate(knn_steps + knn_others)]
                if not video_emb_path(u, key).exists()]
        if todo:
            emb_worker.request({"instruction": instruction, "images": [], "videos": [
                {"path": str(u.path), "stride": stride, "out": str(video_emb_path(u, key)),
                 "label": lab} for u, lab in todo]}, _sub(progress_cb, 0.0, 0.42), cancel)
        if cancel.is_set():
            raise Cancelled("cancelled")

        # -- 3. candidates -------------------------------------------------- #
        progress_cb(0.43, "choosing the candidate stretches")
        tf, T, decoded = _cache.get(video_emb_path(target, key))
        step_embs = [np.load(video_emb_path(u, key)) for u in steps]
        P = np.concatenate([z["emb"].astype(np.float32) for z in step_embs])
        S = T @ P.T
        sd = S.std(0)
        sd[sd < 1e-6] = 1e-6
        emb_score = ((S - S.mean(0)) / sd).max(1)          # per embedded target frame
        fps = float(target.fps or 20.0)
        knn = None
        if use_knn:
            progress_cb(0.43, "kNN vote against the reference frames")
            knn = _knn_region(T, decoded, fps, stride, key, refs, knn_steps, knn_others,
                              int(body.get("knn_k") or 15), float(body.get("knn_threshold") or 0.5))
            region = knn["region"]
        else:
            region = _candidate_region(emb_score, tf, decoded, fps, stride,
                                       float(body.get("coverage") or 0.25))
        centers = [f for f in range(0, decoded, stride) if region[f]]

        # -- 4. classify ---------------------------------------------------- #
        name = (body.get("name") or "").strip() or "the step shown in the examples"
        desc = (body.get("description") or "").strip()
        options = {"A": f'the step "{name}"' + (f": {desc}" if desc else ""),
                   "B": OTHER_TEXT + (" — including the other steps shown in the examples of B"
                                      if others else "")}
        examples = {"A": _examples(steps, step_embs), "B": _examples(others, [
            np.load(video_emb_path(u, key)) for u in others])}
        out = config.TMP_DIR / f"{job_id}_p.npy"
        if centers:
            vlm.request({"op": "classify", "options": options, "examples": examples,
                         "clip_frames": CLIP_FRAMES, "clip_step": CLIP_STEP, "out": str(out),
                         "target": {"path": str(target.path), "fps": fps, "total": decoded,
                                    "centers": centers}},
                        _sub(progress_cb, 0.45, 0.99), cancel)
            try:
                p = np.load(out)
            finally:
                out.unlink(missing_ok=True)
        else:  # nothing passed the candidate stage: nothing to ask the VLM
            p = np.zeros(0)

    runs = _runs(region)
    return {
        "decoded_frames": decoded, "fps": fps, "stride": stride, "key": key,
        "instruction": instruction, "options": options,
        "emb_frames_stride": int(np.median(np.diff(tf))) if len(tf) > 1 else stride,
        "emb_score": np.round(emb_score, 3).tolist(),
        "region": runs,
        "centers": centers,
        "p": [None if np.isnan(v) else round(float(v), 4) for v in p],
        "examples": {k: [{"clip": e["clip"], "frame": e["center"]} for e in v]
                     for k, v in examples.items()},
        "coverage": round(float(region.mean()), 3),
        "candidates": "knn" if use_knn else "similarity",
        "knn": None if knn is None else {
            "score": np.round(knn["score"], 3).tolist(), "k": knn["k"], "threshold": knn["threshold"],
            "step_frames": knn["n_pos"], "other_frames": knn["n_neg"]},
    }


def _candidate_region(emb_score: np.ndarray, tf: np.ndarray, decoded: int, fps: float,
                      stride: int, coverage: float) -> np.ndarray:
    """The frames the VLM will look at: the top `coverage` of the smoothed score,
    runs under 1 s dropped, gaps under 2 s filled, then grown by 2 s each side."""
    if coverage >= 0.999:
        return np.ones(decoded, bool)
    win = max(1, round(0.95 * fps / stride))
    k = np.ones(win) / win
    sm = np.convolve(np.pad(emb_score, win // 2, mode="edge"), k, mode="valid")[: len(emb_score)]
    near = np.clip(np.round(np.arange(decoded) / stride).astype(int), 0, len(sm) - 1)
    per = sm[near]
    mask = per >= np.quantile(per, 1 - coverage)
    mask = _clean(mask, round(1.0 * fps), round(2.0 * fps))
    grow = round(2.0 * fps)
    return np.convolve(mask, np.ones(2 * grow + 1), "same") > 0


def knn_scores(q: np.ndarray, ref: np.ndarray, ref_pos: np.ndarray, k: int,
               balanced: bool = True) -> np.ndarray:
    """match_frames.knn_scores, as the research ran it: the similarity-weighted share of
    step frames among each query frame's k nearest reference frames (softmax over the
    neighbours at temperature 0.02); balanced divides each side's vote by its share of
    the reference, so a rare step is not outvoted just for being rare."""
    sims = q @ ref.T
    k = min(k, ref.shape[0] - 1) if ref.shape[0] > 1 else 1
    idx = np.argpartition(-sims, k, axis=1)[:, :k] if ref.shape[0] > k else np.tile(
        np.arange(ref.shape[0]), (len(q), 1))
    s = np.take_along_axis(sims, idx, axis=1)
    w = np.exp((s - s.max(axis=1, keepdims=True)) / 0.02)
    pos = (w * ref_pos[idx]).sum(1)
    neg = (w * ~ref_pos[idx]).sum(1)
    if balanced:
        pos, neg = pos / ref_pos.mean(), neg / (1 - ref_pos.mean())
    return pos / (pos + neg)


def _knn_region(T: np.ndarray, decoded: int, fps: float, stride: int, key: str, refs: list,
                knn_steps: list, knn_others: list, k: int, thr: float) -> dict:
    """The research's candidates (match_frames.py --balanced), with the marks as labels:
    reference frames inside a marked step range are the step, every other reference
    frame is not (two groups), plus ready-cut clips on either side. Each embedded target
    frame gets the balanced kNN vote; smoothed over ~1 s, >= thr, gaps under 2 s filled
    and runs under 1 s dropped (the research's 17 / 35 / 18 frames at ~17.6 fps), then
    every decoded frame takes its nearest embedded frame's decision. No margin is added:
    the VLM checks exactly these segments, as in vlm_classify.py --candidates."""
    E, POS = [], []
    for up, step_ranges, _ in refs:
        z = np.load(video_emb_path(up, key))
        f = z["frames"].astype(np.int64)
        pos = np.zeros(len(f), bool)
        for a, b in step_ranges:
            pos |= (f >= a) & (f <= b)
        E.append(z["emb"].astype(np.float32))
        POS.append(pos)
    for ups, is_step in ((knn_steps, True), (knn_others, False)):
        for up in ups:
            e = np.load(video_emb_path(up, key))["emb"].astype(np.float32)
            E.append(e)
            POS.append(np.full(len(e), is_step))
    E, POS = np.concatenate(E), np.concatenate(POS)
    if not POS.any():
        raise RuntimeError("the marked step ranges contain no embedded frames: mark longer stretches")
    if POS.all():
        raise RuntimeError("everything in the reference is marked as the step, so the kNN vote "
                           "has nothing to vote against")
    win = lambda frames: max(1, round(frames / stride))
    score = knn_scores(T, E, POS, k, balanced=True)
    sm = _smooth(score, win(0.95 * fps))
    pred = _clean(sm >= thr, win(1.0 * fps), win(2.0 * fps))
    near = np.clip(np.round(np.arange(decoded) / stride).astype(int), 0, len(pred) - 1)
    return {"region": pred[near], "score": sm, "k": k, "threshold": thr,
            "n_pos": int(POS.sum()), "n_neg": int((~POS).sum())}


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(np.pad(x, win // 2, mode="edge"), k, mode="valid")[: len(x)]


def _runs(mask: np.ndarray) -> list[list[int]]:
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append([start, i - 1])
            start = None
    if start is not None:
        out.append([start, len(mask) - 1])
    return out


def _clean(mask: np.ndarray, min_seg: int, max_gap: int) -> np.ndarray:
    """match_frames.clean: fill short gaps, then drop short runs."""
    mask = mask.copy()
    segs = _runs(mask)
    for (_, a1), (b0, _) in zip(segs, segs[1:]):
        if b0 - a1 - 1 <= max_gap:
            mask[a1 + 1:b0] = True
    for a, b in _runs(mask):
        if b - a + 1 < min_seg:
            mask[a:b + 1] = False
    return mask


def _examples(ups: list, embs: list) -> list[dict]:
    """Up to N_EXAMPLES example clip centres across the clips: k-means medoids of
    their frame embeddings (vlm_classify.kmeans_medoids), so the examples cover
    different phases of the step rather than N near-identical moments."""
    if not ups:
        return []
    half = (CLIP_FRAMES - 1) * CLIP_STEP // 2
    items, E = [], []
    for i, (u, z) in enumerate(zip(ups, embs)):
        dec = int(z["decoded"])
        for f, e in zip(z["frames"], z["emb"].astype(np.float32)):
            # prefer centres whose whole clip lies inside the uploaded clip
            if half <= f <= dec - 1 - half or dec <= 2 * half + 1:
                items.append((i, int(f), dec, u))
                E.append(e)
    if not items:
        return []
    E = np.asarray(E)
    k = min(N_EXAMPLES, len(items))
    rng = np.random.default_rng(0)
    cent = E[rng.choice(len(E), k, replace=False)]
    for _ in range(50):
        assign = np.argmax(E @ cent.T, 1)
        new = np.stack([E[assign == j].mean(0) if (assign == j).any() else cent[j] for j in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True)
        if np.allclose(new, cent):
            break
        cent = new
    picks = sorted({int(np.argmax(E @ c)) for c in cent})
    return [{"clip": items[j][0], "center": items[j][1], "total": items[j][2],
             "path": str(items[j][3].path), "fps": float(items[j][3].fps or 20.0)}
            for j in picks]
