"""Qwen3-VL-8B-Instruct worker for segment extraction — runs in the *qwen* venv.

    <qwen venv>/bin/python qwen_vlm_worker.py '{"model": ..., "gpu_util": 0.9, ...}'

Same protocol as qwen_embed_worker.py: loads once, prints `READY`, then serves one
JSON request per stdin line with `PROGRESS <0..1> <stage>` lines and a final
`DONE` or `ERROR <message>`. Imports nothing from the backend.

Requests:

  {"op": "describe", "clips": [{"path": ..., "fps": ...}, ...], "out": "x.json"}
      Watches up to 3 clips of one step and writes {"name", "description"}: the
      step as a short phrase, and what makes it recognisable in another recording.

  {"op": "classify", "options": {"A": "...", "B": "..."},
   "examples": {"A": [{"path", "center", "fps", "total"}...], "B": [...]},
   "target": {"path": ..., "fps": ..., "total": N, "centers": [...]},
   "clip_frames": 8, "clip_step": 5, "out": "p.npy"}
      For each centre, a ~2 s clip of the target goes to the model after the
      labelled example clips; P(A) from the first answer token's logprobs.
      The method of qwen_vl/extraction_project/scripts/vlm_classify.py.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # its JIT needs nvcc, absent here

import cv2
import numpy as np

MAX_SIDE = 640  # the research's frame size; larger frames cost tokens, not accuracy


def say(line: str) -> None:
    print(line, flush=True)


def fit(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def clip_indices(center: int, n: int, step: int, total: int) -> list[int]:
    """n frames `step` apart centred on `center`, clamped inside the video."""
    span = (n - 1) * step
    start = min(max(0, center - span // 2), max(0, total - 1 - span))
    return [min(total - 1, start + i * step) for i in range(n)]


def as_video(frames: list, idx: list[int], fps: float):
    """(array, metadata) as vLLM takes it; timestamps start at 0 for every clip."""
    rel = [i - idx[0] for i in idx]
    return np.stack(frames), {
        "fps": fps, "duration": (rel[-1] + 1) / fps, "total_num_frames": rel[-1] + 1,
        "frames_indices": rel, "video_backend": "opencv", "do_sample_frames": False}


def read_frames(path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """The wanted frame indices of a video, by one sequential decode (exact indices)."""
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(path)
    idx, last = 0, max(wanted)
    try:
        while idx <= last:
            if idx in wanted:
                ok, bgr = cap.read()
                if not ok:
                    break
                out[idx] = fit(bgr)
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
    return out


def frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def letter_probs(logprobs: dict, letters: list[str]) -> dict[str, float]:
    p = {k: 0.0 for k in letters}
    for lp in logprobs.values():
        t = lp.decoded_token.strip().upper()
        if t in p:
            p[t] += math.exp(lp.logprob)
    s = sum(p.values())
    return {k: (v / s if s else 1 / len(letters)) for k, v in p.items()}


class VLM:
    def __init__(self, cfg: dict) -> None:
        from vllm import LLM

        t0 = time.time()
        self.llm = LLM(model=cfg["model"], dtype="bfloat16",
                       max_model_len=int(cfg.get("max_len", 16384)),
                       gpu_memory_utilization=float(cfg["gpu_util"]), max_num_seqs=8,
                       enforce_eager=True, enable_prefix_caching=True, disable_log_stats=True,
                       limit_mm_per_prompt={"image": 0, "video": int(cfg.get("max_videos", 12))})
        self.tok = self.llm.get_tokenizer()
        print(f"model up in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    def prompt(self, content: list) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": content}],
                                            tokenize=False, add_generation_prompt=True)

    # -- describe ------------------------------------------------------------ #
    def describe(self, req: dict) -> None:
        from vllm import SamplingParams

        clips = req["clips"][:3]
        vids = []
        for i, c in enumerate(clips):
            n = frame_count(c["path"])
            if n == 0:
                raise RuntimeError(f"clip {i + 1} has no decodable frames")
            idx = sorted(set(np.linspace(0, n - 1, min(16, n)).round().astype(int).tolist()))
            frames = read_frames(c["path"], set(idx))
            idx = [f for f in idx if f in frames]
            vids.append(as_video([frames[f] for f in idx], idx, float(c.get("fps") or 20)))
            say(f"PROGRESS {0.1 + 0.5 * (i + 1) / len(clips):.3f} watching clip {i + 1}/{len(clips)}")
        content = [{"type": "text", "text":
                    f"Here {'is a video clip' if len(vids) == 1 else f'are {len(vids)} video clips'} of "
                    "the same step of a manual task, filmed by a fixed camera."}]
        for i in range(len(vids)):
            content += [{"type": "text", "text": f"Clip {i + 1}:"}, {"type": "video"}]
        content.append({"type": "text", "text":
            "Describe this step so that someone could recognise it in another recording of the "
            "same station and tell it apart from the steps before and after it. Answer only with "
            'JSON: {"name": "<the step as a short imperative phrase, under 12 words>", '
            '"description": "<2-4 sentences: the state and orientation of the workpiece, the '
            "tools and parts in use, where on the workpiece the hands work, and what visibly "
            'changes>"}'})
        say("PROGRESS 0.7 describing the step")
        out = self.llm.generate([{"prompt": self.prompt(content), "multi_modal_data": {"video": vids}}],
                                SamplingParams(temperature=0.0, max_tokens=320),
                                use_tqdm=False)[0].outputs[0].text
        m = re.search(r"\{.*\}", out, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        d = {"name": str(d.get("name") or "").strip(),
             "description": str(d.get("description") or "").strip() or out.strip()}
        json.dump(d, open(req["out"], "w"))

    # -- classify ------------------------------------------------------------ #
    def classify(self, req: dict) -> None:
        from vllm import SamplingParams

        n, step = int(req.get("clip_frames", 8)), int(req.get("clip_step", 5))
        letters = ["A", "B"]
        opts = req["options"]
        ex = req["examples"]

        # the example clips, the same ones in every prompt (prefix-cached)
        ex_items, ex_content = [], []
        for letter in letters:
            for e in ex.get(letter, []):
                idx = clip_indices(int(e["center"]), n, step, int(e["total"]))
                fr = read_frames(e["path"], set(idx))
                idx = [f for f in idx if f in fr]
                if len(idx) < 2:
                    continue
                ex_items.append(as_video([fr[f] for f in idx], idx, float(e["fps"])))
                ex_content += [{"type": "text", "text": f"Example of {letter}:"}, {"type": "video"}]
        with_b = any(c["text"] == "Example of B:" for c in ex_content if c["type"] == "text")
        content = [{"type": "text", "text":
                    "These clips come from fixed cameras at a manual workstation. Each clip is about "
                    "two seconds long. The options are:\n"
                    f"A = {opts['A']}\nB = {opts['B']}\n"
                    "Here are labelled example clips from other recordings of the same station.\n"}]
        content += ex_content
        content += [{"type": "text", "text": "Now the clip to classify:"}, {"type": "video"},
                    {"type": "text", "text":
                     "Which option does this last clip show? Compare the state and orientation of "
                     "the workpiece, the tool, and where the hands work with the examples"
                     + (" of both options" if with_b else "") +
                     ". Answer with a single letter: A or B."}]
        prompt = self.prompt(content)
        say("PROGRESS 0.02 example clips ready")

        # target clips: one sequential decode, each clip handed off as soon as it is whole
        tgt = req["target"]
        total, fps = int(tgt["total"]), float(tgt["fps"])
        centers = [int(c) for c in tgt["centers"]]
        clips = [clip_indices(c, n, step, total) for c in centers]
        need: dict[int, list[int]] = {}
        for q, idx in enumerate(clips):
            for f in idx:
                need.setdefault(f, []).append(q)
        last_of = [max(idx) for idx in clips]
        by_last: dict[int, list[int]] = {}
        for q, f in enumerate(last_of):
            by_last.setdefault(f, []).append(q)
        probs = np.full(len(clips), np.nan, np.float32)
        store: dict[int, np.ndarray] = {}
        batch: list[int] = []
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        done = 0
        t0 = time.time()

        def flush() -> None:
            nonlocal done
            if not batch:
                return
            inputs = [{"prompt": prompt, "multi_modal_data": {"video": ex_items + [
                as_video([store[f] for f in clips[q]], clips[q], fps)]}} for q in batch]
            outs = self.llm.generate(inputs, sp, use_tqdm=False)
            for q, o in zip(batch, outs):
                probs[q] = letter_probs(o.outputs[0].logprobs[0], letters)["A"]
            done += len(batch)
            batch.clear()
            # frames no pending clip needs any more
            keep = {f for q in range(len(clips)) if np.isnan(probs[q]) for f in clips[q]}
            for f in [f for f in store if f not in keep]:
                del store[f]
            rate = done / max(time.time() - t0, 1e-6)
            say(f"PROGRESS {0.02 + 0.98 * done / len(clips):.4f} "
                f"classifying clips {done}/{len(clips)} ({rate:.1f}/s)")

        if clips:
            cap = cv2.VideoCapture(tgt["path"])
            idx, last = 0, max(last_of)
            try:
                while idx <= last:
                    if idx in need:
                        ok, bgr = cap.read()
                        if not ok:
                            break
                        store[idx] = fit(bgr)
                    elif not cap.grab():
                        break
                    for q in by_last.get(idx, []):
                        batch.append(q)
                    if len(batch) >= 32:
                        flush()
                    idx += 1
            finally:
                cap.release()
            flush()
        np.save(req["out"], probs)

    def handle(self, req: dict) -> None:
        {"describe": self.describe, "classify": self.classify}[req["op"]](req)


def main(cfg: dict) -> None:
    vlm = VLM(cfg)
    say("READY")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            vlm.handle(json.loads(line))
            say("DONE")
        except Exception as exc:  # noqa: BLE001 - report it, stay up for the next request
            traceback.print_exc(file=sys.stderr)
            say(f"ERROR {type(exc).__name__}: {exc}".replace("\n", " "))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(json.loads(sys.argv[1]))
