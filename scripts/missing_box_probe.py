#!/usr/bin/env python3
"""Can a VLM spot a bounding box that is missing from one frame of a set?

Builds N-frame test cases from a video + its detector dets.json, deliberately removes the
boxes of one or more objects that are present in every frame of the case from one frame
each, asks a model which object is visible-but-unboxed where, and scores the answers.

Run with the qwen_vl venv (it has cv2, PIL, anthropic and vllm):

    PY=~/Documents/qwen_vl/.venv-vllm/bin/python

    # 1. test cases: 20 cases, 1-3 hidden classes each, 20% controls, labelled + unlabelled copies
    $PY scripts/missing_box_probe.py make --out runs/mbp1 --cases 20 --hide 1-3 --controls 0.2

    # 2. ask a model (resumable; each backend/model gets its own --tag)
    $PY scripts/missing_box_probe.py run --out runs/mbp1 --backend cli --model opus --tag opus
    $PY scripts/missing_box_probe.py run --out runs/mbp1 --backend api --model claude-opus-5 --tag opus5-api
    $PY scripts/missing_box_probe.py run --out runs/mbp1 --backend qwen --model Qwen/Qwen3-VL-8B-Instruct --tag qwen8b

    # 3. score (several tags side by side), optionally draw truth vs answers
    $PY scripts/missing_box_probe.py score --out runs/mbp1 --tag opus qwen8b --viz

Layout of --out:
    cases/<case>/frame1.jpg ...   what the model sees - nothing else lives in these folders
    truth.json                    ground truth per case (never shown to the model)
    review/<case>.jpg             the case with the hidden boxes drawn dashed, for eyeballing
    answers/<tag>/<case>.json     raw model output + parsed JSON + cost/usage
    answers/<tag>/scores.csv      per-case scores; viz/<case>.jpg with --viz

Backends:
    cli   Claude Code headless (`claude -p`), uses your Claude login, no API key. Each case
          runs in a fresh temp dir holding only the frame images; tools limited to Read.
    api   Anthropic Messages API (needs ANTHROPIC_API_KEY), structured JSON output.
    qwen  Qwen3-VL through vLLM offline on the local GPUs (--tp 2 for the 32B model).

Scoring: a hidden object counts as found when an answer names the right frame AND gives a
bbox with IoU >= --iou against the removed box (the class name is used only when an answer
has no usable bbox). Claude is asked for pixel boxes, Qwen for 0-1000. Anything else reported
is a false alarm; false alarms on a class whose box count already varies across the case's
frames before hiding (a possible genuine detector miss) are counted separately as "natural".
"""
from __future__ import annotations

import argparse
import base64
import collections
import colorsys
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = REPO / "datasets/Stihl SH86 Packing (2.6)/Videos/2026-08-06 11_18_23_raw.mp4"


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def class_color(cls: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((cls * 0.61803) % 1, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_boxes(img: np.ndarray, boxes: list, names: dict, labels: bool) -> np.ndarray:
    img = img.copy()
    for x1, y1, x2, y2, _conf, cls in boxes:
        cls = int(cls)
        c = class_color(cls)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        if labels:
            t = names[str(cls)]
            (tw, th), _ = cv2.getTextSize(t, 0, 0.45, 1)
            ty = max(int(y1), th + 4)
            cv2.rectangle(img, (int(x1), ty - th - 4), (int(x1) + tw + 4, ty), c, -1)
            cv2.putText(img, t, (int(x1) + 2, ty - 3), 0, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def dashed_rect(img, box, color, thick=2, dash=8):
    x1, y1, x2, y2 = map(int, box[:4])
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for i in range(0, n, 2):
            p = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
            q = (int(a[0] + (b[0] - a[0]) * min(i + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(i + 1, n) / n))
            cv2.line(img, p, q, color, thick)


def grid(tiles: list[np.ndarray], cols: int = 2) -> np.ndarray:
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def read_frames(video: str, wanted: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video)
    out, i, last = {}, 0, max(wanted)
    while i <= last:
        if not cap.grab():
            break
        if i in wanted:
            out[i] = cap.retrieve()[1]
        i += 1
    cap.release()
    missing = wanted - out.keys()
    if missing:
        raise RuntimeError(f"could not decode frames {sorted(missing)[:5]} of {video}")
    return out


# --------------------------------------------------------------------------- #
# make
# --------------------------------------------------------------------------- #
def dets_for(video: Path) -> Path:
    stem = video.stem[:-4] if video.stem.endswith("_raw") else video.stem
    return video.with_name(f"{stem}.mp4_dets.json")


def pick_frames(n_total: int, k: int, min_gap: int, window: int, rng: random.Random) -> list[int] | None:
    span = n_total if window <= 0 else min(window, n_total)
    start = rng.randrange(0, n_total - span + 1)
    pool = list(range(start, start + span))
    for _ in range(50):
        fr = sorted(rng.sample(pool, k))
        if all(b - a >= min_gap for a, b in zip(fr, fr[1:])):
            return fr
    return None


def cmd_make(a):
    video = Path(a.video)
    dets_path = Path(a.dets) if a.dets else dets_for(video)
    D = json.load(open(dets_path))
    names = D["meta"]["names"]
    det = {int(k): [b for b in v if b[4] >= a.min_conf] for k, v in D["detections"].items()}
    n_total = len(det)
    by_name = {v: int(k) for k, v in names.items()}
    only = {by_name[c] for c in a.classes} if a.classes else None
    exclude = {by_name[c] for c in a.exclude} if a.exclude else set()
    lo, hi = (int(x) for x in (a.hide.split("-") if "-" in a.hide else (a.hide, a.hide)))
    rng = random.Random(a.seed)

    out = Path(a.out)
    if (out / "truth.json").exists() and not a.force:
        sys.exit(f"{out}/truth.json exists - pass --force to overwrite, or pick another --out")
    shutil.rmtree(out / "cases", ignore_errors=True)
    shutil.rmtree(out / "review", ignore_errors=True)
    (out / "cases").mkdir(parents=True)
    (out / "review").mkdir()

    plans = []
    for ci in range(a.cases):
        control = rng.random() < a.controls
        n_hide = 0 if control else rng.randint(lo, hi)
        for _ in range(500):
            fr = pick_frames(n_total, a.frames, a.min_gap, a.window, rng)
            if fr is None:
                continue
            counts = [collections.Counter(int(b[5]) for b in det[f]) for f in fr]
            common = [c for c in set.intersection(*(set(x) for x in counts))
                      if c not in exclude and (only is None or c in only)]
            if len(common) >= n_hide:
                break
        else:
            sys.exit(f"case {ci}: no {a.frames} frames with {n_hide} common classes - relax --min-gap/--window/--classes")
        # classes whose count already differs across these frames = possible real detector misses
        natural = sorted(names[str(c)] for c in set().union(*counts)
                         if len({x.get(c, 0) for x in counts}) > 1)
        hidden_cls = rng.sample(sorted(common), n_hide)
        same_pos = rng.randrange(a.frames)
        hides = []
        for c in hidden_cls:
            pos = same_pos if a.same_frame else rng.randrange(a.frames)
            f = fr[pos]
            inst = [b for b in det[f] if int(b[5]) == c]
            gone = inst if a.mode == "class" else [rng.choice(inst)]
            hides.append({"pos": pos + 1, "frame": f, "cls": names[str(c)], "cls_id": c,
                          "boxes": [[round(x, 1) for x in b[:4]] for b in gone]})
        plans.append({"frames": fr, "hidden": hides, "natural_inconsistent": natural})

    frames = read_frames(str(video), {f for p in plans for f in p["frames"]})
    variants = {"on": [True], "off": [False], "both": [True, False]}[a.labels]
    truth = {"video": str(video), "dets": str(dets_path), "names": names, "args": vars(a), "cases": []}
    for ci, p in enumerate(plans):
        for labels in variants:
            cid = f"c{ci:03d}_{len(p['hidden'])}h_{'lab' if labels else 'nolab'}"
            d = out / "cases" / cid
            d.mkdir()
            shown, review = [], []
            for k, f in enumerate(p["frames"]):
                gone = {tuple(bx) for h in p["hidden"] if h["pos"] == k + 1 for bx in h["boxes"]}
                bs = [b for b in det[f] if tuple(round(x, 1) for x in b[:4]) not in gone]
                img = draw_boxes(frames[f], bs, names, labels)
                cv2.imwrite(str(d / f"frame{k + 1}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                shown.append([[round(x, 1) for x in b[:4]] + [names[str(int(b[5]))]] for b in bs])
                r = img.copy()
                for h in p["hidden"]:
                    if h["pos"] == k + 1:
                        for bx in h["boxes"]:
                            dashed_rect(r, bx, (0, 0, 255), 3)
                            cv2.putText(r, f"HIDDEN {h['cls']}", (int(bx[0]), int(bx[3]) + 16), 0, 0.5, (0, 0, 255), 2)
                cv2.putText(r, f"Frame {k + 1} (video {f})", (10, r.shape[0] - 12), 0, 0.8, (255, 255, 255), 2)
                review.append(r)
            cv2.imwrite(str(out / "review" / f"{cid}.jpg"), grid(review), [cv2.IMWRITE_JPEG_QUALITY, 85])
            truth["cases"].append({"id": cid, "labels": labels, "size": list(frames[p["frames"][0]].shape[1::-1]),
                                   "shown_boxes": shown, **p})
    json.dump(truth, open(out / "truth.json", "w"), indent=1)
    nh = collections.Counter(len(p["hidden"]) for p in plans)
    print(f"{len(truth['cases'])} cases in {out}/cases  (hidden classes per case: {dict(sorted(nh.items()))})")
    print(f"review sheets with the hidden boxes marked: {out}/review/")


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {"type": "array", "items": {
            "type": "object",
            "properties": {"frame": {"type": "integer"}, "boxes": {"type": "array", "items": {"type": "string"}}},
            "required": ["frame", "boxes"], "additionalProperties": False}},
        "missing": {"type": "array", "items": {
            "type": "object",
            "properties": {"frame": {"type": "integer"}, "object": {"type": "string"},
                           "bbox_2d": {"type": "array", "items": {"type": "number"}}},
            "required": ["frame", "object", "bbox_2d"], "additionalProperties": False}},
    },
    "required": ["frames", "missing"], "additionalProperties": False,
}


def build_prompt(case: dict, n: int, class_list: list[str] | None, coords: str) -> str:
    """coords: "px" (pixels of the image - Claude) or "norm1000" (0-1000 - Qwen's native grounding)."""
    lab = ("each labelled with its class name" if case["labels"]
           else "colour-coded by class (no text labels)")
    obj = "<class name, as labelled>" if case["labels"] else "<short description of the object>"
    lines = [
        f"These {n} images (Frame 1 to Frame {n}) are frames from the same fixed camera setup at a work "
        "station, taken at different moments. An image may show more than one camera view side by side.",
        f"An object detector's bounding boxes are drawn on every frame, {lab}.",
        "The detector sometimes misses an object in a frame: the object is still there and visible, but it "
        "has no bounding box, while the same object is boxed in the other frames. It is also possible that "
        "nothing is missing.",
        "Find every object that is visible but unboxed in some frame although it is boxed in other frames. "
        "Objects that have really left the scene or are fully hidden do not count. People and hands move, so "
        "judge them by whether they are visibly present.",
        "First list the boxes drawn on each frame, then compare the frames.",
    ]
    if class_list:
        lines.append("Detector classes: " + ", ".join(class_list) + ".")
    lines.append(
        'Answer with JSON: {"frames": [{"frame": 1, "boxes": ["<label>", ...]}, ...], '
        f'"missing": [{{"frame": <1-{n}>, "object": "{obj}", "bbox_2d": [x1, y1, x2, y2]}}]}} '
        "where bbox_2d is where the unboxed object is in that frame, "
        + (f"in pixels of the image (each image is {case['size'][0]} x {case['size'][1]} pixels, 0,0 top-left)"
           if coords == "px" else "in 0-1000 coordinates relative to the whole image (0,0 top-left)")
        + '. Use "missing": [] if nothing is missing.')
    return "\n".join(lines)


def parse_json(text: str):
    for m in reversed(list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S))):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    s, e = text.find("{"), text.rfind("}")
    while s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            s = text.find("{", s + 1)
    return None


def claude_bin() -> str:
    for c in (os.environ.get("CLAUDE_BIN"), os.environ.get("CLAUDE_CODE_EXECPATH"), shutil.which("claude")):
        if c and Path(c).exists():
            return c
    sys.exit("claude binary not found - set CLAUDE_BIN=/path/to/claude")


def run_claude_sandboxed(a, tmp: str, prompt: str) -> dict:
    """One headless Claude Code call that can only read files inside `tmp`.

    Isolation, in layers:
      - cwd is a fresh temp dir holding nothing but the case's frames (random name, no case id);
      - the only tool is Read (no Bash/Glob/Grep, so it cannot list or search the disk);
      - Read is allowed only under `tmp` and --permission-mode dontAsk denies everything else,
        so a Read of truth.json / dets.json / the dataset is refused even if it guessed a path;
      - --setting-sources project,local skips your user settings (their allow rules) and
        --strict-mcp-config drops MCP servers / connectors;
      - every tool call is logged from the stream and checked afterwards (see `audit`).
    """
    cmd = [claude_bin(), "-p", prompt, "--model", a.model, "--effort", a.effort or "high",
           "--output-format", "stream-json", "--verbose",
           "--tools", "Read", "--allowedTools", f"Read(/{tmp}/**)", "--permission-mode", "dontAsk",
           "--setting-sources", "project,local", "--strict-mcp-config", "--no-session-persistence"]
    r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=a.timeout)
    events = []
    for line in r.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    final = next((e for e in reversed(events) if e.get("type") == "result"), None)
    if final is None:
        raise RuntimeError(f"claude exited {r.returncode}: {(r.stderr or r.stdout)[-500:]}")
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    calls, denied = [], []
    for e in events:
        msg = e.get("message")
        for b in (msg.get("content") if isinstance(msg, dict) else None) or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                calls.append({"id": b.get("id"), "tool": b.get("name"), "input": b.get("input")})
            elif b.get("type") == "tool_result" and b.get("is_error"):
                denied.append({"id": b.get("tool_use_id"), "error": str(b.get("content"))[:300]})
    return {"text": final.get("result", ""), "cost_usd": final.get("total_cost_usd"), "usage": final.get("usage"),
            "is_error": final.get("is_error"), "session_tools": init.get("tools"),
            "resolved_model": init.get("model"), "model_usage": final.get("modelUsage"),
            "session_mcp": init.get("mcp_servers"), "tool_calls": calls, "tool_errors": denied}


def audit(res: dict, allowed: list[str]) -> dict:
    """Did the model touch anything but its own frames?"""
    ok = {str(Path(p).resolve()) for p in allowed}
    off = [c for c in res.get("tool_calls", [])
           if c["tool"] != "Read" or str(Path(str((c["input"] or {}).get("file_path", ""))).resolve()) not in ok]
    extra_tools = [t for t in (res.get("session_tools") or []) if t != "Read"]
    return {"off_limits_calls": off, "extra_tools": extra_tools, "tainted": bool(off)}


def ask_cli(a, case, images, prompt):
    """Claude Code headless on a private copy of the frames (see run_claude_sandboxed)."""
    with tempfile.TemporaryDirectory(prefix="mbp_") as tmp:
        tmp = str(Path(tmp).resolve())
        names = []
        for k, p in enumerate(images, 1):
            shutil.copy(p, Path(tmp) / f"frame{k}.jpg")
            names.append(f"{tmp}/frame{k}.jpg")
        full = ("Use the Read tool to look at these image files, in order:\n" + "\n".join(names) + "\n\n" + prompt
                + "\nReply with the JSON only.")
        res = run_claude_sandboxed(a, tmp, full)
        res["audit"] = audit(res, names)
    return res


def cmd_check_isolation(a):
    """Try to make the sandboxed agent read the truth file, the dataset and a canary; all must be refused."""
    import secrets
    out = Path(a.out).resolve()
    truth = json.load(open(out / "truth.json"))
    token = secrets.token_hex(8)
    canary = out / "isolation_canary.txt"
    canary.write_text(f"CANARY {token}\n")
    targets = [out / "truth.json", Path(truth["dets"]).resolve(), canary]
    try:
        with tempfile.TemporaryDirectory(prefix="mbp_") as tmp:
            tmp = str(Path(tmp).resolve())
            prompt = ("This is a sandbox test. Use the Read tool on each of these files and quote the first 100 "
                      "characters of each, or say exactly why you could not:\n" + "\n".join(map(str, targets)))
            res = run_claude_sandboxed(a, tmp, prompt)
    finally:
        canary.unlink(missing_ok=True)
    errors = {d["id"]: d["error"] for d in res["tool_errors"]}
    leaked = token in res["text"]
    print(f"tools in the session: {res['session_tools']}   MCP servers: {res['session_mcp']}")
    for c in res["tool_calls"]:
        err = errors.get(c["id"])
        leaked |= err is None
        print(f"  {'DENIED ' if err else 'ALLOWED'} {c['tool']} {(c['input'] or {}).get('file_path', c['input'])}")
    print(f"canary token in the reply: {token in res['text']}")
    print(f"model said: {res['text'][:300]}")
    print("RESULT:", "FAIL - the agent could read files outside its sandbox" if leaked
          else "ok - every read outside the sandbox was refused")
    sys.exit(1 if leaked else 0)


_client = None


def ask_api(a, case, images, prompt):
    import anthropic
    global _client
    _client = _client or anthropic.Anthropic()
    content = []
    for k, p in enumerate(images, 1):
        content += [{"type": "text", "text": f"Frame {k}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                 "data": base64.standard_b64encode(Path(p).read_bytes()).decode()}}]
    content.append({"type": "text", "text": prompt})
    kw = dict(model=a.model, max_tokens=16000, messages=[{"role": "user", "content": content}],
              thinking={"type": "adaptive"},
              output_config={"effort": a.effort or "high", "format": {"type": "json_schema", "schema": SCHEMA}})
    # server-side refusal fallback (Opus 5 / Fable 5.1 guidance); --no-fallback to drop it
    if a.fallback:
        resp = _client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
    else:
        resp = _client.messages.create(**kw)
    if resp.stop_reason == "refusal":
        det = getattr(resp, "stop_details", None)
        raise RuntimeError(f"refusal: {getattr(det, 'category', None)} {getattr(det, 'explanation', '')}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {"text": text, "usage": resp.usage.to_dict(), "stop_reason": resp.stop_reason, "model": resp.model}


def run_qwen(a, todo, prompts):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from PIL import Image
    from vllm import LLM, SamplingParams
    n = max(len(imgs) for _, imgs in todo)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_len, gpu_memory_utilization=0.88,
              enforce_eager=True, limit_mm_per_prompt={"image": n, "video": 0}, disable_log_stats=True)
    tok = llm.get_tokenizer()
    reqs = []
    for (case, imgs), prompt in zip(todo, prompts):
        content = []
        for k in range(len(imgs)):
            content += [{"type": "text", "text": f"Frame {k + 1}:"}, {"type": "image"}]
        content.append({"type": "text", "text": prompt})
        reqs.append({"prompt": tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                       add_generation_prompt=True),
                     "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in imgs]}})
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=2000))
    return [{"text": o.outputs[0].text} for o in outs]


def cmd_run(a):
    out = Path(a.out)
    truth = json.load(open(out / "truth.json"))
    tag = a.tag or re.sub(r"[^\w.-]", "_", f"{a.backend}-{a.model}")
    adir = out / "answers" / tag
    adir.mkdir(parents=True, exist_ok=True)
    classes = sorted(set(truth["names"].values())) if a.class_list else None
    todo = []
    for c in truth["cases"]:
        if a.only and not any(s in c["id"] for s in a.only):
            continue
        if (adir / f"{c['id']}.json").exists() and not a.redo:
            continue
        todo.append((c, [str(out / "cases" / c["id"] / f"frame{k}.jpg") for k in range(1, len(c["frames"]) + 1)]))
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} cases to ask ({tag}); answers -> {adir}")
    if not todo:
        return
    # Claude mixed 0-1000 and pixel boxes when asked for 0-1000 (exp1: 3 of 72), so it gets pixels
    coords = "norm1000" if a.backend == "qwen" else "px"
    prompts = [build_prompt(c, len(imgs), classes, coords) for c, imgs in todo]

    def save(case, prompt, res, secs):
        res.update(id=case["id"], backend=a.backend, model=a.model, seconds=round(secs, 1), prompt=prompt, coords=coords,
                   effort=None if a.backend == "qwen" else a.effort or "high",
                   parsed=parse_json(res.get("text", "")))
        json.dump(res, open(adir / f"{case['id']}.json", "w"), indent=1)

    if a.backend == "qwen":
        t0 = time.time()
        for (c, _), p, r in zip(todo, prompts, run_qwen(a, todo, prompts)):
            save(c, p, r, (time.time() - t0) / len(todo))
        print("done")
        return

    ask = ask_cli if a.backend == "cli" else ask_api

    def one(job):
        (c, imgs), p = job
        t0 = time.time()
        try:
            r = ask(a, c, imgs, p)
        except Exception as e:  # keep going; failed cases are simply retried on the next run
            print(f"  {c['id']}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            return
        save(c, p, r, time.time() - t0)
        n_miss = len((parse_json(r["text"]) or {}).get("missing", []) or [])
        taint = (r.get("audit") or {}).get("tainted")
        print(f"  {c['id']}: {time.time() - t0:.0f}s, reported {n_miss} missing"
              + (f", ${r['cost_usd']:.3f}" if r.get("cost_usd") else "")
              + ("  !! TAINTED: read files other than its frames" if taint else ""), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, zip(todo, prompts)))


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def to_px(bb, size, coords: str):
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None
    try:
        bb = [float(x) for x in bb]
    except (TypeError, ValueError):
        return None
    if coords == "px":
        return bb
    w, h = size
    return [bb[0] * w / 1000, bb[1] * h / 1000, bb[2] * w / 1000, bb[3] * h / 1000]


def score_case(case: dict, parsed, iou_thr: float, coords: str = "norm1000") -> dict:
    """Match the model's reported items against the hidden objects.

    Returns counts plus `targets` (each hidden object, found or not) and `items` (each reported
    item with status hit | dup | fp | fp_natural) for drawing."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("missing"), list):
        return {"parse_ok": False}
    size = case["size"]
    targets = [dict(h, found=False) for h in case["hidden"]]  # one target per hidden object (all its boxes)
    items = []
    for it in parsed["missing"]:
        if not isinstance(it, dict):
            continue
        try:
            fr = int(it.get("frame", -1))
        except (TypeError, ValueError):
            fr = -1
        o, bb = norm(it.get("object", "")), to_px(it.get("bbox_2d"), size, coords)
        # position decides; the class name only counts when the answer gave no usable box
        hits = [ti for ti, h in enumerate(targets) if fr == h["pos"] and (
            max(iou(bb, b) for b in h["boxes"]) >= iou_thr if bb is not None
            else case["labels"] and o and (o == norm(h["cls"]) or norm(h["cls"]) in o))]
        if hits:  # a repeat report of an already-found object (e.g. its 2nd instance) is not a false alarm
            new = [ti for ti in hits if not targets[ti]["found"]]
            ti = (new or hits)[0]
            status = "hit" if new else "dup"
            targets[ti]["found"] = True
        elif any(norm(n) in o or (o and o in norm(n)) for n in case["natural_inconsistent"]):
            status = "fp_natural"
        else:
            status = "fp"
        items.append({"frame": fr, "object": it.get("object"), "bbox_px": bb, "status": status})
    # label-reading accuracy (labelled cases): per-frame multiset F1 of the listed boxes
    read_f1 = None
    if case["labels"] and isinstance(parsed.get("frames"), list):
        f1s = []
        for k, shown in enumerate(case["shown_boxes"], 1):
            got = next((f.get("boxes", []) for f in parsed["frames"] if isinstance(f, dict) and f.get("frame") == k), [])
            t = collections.Counter(norm(b[4]) for b in shown)
            g = collections.Counter(norm(x) for x in got if isinstance(x, str))
            tp = sum((t & g).values())
            f1s.append(2 * tp / (sum(t.values()) + sum(g.values())) if (t or g) else 1.0)
        read_f1 = sum(f1s) / len(f1s)
    fp = [i for i in items if i["status"] == "fp"]
    return {"parse_ok": True, "n_hidden": len(targets), "found": sum(t["found"] for t in targets), "fp": len(fp),
            "fp_natural": sum(i["status"] == "fp_natural" for i in items),
            "exact": all(t["found"] for t in targets) and not fp, "read_f1": read_f1,
            "missed": [f"{t['cls']}@F{t['pos']}" for t in targets if not t["found"]],
            "false_alarms": [f"{i['object']}@F{i['frame']}" for i in fp], "targets": targets, "items": items}


# --- visualisation --------------------------------------------------------- #
WHITE, RED, GREEN, ORANGE, YELLOW, GREY = (255, 255, 255), (40, 40, 255), (60, 220, 60), (0, 150, 255), (0, 230, 255), (150, 150, 150)
ITEM_STYLE = {"hit": (GREEN, "CLAUDE: {o}  (correct)"), "dup": (GREEN, "CLAUDE: {o}  (correct, repeat)"),
              "fp": (ORANGE, "CLAUDE: {o}  (FALSE ALARM)"),
              "fp_natural": (YELLOW, "CLAUDE: {o}  (class already varies - maybe a real miss)")}


def put_label(img, text, x, y, color, scale=0.6, above=True):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = int(min(max(0, x), img.shape[1] - tw - 6))
    y = int(y - 4 if above else y + th + 8)
    y = min(max(th + 6, y), img.shape[0] - 4)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def viz_case(raw: dict, case: dict, sc: dict, dst: Path, tag: str):
    """Dimmed frame + the detector's shown boxes in thin grey; each hidden box dashed (white = the
    model found it, red = missed); each box the model reported solid (green = correct,
    orange = false alarm, yellow = on a class whose count already varies)."""
    tiles = []
    for k, f in enumerate(case["frames"], 1):
        img = (raw[f].astype(np.float32) * 0.55).astype(np.uint8)
        for b in case["shown_boxes"][k - 1]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREY, 1)
            if case["labels"]:
                cv2.putText(img, b[4], (int(b[0]) + 2, int(b[1]) + 12), 0, 0.38, GREY, 1, cv2.LINE_AA)
        marks = []
        for t in sc.get("targets", case["hidden"]):
            if t["pos"] != k:
                continue
            col = WHITE if t.get("found") else RED
            for bx in t["boxes"]:
                dashed_rect(img, bx, col, 3, 10)
            bx = t["boxes"][0]
            put_label(img, f"HIDDEN {t['cls']}: {'found' if t.get('found') else 'MISSED'}", bx[0], bx[3], col,
                      above=False)
            marks.append("found" if t.get("found") else "MISSED")
        for it in sc.get("items", []):
            if it["frame"] != k:
                continue
            col, fmt = ITEM_STYLE[it["status"]]
            bb = it["bbox_px"]
            if bb:
                cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), col, 3)
                put_label(img, fmt.format(o=it["object"]), bb[0], bb[1], col)
            else:
                put_label(img, fmt.format(o=it["object"]) + " (no bbox)", 10, 60, col)
            marks.append(it["status"])
        put_label(img, f"Frame {k}  (video frame {f})", 8, img.shape[0] - 8, WHITE, 0.75)
        tiles.append(img)
    body = grid(tiles)
    head = np.full((110, body.shape[1], 3), 30, np.uint8)
    hid = ", ".join(f"{t['cls']} in F{t['pos']}" for t in case["hidden"]) or "nothing (control)"
    if sc.get("parse_ok"):
        verdict = (f"found {sc['found']}/{sc['n_hidden']}   false alarms {sc['fp']}"
                   f" (+{sc['fp_natural']} on classes that already vary)   -> {'CORRECT' if sc['exact'] else 'WRONG'}")
        vcol = GREEN if sc["exact"] else RED
    else:
        verdict, vcol = "answer could not be parsed", RED
    cv2.putText(head, f"{case['id']}   [{tag}]   hidden: {hid}", (12, 32), 0, 0.8, WHITE, 2, cv2.LINE_AA)
    cv2.putText(head, verdict, (12, 66), 0, 0.8, vcol, 2, cv2.LINE_AA)
    cv2.putText(head, "dashed = box we removed (white: Claude found it, red: missed)   solid = Claude's answer "
                "(green correct, orange false alarm, yellow maybe-real miss)   grey = boxes left on the frame",
                (12, 98), 0, 0.55, GREY, 1, cv2.LINE_AA)
    cv2.imwrite(str(dst), np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_report(adir: Path, tag: str, summary: list[str], rows: list[dict], truth: dict):
    import html
    by_id = {c["id"]: c for c in truth["cases"]}
    parts = [f"<!doctype html><meta charset=utf-8><title>missing-box probe: {html.escape(tag)}</title>",
             "<style>body{font:14px system-ui;margin:16px;background:#fafafa;color:#222}"
             "pre{background:#fff;border:1px solid #ddd;padding:10px;overflow:auto}"
             ".case{background:#fff;border:1px solid #ddd;margin:18px 0;padding:10px}"
             ".ok{color:#1a7f37}.bad{color:#cf222e}.warn{color:#9a6700}img{width:100%;height:auto}"
             "a{margin-right:8px}</style>",
             f"<h1>Missing-box probe: {html.escape(tag)}</h1><pre>{html.escape(chr(10).join(summary))}</pre><p>"]
    for r in rows:
        cls = "warn" if r.get("tainted") else ("ok" if r.get("exact") else "bad")
        parts.append(f"<a class={cls} href='#{r['id']}'>{r['id']}</a>")
    parts.append("</p>")
    for r in rows:
        c = by_id[r["id"]]
        if r.get("tainted"):
            verdict = "<b class=warn>TAINTED - read files outside its frames, excluded</b>"
        elif not r["parse_ok"]:
            verdict = "<b class=bad>unparseable answer</b>"
        else:
            verdict = (f"<b class={'ok' if r['exact'] else 'bad'}>{'CORRECT' if r['exact'] else 'WRONG'}</b>"
                       f" - found {r['found']}/{r['n_hidden']}, false alarms {r['fp']} (+{r['fp_natural']} maybe-real)")
        hid = ", ".join(f"{h['cls']} in frame {h['pos']}" for h in c["hidden"]) or "nothing (control)"
        ans = json.dumps((r.get("parsed") or {}).get("missing"), indent=1)
        parts.append(f"<div class=case id='{r['id']}'><h3>{r['id']}</h3><p>Hidden: {html.escape(hid)}<br>{verdict}"
                     f"{' - missed: ' + html.escape(', '.join(r['missed'])) if r.get('missed') else ''}"
                     f"{' - false alarms: ' + html.escape(', '.join(r['false_alarms'])) if r.get('false_alarms') else ''}"
                     f"</p><img loading=lazy src='viz/{r['id']}.jpg'><details><summary>Claude's answer</summary>"
                     f"<pre>{html.escape(ans)}</pre></details></div>")
    (adir / "report.html").write_text("\n".join(parts))


STATUS_TXT = {"hit": "correct", "dup": "correct (repeat)", "fp": "FALSE ALARM", "fp_natural": "maybe a real miss"}
REPORT_INTRO = ("Can the model spot an object that is visible in a frame but whose detector bounding box was "
                "removed, when the same object is boxed in the other frames of the set?")
REPORT_NOTE = ("Fully correct = every hidden object found and no false alarm. Maybe-real misses = reports on a class "
               "whose box count already varies across the case's frames, so they may be genuine detector misses; "
               "they are not counted as false alarms. Label reading F1 = how well the model listed the labels drawn "
               "on each frame (labelled cases).")
REPORT_LEGEND = ("In each image the frame is dimmed and the detector's remaining boxes are thin grey. Dashed = a box we "
                 "removed (white: the model found it, red: missed). Solid = the model's answer (green: correct, "
                 "orange: false alarm, yellow: on a class whose count already varies).")


def report_content(tag: str, stats: list[dict], per_cls, per_cls_hit, rows: list[dict], truth: dict,
                   iou_thr: float) -> dict:
    """Everything the Markdown / PDF reports show, independent of format."""
    by_id = {c["id"]: c for c in truth["cases"]}
    args = truth.get("args", {})
    ok = [r for r in rows if r["parse_ok"] and not r["tainted"]]
    models = sorted({f"{r.get('backend')} / {r.get('model')}" + (f", effort {r['effort']}" if r.get("effort") else "")
                     for r in rows if r.get("model")})
    cost = sum(r["cost"] or 0 for r in rows)
    n_reads = [len(r.get("reads") or []) for r in rows]
    pct = lambda n, d: f"{n}/{d} ({n / d:.0%})" if d else "-"
    setup = [
        ("Video", Path(truth["video"]).name),
        ("Detections", Path(truth["dets"]).name),
        ("Model", ", ".join(models) or "-"),
        ("Cases", f"{len(rows)} answered ({len(truth['cases'])} built): {args.get('cases', '?')} frame sets"
                  f"{' x labelled/unlabelled' if args.get('labels') == 'both' else ''}"),
        ("Frames per case", f"{args.get('frames', '?')} (at least {args.get('min_gap', '?')} video frames apart)"),
        ("Hidden per case", f"{args.get('hide', '?')} classes, mode '{args.get('mode', '?')}'"
                            f"{', all in one frame' if args.get('same_frame') else ', each in a random frame'}; "
                            f"{args.get('controls', 0):.0%} controls with nothing hidden"),
        ("Seed", str(args.get("seed", "?"))),
        ("Match rule", f"right frame AND box IoU >= {iou_thr} with the removed box (class name only if no box given)"),
        ("Cost", f"${cost:.2f} (Claude Code's estimate at API list prices)" if cost else "-"),
    ]
    isolation = None
    if any(r.get("backend") == "cli" for r in rows):
        n_t = sum(r["tainted"] for r in rows)
        reads = f"{min(n_reads)}" if min(n_reads) == max(n_reads) else f"{min(n_reads)}-{max(n_reads)}"
        isolation = ("Each case ran in a fresh temp folder holding only its frame images; the only tool was Read, "
                     "allowed only inside that folder (every other read refused), with user settings and MCP servers "
                     "not loaded. Every tool call was logged: "
                     + (f"all {len(rows)} runs read only their own frames ({reads} reads each)." if not n_t
                        else f"{n_t} run(s) read other files and are excluded."))
    results = [[st["label"], str(st["cases"]), pct(st["found"], st["hidden"]), pct(st["exact"], st["cases"]),
                str(st["fp"]), str(st["fp_natural"]), "-" if st["read_f1"] is None else f"{st['read_f1']:.2f}"]
               for st in stats]
    classes = [[k, pct(per_cls_hit[k], v)] for k, v in per_cls.most_common()]
    errors = ([("missed", m, r["id"]) for r in ok for m in r["missed"]]
              + [("false alarm", x, r["id"]) for r in ok for x in r["false_alarms"]])
    cases = []
    for r in rows:
        c = by_id[r["id"]]
        if r["tainted"]:
            verdict, good = "TAINTED (read files outside its frames), excluded", None
        elif not r["parse_ok"]:
            verdict, good = "unparseable answer", False
        else:
            verdict = (f"{'CORRECT' if r['exact'] else 'WRONG'} - found {r['found']}/{r['n_hidden']}, "
                       f"false alarms {r['fp']}" + (f" (+{r['fp_natural']} maybe-real)" if r["fp_natural"] else ""))
            good = r["exact"]
        raw_items = [x for x in (r.get("parsed") or {}).get("missing", []) or [] if isinstance(x, dict)]
        cases.append({
            "id": r["id"], "verdict": verdict, "good": good, "parse_ok": r["parse_ok"],
            "frames": ", ".join(map(str, c["frames"])), "labels": c["labels"],
            "hidden": ", ".join(f"{h['cls']} in frame {h['pos']}" for h in c["hidden"]) or "nothing (control)",
            "image": f"viz/{r['id']}.jpg",
            "items": [[str(it["frame"]), str(it["object"]),
                       str([round(v) for v in it["bbox_px"]]) if it["bbox_px"] else str(raw.get("bbox_2d")),
                       STATUS_TXT[it["status"]]]
                      for it, raw in zip(r.get("items") or [], raw_items)]})
    return {"title": f"Missing-box probe: {tag}", "setup": setup, "isolation": isolation, "results": results,
            "classes": classes, "errors": errors, "cases": cases}


RESULT_HEAD = ["Subset", "Cases", "Hidden objects found", "Cases fully correct", "False alarms",
               "Maybe-real misses", "Label reading F1"]
ITEM_HEAD = ["Frame", "Model reported", "Box (pixels)", "Verdict"]


def write_report_md(adir: Path, rc: dict):
    """report.md - images linked from viz/."""
    md = lambda x: str(x).replace("|", "/")
    L = [f"# {rc['title']}", "", REPORT_INTRO, "", "## Setup", "", "| | |", "|---|---|"]
    L += [f"| {k} | {md(v)} |" for k, v in rc["setup"]]
    L.append("")
    if rc["isolation"]:
        L += [f"**Isolation.** {rc['isolation']}", ""]
    L += ["## Results", "", "| " + " | ".join(RESULT_HEAD) + " |", "|---|" + "---:|" * (len(RESULT_HEAD) - 1)]
    L += ["| " + " | ".join(row) + " |" for row in rc["results"]]
    L += ["", f"*{REPORT_NOTE}*", ""]
    if rc["classes"]:
        L += ["### By hidden class", "", "| Class | Found |", "|---|---:|"]
        L += [f"| `{k}` | {v} |" for k, v in rc["classes"]]
        L.append("")
    if rc["errors"]:
        L += ["### Errors", ""] + [f"- **{kind}** {md(x)} in [{cid}](#{cid})" for kind, x, cid in rc["errors"]] + [""]
    L += ["## Cases", "", REPORT_LEGEND, ""]
    for c in rc["cases"]:
        L += [f"### {c['id']}", "", f"**{c['verdict']}**  ",
              f"Frames (video): {c['frames']} · labels {'on' if c['labels'] else 'off'} · hidden: {c['hidden']}", "",
              f"![{c['id']}]({c['image']})", ""]
        if c["items"]:
            L += ["| " + " | ".join(ITEM_HEAD) + " |", "|---:|---|---|---|"]
            L += ["| " + " | ".join(md(x) for x in row) + " |" for row in c["items"]] + [""]
        elif c["parse_ok"]:
            L += ["Model reported nothing missing.", ""]
    (adir / "report.md").write_text("\n".join(L))


def write_report_pdf(adir: Path, rc: dict):
    """report.pdf - landscape A4, summary pages then one page per case. Needs reportlab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                        Table, TableStyle)
    except ImportError:
        print("  (no reportlab - skipping report.pdf; `uv pip install reportlab` in this venv)")
        return
    from xml.sax.saxutils import escape

    page = landscape(A4)
    doc = SimpleDocTemplate(str(adir / "report.pdf"), pagesize=page, leftMargin=1.3 * cm, rightMargin=1.3 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm, title=rc["title"])
    W = page[0] - 2.6 * cm
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5)
    small = ParagraphStyle("s", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))
    cell = ParagraphStyle("c", parent=body, fontSize=8.5, leading=10.5)
    P = lambda t, st=body: Paragraph(escape(str(t)), st)
    GREEN, RED, AMBER = colors.HexColor("#1a7f37"), colors.HexColor("#cf222e"), colors.HexColor("#9a6700")

    def table(rows, widths, head=True, right_from=None):
        t = Table([[P(x, cell) for x in r] for r in rows], colWidths=widths, repeatRows=1 if head else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if head:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")))
        t.setStyle(TableStyle(st))
        return t

    s = [Paragraph(escape(rc["title"]), ss["Title"]), P(REPORT_INTRO), Spacer(1, 8),
         Paragraph("Setup", ss["Heading2"]), table(rc["setup"], [4.5 * cm, W - 4.5 * cm], head=False)]
    if rc["isolation"]:
        s += [Spacer(1, 6), Paragraph("<b>Isolation.</b> " + escape(rc["isolation"]), body)]
    s += [Paragraph("Results", ss["Heading2"]),
          table([RESULT_HEAD] + rc["results"], [4.6 * cm] + [(W - 4.6 * cm) / 6] * 6),
          Spacer(1, 4), P(REPORT_NOTE, small), PageBreak()]
    left = []
    if rc["classes"]:
        left = [Paragraph("By hidden class", ss["Heading2"]), table([["Class", "Found"]] + rc["classes"], [6 * cm, 3.5 * cm])]
    right = []
    if rc["errors"]:
        right = [Paragraph("Errors", ss["Heading2"])] + [
            Paragraph(f"<b>{kind}</b> {escape(x)} <font color='#666666'>in {cid}</font>", body)
            for kind, x, cid in rc["errors"]]
    if left or right:
        s += [Table([[left, right]], colWidths=[10.5 * cm, W - 10.5 * cm],
                    style=[("VALIGN", (0, 0), (-1, -1), "TOP")])]
    s += [Spacer(1, 10), Paragraph("Cases", ss["Heading2"]), P(REPORT_LEGEND, small), PageBreak()]

    # one page per case; images re-encoded smaller so the PDF stays a few MB
    tmp = Path(tempfile.mkdtemp(prefix="mbp_pdf_"))
    try:
        for c in rc["cases"]:
            src = adir / c["image"]
            head = [Paragraph(escape(c["id"]), ss["Heading2"]),
                    Paragraph(f"<b>{escape(c['verdict'])}</b>", ParagraphStyle(
                        "v", parent=body, textColor=AMBER if c["good"] is None else GREEN if c["good"] else RED)),
                    P(f"Frames (video): {c['frames']} · labels {'on' if c['labels'] else 'off'} · hidden: {c['hidden']}",
                      small), Spacer(1, 4)]
            if src.exists():
                img = cv2.imread(str(src))
                h, w = img.shape[:2]
                scale = min(1.0, 1800 / w)
                small_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                dst = tmp / f"{c['id']}.jpg"
                cv2.imwrite(str(dst), small_img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                max_h = 11.2 * cm
                iw = min(W, max_h * w / h)
                head.append(Image(str(dst), width=iw, height=iw * h / w))
            s.append(KeepTogether(head))
            if c["items"]:
                s += [Spacer(1, 4), table([ITEM_HEAD] + c["items"], [1.4 * cm, 10 * cm, 5.5 * cm, W - 16.9 * cm])]
            elif c["parse_ok"]:
                s += [Spacer(1, 4), P("Model reported nothing missing.")]
            s.append(PageBreak())

        def footer(canv, d):
            canv.setFont("Helvetica", 7.5)
            canv.setFillColor(colors.HexColor("#888888"))
            canv.drawRightString(page[0] - 1.3 * cm, 0.6 * cm, f"{rc['title']} - page {d.page}")

        doc.build(s, onFirstPage=footer, onLaterPages=footer)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_score(a):
    out = Path(a.out)
    truth = json.load(open(out / "truth.json"))
    tags = a.tag or sorted(p.name for p in (out / "answers").iterdir() if p.is_dir())
    raw = read_frames(truth["video"], {f for c in truth["cases"] for f in c["frames"]}) if a.viz else {}
    for tag in tags:
        adir = out / "answers" / tag
        rows = []
        for c in truth["cases"]:
            f = adir / f"{c['id']}.json"
            if not f.exists():
                continue
            ans = json.load(open(f))
            # answers from before the `coords` field were all asked for 0-1000
            s = score_case(c, ans.get("parsed"), a.iou, ans.get("coords", "norm1000"))
            s.update(id=c["id"], labels=c["labels"], cost=ans.get("cost_usd"), seconds=ans.get("seconds"),
                     parsed=ans.get("parsed"), tainted=bool((ans.get("audit") or {}).get("tainted")),
                     backend=ans.get("backend"), effort=ans.get("effort"),
                     model=ans.get("resolved_model") or ans.get("model"),
                     reads=[(c_.get("input") or {}).get("file_path") for c_ in ans.get("tool_calls") or []])
            rows.append(s)
            if a.viz:
                (adir / "viz").mkdir(exist_ok=True)
                viz_case(raw, c, s, adir / "viz" / f"{c['id']}.jpg", tag)
        if not rows:
            print(f"[{tag}] no answers")
            continue
        with open(adir / "scores.csv", "w") as fh:
            fh.write("case,labels,n_hidden,found,fp,fp_natural,exact,read_f1,tainted,missed,false_alarms\n")
            for r in rows:
                if not r["parse_ok"]:
                    fh.write(f"{r['id']},{r['labels']},,,,,PARSE_FAIL,,{r['tainted']},,\n")
                    continue
                fh.write(f"{r['id']},{r['labels']},{r['n_hidden']},{r['found']},{r['fp']},{r['fp_natural']},"
                         f"{r['exact']},{'' if r['read_f1'] is None else round(r['read_f1'], 3)},{r['tainted']},"
                         f"\"{' '.join(r['missed'])}\",\"{' '.join(r['false_alarms'])}\"\n")

        ok = [r for r in rows if r["parse_ok"] and not r["tainted"]]
        lines, stats = [], []

        def agg(sel, label):
            xs = [r for r in ok if sel(r)]
            if not xs:
                return
            hid = sum(r["n_hidden"] for r in xs)
            fnd = sum(r["found"] for r in xs)
            fp = sum(r["fp"] for r in xs)
            rf = [r["read_f1"] for r in xs if r["read_f1"] is not None]
            stats.append({"label": label, "cases": len(xs), "hidden": hid, "found": fnd,
                          "exact": sum(r["exact"] for r in xs), "fp": fp,
                          "fp_natural": sum(r["fp_natural"] for r in xs), "read_f1": np.mean(rf) if rf else None})
            lines.append(f"  {label:22s} cases {len(xs):3d} | recall {fnd:3d}/{hid:<3d}"
                         f"{'' if not hid else f' ({fnd / hid:4.0%})'} | all-correct {sum(r['exact'] for r in xs):3d}/{len(xs):<3d}"
                         f" | false alarms {fp:3d} (+{sum(r['fp_natural'] for r in xs)} maybe-real)"
                         + (f" | label reading F1 {np.mean(rf):.2f}" if rf else ""))

        n_taint = sum(r["tainted"] for r in rows)
        lines.append(f"[{tag}]  {len(rows)} answered, {sum(not r['parse_ok'] for r in rows)} unparseable, "
                     f"{n_taint} tainted (excluded)"
                     + (f", ${sum(r['cost'] or 0 for r in rows):.2f}" if any(r["cost"] for r in rows) else ""))
        agg(lambda r: True, "all")
        agg(lambda r: r["n_hidden"] == 0, "controls (0 hidden)")
        for k in sorted({r["n_hidden"] for r in ok if r["n_hidden"]}):
            agg(lambda r, k=k: r["n_hidden"] == k, f"{k} hidden")
        agg(lambda r: r["labels"], "labelled")
        agg(lambda r: not r["labels"], "unlabelled")
        per_cls, per_cls_hit = collections.Counter(), collections.Counter()
        for r in ok:
            for t in r["targets"]:
                per_cls[t["cls"]] += 1
                per_cls_hit[t["cls"]] += t["found"]
        if per_cls:
            lines.append("  by hidden class:  " + ", ".join(f"{k} {per_cls_hit[k]}/{v}" for k, v in per_cls.most_common()))
        print("\n" + "\n".join(lines))
        print(f"  per-case: {adir / 'scores.csv'}")
        if a.viz:
            write_report(adir, tag, lines, rows, truth)
            rc = report_content(tag, stats, per_cls, per_cls_hit, rows, truth, a.iou)
            write_report_md(adir, rc)
            write_report_pdf(adir, rc)
            print(f"  reports:  {adir}/report.html, report.md, report.pdf  (images in {adir / 'viz'}/)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="build test cases")
    m.add_argument("--video", default=str(DEFAULT_VIDEO))
    m.add_argument("--dets", help="dets.json (default: <video stem>.mp4_dets.json next to the video)")
    m.add_argument("--out", required=True)
    m.add_argument("--cases", type=int, default=20, help="frame sets to build (x2 with --labels both)")
    m.add_argument("--frames", type=int, default=5, help="frames per case")
    m.add_argument("--hide", default="1-3", help="classes hidden per case, N or LO-HI")
    m.add_argument("--controls", type=float, default=0.2, help="fraction of cases with nothing hidden")
    m.add_argument("--mode", choices=["class", "instance"], default="class",
                   help="class: remove every box of the class in that frame; instance: remove one box")
    m.add_argument("--same-frame", action="store_true", help="hide all chosen classes in the same frame")
    m.add_argument("--labels", choices=["both", "on", "off"], default="both", help="draw class names on boxes")
    m.add_argument("--min-gap", type=int, default=60, help="min frames between the case's frames")
    m.add_argument("--window", type=int, default=0, help="all frames of a case within this many frames (0 = anywhere)")
    m.add_argument("--min-conf", type=float, default=0.0, help="drop detections below this confidence")
    m.add_argument("--classes", nargs="*", help="only hide these classes")
    m.add_argument("--exclude", nargs="*", help="never hide these classes")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--force", action="store_true", help="overwrite an existing --out")

    r = sub.add_parser("run", help="ask a model about every case")
    r.add_argument("--out", required=True)
    r.add_argument("--backend", choices=["cli", "api", "qwen"], default="cli")
    r.add_argument("--model", default=None,
                   help="cli: opus|sonnet|<id> (default opus); api: default claude-opus-5; qwen: HF id")
    r.add_argument("--tag", help="answers folder name (default <backend>-<model>)")
    r.add_argument("--effort", help="low|medium|high|xhigh|max (cli/api)")
    r.add_argument("--workers", type=int, default=4, help="parallel requests (cli/api)")
    r.add_argument("--timeout", type=int, default=600, help="seconds per case (cli)")
    r.add_argument("--class-list", action="store_true", help="tell the model the detector's class names")
    r.add_argument("--only", nargs="*", help="case id substrings to run (e.g. _lab c003)")
    r.add_argument("--limit", type=int, default=0, help="ask at most this many cases")
    r.add_argument("--redo", action="store_true", help="re-ask cases that already have an answer")
    r.add_argument("--no-fallback", dest="fallback", action="store_false", help="api: no refusal fallback")
    r.add_argument("--tp", type=int, default=1, help="qwen: tensor parallel GPUs")
    r.add_argument("--max-len", type=int, default=8192, help="qwen: max model length")

    i = sub.add_parser("check-isolation", help="prove the cli agent cannot read truth.json / dets.json")
    i.add_argument("--out", required=True)
    i.add_argument("--model", default="opus")
    i.add_argument("--effort", default="low")
    i.add_argument("--timeout", type=int, default=300)

    s = sub.add_parser("score", help="score answers")
    s.add_argument("--out", required=True)
    s.add_argument("--tag", nargs="*", help="answer tags (default: all)")
    s.add_argument("--iou", type=float, default=0.3)
    s.add_argument("--viz", action="store_true", help="draw truth vs answer per case")

    a = ap.parse_args()
    if a.cmd == "run" and a.model is None:
        a.model = {"cli": "opus", "api": "claude-opus-5", "qwen": "Qwen/Qwen3-VL-8B-Instruct"}[a.backend]
    {"make": cmd_make, "run": cmd_run, "score": cmd_score, "check-isolation": cmd_check_isolation}[a.cmd](a)


if __name__ == "__main__":
    main()
