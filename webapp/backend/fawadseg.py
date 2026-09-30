"""Frame extraction fawad segment: the research's `class_N_desc_vlm_hints` pipeline, run as is.

qwen_vl/extraction_project made outputs/<station>/<target>/class_N_desc_vlm_hints with
four scripts. This runs byte-identical copies of them (fawadseg_scripts/, via its
launch.py) with the arguments the research used, so the model logic, the inputs and
the output files are the research's own:

  inputs  reference video + its preds.json (per-frame step labels), detector.json
          (step names), the target step, its confusable steps, a description of the
          step (embedding instruction), per-step visual hints (VLM options), the
          target video, and optionally the target's preds.json (metrics only;
          selection never reads it)

  1. extract_frames.py   every decodable frame of both videos, as JPEG + meta.json
  2. embed_frames.py --stride 5 [--describe]          Qwen3-VL-Embedding-8B
  3. match_frames.py --balanced                        -> class_N[_desc]_balanced/
  4. vlm_classify.py target --confusers --hints --candidates
                                                       -> class_N[_desc]_vlm[_hints]/

Each run gets the research's folder layout (work/frames/<video stem>,
work/embeddings/<stem>__classNdesc.npz), because match_frames finds the reference's
meta.json and names its outputs from those paths. The frames and embeddings are
symlinks into a cache keyed by file content, so a video already seen is not decoded
or embedded again.

Without target labels a placeholder preds.json (no labels anywhere) stands in, the
GT-only files (metrics.json, the candidates' report.md) are dropped, and the
selected-frames video is drawn without its GT / TP / FP text (launch.py).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

from . import config
from .extraction import Cancelled, _qwen_gpu_mem, _reap, digest
from .segx import _qwen_gpu

log = logging.getLogger("sam3webapp.fawadseg")

ProgressCb = Callable[[float, str], None]

SCRIPTS = Path(__file__).resolve().parent / "fawadseg_scripts"
LAUNCH = SCRIPTS / "launch.py"
FRAMES = config.FAWADSEG_WORK / "frames"
EMBEDS = config.FAWADSEG_WORK / "embeddings"
SOURCES = config.FAWADSEG_WORK / "src"

# the research's fixed settings (script defaults, plus what its pipeline passed)
STRIDE = 5
# gpu_memory_utilization the scripts ask vLLM for, and what an override leaves spare
# and needs at least (GB) — the same margins as extraction._KINDS
_GPU = {"embed": (0.90, 2.5, config.QWEN_MIN_GPU_GB), "vlm": (0.92, 1.2, 21.0)}


def dir_names(cls: int, described: bool, hinted: bool) -> tuple[str, str]:
    """The research's output folder names for this variant of the pipeline."""
    d = "_desc" if described else ""
    return f"class_{cls}{d}_balanced", f"class_{cls}{d}_vlm{'_hints' if hinted else ''}"


def _instruction_key(description: str) -> str:
    """What decides embed_frames.py's instruction: --describe's text, whitespace-normalised."""
    desc = " ".join(description.split())
    return f"describe:{desc}" if desc else "default"


def _json(text: str, what: str) -> dict:
    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} is not valid JSON: {exc}") from None
    if not isinstance(d, dict):
        raise ValueError(f"{what} should be a JSON object")
    return d


def validate(body: dict) -> dict:
    """Parse and check the JSON inputs before anything runs. Returns the step names."""
    det = _json(body["detector"], "detector.json")
    steps = det.get("cycle_steps")
    if not isinstance(steps, dict) or not steps:
        raise ValueError("detector.json has no cycle_steps")
    id2step = {int(v): k for k, v in steps.items()}
    cls = int(body["cls"])
    conf = sorted(set(int(c) for c in body["confusers"]) - {cls})
    if not conf:
        raise ValueError("pick at least one confusable step (the research used 3 and 7)")
    for c in [cls, *conf]:
        if c not in id2step:
            raise ValueError(f"step {c} is not in detector.json's cycle_steps")
    for text, what in [(body["ref_preds"], "the reference preds.json")] + (
            [(body["tgt_preds"], "the target preds.json")] if body.get("tgt_preds") else []):
        if not isinstance(_json(text, what).get("preds"), dict):
            raise ValueError(f"{what} has no \"preds\" object")
    return id2step


# --------------------------------------------------------------------------- #
# running a script
# --------------------------------------------------------------------------- #
class _Script:
    """One launch.py run: its own process group (vLLM forks an EngineCore that holds
    the GPU), all output to the run's log, and its lines — tqdm's \\r updates too —
    handed to `on_line` for progress."""

    def __init__(self, name: str, args: list[str], log_path: Path, env_extra: dict,
                 cancel: threading.Event, on_line: Callable[[str], None],
                 poll: Optional[Callable[[], None]] = None) -> None:
        self.name, self.args, self.log_path = name, args, log_path
        self.env_extra, self.cancel, self.on_line, self.poll = env_extra, cancel, on_line, poll

    def run(self) -> None:
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "CUDA_VISIBLE_DEVICES": config.QWEN_GPU,
               "VLLM_USE_FLASHINFER_SAMPLER": "0",  # vlm_classify's own default
               "QWEN_EMB_MODEL": config.QWEN_EMB_MODEL, "QWEN_VL_MODEL": config.QWEN_VLM_MODEL,
               "PYTHONUNBUFFERED": "1", **self.env_extra}
        cmd = [str(config.QWEN_PYTHON), str(LAUNCH), self.name, *self.args]
        with self.log_path.open("a") as lf:
            lf.write(f"\n$ {' '.join(_q(a) for a in cmd[2:])}\n")
            lf.write("".join(f"  {k}={v}\n" for k, v in self.env_extra.items()))
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=str(SCRIPTS))
        lines: "queue.Queue[Optional[str]]" = queue.Queue()

        def pump() -> None:
            buf = b""
            with self.log_path.open("ab") as lf:
                while chunk := proc.stdout.read1(65536):  # type: ignore[union-attr]
                    lf.write(chunk)
                    lf.flush()
                    buf += chunk
                    *done, buf = re.split(rb"[\r\n]", buf)
                    for d in done:
                        if d.strip():
                            lines.put(d.decode("utf-8", "replace"))
            if buf.strip():
                lines.put(buf.decode("utf-8", "replace"))
            lines.put(None)

        threading.Thread(target=pump, daemon=True, name=f"fawadseg-{self.name}").start()
        tail: list[str] = []
        try:
            while True:
                if self.cancel.is_set():
                    raise Cancelled("cancelled")
                try:
                    line = lines.get(timeout=0.5)
                except queue.Empty:
                    if self.poll:
                        self.poll()
                    continue
                if line is None:
                    break
                tail = (tail + [line])[-30:]
                self.on_line(line)
            rc = proc.wait()
        finally:
            # also after a clean exit: vLLM's EngineCore child can outlive its parent
            # for a moment, holding the GPU memory the next stage needs
            _reap(proc)
        if rc != 0:
            err = next((l for l in reversed(tail) if re.match(r"\w*(Error|Exception)\b", l.strip())),
                       tail[-1] if tail else f"exit code {rc}")
            raise RuntimeError(f"{self.name}.py failed: {err.strip()[:400]}")


def _q(a: str) -> str:
    return f'"{a}"' if (" " in a or not a) else a


def _gpu_env(kind: str, progress_cb: ProgressCb, frac: float) -> dict:
    """The script's own gpu_memory_utilization when that much is free (always, on an
    idle dedicated card); otherwise the most that is, if it is enough."""
    want, spare, least = _GPU[kind]
    gib = 1024 ** 3
    deadline = time.time() + 15  # memory of a process that just exited takes a moment
    while True:
        free, total = _qwen_gpu_mem()
        if free >= want * total + 0.3 * gib:
            return {}
        if time.time() > deadline:
            break
        time.sleep(1)
    budget = free - spare * gib
    if budget < least * gib:
        raise RuntimeError(
            f"only {free / gib:.1f} GB is free on GPU {config.QWEN_GPU}; the Qwen "
            f"{'VLM' if kind == 'vlm' else 'embedding model'} needs about {least:.0f} GB. "
            "Something else is using that GPU.")
    util = round(min(want, budget / total), 3)
    log.info("fawadseg %s: %.1f GB free, gpu_memory_utilization %s instead of %s",
             kind, free / gib, util, want)
    progress_cb(frac, f"GPU {config.QWEN_GPU} is partly in use: vLLM gets {util:.0%} of it, not {want:.0%}")
    return {"FAWADSEG_GPU_UTIL": str(util)}


# --------------------------------------------------------------------------- #
# caches: frames and embeddings by file content
# --------------------------------------------------------------------------- #
def _frames_cache(up) -> Path:
    return FRAMES / digest(up.path)


def _emb_cache(up, description: str) -> Path:
    key = hashlib.sha1(f"{config.QWEN_EMB_MODEL}|{STRIDE}|{_instruction_key(description)}"
                       .encode()).hexdigest()[:10]
    return EMBEDS / f"{digest(up.path)}__{key}.npz"


def _link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def _stem(name: str, taken: str = "") -> str:
    """A video's name as the research's folder names use it. `__` would break
    match_frames' split of the embedding name, and the two videos need different names."""
    s = (Path(name).stem or "video").replace("__", "_").replace("/", "_").strip() or "video"
    return f"{s} (target)" if s == taken else s


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
def run(engine, workers: list, ref, target, body: dict, job_id: str,
        progress_cb: ProgressCb, cancel: threading.Event) -> dict:
    id2step = validate(body)
    cls = int(body["cls"])
    confusers = sorted(set(int(c) for c in body["confusers"]) - {cls})
    description = body.get("description") or ""
    hints = {k: v.strip() for k, v in (body.get("hints") or {}).items() if v and v.strip()}
    has_gt = bool((body.get("tgt_preds") or "").strip())
    cand_name, final_name = dir_names(cls, bool(description.strip()), bool(hints))

    run_dir = config.FAWADSEG_RUNS / job_id
    inputs, work = run_dir / "inputs", run_dir / "work"
    inputs.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"
    ref_stem = _stem(ref.name or ref.path.name)
    tgt_stem = _stem(target.name or target.path.name, taken=ref_stem)
    ref_preds = inputs / f"{ref_stem}.mp4_preds.json"
    tgt_preds = inputs / (f"{tgt_stem}.mp4_preds.json" if has_gt else "no_target_labels_preds.json")
    ref_preds.write_text(body["ref_preds"])
    detector = inputs / "detector.json"
    detector.write_text(body["detector"])
    if has_gt:
        tgt_preds.write_text(body["tgt_preds"])
    desc_file = inputs / f"class{cls}_description.txt"
    if description.strip():
        desc_file.write_text(description)
    hints_file = inputs / "step_hints.json"
    if hints:
        hints_file.write_text(json.dumps(hints, indent=2))

    # one Qwen model on the card at a time: the other tabs' workers go first
    progress_cb(0.005, "freeing the Qwen GPU")
    for w in workers:
        w.stop()

    frames = {"ref": work / "frames" / ref_stem, "tgt": work / "frames" / tgt_stem}
    embs_dir = work / "embeddings"
    suffix = f"__class{cls}desc" if description.strip() else ""
    embs = {"ref": embs_dir / f"{ref_stem}{suffix}.npz", "tgt": embs_dir / f"{tgt_stem}{suffix}.npz"}
    ups = {"ref": ref, "tgt": target}
    cmds: list[str] = []

    def script(name: str, args: list[str], done_at: float, env_extra: Optional[dict] = None,
               on_line: Optional[Callable[[str], None]] = None, poll=None) -> None:
        cmds.append(" ".join(_q(a) for a in [f"scripts/{name}.py", *args]))
        _Script(name, args, log_path, env_extra or {}, cancel,
                on_line or (lambda _l: None), poll).run()
        progress_cb(done_at, f"{name} done")

    t0 = time.time()
    with _qwen_gpu(engine, []):
        # -- 1. frames (CPU) ------------------------------------------------ #
        for i, k in enumerate(("ref", "tgt")):
            cache = _frames_cache(ups[k])
            if not (cache / "meta.json").exists():
                # the video under its own name, which meta.json and report.md show
                src = SOURCES / digest(ups[k].path) / (Path(ups[k].name or "").name or ups[k].path.name)
                _link(src, ups[k].path.resolve())
                lo, hi = 0.01 + 0.07 * i, 0.08 + 0.07 * i
                want = max(1, int(ups[k].frames or 1))
                label = "reference" if k == "ref" else "target"

                def poll(cache=cache, lo=lo, hi=hi, want=want, label=label) -> None:
                    n = sum(1 for _ in cache.glob("*.jpg")) if cache.is_dir() else 0
                    progress_cb(lo + (hi - lo) * min(1.0, n / want),
                                f"decoding the {label} video: frame {n}")

                progress_cb(lo, f"decoding the {label} video")
                script("extract_frames", [str(src), str(cache)], hi, poll=poll)
            os.utime(cache / "meta.json")  # the sweeper goes by this
            _link(frames[k], cache)
        metas = {k: json.loads((frames[k] / "meta.json").read_text()) for k in frames}
        _check_labels(ref_preds, metas["ref"]["decoded_frames"], "reference", STRIDE)
        if has_gt:
            _check_labels(tgt_preds, metas["tgt"]["decoded_frames"], "target", 1)
        else:
            tgt_preds.write_text(json.dumps(
                {"_note": "placeholder: the target has no labels", "preds": {
                    str(f): {"pred": []} for f in range(metas["tgt"]["decoded_frames"])}}))
        _check_steps(ref_preds, metas["ref"]["decoded_frames"], [cls, *confusers], id2step)
        if cancel.is_set():
            raise Cancelled("cancelled")

        # -- 2. embeddings (GPU) ------------------------------------------- #
        todo = [k for k in ("ref", "tgt") if not _emb_cache(ups[k], description).exists()]
        if todo:
            want = {frames[k].name: math.ceil(metas[k]["decoded_frames"] / STRIDE) for k in todo}
            done = {n: 0 for n in want}

            def on_embed(line: str) -> None:
                m = re.match(r"^(.*): (\d+)/(\d+) \(([\d.]+) img/s\)", line)
                if m and m.group(1) in done:
                    done[m.group(1)] = int(m.group(2))
                    f = sum(done.values()) / max(1, sum(want.values()))
                    progress_cb(0.16 + 0.34 * f, f"embedding frames {sum(done.values())}/"
                                f"{sum(want.values())} ({m.group(4)} img/s)")
                elif "Loading safetensors" in line or "non-default args" in line:
                    progress_cb(0.16, f"loading Qwen3-VL-Embedding on GPU {config.QWEN_GPU}")

            args = ["--stride", str(STRIDE)]
            if description.strip():
                args += ["--describe", str(desc_file), "--suffix", suffix]
            args += [str(frames[k]) for k in todo]
            progress_cb(0.16, f"loading Qwen3-VL-Embedding on GPU {config.QWEN_GPU}")
            script("embed_frames", args, 0.5, _gpu_env("embed", progress_cb, 0.16), on_embed)
            EMBEDS.mkdir(parents=True, exist_ok=True)
            for k in todo:  # written into the run's work dir: move into the cache
                embs[k].replace(_emb_cache(ups[k], description))
        for k in ("ref", "tgt"):
            os.utime(_emb_cache(ups[k], description))
            _link(embs[k], _emb_cache(ups[k], description))
        if cancel.is_set():
            raise Cancelled("cancelled")

        # -- 3. candidates: class-balanced kNN (CPU) ------------------------ #
        no_gt = {} if has_gt else {"FAWADSEG_NO_GT": "1"}
        cand_dir, final_dir = run_dir / cand_name, run_dir / final_name
        progress_cb(0.5, "kNN candidates (match_frames --balanced)")
        script("match_frames", [
            "--ref-emb", str(embs["ref"]), "--ref-preds", str(ref_preds),
            "--tgt-emb", str(embs["tgt"]), "--tgt-frames", str(frames["tgt"]),
            "--tgt-preds", str(tgt_preds), "--detector", str(detector),
            "--cls", str(cls), "--balanced", "--out", str(cand_dir)], 0.55, no_gt)

        # -- 4. VLM classification of the candidates (GPU) ------------------ #
        n_q = [0]

        def on_vlm(line: str) -> None:
            m = re.match(r"^(\d+) query clips", line)
            if m:
                n_q[0] = int(m.group(1))
                progress_cb(0.56, f"{n_q[0]} candidate clips; loading Qwen3-VL-8B on GPU {config.QWEN_GPU}")
            m = re.search(r"Processed prompts:\s+(\d+)%.*?(\d+)/(\d+)", line)
            if m:
                progress_cb(0.6 + 0.38 * int(m.group(1)) / 100,
                            f"VLM classifying clips {m.group(2)}/{m.group(3)}")

        args = ["target", "--ref-emb", str(embs["ref"]), "--ref-frames", str(frames["ref"]),
                "--ref-preds", str(ref_preds), "--tgt-frames", str(frames["tgt"]),
                "--tgt-preds", str(tgt_preds), "--detector", str(detector),
                "--cls", str(cls), "--confusers", ",".join(map(str, confusers))]
        if hints:
            args += ["--hints", str(hints_file)]
        args += ["--candidates", str(cand_dir / "selected_frames.json"), "--out", str(final_dir)]
        final_dir.mkdir(parents=True, exist_ok=True)
        progress_cb(0.56, f"loading Qwen3-VL-8B on GPU {config.QWEN_GPU}")
        script("vlm_classify", args, 0.98, {**_gpu_env("vlm", progress_cb, 0.56), **no_gt}, on_vlm)

    # -- outputs ------------------------------------------------------------ #
    progress_cb(0.985, "packing the outputs")
    if not has_gt:  # GT-only files, meaningless against the placeholder
        for p in (cand_dir / "metrics.json", cand_dir / "report.md", final_dir / "metrics.json"):
            p.unlink(missing_ok=True)
    (run_dir / "commands.txt").write_text(
        "# run from backend/fawadseg_scripts (byte-identical copies of the research's scripts/)\n"
        + "\n".join(cmds) + "\n")
    zip_path = run_dir / f"{tgt_stem}_{final_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in (inputs, cand_dir, final_dir):
            for p in sorted(d.iterdir()):
                zf.write(p, f"{d.name}/{p.name}")
        for p in (log_path, run_dir / "commands.txt"):
            zf.write(p, p.name)

    final_sel = json.loads((final_dir / "selected_frames.json").read_text())
    cand_sel = json.loads((cand_dir / "selected_frames.json").read_text())
    tm = metas["tgt"]
    result = {
        "run_id": job_id, "has_gt": has_gt,
        "class_id": cls, "class_name": id2step[cls], "confusers": confusers,
        "folders": {"candidates": cand_name, "final": final_name},
        "target": {"name": target.name, "stem": tgt_stem, "decoded_frames": tm["decoded_frames"],
                   "header_frames": tm["header_frames"], "fps": tm["fps"]},
        "reference": {"name": ref.name, "stem": ref_stem, "decoded_frames": metas["ref"]["decoded_frames"],
                      "header_frames": metas["ref"]["header_frames"]},
        "stride": STRIDE,
        "segments": final_sel.get("segments", []),
        "candidate_segments": cand_sel.get("segments", []),
        "options": final_sel.get("options", {}),
        "examples": final_sel.get("examples", {}),
        "scores": _scores(final_dir / "vlm_scores.csv"),
        "metrics": _read(final_dir / "metrics.json"),
        "candidate_metrics": _read(cand_dir / "metrics.json"),
        "files": {"candidates": _files(job_id, cand_dir), "final": _files(job_id, final_dir),
                  "inputs": _files(job_id, inputs),
                  "run": _files(job_id, run_dir, [log_path.name, "commands.txt"])},
        "zip_url": _url(job_id, zip_path.name),
        "commands": cmds,
        "seconds": round(time.time() - t0, 1),
    }
    (run_dir / "result.json").write_text(json.dumps(result))
    return result


def _check_labels(preds_path: Path, decoded: int, which: str, stride: int) -> None:
    """match_frames reads a label for every embedded reference frame (every target
    frame), and stops with a bare KeyError on the first missing one."""
    preds = json.loads(preds_path.read_text())["preds"]
    missing = [f for f in range(0, decoded, stride) if str(f) not in preds]
    if missing:
        raise ValueError(f"the {which} preds.json has no label for {len(missing)} of the frames "
                         f"it needs (first: frame {missing[0]}); is it the preds.json of this video?")


def _check_steps(preds_path: Path, decoded: int, steps: list[int], id2step: dict) -> None:
    """vlm_classify.pick_examples takes 2 example clips of every option from the
    reference, centred on embedded frames whose whole ~2 s clip carries only that
    label; with fewer than 2 such frames it stops with a bare numpy error."""
    preds = json.loads(preds_path.read_text())["preds"]
    lab = {int(k): tuple(v["pred"]) for k, v in preds.items() if int(k) < decoded}
    half = (8 - 1) * 5 // 2  # vlm_classify's clip_half at its default --clip-frames/--clip-step
    short = [c for c in steps
             if sum(all(lab.get(g, ()) == (c,) for g in range(f - half, f + half + 1))
                    for f in range(0, decoded, STRIDE)) < 2]
    if short:
        raise ValueError("the reference video has no clean ~2 s stretch labelled only with step "
                         + ", ".join(f"{c} ({' '.join(id2step[c].split())})" for c in short)
                         + " — the VLM's example clips of each option come from there")


def _scores(csv_path: Path) -> dict:
    """vlm_scores.csv as columns (frame, p_A.., p_target_smoothed)."""
    if not csv_path.exists():
        return {"columns": [], "rows": []}
    lines = csv_path.read_text().splitlines()
    head = lines[0].split(",")[:-1]  # the last column is the GT label, quoted
    rows = [[float(v) for v in l.split(",")[:len(head)]] for l in lines[1:] if l.strip()]
    return {"columns": head, "rows": rows}


def _read(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _url(job_id: str, rel: str) -> str:
    return f"/api/fawadseg/{job_id}/{quote(rel)}"


def _files(job_id: str, d: Path, only: Optional[list[str]] = None) -> list[dict]:
    run_dir = config.FAWADSEG_RUNS / job_id
    ps = [d / n for n in only] if only else sorted(d.iterdir())
    return [{"name": p.name, "url": _url(job_id, str(p.relative_to(run_dir))), "bytes": p.stat().st_size}
            for p in ps if p.is_file()]


# --------------------------------------------------------------------------- #
# sweeping
# --------------------------------------------------------------------------- #
def sweep(ttl: float) -> None:
    """Runs, cached frames and embeddings unused for `ttl` seconds."""
    now = time.time()
    for d in config.FAWADSEG_RUNS.iterdir() if config.FAWADSEG_RUNS.is_dir() else []:
        if d.is_dir() and now - d.stat().st_mtime > ttl:
            shutil.rmtree(d, ignore_errors=True)
    for d in FRAMES.iterdir() if FRAMES.is_dir() else []:
        meta = d / "meta.json"
        age = now - (meta.stat().st_mtime if meta.exists() else d.stat().st_mtime)
        if d.is_dir() and age > ttl:
            shutil.rmtree(d, ignore_errors=True)
    for p in EMBEDS.iterdir() if EMBEDS.is_dir() else []:
        if now - p.stat().st_mtime > ttl:
            p.unlink(missing_ok=True)
    for d in SOURCES.iterdir() if SOURCES.is_dir() else []:  # links to swept uploads
        if d.is_dir() and not any(p.exists() for p in d.iterdir()):
            shutil.rmtree(d, ignore_errors=True)
