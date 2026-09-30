# SAM 3 web app

Tools, switched in the top bar: **object tracking** with SAM 3 (below),
**frame extraction** with Qwen3-VL-Embedding (see [Frame extraction](#frame-extraction)),
**segment extraction**, and **frame extraction fawad segment** (the research's
step-retrieval pipeline as is, see [its section](#frame-extraction-fawad-segment)).
Both stay mounted, so switching never loses work.

**Object tracking** is laid out as three steps:

1. **Upload video** — drop a clip (and optionally a `classes.txt`). The moment
   the upload lands the backend embeds frame 0, so step 2 has nothing to wait
   for.
2. **Detect objects** — prompt SAM 3 on a frame to define instances.
3. **Track & export** — propagate them through the clip; out comes one fused
   annotated video + a combined per-frame JSON.

The rendered video draws each instance's **segmentation mask** — tinted and
outlined in its own colour — not its bounding box. The mask is what the tracker
actually produces; the box is derived from it. Masks are carried through
compositing as polygons (full-res masks for every frame of every instance would
be gigabytes) and dropped before the JSON is written, which still carries
per-frame boxes. An instance whose mask comes back empty falls back to its box.

Each object you define is one **instance** — one tracked identity, one class,
one colour — and every instance gets its own row in the output JSON. Define as
many as you like on the same frame.

### The three prompt types

Offered first, in this order, because each one starts from nothing:

- **Click to segment** — click *one* object and SAM 3 segments it, the way the
  SAM 3 demo does. Each click re-segments the frame and draws the resulting
  mask, so you refine until it looks right: positive clicks grow the selection,
  negative clicks (the toggle, right-click, or alt-click) carve pieces off, and
  clicking a point removes it. Then **commit** it — pick its class and it
  becomes one instance, the clicks clear, and you start the next object. Clicks
  never merge across objects: each committed mask is its own tracked identity.
  The preview runs the same tracker the job does, so the mask you commit is the
  one that gets propagated.
- **Draw a bounding box** — click two opposite corners, then commit it the same
  way. Seeded as a rectangular mask, so the tracked shape starts as that
  rectangle rather than a real segmentation.
- **Text prompt** — a phrase, then either **detect on this frame** (SAM 3
  segments every match here and you commit the ones you want) or **add as phrase
  object**, which hands the concept to the detector to find and track across the
  whole clip under one row.

### Similar-object search

Once at least one instance exists, each row in the instance list offers **Find
similar objects**: it uses that instance's own mask as a visual exemplar and
proposes every matching object on the frame with a score. A confidence-threshold
slider filters them live (included = solid, excluded = dashed grey) and the ✕ on
any candidate discards it outright; whatever survives is committed as its own
instance under the class you pick. Two methods:

- **SAM 3 exemplar** (default) — single-image visual exemplar / PCS. Tight boxes
  when it hits; reliable for clean uniform groups, noisier among clutter.
- **YOLOE visual prompt** — localises the sibling objects with far less
  cross-frame clutter, but boxes are looser and confidences run low and
  uncalibrated, so the slider starts low and is rank-ordered — review visually.
  Needs `ultralytics` (see setup); the model loads lazily on first use and
  auto-downloads its weights (or drop `checkpoints/yoloe-v8l-seg.pt`).

### Removing things

Everything drawn on the frame carries a ✕: click it to delete that instance (or
discard that candidate). The instance list has the same ✕ on each row.

### Preprocessing while you annotate

Two different things are prepared ahead of time, on two different workers, so
neither the first prompt nor the Track button waits on groundwork.

**The tracking window** (`POST /jobs/prep-clip`, `clipprep.py`) — a tracking run
is trim → decode → resize every frame to 1008² → GPU propagation, and only the
last step needs the GPU. The rest is ~30 ms/frame of pure CPU work (8.7 s for
300 frames) that grows with the window, so it runs the moment the upload lands
and again, debounced, whenever Max frames grows. It has **its own worker thread**
(`app.state.cpu_jobs`, a second `JobRegistry`), so preparing a long clip never
delays a click preview on the inference thread. `run_multitrack` takes the
finished tensor as `images=` and skips straight to seeding.

Only the newest prepared clip is kept — at ~6 MB a frame (float16, 1008²), a
second one is a lot of RAM for something nothing will read.

Two things this deliberately does **not** do:

- It does not precompute per-frame vision features. The tracker's
  `cached_features` holds exactly one frame and is replaced wholesale on every
  miss (`_get_image_feature`), so a whole-clip set would be thrown away on first
  use — and at ~40 MB/frame it would not fit anyway.
- It does not hand `load_video_frames` the `.mp4` directly, tempting as that is
  (3.2 s instead of 6.2 s for 300 frames). **Its video-file branch never scales
  pixels to [0, 1] before applying `img_mean`/`img_std`**, so it returns roughly
  `[-1, 509]` where the image-folder branch returns `[-1, 1]`. Prep goes through
  the JPEG-folder route, which is exactly what an un-prepared run does — the
  tensor comes out `torch.equal` to it, so preparing ahead cannot change a
  result.

### Frame embeddings (for prompting)

Encoding a frame is most of the cost of a prompt, and it depends only on the
pixels, so the backend caches it per `(upload, frame)`: the image model's vision
features for the text/exemplar search, and a one-frame tracker inference state
for the click preview. `POST /api/jobs/prepare` builds both. This one *does* run
on the inference worker, since it is GPU work. The frontend fires
it as soon as a video is uploaded (frame 0) and again whenever the viewer settles
on a new frame — but never while the model is busy with something the user is
waiting on, since every job shares one worker thread. Three frames are kept
(LRU); a tracking run clears them all to free the GPU. A prompt that fails on a
cached frame re-encodes once before giving up.

### Frames tracked, and the detection threshold

There is no policy cap on **Max frames** — the slider spans the whole clip and
the route clamps only to the upload's own frame count. The cost is time (linear
in frames) rather than memory: at 1080p a frame is ~6 MB in the tracker's
float16 buffer plus ~6 MB decoded, so even a few thousand frames sit well inside
this box's RAM.

The frame scrubber is deliberately **not** tied to Max frames, so you can seed
anywhere in the clip. Committing a prompt past the current window widens the
window to include it, and a prompt left outside one offers a one-click extend.

**Detection threshold** only affects phrase objects — it is
`score_threshold_detection` on the detector, set inside `_track_text`. Click,
box and exemplar instances never see it, which is why it lives in the text-prompt
panel rather than in the general tracking settings. Unlike the confidence slider
on a similar-object search — which re-filters candidates already sitting in the
browser — this one is consumed by the model before it tracks, so changing it
means re-running the job.

### Run history

Every `video-track` job writes a record to `var/runs.json` (`runs.py`) — the
source clip, its dimensions, the settings, the instances that were prompted, and
the outputs. Going back from the results screen to detect more objects and
tracking again is a **new job**, so it gets its own record rather than replacing
the first; the second record carries the full instance list at that point, which
is what was actually tracked.

Open it from **History** beside the step rail. A run expands to its video,
downloads, settings, and a **Reuse** button: settings always, plus the instances
when the run belongs to the upload you have open (their coordinates mean nothing
against a different video, so that case reuses settings only).

Records are small — the prompts, not the pixels — and capped at 200. A corrupt
`runs.json` is tolerated: the app logs and starts an empty history rather than
refusing to boot.

**Deleting videos.** Three ways:

- **One run** — *Delete this run* in the History drawer (`DELETE /api/runs/{id}`)
  removes the record and unlinks `var/results/<id>.*`.
- **Automatically** — the sweeper every 30 min: results and uploads after
  `RESULT_TTL_SECONDS` / `UPLOAD_TTL_SECONDS` (both 30 days), always sparing the
  newest `KEEP_LAST_RUNS`. In-memory job records go after `JOB_TTL_SECONDS` (6 h)
  — the durable copy is the history.
- **By hand** — `rm webapp/backend/var/results/*`. `outputs_present` re-checks
  the disk on every read, so history reports "cleared out" instead of showing a
  dead player.

`KEEP_LAST_RUNS` counts **runs, not files**: a run writes `<id>.mp4` *and*
`<id>.json`, and the old file-counting version kept half as many runs as it
claimed.

### One prompt per frame

An instance carries **one prompt per frame** (it is a single tracked identity,
and SAM 3 takes one prompt per object per frame). To re-anchor a drifting track,
open the instance, move to another frame and hit **re-seed on frame N** — the
reseed pattern from `scripts/track_segmented_box_as_mask.py`. All of this uses
SAM 3's interactive tracker (`model.tracker`), no detector.

**Phrase objects** are the exception: instead of prompting anything, a concept
phrase goes to the detector, which finds and tracks every matching instance, all
under that one row.

### Classes

Upload a `classes.txt` (one name per line) on step 1. You pick the class at the
moment you **commit** a mask or box, not before — prompt first, label second. The
choice sticks, so a run of the same class stays one click per object, and colours
follow the class so every instance of a class reads the same. Open an instance to
change its class afterwards. Line N of the file is class id N (the YOLO
convention); blank lines, `#` comments, and a leading `0 ` / `0: ` index are all
tolerated. The list lives on the server (`var/classes.json`), so it survives a
reload, and both the id and the resolved name land in the result JSON. Classes
are optional — instances left unclassified just track as usual.

Optionally also track backwards from each seed frame (`bidirectional`).

- **backend** — FastAPI, single worker, SAM 3 pinned to **GPU 1**
  (`CUDA_VISIBLE_DEVICES=1`). All SAM 3 inference is serialized on one thread.
  Frame extraction's Qwen process uses GPU 0 (see [Frame extraction](#frame-extraction)).
- **frontend** — Vite + React + TS. Talks to `/api` (Vite proxy in dev,
  same-origin when the backend serves the build).

Everything runs on this box; the browser is on the box too (AnyDesk). No auth,
no network exposure. To later host the frontend on a website, set
`VITE_API_BASE` (frontend) + `SAM3_CORS_ORIGINS` (backend).

## Frame extraction

Give a video and a few reference images; get back every stretch of the video
that looks like them, as segments you can review, correct and export. It is the
method that held up in `~/Documents/qwen_vl/extraction_project`, minus the need
for a labelled reference video.

**Method.** Qwen3-VL-Embedding-8B embeds every 5th frame and each reference
(same instruction for both). Each reference's cosine similarity to the frames is
**z-scored against the video's own distribution**; a frame's score is the max
over references, smoothed over ~1 s, thresholded, gaps under 2 s filled, runs
under 1 s dropped. The z-scoring matters: it puts references of different
"typicality" on one scale, so one threshold fits all. On the research data
(5 reference images, image-only, no VLM stage) z ≥ 1.5 gave F1 0.73–0.92 on
the distinctive steps (3, 7). Hard, look-alike steps (5) stay weak (~0.3), as
they were in the research without its VLM verification stage. Adding "not
this" negative images made things worse, so there are none.

**What costs what.**

| action | where | time |
| --- | --- | --- |
| first run on a video | GPU 0 | ~45 s to load the model (if it is not loaded yet) + frames ÷ 5 ÷ 9.8 per second (≈4 min for 12k frames) |
| new reference *image* | GPU 0 | under a second while the model is loaded; ~45 s if it has to load first |
| reference picked *from the video* | CPU | milliseconds — it is a row of the cached embedding |
| threshold / smoothing / exclusions | browser | live |
| export | CPU worker | seconds |

Embeddings are cached in `var/embeds/` by **file content hash** (+ model,
instruction, stride), so re-uploading the same video — or re-opening it after a
restart, which forgets every upload — costs nothing.

**One model per GPU.** SAM 3 lives on GPU 1 (the backend's own
`CUDA_VISIBLE_DEVICES`); Qwen runs on **GPU 0** (`SAM3_QWEN_GPU`, nvidia-smi / PCI
numbering) in its own process, the qwen venv's python
(`~/Documents/qwen_vl/.venv-vllm`; vLLM needs its own torch). Embedding jobs run
on a third worker thread (`app.state.qwen_jobs`), so **tracking and extraction
run side by side** — a click-to-segment answered in 0.7 s mid-embedding.

The worker (`qwen_embed_worker.py`, driven by `extraction.QwenWorker`) loads the
model once, then serves requests over stdin. It **stays loaded for
`SAM3_QWEN_KEEP_WARM_S`** (default 600 s) after its last request, then exits
and gives GPU 0 back — `0` makes it exit after every run. `gpu_memory_utilization`
is sized from what nvidia-smi says is free, minus 2.5 GB (with 0.7 GB spare vLLM
OOMed on the first batch). It runs in its own process group, and stopping or
cancelling kills the whole group — vLLM's EngineCore child holds the VRAM.
`/health` reports it under `qwen` (`off | starting | ready | busy`), and the status
pill shows it.

**Shared-GPU fallback.** Set `SAM3_QWEN_GPU=1` (the backend's own GPU) and
Qwen shares GPU 1 by taking turns: the job moves to the inference thread,
`engine.offload()` parks both SAM 3 models in CPU RAM, the worker runs and is
stopped, and `engine.restore()` brings SAM 3 back in a `finally`. `/health` says
`busy` meanwhile and the tracking UI waits. Use it if GPU 0 is needed for
training again.

**Refining.** Each segment row has ＋ (right match: add its best frame as a
reference, re-scored instantly) and ✕ (wrong match: exclude those frames; the
exclusion survives threshold changes). A thumbnail in the reference grid can be
switched off without deleting it.

**Playback.** Segments play in the browser's own `<video>` decoder, not as a
stream of JPEGs (that managed ~5 frames/s: one ~50 ms request per frame, strictly
in turn). On upload, the CPU worker makes a playback copy
(`POST /jobs/extract-playback`, ~8 s per 5 min of 640×480, cached in
`var/proxies/` by content hash): H.264, one keyframe a second, and **re-timed to
a constant frame rate** so frame N is at exactly N/fps — the source's own
timestamps drift, and `currentTime` would land on the wrong frame. Verified
frame-exact against cv2's numbering. During playback `requestVideoFrameCallback`
reports each presented frame back, so the timeline and badges follow; stopping
leaves the video up until the JPEG of that exact frame has loaded under it.
Scrubbing still uses JPEGs, which are exact for any frame.

**Truncated files.** Some recordings list more frames in their header than they
contain. The worker reports how many actually decode and the page uses that.

**Export** (`var/results/extract_<job>.*`): a JSON of the segments (frames,
seconds, peak/mean score, closest reference) with the settings and references
behind them; optionally a clip of just the selected frames, each labelled with
its segment and source position; optionally a ZIP of every Nth frame as JPEGs,
one folder per segment.

## Segment extraction

Find where a *step* happens in other recordings, from an occurrence of it marked
on a reference video. Same models and GPU as frame extraction, plus the
Qwen3-VL-8B-Instruct VLM; the method is the research's
`class_N_desc_vlm_hints` pipeline (see `backend/segx.py`). Three steps:

1. **Mark the step.** Upload a reference video, scrub to where the step starts
   and press *Set start* (`I`), then to where it ends and *Set end* (`O`); drag
   the flags on the timeline to fine-tune, ←/→ step a frame (shift: 10). *Add as
   the step* cuts that range, frame-exact, into a clip on the server
   (`POST /segx/cut`). Mark two or three occurrences, from one video or several
   (*Use another video* keeps what is marked). *Add as another step* marks
   counter-examples (the steps before and after, look-alikes); in the research
   they made the difference on look-alike steps. Ready-cut clips can be uploaded
   instead.
2. **Describe it.** The VLM watches up to three step clips and writes a step
   name and description, editable. It goes into the VLM's question and,
   optionally, the embedding instruction.
3. **Search a video.** Upload another recording and *Find the step*: embed every
   5th frame, keep the 25% most like the step clips, have the VLM classify a ~2 s
   clip every 5 frames there, P(step) → live threshold/smoothing → segments,
   export as in frame extraction. *Search another* keeps the step and description.

## Frame extraction fawad segment

The research's `class_N_desc_vlm_hints` pipeline
(`~/Documents/qwen_vl/extraction_project`, e.g.
`outputs/marmon_station_1/2026-08-20 12_45_33/class_5_desc_vlm_hints`), **run as
is** rather than adapted: `backend/fawadseg_scripts/` holds byte-identical copies
of the research's `extract_frames.py`, `embed_frames.py`, `match_frames.py` and
`vlm_classify.py`, and `backend/fawadseg.py` runs them in the qwen venv with the
arguments the research's pipeline used. Segment extraction (above) is the
adaptation that needs no labels; this one needs what the research needed.

**Inputs** (the research's):

| input | goes to |
| --- | --- |
| reference video + its `preds.json` (per-frame step labels) | kNN vote, VLM example clips |
| `detector.json` | step names (`cycle_steps`) |
| step to find, confusable steps (default 3, 7) | `--cls`, `--confusers` |
| step description (`classN_description.txt`) | `embed_frames.py --describe` |
| per-step hints (`step_hints.json`) | `vlm_classify.py --hints` |
| target video | what is searched |
| target `preds.json` — optional | metrics only; selection never reads it |

**Pipeline**: `extract_frames.py` (both videos) → `embed_frames.py --stride 5
--describe` → `match_frames.py --balanced` → `vlm_classify.py target --confusers
--hints --candidates`. **Outputs** are the scripts' own files, in the research's
folders: `class_N_desc_balanced/` (candidates: `selected_frames.json`,
`frame_scores.csv`, `metrics.json`, `report.md`, `timeline.png`, the clip) and
`class_N_desc_vlm_hints/` (`selected_frames.json`, `vlm_scores.csv`,
`metrics.json`, `timeline.png`, `vlm_examples.jpg`, `<target>_classN_selected.mp4`),
plus `inputs/`, `pipeline.log`, `commands.txt` and a ZIP of all of it. The page
shows the segments on the target video with P(step), the metrics, the pictures
and the files. There are no live sliders: the selection is the script's.

Each run lives in `var/fawadseg/runs/<job>/` (served at `/api/fawadseg/`) with
the research's layout — `work/frames/<video stem>`, `work/embeddings/<stem>__classNdesc.npz`
— because `match_frames.py` finds the reference's `meta.json` and names its
outputs from those paths. They are symlinks into `var/fawadseg/work/`, a cache
of decoded frames (~50 kB a frame) and embeddings keyed by file content, so a
video already seen is not decoded or embedded again. Swept after
`RESULT_TTL_SECONDS` unused.

Two opt-in hooks in `fawadseg_scripts/launch.py`, the only code around the
scripts: when the Qwen GPU has less free memory than the scripts ask vLLM for
(0.9 / 0.92 of the card), a smaller `gpu_memory_utilization` (KV-cache size
only); and without target labels, a placeholder `preds.json` with no labels
stands in, `metrics.json` / `report.md` are dropped and the selected-frames video
is drawn without GT / TP / FP text. The other tabs' Qwen workers are stopped
first (one model on the card at a time); on a shared GPU SAM 3 is parked as for
the other Qwen jobs.

## One-time setup

```sh
VIRTUAL_ENV=../.venv uv pip install fastapi "uvicorn[standard]" python-multipart
VIRTUAL_ENV=../.venv uv pip install ultralytics   # optional — only for "find similar" → YOLOE
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
nvm install 20
cd frontend && npm install && cd ..
```

Frame extraction also needs the qwen vLLM venv at `~/Documents/qwen_vl/.venv-vllm`
(or `SAM3_QWEN_PYTHON`) with `Qwen/Qwen3-VL-Embedding-8B` in the HF cache. Without
it the page says so and tracking is unaffected.

Needs `checkpoints/sam3.pt` and an `ffmpeg` with `libx264` (the backend finds
`/usr/bin/ffmpeg` automatically; the conda ffmpeg lacks it, and the fallback is
VP8/webm which still plays in Chrome/Firefox).

### "bad interpreter: .venv/bin/python3: No such file or directory"

`.venv` was built by `uv` with the Python that lives inside the VS Code snap
(`~/snap/code/<rev>/...`); VS Code self-updates and deletes the old `<rev>`,
orphaning the interpreter. `run.sh` auto-heals this, or run it by hand:

```sh
./fix-venv.sh    # copies a stable python-build-standalone 3.12 to
                 # ~/.local/share/python-standalone/ and re-points .venv at it
```

The 7 GB of site-packages are kept; only the interpreter is relinked.

## Run

**As a service (the default on the training box).** `run.sh` ties the backend and
Vite to its terminal: when VS Code's remote connection reconnects or reloads, the
terminal gets SIGHUP and takes both down ("Backend unreachable"). As systemd user
services they run on their own, restart within seconds if they crash, and log to
journald (lingering is enabled, so they also survive the last logout):

```sh
webapp/service.sh install     # once: write the units, enable, start -> http://127.0.0.1:5173
webapp/service.sh restart     # after a backend change (Vite hot-reloads the frontend itself)
webapp/service.sh logs        # follow both logs
webapp/service.sh status | stop | start | uninstall
```

While the services are active, `run.sh` just says so and exits.

**The browser does the media work.** A video you pick is opened in the browser
itself (`src/lib/localVideo.ts`): mp4box.js reads its frame times from the moov
box (~40 ms), and playback and scrubbing run on that local copy — no JPEG per
frame and no playback copy over the network, frame-exact with the server's cv2
numbering (checked on three recordings). The reference video in segment
extraction is never uploaded: a marked range is grabbed frame by frame in the
browser and only those frames go up (`POST /segx/clip-frames`). A video to search
is uploaded in the background while it is already on screen. MP4/MOV the browser
can decode only; anything else falls back to uploading it and the server's frames
and playback copy.

```sh
./run.sh                 # dev: uvicorn :8000 + Vite :5173 (HMR) -> http://127.0.0.1:5173
make backend             # backend only, no reload
make serve               # single process: build the frontend, serve it on :8000

Under `run.sh` the backend on :8000 serves the API only (`SAM3_SERVE_FRONTEND=0`),
so the one place to open the app is **:5173** — a build on :8000 would go stale
the moment the source changes. `make serve` is unaffected.
```

`run.sh` clears its own leftovers before starting. A previous run that was
SIGKILLed — or whose terminal simply closed — never fires its EXIT trap, so its
uvicorn outlives it, keeps `:8000` and ~8 GB of GPU, and every later start comes
up with Vite in front of a dead backend. The preflight stops earlier `run.sh`
instances, then reclaims `:8000` and `:5173`, then sweeps any backend that is
alive but failed to bind. It only ever kills **our own** processes: the port's
owner is matched against this repo's uvicorn/Vite command lines, and anything
else on the port stops the script with a message instead of being killed. Ports
are overridable with `SAM3_PORT` / `SAM3_VITE_PORT`.

**Anything the old backend was working on dies with it** — an in-flight tracking
job is lost, not resumed.

First start loads the model (~1 min); the status pill turns green when ready.
Model load and every track job share one worker, so a second job queues behind a
running one (the UI shows "N jobs ahead").

### Reaching it from another machine

Both servers bind to `127.0.0.1` only, so nothing is on the LAN. Tailnet peers
get in through `tailscale serve`, which proxies to the Vite port from inside
tailscaled — no bind change, no LAN exposure:

```sh
tailscale serve --bg --http=8080 5173     # -> http://<host>.<tailnet>.ts.net:8080/
tailscale serve status                    # what is currently proxied
tailscale serve --http=8080 off           # stop
```

Vite rejects unknown `Host` headers, so `server.allowedHosts` in
`frontend/vite.config.ts` carries `.ts.net`; without it the tailnet URL answers
"Blocked request. This host is not allowed." The serve config lives in tailscaled
and survives a `run.sh` restart — but it points at **5173**, so it only works in
dev mode. For `make serve` (single process on :8000) point it at 8000 instead.

Anyone on the tailnet can then reach it, and the app has no auth.

## API (all under `/api`, coords are pixels in the video's native resolution)

| | |
| --- | --- |
| `GET /health` | status, model, GPU, ffmpeg, config |
| `POST /uploads` (multipart `file`) | → `{upload_id, kind:"video", width, height, frames, fps}` |
| `GET /uploads/{id}/frame/{idx}.jpg` | a video frame as JPEG (for drawing on) |
| `GET /classes` | → `{classes:[{id,name}], count, source}` |
| `POST /classes` (multipart `file`) | replace the label set from a classes.txt → same shape (+ `warning` on duplicate names) |
| `DELETE /classes` | clear the label set |
| `POST /jobs/prep-clip` | `{upload_id, frames}` → `{job_id}`; runs a tracking job's CPU preamble (trim + resize to the tracker's input tensor) on the CPU worker. Result `{frames, reused, message}` |
| `POST /jobs/prepare` | `{upload_id, frame}` → `{job_id}`; embeds that frame (image features + one-frame tracker state) so the next prompt on it skips the encode. Fire-and-forget: the result is just `{frame, width, height, ready}` |
| `POST /jobs/click-preview` | `{upload_id, frame, points:[[x,y],…], labels:[1\|0,…], box?}` → `{job_id}`; result `{polygons:[[[x,y],…],…], box, area, coverage, frame, width, height, message}` |
| `POST /jobs/exemplar` | `{upload_id, frame, box:[x1,y1,x2,y2], neg_boxes?, method?:"sam3"\|"yoloe"}` → `{job_id}`; result `{candidates:[{box,score}], frame, width, height, method, suggest_threshold}` |
| `POST /jobs/video-track` | see below → `{job_id}` |
| `GET /runs` | → `{runs:[…], count}` — history, newest first, without the prompt payloads |
| `GET /runs/{id}` | one run in full, including the instances it tracked |
| `DELETE /runs/{id}` | forget a run and delete the video/JSON it produced |
| `GET /jobs/{id}` | `{status, progress, stage, queued_ahead, result, error}` |
| `DELETE /jobs/{id}` | cancel |
| `GET /api/files/{name}` | result `.mp4` and `.json` |
| `POST /jobs/extract-embed` | `{upload_id, image_ids:[…], stride?, instruction?}` → `{job_id, key}`; embeds whatever is not cached (Qwen, GPU 0). Result `{key, stride, decoded_frames, embedded_frames, embedded_now}` |
| `POST /jobs/extract-playback` | `{upload_id}` → `{job_id}`; result `{url, fps}` — the H.264 playback copy, served from `/api/proxies/` (CPU worker) |
| `POST /extract/score` | `{upload_id, key, refs:[{kind:"image",id}\|{kind:"frame",frame}]}` → `{frames, decoded_frames, rows}` — one z-scored row per ref. Synchronous, CPU; 409 if not embedded |
| `POST /segx/clip-frames` | multipart `frames` (JPEGs, in order), `fps`, `name` → an upload (video) of exactly those frames — a range marked on a video that only the browser has |
| `POST /segx/cut` | `{upload_id, start, end}` → an upload (video) of frames start..end inclusive, plus `{source_id, start, end}`. Synchronous, CPU, frame-exact (cv2 frame numbering) |
| `POST /jobs/segx-describe` | `{clip_ids:[…]}` → `{job_id}`; result `{name, description, clips_watched}` (VLM, GPU 0) |
| `POST /jobs/segx-search` | `{upload_id, step_ids, other_ids?, name?, description?, use_description?, coverage?, stride?}` → `{job_id}`; result `{decoded_frames, fps, stride, region, centers, p, options, …}` |
| `POST /jobs/extract-export` | `{upload_id, segments:[{start,end,peak?,mean?,best_ref?}], video?, zip_every?, settings?, references?}` → `{job_id}`; result `{json_url, video_url?, zip_url?, …}` (CPU worker) |
| `POST /jobs/fawadseg-run` | `{ref_upload_id, upload_id, ref_preds, detector, tgt_preds?, cls, confusers, description?, hints?}` (the JSON files as their text) → `{job_id}`; result `{segments, candidate_segments, scores, metrics, options, examples, files, zip_url, commands, …}` |

`POST /jobs/video-track` body:
```jsonc
{
  "upload_id": "…",
  "max_frames": 120,          // clamped to [10, the clip's own frame count]
  "threshold": 0.5,           // detector score threshold — phrase objects ONLY
  "bidirectional": false,
  "objects": [
    // one prompt per frame: a `box` alone, or `points`+`labels` (a click
    // segment) optionally refining a `box`. `cls` indexes the uploaded
    // classes.txt; omit it for an unclassified instance.
    { "name": "plate 1", "color": [80,180,255], "kind": "box", "cls": 2,
      "seeds": [ { "frame": 0,  "box": [81,78,340,352] },
                 { "frame": 18, "points": [[240,210],[300,90]], "labels": [1,0] } ] },
    { "name": "plate 2", "color": [80,180,255], "kind": "box", "cls": 2,
      "seeds": [ { "frame": 0, "points": [[420,200]], "labels": [1] } ] },
    { "name": "people", "color": [255,120,80], "kind": "text", "cls": null,
      "phrase": "person", "prompt_frame": 0 }
  ]
}
```

Result:
```jsonc
{
  "tracked_video_url": "/api/files/<job>.mp4",
  "json_url": "/api/files/<job>.json",
  // {classes:{"2":"plate"},
  //  objects:[{name,color,kind,cls,class_name,per_frame:[[x1,y1,x2,y2],…]|[]}]}
  "objects": 2, "frames": 120, "message": "…", "codec_warning": null
}
```

## Layout

```
backend/
  main.py       app, lifespan (warm-load + sweeper), routers, static mounts
  config.py     paths, tunables, require_single_gpu()
  engine.py     run_multitrack: instances (box/click seeds) via model.tracker +
                phrase objects via the detector session, composited & fused;
                click_preview: one frame's mask from a set of clicks;
                find_similar: text phrase or single-image visual exemplar;
                prepare_frame + the (upload, frame) encoder cache behind all three
  clipprep.py   the CPU preamble of a tracking run, prepared while you annotate
  extraction.py frame extraction: embedding cache, the Qwen worker, scoring, export
  fawadseg.py   frame extraction fawad segment: runs fawadseg_scripts/ (the research's
                scripts, verbatim, + launch.py) as the research's pipeline did
  qwen_embed_worker.py  the Qwen3-VL-Embedding subprocess (runs in the qwen venv)
  runs.py       run history, persisted to var/runs.json
  classes.py    the uploaded classes.txt label set, persisted to var/classes.json
  jobs.py       Job + JobRegistry + the single-thread executor
  media.py      cv2 probe / frame extract / trim, ffmpeg finalize
  store.py      uploaded-file registry
  schemas.py    request models
  routes/       health, uploads, classes, jobs
  var/          runtime uploads/frames/results/tmp (gitignored, auto-swept)
frontend/       Vite + React + TS
  src/App.tsx           shell: top bar, status, step rail
  src/lib/workspace.ts  the state that outlives a step change, + frame embedding
  src/steps/            StepVideo (1) · StepDetect (2) · StepResult (3)
  src/components/       BoxCanvas (the frame + overlays + ✕ badges), ObjectList,
                        FrameBar, Dropzone, LabelsCard, Progress, Slider, …
  src/lib/objects.ts    instance model, naming, colours, request payloads
  src/extract/          frame extraction page: ExtractPage, Timeline (canvas),
                        RefGrid, SegmentList; segment extraction: SegxPage,
                        MarkTimeline (start/end marking); src/lib/segx.ts its state;
                        FawadSegPage + src/lib/fawadseg.ts (fawad segment)
  src/lib/extraction.ts its state; src/lib/segments.ts the scoring (a port of
                        smooth/clean from the research's match_frames.py)
```

## Env vars

| var | default | |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `1` (set by `run.sh`) | must resolve to exactly one device |
| `SAM3_CHECKPOINT` | `../checkpoints/sam3.pt` | |
| `SAM3_CORS_ORIGINS` | unset | comma list; enables CORS for a hosted frontend |
| `SAM3_FFMPEG` | auto | force a specific ffmpeg binary |
| `SAM3_ALLOW_MULTI_GPU` | unset | bypass the single-GPU guard (don't) |
| `SAM3_QWEN_PYTHON` | `~/Documents/qwen_vl/.venv-vllm/bin/python` | interpreter for the embedding worker |
| `SAM3_QWEN_EMB_MODEL` | `Qwen/Qwen3-VL-Embedding-8B` | |
| `SAM3_QWEN_GPU` | `0` | physical GPU for Qwen; set to the backend's own (`1`) to share it with SAM 3 |
| `SAM3_QWEN_KEEP_WARM_S` | `600` | keep Qwen loaded this long after its last run; `0` = unload every time |
