import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type HealthConfig, type QwenStatus } from '../api/client'
import { Dropzone } from '../components/Dropzone'
import { Progress } from '../components/Progress'
import { Slider } from '../components/Slider'
import { Stepper } from '../components/Stepper'
import {
  markedSegments, SEGX_DEFAULTS, SEGX_STEPS, useSegx, type Clip, type ClipKind, type Segx,
} from '../lib/segx'
import type { Segment } from '../lib/segments'
import { ExportCard } from './ExportCard'
import { clock } from './format'
import { MarkTimeline } from './MarkTimeline'
import { SegmentList } from './SegmentList'
import { Timeline } from './Timeline'
import { VideoViewer, type Play } from './VideoViewer'
import './extract.css'

// measured: ~45 s to load each Qwen model, ~9.8 frames/s to embed, ~2.2 clips/s to classify
const LOAD_S = 45
const EMBED_FPS = 9.8
const CLIPS_PER_S = 2.2

function duration(s: number): string {
  return s < 90 ? `${Math.max(1, Math.round(s / 10) * 10)} s` : `${Math.round(s / 60)} min`
}

export function SegxPage({ cfg, qwen }: { cfg: HealthConfig; qwen?: QwenStatus }) {
  const x = useSegx(cfg)

  if (!cfg.extract?.available) {
    return (
      <main>
        <div className="page narrow">
          <p className="note danger">
            Segment extraction needs the Qwen vLLM environment, and the backend could not find it.
            Point <code>SAM3_QWEN_PYTHON</code> at its python and restart the backend.
          </p>
        </div>
      </main>
    )
  }

  return (
    <>
      <div className="steprow">
        <Stepper steps={SEGX_STEPS} current={x.step} reached={x.reached} onGo={x.goto} />
      </div>
      <main>{x.step === 'search' ? <SearchView x={x} qwen={qwen} /> : <ReferenceView x={x} />}</main>
    </>
  )
}

// =========================================================================== //
// Steps 1 and 2: the reference video on the left
// =========================================================================== //
function ReferenceView({ x }: { x: Segx }) {
  const [playing, setPlaying] = useState<number | null>(null)
  const play: Play = { playing, setPlaying }
  const pageRef = useRef<HTMLDivElement>(null)
  const v = x.ref.info
  const fps = x.refFps

  // what the viewer can play and step through: marked ranges, then the selection
  // (memoised: a new array would restart playback on every frame it reports)
  const vid = v?.id
  const stepSegs = useMemo(() => markedSegments(x.steps, vid), [x.steps, vid])
  const otherSegs = useMemo(() => markedSegments(x.others, vid), [x.others, vid])
  const segs = useMemo(() => {
    const all: Segment[] = [...stepSegs, ...otherSegs]
    if (x.selection) {
      const [start, end] = x.selection
      all.push({ start, end, peak: 1, peakFrame: start, mean: 1, bestRef: 0 })
    }
    return all
  }, [stepSegs, otherSegs, x.selection])
  const selIdx = x.selection ? segs.length - 1 : -1

  // I / O set the start and end at the current frame; arrows step a frame (shift: 10)
  const { refFrame, setMarkIn, setMarkOut, setRefFrame, step } = x
  useEffect(() => {
    if (step !== 'reference') return
    const onKey = (e: KeyboardEvent) => {
      if (!pageRef.current?.offsetParent || e.ctrlKey || e.metaKey || e.altKey) return
      const t = e.target as HTMLElement
      if (t.closest('input, textarea, select, [contenteditable]')) return
      const k = e.key.toLowerCase()
      if (k === 'i') setMarkIn(refFrame)
      else if (k === 'o') setMarkOut(refFrame)
      else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        if (t.getAttribute('role') === 'slider') return // the timelines handle their own
        setPlaying(null)
        setRefFrame(refFrame + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 10 : 1))
      } else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [step, refFrame, setMarkIn, setMarkOut, setRefFrame])

  const inStep = stepSegs.findIndex((s) => x.refFrame >= s.start && x.refFrame <= s.end)
  const inOther = otherSegs.some((s) => x.refFrame >= s.start && x.refFrame <= s.end)
  const inSel = !!x.selection && x.refFrame >= x.selection[0] && x.refFrame <= x.selection[1]

  return (
    <div className="page extract" ref={pageRef}>
      <VideoViewer
        video={x.ref.video}
        local={x.ref.local}
        onLocalFail={x.ref.localFail}
        empty={
          <div className="xempty">
            <Dropzone
              accept="video/*"
              title="Drop a reference video"
              hint="A recording in which the step happens. You will mark where it starts and ends. MP4, MOV, AVI, MKV or WEBM, up to 500 MB"
              busy={x.ref.busy}
              onFile={(f) => void x.loadRef(f)}
            />
            {x.ref.error && <p className="note danger">{x.ref.error}</p>}
          </div>
        }
        frame={x.refFrame}
        setFrame={x.setRefFrame}
        lastFrame={x.refLast}
        fps={fps}
        segments={segs}
        playbackUrl={x.ref.playbackUrl}
        onPlaybackFail={x.ref.playbackFail}
        playbackError={x.ref.playbackError}
        play={play}
        marked={[x.markIn, x.markOut].filter((f): f is number => f !== null)}
        badges={
          <>
            {inStep >= 0 && <span className="xbadge in">Step clip {inStep + 1}</span>}
            {inOther && <span className="xbadge other">Other step</span>}
            {inSel && <span className="xbadge sel">Selection</span>}
          </>
        }
        tools={() =>
          x.step === 'reference' && (
            <>
              <button
                type="button"
                className="btn"
                onClick={() => x.setMarkIn(x.refFrame)}
                title="Set the start of the selection to this frame (I)"
              >
                ⟦ Set start <kbd>I</kbd>
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => x.setMarkOut(x.refFrame)}
                title="Set the end of the selection to this frame (O)"
              >
                Set end ⟧ <kbd>O</kbd>
              </button>
            </>
          )
        }
        timeline={(seek) => (
          <MarkTimeline
            total={x.refLast + 1}
            fps={fps}
            frame={x.refFrame}
            steps={stepSegs.map((s) => [s.start, s.end])}
            others={otherSegs.map((s) => [s.start, s.end])}
            markIn={x.step === 'reference' ? x.markIn : null}
            markOut={x.step === 'reference' ? x.markOut : null}
            onSeek={seek}
            onMarkIn={x.setMarkIn}
            onMarkOut={x.setMarkOut}
          />
        )}
      />

      <aside className="panel">
        {x.step === 'reference' ? (
          <MarkPanel x={x} play={play} selIdx={selIdx} segs={segs} />
        ) : (
          <DescribePanel x={x} />
        )}
      </aside>
    </div>
  )
}

// -- step 1 ------------------------------------------------------------------ //
function MarkPanel({ x, play, selIdx, segs }: { x: Segx; play: Play; selIdx: number; segs: Segment[] }) {
  const v = x.ref.info
  const fps = x.refFps
  // index into the viewer's segments, for the play buttons
  const playIdx = (c: Clip) =>
    c.src && c.src.id === v?.id ? segs.findIndex((s) => s.start === c.src!.start && s.end === c.src!.end) : -1

  return (
    <>
      <header className="xhead">
        <h2>Mark the step</h2>
        <p className="muted small">
          On a reference video, find one occurrence of the step and mark where it starts and where it
          ends. That range is what gets searched for in other videos. Two or three occurrences, from
          one video or several, make the search more reliable.
        </p>
      </header>

      {v && (
        <section className="card">
          <div className="card-head">
            <div>
              <h3 className="xtitle" title={v.name}>
                {v.name || 'Reference video'}
              </h3>
              <p className="card-sub">
                {v.width} × {v.height} · {v.frames} frames · {v.fps} fps · {clock(v.frames, fps)}
                {x.ref.local ? ' · playing from your computer' : ''}
              </p>
            </div>
            <label className="btn-text xreplace" title="Marked clips are kept">
              Use another video
              <input
                type="file"
                accept="video/*"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) void x.loadRef(f)
                  e.target.value = ''
                }}
              />
            </label>
          </div>
        </section>
      )}

      {v && <SelectionCard x={x} play={play} selIdx={selIdx} />}

      <ClipsCard x={x} kind="step" play={play} playIdx={playIdx} />
      {(x.others.length > 0 || x.steps.length > 0) && (
        <ClipsCard x={x} kind="other" play={play} playIdx={playIdx} />
      )}

      <details className="card options">
        <summary>Already have the step cut into clips?</summary>
        <p className="muted small">
          Upload clips that show the step and nothing else, instead of or as well as marking them.
        </p>
        <Dropzone
          compact
          multiple
          accept="video/*"
          title="Drop step clips"
          hint="video files, several at once"
          busy={x.uploading > 0}
          busyLabel={`Uploading ${x.uploading}…`}
          onFiles={(fs) => void x.addClips(fs, 'step')}
        />
        <Dropzone
          compact
          multiple
          accept="video/*"
          title="Drop clips of other steps"
          hint="optional counter-examples"
          busy={x.uploading > 0}
          busyLabel={`Uploading ${x.uploading}…`}
          onFiles={(fs) => void x.addClips(fs, 'other')}
        />
      </details>

      <div className="xnav">
        <span />
        <button
          type="button"
          className="btn primary big"
          disabled={!x.steps.length}
          onClick={() => x.goto('describe')}
        >
          {x.steps.length ? 'Next: describe the step →' : 'Mark the step first'}
        </button>
      </div>
    </>
  )
}

function SelectionCard({ x, play, selIdx }: { x: Segx; play: Play; selIdx: number }) {
  const fps = x.refFps
  const sel = x.selection
  const end = (f: number | null, label: string, set: () => void, clear: () => void) => (
    <div className="markend">
      <span className="markend-label">{label}</span>
      {f === null ? (
        <span className="muted">not set</span>
      ) : (
        <button type="button" className="btn-text markend-time" onClick={() => x.setRefFrame(f)} title="Go there">
          {clock(f, fps, true)} <em>#{f}</em>
        </button>
      )}
      <span className="row gap-sm">
        <button type="button" className="btn small" onClick={set}>
          Set here
        </button>
        {f !== null && (
          <button type="button" className="btn icon small" onClick={clear} aria-label={`Clear the ${label.toLowerCase()}`}>
            ✕
          </button>
        )}
      </span>
    </div>
  )
  return (
    <section className={`card${sel ? ' accent' : ''}`}>
      <div className="card-head">
        <div>
          <h3>Selection</h3>
          <p className="card-sub">
            {sel
              ? `${sel[1] - sel[0] + 1} frames · ${((sel[1] - sel[0] + 1) / fps).toFixed(1)} s`
              : 'Scrub to where the step starts and press Set start (I), then to where it ends and press Set end (O). Drag the flags on the timeline to fine-tune.'}
          </p>
        </div>
        {sel && (
          <button
            type="button"
            className={`btn icon${play.playing === selIdx ? ' on' : ''}`}
            onClick={() => play.setPlaying(play.playing === selIdx ? null : selIdx)}
            title="Play the selection"
            aria-label="Play the selection"
          >
            {play.playing === selIdx ? '■' : '▶'}
          </button>
        )}
      </div>
      {end(x.markIn, 'Start', () => x.setMarkIn(x.refFrame), () => x.setMarkIn(null))}
      {end(x.markOut, 'End', () => x.setMarkOut(x.refFrame), () => x.setMarkOut(null))}
      {sel && sel[1] - sel[0] + 1 < Math.round(fps) && (
        <p className="note warn">Under a second. The VLM looks at ~2 s at a time; a longer range describes the step better.</p>
      )}
      <div className="row gap">
        <button type="button" className="btn primary" disabled={!sel || !!x.cutting} onClick={() => void x.addMarked('step')}>
          {x.cutting ? `Cutting ${x.cutting.done}/${x.cutting.total}…` : 'Add as the step'}
        </button>
        <button
          type="button"
          className="btn"
          disabled={!sel || !!x.cutting}
          onClick={() => void x.addMarked('other')}
          title="A different step, or something easily mistaken for this one: shown to the VLM as a counter-example"
        >
          Add as another step
        </button>
      </div>
      {x.error && <p className="note danger">{x.error}</p>}
    </section>
  )
}

function ClipsCard({
  x, kind, play, playIdx,
}: { x: Segx; kind: ClipKind; play: Play; playIdx: (c: Clip) => number }) {
  const clips = kind === 'step' ? x.steps : x.others
  const step = kind === 'step'
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>
            {step ? 'The step' : 'Other steps'} <span className="count">{clips.length}</span>
            {!step && <span className="badge xopt">optional</span>}
          </h3>
          <p className="card-sub">
            {step
              ? clips.length
                ? 'What will be searched for. Each should show the step and nothing else.'
                : 'Nothing marked yet.'
              : 'Counter-examples: the steps just before and after it, or anything easily mistaken for it. In the research, these made the difference on look-alike steps.'}
          </p>
        </div>
      </div>
      {clips.length > 0 && (
        <ul className="cliplist">
          {clips.map((c, i) => (
            <ClipRow
              key={c.uid}
              c={c}
              i={i}
              label={`${step ? 'step' : 'other-step'} clip ${i + 1}`}
              current={c.src?.id === x.ref.info?.id}
              playing={play.playing !== null && play.playing === playIdx(c)}
              onSeek={() => c.src && x.setRefFrame(c.src.start)}
              onPlay={() => {
                const j = playIdx(c)
                play.setPlaying(play.playing === j ? null : j)
              }}
              onRemove={() => x.removeClip(c.uid)}
            />
          ))}
        </ul>
      )}
    </section>
  )
}

function ClipRow(p: {
  c: Clip
  i: number
  /** for screen readers: which list and number */
  label: string
  current: boolean
  playing: boolean
  onSeek: () => void
  onPlay: () => void
  onRemove: () => void
}) {
  const { c, i } = p
  const n = c.up.frames ?? 0
  const fps = c.up.fps || 20
  const thumb = <img src={api.frameUrl(c.up.upload_id, Math.floor(n / 2))} alt="" loading="lazy" />
  const text = (
    <span className="cliptext">
      <span className="clipname" title={c.src ? `${c.src.name}, frames ${c.src.start}–${c.src.end}` : c.up.name}>
        {i + 1}.{' '}
        {c.src ? (
          <>
            {clock(c.src.start, fps, true)} – {clock(c.src.end + 1, fps, true)}
          </>
        ) : (
          c.up.name
        )}
      </span>
      <span className="segmeta">
        {(n / fps).toFixed(1)} s · {n} frames
        {c.src ? (p.current ? '' : ` · ${c.src.name}`) : ' · uploaded clip'}
      </span>
    </span>
  )
  return (
    <li className="cliprow">
      {p.current ? (
        <button type="button" className="clipmain" onClick={p.onSeek} title="Go to its start">
          {thumb}
          {text}
        </button>
      ) : (
        <span className="clipmain">
          {thumb}
          {text}
        </span>
      )}
      {p.current && (
        <button
          type="button"
          className={`btn icon${p.playing ? ' on' : ''}`}
          onClick={p.onPlay}
          aria-label={p.playing ? 'Stop' : `Play ${p.label}`}
          title={p.playing ? 'Stop' : 'Play'}
        >
          {p.playing ? '■' : '▶'}
        </button>
      )}
      <button type="button" className="btn icon danger" onClick={p.onRemove} aria-label={`Remove ${p.label}`}>
        ✕
      </button>
    </li>
  )
}

// -- step 2 ------------------------------------------------------------------ //
function DescribePanel({ x }: { x: Segx }) {
  const job = x.describeJob
  const watched = Math.min(3, x.steps.length)
  return (
    <>
      <header className="xhead">
        <h2>Describe the step</h2>
        <p className="muted small">
          Qwen3-VL watches {watched === 1 ? 'the marked clip' : `${watched} of the marked clips`} and
          names the step in natural language. The description goes into the VLM's question for every
          clip it checks and, if ticked, into the embeddings. Correct it if it is wrong: the VLM
          describes what it sees, and can mistake the tool or the part.
        </p>
      </header>

      <section className="card">
        {job.busy ? (
          <Progress job={job} idleLabel="Waiting for the GPU" onCancel={job.cancel} />
        ) : (
          job.status === 'error' && <Progress job={job} />
        )}
        {x.describeStale && !job.busy && (
          <p className="note warn">
            The marked clips changed since this description was written.{' '}
            <button type="button" className="btn-text" onClick={x.describe}>
              Describe again
            </button>
          </p>
        )}
        <label className="field">
          Step name
          <input
            type="text"
            value={x.name}
            placeholder={job.busy ? 'The VLM is watching the clips…' : 'e.g. Install the corner rivnuts'}
            onChange={(e) => x.setName(e.target.value)}
          />
        </label>
        <label className="field">
          What it looks like
          <textarea
            className="xtext"
            rows={7}
            value={x.description}
            placeholder={job.busy ? '' : 'The state of the workpiece, the tool, where the hands work…'}
            onChange={(e) => x.setDescription(e.target.value)}
          />
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={x.useDescription}
            onChange={(e) => x.setUseDescription(e.target.checked)}
          />
          <span>
            Use the description in the embeddings too
            <em>
              On the research data this kept 95–100% of the step in the candidates, against 66–100% without
            </em>
          </span>
        </label>
        <button type="button" className="btn-text" onClick={x.describe} disabled={job.busy}>
          {x.edited ? 'Discard my edits and describe again' : 'Describe again'}
        </button>
      </section>

      <div className="xnav">
        <button type="button" className="btn ghost" onClick={() => x.goto('reference')}>
          ← Back to marking
        </button>
        <button type="button" className="btn primary big" disabled={job.busy} onClick={() => x.goto('search')}>
          {job.busy ? 'Describing…' : 'Next: search a video →'}
        </button>
      </div>
    </>
  )
}

// =========================================================================== //
// Step 3: the video to search on the left
// =========================================================================== //
function SearchView({ x, qwen }: { x: Segx; qwen?: QwenStatus }) {
  const [playing, setPlaying] = useState<number | null>(null)
  const play: Play = { playing, setPlaying }

  const segIdx = x.segments.findIndex((s) => x.frame >= s.start && x.frame <= s.end)
  const p = x.scored?.perFrame[x.frame]
  const inRegion = !!x.result?.region.some(([a, b]) => x.frame >= a && x.frame <= b)

  return (
    <div className="page extract">
      <VideoViewer
        video={x.video}
        local={x.target.local}
        onLocalFail={x.target.localFail}
        empty={
          <div className="xempty">
            <Dropzone
              accept="video/*"
              title="Drop the video to search"
              hint="Another recording of the station, to find the step in. MP4, MOV, AVI, MKV or WEBM, up to 500 MB"
              busy={x.videoBusy}
              onFile={(f) => void x.loadVideo(f)}
            />
            {x.videoError && <p className="note danger">{x.videoError}</p>}
          </div>
        }
        frame={x.frame}
        setFrame={x.setFrame}
        lastFrame={x.lastFrame}
        fps={x.fps}
        segments={x.segments}
        playbackUrl={x.playbackUrl}
        onPlaybackFail={x.playbackFail}
        playbackError={x.playbackError}
        play={play}
        marked={[]}
        badges={
          <>
            {segIdx >= 0 ? (
              <span className="xbadge in">Segment {segIdx + 1}</span>
            ) : x.scored ? (
              <span className="xbadge">Not the step</span>
            ) : null}
            {x.scored && (
              <span className="xbadge">
                {inRegion ? `P(step) ${(p ?? 0).toFixed(2)}` : 'not checked by the VLM'}
              </span>
            )}
          </>
        }
        timeline={(seek) =>
          x.scored && x.result ? (
            <Timeline
              score={x.scored.score}
              stride={x.result.stride}
              total={x.lastFrame + 1}
              fps={x.fps}
              threshold={x.threshold}
              range={[0, 1]}
              bands={x.result.region}
              scoreLabel="P(step)"
              segments={x.segments}
              excluded={x.excluded}
              refFrames={[]}
              frame={x.frame}
              onSeek={seek}
              onThreshold={(v) => x.setThreshold(Math.max(0.05, Math.min(0.95, v)))}
            />
          ) : (
            <div className="timeline placeholder">
              {x.searchJob.busy
                ? 'Searching… the step probability appears here when the VLM is done.'
                : 'Run the search to see where the step happens. Shaded stretches are the ones the VLM checked.'}
            </div>
          )
        }
      />

      <aside className="panel">
        <header className="xhead">
          <h2>Search a video</h2>
          <p className="muted small">
            Upload another recording. The step is found in it and its start and end are refined clip
            by clip with the vision-language model. Replace the video to search the next one; the
            marked step and its description stay.
          </p>
        </header>
        <QueryCard x={x} />
        <TargetCard x={x} />
        <RunCard x={x} qwen={qwen} />
        {x.scored && <ResultsCard x={x} play={play} />}
        {x.scored && x.segments.length > 0 && <ExportCard x={x} />}
        <AdvancedCard x={x} />
      </aside>
    </div>
  )
}

function QueryCard({ x }: { x: Segx }) {
  return (
    <section className="card query">
      <div className="card-head">
        <div>
          <h3>{x.name || 'The marked step'}</h3>
          <p className="card-sub">
            {x.steps.length} step clip{x.steps.length === 1 ? '' : 's'}
            {x.others.length > 0 && ` · ${x.others.length} other-step clip${x.others.length === 1 ? '' : 's'}`}
            {x.description ? '' : ' · no description'}
          </p>
        </div>
        <span className="row gap-sm">
          <button type="button" className="btn-text" onClick={() => x.goto('reference')}>
            Clips
          </button>
          <button type="button" className="btn-text" onClick={() => x.goto('describe')}>
            Description
          </button>
        </span>
      </div>
      {x.description && <p className="query-desc">{x.description}</p>}
    </section>
  )
}

function TargetCard({ x }: { x: Segx }) {
  const video = x.target.info
  if (!video) return null
  const n = x.lastFrame + 1
  const up = x.target.uploadPct
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3 className="xtitle" title={video.name}>
            {video.name || 'Video'}
          </h3>
          <p className="card-sub">
            {video.width} × {video.height} · {n} frames · {video.fps} fps · {clock(n, x.fps)}
            {x.result && video.frames && x.result.decoded_frames < video.frames && (
              <> · only {x.result.decoded_frames} of {video.frames} frames decode</>
            )}
          </p>
          {up !== null && (
            <div className="xupload">
              <div className="xupload-bar">
                <span style={{ width: `${Math.round(up * 100)}%` }} />
              </div>
              <span className="muted small">
                Uploading for the GPU {Math.round(up * 100)}%
                {x.target.local ? ' · you can already play and scrub it' : ''}
              </span>
            </div>
          )}
          {x.target.error && up === null && !x.video && (
            <p className="note danger">
              {x.target.error}{' '}
              <button type="button" className="btn-text" onClick={x.target.retryUpload}>
                Retry upload
              </button>
            </p>
          )}
        </div>
        <label className="btn-text xreplace">
          Search another
          <input
            type="file"
            accept="video/*"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void x.loadVideo(f)
              e.target.value = ''
            }}
          />
        </label>
      </div>
    </section>
  )
}

function RunCard({ x, qwen }: { x: Segx; qwen?: QwenStatus }) {
  const { searchJob: job, video } = x
  if (job.busy) {
    return (
      <section className="card accent">
        <Progress job={job} idleLabel="Waiting for the GPU" onCancel={job.cancel} />
        <p className="muted small">
          Runs on GPU {qwen?.gpu ?? 0}: the embedding model, then the VLM (they take turns on the
          card). Object tracking on GPU 1 carries on as normal.
        </p>
      </section>
    )
  }
  if (!x.target.info) return <p className="muted small">Upload the video to search on the left.</p>
  const ready = !!video && x.steps.length > 0 && !x.describeJob.busy
  const n = x.lastFrame + 1
  const clipFrames = x.steps.concat(x.others).reduce((a, c) => a + (c.up.frames ?? 0), 0)
  const knn = x.candidates === 'knn'
  const marked = x.steps.some((c) => c.src)
  // kNN: the reference videos are embedded too, and the VLM checks roughly what it keeps
  const est =
    2 * LOAD_S + (n + clipFrames) / x.stride / EMBED_FPS + (knn ? n / x.stride / EMBED_FPS : 0) +
    (Math.min(1, knn ? 0.3 : x.coverage * 1.5) * n) / x.stride / CLIPS_PER_S
  const up = x.refUpload
  return (
    <section className={`card${x.result && !x.stale ? '' : ' accent'}`}>
      {job.status === 'error' && <Progress job={job} />}
      {x.stale && <p className="note warn">The inputs changed since this result. Run again to update it.</p>}
      <div className="field">
        <span>What the VLM checks</span>
        <div className="switch" role="radiogroup" aria-label="Candidate stretches">
          <button
            type="button"
            role="radio"
            aria-checked={!knn}
            className={knn ? '' : 'on'}
            onClick={() => x.setCandidates('similarity')}
            title="The stretches most similar to the marked clips (top share set under Advanced)"
          >
            Most similar
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={knn}
            className={knn ? 'on' : ''}
            onClick={() => x.setCandidates('knn')}
            disabled={!marked}
            title={marked
              ? 'The research pipeline (match_frames.py --balanced): a kNN vote of the marked step frames against the rest of the reference videos'
              : 'Needs the step marked on a reference video, not only uploaded clips'}
          >
            kNN, 2 groups (research)
          </button>
        </div>
      </div>
      {knn && (
        <p className="muted small">
          Marked ranges are the step; every other frame of the reference video
          {new Set(x.steps.concat(x.others).flatMap((c) => (c.src ? [c.src.id] : []))).size > 1 ? 's' : ''} is
          not. Each frame of this video takes the vote of its {x.knnK} most similar reference frames
          (balanced for how rare the step is), and the stretches that vote "step" go to the VLM. The
          reference video{x.steps.length > 1 ? 's are' : ' is'} uploaded and embedded for this. If the step
          recurs in the reference, mark every occurrence: an unmarked one counts as "not the step".
        </p>
      )}
      {up && (
        <div className="xupload">
          <div className="xupload-bar">
            <span style={{ width: `${Math.round(up.pct * 100)}%` }} />
          </div>
          <span className="muted small">
            Uploading reference video {up.i} of {up.n}: {Math.round(up.pct * 100)}%
          </span>
        </div>
      )}
      {x.step === 'search' && x.error && <p className="note danger">{x.error}</p>}
      <button type="button" className="btn primary big" disabled={!ready || !!up} onClick={() => void x.run()}>
        {up ? 'Uploading the reference…' : !x.steps.length
          ? 'Mark the step first'
          : !video
            ? x.target.uploadPct !== null
              ? `Uploading ${Math.round(x.target.uploadPct * 100)}%…`
              : 'Waiting for the upload'
            : x.describeJob.busy
            ? 'Waiting for the description…'
            : x.result
              ? 'Search again'
              : 'Find the step'}
      </button>
      <p className="muted small">
        Up to about {duration(est)} (less when the videos are already embedded): embed every{' '}
        {x.stride}th frame,{' '}
        {knn
          ? `keep the stretches the kNN vote calls the step (threshold ${x.knnThreshold})`
          : `pick the ${Math.round(x.coverage * 100)}% of the video most like the clips`}
        , then have the VLM check a 2-second clip every {x.stride} frames there.
      </p>
    </section>
  )
}

function ResultsCard({ x, play }: { x: Segx; play: Play }) {
  const { segments, fps } = x
  const frames = segments.reduce((a, s) => a + s.end - s.start + 1, 0)
  const total = x.lastFrame + 1
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>
            Segments <span className="count">{segments.length}</span>
          </h3>
          <p className="card-sub">
            {frames} frames · {(frames / fps).toFixed(1)} s · {((100 * frames) / total).toFixed(1)}% of the
            video · the VLM checked {Math.round((x.result?.coverage ?? 0) * 100)}%
            {x.result?.knn &&
              ` (kNN candidates: ${x.result.knn.step_frames} step vs ${x.result.knn.other_frames} other reference frames)`}
          </p>
        </div>
      </div>
      <Slider
        label="Step probability threshold"
        help="Frames whose smoothed P(step) from the VLM reaches this count as the step. Drag the dashed line on the timeline to do the same."
        value={x.threshold}
        min={0.05}
        max={0.95}
        step={0.05}
        format={(v) => v.toFixed(2) + (v === SEGX_DEFAULTS.threshold ? ' (default)' : '')}
        minLabel="Wider segments"
        maxLabel="Only the surest"
        onChange={x.setThreshold}
      />
      {x.excluded.length > 0 && (
        <p className="muted small">
          {x.excluded.length} range{x.excluded.length > 1 ? 's' : ''} left out by hand.{' '}
          <button type="button" className="btn-text" onClick={x.clearExcluded}>
            Put them back
          </button>
        </p>
      )}
      <SegmentList
        segments={segments}
        active={[]}
        refIndex={() => 0}
        videoId={x.video?.upload_id}
        fps={fps}
        frame={x.frame}
        playing={play.playing}
        onSeek={(f) => {
          play.setPlaying(null)
          x.setFrame(f)
        }}
        onPlay={(i) => play.setPlaying(play.playing === i ? null : i)}
        onExclude={x.exclude}
      />
    </section>
  )
}

function AdvancedCard({ x }: { x: Segx }) {
  return (
    <details className="card options">
      <summary>Advanced</summary>
      {x.candidates === 'knn' && (
        <>
          <Slider
            label="kNN: neighbours (k)"
            help="How many of the most similar reference frames vote. The research used 15."
            value={x.knnK}
            min={3}
            max={50}
            step={1}
            format={(v) => `${v}${v === 15 ? ' (research)' : ''}`}
            onChange={x.setKnnK}
          />
          <Slider
            label="kNN: vote threshold"
            help="The balanced share of 'step' votes a stretch needs (after ~1 s smoothing) to go to the VLM. The research used 0.5; lower sends more of the video."
            value={x.knnThreshold}
            min={0.1}
            max={0.9}
            step={0.05}
            format={(v) => v.toFixed(2) + (v === 0.5 ? ' (research)' : '')}
            onChange={x.setKnnThreshold}
          />
        </>
      )}
      {x.candidates === 'similarity' && <Slider
        label="VLM coverage"
        help="The share of the video, ranked by embedding similarity to the clips, that the VLM checks (plus 2 s either side). 25% kept 95–100% of the step on the research data; raise it if a whole occurrence is missing."
        value={x.coverage}
        min={0.1}
        max={1}
        step={0.05}
        format={(v) => (v >= 1 ? 'the whole video' : `${Math.round(v * 100)}%`)}
        onChange={x.setCoverage}
      />}
      <Slider
        label="Smoothing"
        help="P(step) is averaged over this window before thresholding."
        value={x.smoothSec}
        min={0}
        max={5}
        step={0.05}
        format={(v) => (v === 0 ? 'off' : `${v.toFixed(2)} s`)}
        onChange={x.setSmoothSec}
      />
      <Slider
        label="Shortest segment"
        help="Stretches shorter than this are dropped."
        value={x.minSegSec}
        min={0}
        max={10}
        step={0.1}
        format={(v) => `${v.toFixed(1)} s · ${Math.round(v * x.fps)} frames`}
        onChange={x.setMinSegSec}
      />
      <Slider
        label="Merge gaps up to"
        help="Two stretches separated by less than this become one segment."
        value={x.gapSec}
        min={0}
        max={20}
        step={0.25}
        format={(v) => `${v.toFixed(2)} s · ${Math.round(v * x.fps)} frames`}
        onChange={x.setGapSec}
      />
      <label className="field">
        Embed and check every Nth frame
        <select value={x.stride} onChange={(e) => x.setStride(Number(e.target.value))}>
          {[2, 3, 5, 10].map((s) => (
            <option key={s} value={s}>
              every {s}th frame{s === 5 ? ' (tested default)' : ''}
            </option>
          ))}
        </select>
      </label>
      <p className="muted small">
        Coverage, stride and the description change what runs on the GPU; the rest applies
        instantly.
      </p>
    </details>
  )
}
