import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type HealthConfig, type QwenStatus } from '../api/client'
import { Dropzone } from '../components/Dropzone'
import { Progress } from '../components/Progress'
import { Slider } from '../components/Slider'
import { Stepper } from '../components/Stepper'
import {
  BG_COLOR, MAX_CLASSES, MAX_TARGETS, MC_COLORS, MC_STEPS, useMcseg, type McClass, type Mcseg,
} from '../lib/mcseg'
import { SEGX_DEFAULTS } from '../lib/segx'
import type { Segment } from '../lib/segments'
import { ExportCard } from './ExportCard'
import { clock } from './format'
import { MarkTimeline } from './MarkTimeline'
import { McTimeline } from './McTimeline'
import { VideoViewer, type Play } from './VideoViewer'
import './extract.css'
import './mcseg.css'

// as SegxPage: ~45 s to load each Qwen model, ~12 frames/s to embed, ~4 clips/s to classify
const LOAD_S = 45
const EMBED_FPS = 12
const CLIPS_PER_S = 4

function duration(s: number): string {
  return s < 90 ? `${Math.max(1, Math.round(s / 10) * 10)} s` : `${Math.round(s / 60)} min`
}

const color = (i: number) => MC_COLORS[i % MC_COLORS.length]
const label = (c: McClass, i: number) => c.name.trim() || `Step ${i + 1}`

export function McsegPage({ cfg, qwen }: { cfg: HealthConfig; qwen?: QwenStatus }) {
  const x = useMcseg(cfg)

  if (!cfg.extract?.available) {
    return (
      <main>
        <div className="page narrow">
          <p className="note danger">
            Multiple class segmentation needs the Qwen vLLM environment, and the backend could not find it.
            Point <code>SAM3_QWEN_PYTHON</code> at its python and restart the backend.
          </p>
        </div>
      </main>
    )
  }

  return (
    <>
      <div className="steprow">
        <Stepper steps={MC_STEPS} current={x.step} reached={x.reached} onGo={x.goto} />
      </div>
      <main>{x.step === 'search' ? <SearchView x={x} qwen={qwen} /> : <ReferenceView x={x} />}</main>
    </>
  )
}

function Chip({ i, bg }: { i: number; bg?: boolean }) {
  return (
    <span className="mcchip" style={{ background: bg ? BG_COLOR : color(i) }} aria-hidden>
      {bg ? '–' : i + 1}
    </span>
  )
}

// =========================================================================== //
// Steps 1 and 2: the reference video on the left
// =========================================================================== //
function ReferenceView({ x }: { x: Mcseg }) {
  const [playing, setPlaying] = useState<number | null>(null)
  const play: Play = { playing, setPlaying }
  const pageRef = useRef<HTMLDivElement>(null)
  const fps = x.refFps

  // the viewer plays the classes (in their order), then the selection
  const segs = useMemo(() => {
    const all: Segment[] = x.classes.map((c, i) => ({
      start: c.start, end: c.end, peak: 1, peakFrame: c.start, mean: 1, bestRef: i,
    }))
    if (x.selection) {
      const [start, end] = x.selection
      all.push({ start, end, peak: 1, peakFrame: start, mean: 1, bestRef: -1 })
    }
    return all
  }, [x.classes, x.selection])
  const selIdx = x.selection ? segs.length - 1 : -1

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
        if (t.getAttribute('role') === 'slider') return
        setPlaying(null)
        setRefFrame(refFrame + (e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 10 : 1))
      } else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [step, refFrame, setMarkIn, setMarkOut, setRefFrame])

  const here = x.classes.findIndex((c) => x.refFrame >= c.start && x.refFrame <= c.end)
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
              hint="A recording in which the steps happen. You will mark one range per step. MP4, MOV, AVI, MKV or WEBM, up to 500 MB"
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
            {here >= 0 && (
              <span className="xbadge mcbadge" style={{ background: color(here), borderColor: color(here) }}>
                {here + 1}. {label(x.classes[here], here)}
              </span>
            )}
            {here < 0 && x.classes.length > 0 && <span className="xbadge">Background</span>}
            {inSel && <span className="xbadge sel">Selection</span>}
          </>
        }
        tools={() =>
          x.step === 'reference' && (
            <>
              <button type="button" className="btn" onClick={() => x.setMarkIn(x.refFrame)} title="Set the start of the selection to this frame (I)">
                ⟦ Set start <kbd>I</kbd>
              </button>
              <button type="button" className="btn" onClick={() => x.setMarkOut(x.refFrame)} title="Set the end of the selection to this frame (O)">
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
            steps={[]}
            others={[]}
            colored={x.classes.map((c, i) => ({ start: c.start, end: c.end, color: color(i) }))}
            markIn={x.step === 'reference' ? x.markIn : null}
            markOut={x.step === 'reference' ? x.markOut : null}
            onSeek={seek}
            onMarkIn={x.setMarkIn}
            onMarkOut={x.setMarkOut}
          />
        )}
      />

      <aside className="panel">
        {x.step === 'reference' ? <MarkPanel x={x} play={play} selIdx={selIdx} /> : <NamePanel x={x} />}
      </aside>
    </div>
  )
}

// -- step 1 ------------------------------------------------------------------ //
function MarkPanel({ x, play, selIdx }: { x: Mcseg; play: Play; selIdx: number }) {
  const v = x.ref.info
  const fps = x.refFps
  const up = x.ref.uploadPct
  return (
    <>
      <header className="xhead">
        <h2>Mark the steps</h2>
        <p className="muted small">
          On one reference video, mark one occurrence of each step you want to find: where it starts and
          where it ends. Every frame you do not mark is <b>background</b>, the class that keeps idle time,
          fetching parts and the other steps apart from yours, so mark each step fully and leave the rest.
        </p>
      </header>

      {v && (
        <section className="card">
          <div className="card-head">
            <div>
              <h3 className="xtitle" title={v.name}>{v.name || 'Reference video'}</h3>
              <p className="card-sub">
                {v.width} × {v.height} · {v.frames} frames · {v.fps} fps · {clock(v.frames, fps)}
              </p>
              {up !== null && (
                <div className="xupload">
                  <div className="xupload-bar"><span style={{ width: `${Math.round(up * 100)}%` }} /></div>
                  <span className="muted small">
                    Uploading for the search {Math.round(up * 100)}% · you can mark steps already
                  </span>
                </div>
              )}
            </div>
            <label className="btn-text xreplace" title="The marked steps belong to this video and are cleared">
              Use another video
              <input
                type="file"
                accept="video/*"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f && (!x.classes.length || window.confirm('Another reference clears the steps marked on this one. Go on?')))
                    void x.loadRef(f)
                  e.target.value = ''
                }}
              />
            </label>
          </div>
        </section>
      )}

      {v && <SelectionCard x={x} play={play} selIdx={selIdx} />}
      {v && <StepsCard x={x} play={play} />}

      <div className="xnav">
        <span />
        <button type="button" className="btn primary big" disabled={!x.classes.length} onClick={() => x.goto('describe')}>
          {x.classes.length ? 'Next: name the steps →' : 'Mark a step first'}
        </button>
      </div>
    </>
  )
}

function SelectionCard({ x, play, selIdx }: { x: Mcseg; play: Play; selIdx: number }) {
  const fps = x.refFps
  const sel = x.selection
  const [target, setTarget] = useState('')
  const hit = sel ? x.overlapping(sel[0], sel[1], target || undefined) : undefined
  const full = x.classes.length >= MAX_CLASSES
  const end = (f: number | null, name: string, set: () => void, clearIt: () => void) => (
    <div className="markend">
      <span className="markend-label">{name}</span>
      {f === null ? (
        <span className="muted">not set</span>
      ) : (
        <button type="button" className="btn-text markend-time" onClick={() => x.setRefFrame(f)} title="Go there">
          {clock(f, fps, true)} <em>#{f}</em>
        </button>
      )}
      <span className="row gap-sm">
        <button type="button" className="btn small" onClick={set}>Set here</button>
        {f !== null && (
          <button type="button" className="btn icon small" onClick={clearIt} aria-label={`Clear the ${name.toLowerCase()}`}>✕</button>
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
              : 'Scrub to where a step starts and press Set start (I), then to where it ends and press Set end (O).'}
          </p>
        </div>
        {sel && (
          <button
            type="button"
            className={`btn icon${play.playing === selIdx ? ' on' : ''}`}
            onClick={() => play.setPlaying(play.playing === selIdx ? null : selIdx)}
            aria-label="Play the selection"
          >
            {play.playing === selIdx ? '■' : '▶'}
          </button>
        )}
      </div>
      {end(x.markIn, 'Start', () => x.setMarkIn(x.refFrame), () => x.setMarkIn(null))}
      {end(x.markOut, 'End', () => x.setMarkOut(x.refFrame), () => x.setMarkOut(null))}
      {sel && sel[1] - sel[0] + 1 < Math.round(1.5 * fps) && (
        <p className="note warn">
          Under 1.5 s. The VLM looks at ~2 s at a time and every 5th frame is embedded: a longer range
          shows the step better.
        </p>
      )}
      {hit && <p className="note warn">Overlaps step “{hit.name || 'unnamed'}”: each frame can belong to one step only.</p>}
      <label className="field">
        {target ? 'This range becomes the new range of' : 'Name of the new step'}
        {x.classes.length > 0 && (
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="">a new step</option>
            {x.classes.map((c, i) => (
              <option key={c.uid} value={c.uid}>replace the range of {i + 1}. {label(c, i)}</option>
            ))}
          </select>
        )}
        {!target && (
          <input
            type="text"
            value={x.newName}
            placeholder="optional: leave empty and the VLM names it"
            onChange={(e) => x.setNewName(e.target.value)}
          />
        )}
      </label>
      <button
        type="button"
        className="btn primary"
        disabled={!sel || !!x.cutting || !!hit || (!target && full)}
        onClick={() => void x.markClass(target || undefined).then(() => setTarget(''))}
      >
        {x.cutting
          ? `Cutting ${x.cutting.done}/${x.cutting.total}…`
          : target
            ? 'Replace its range'
            : full
              ? `At most ${MAX_CLASSES} steps`
              : `Add as step ${x.classes.length + 1}`}
      </button>
      {x.error && <p className="note danger">{x.error}</p>}
    </section>
  )
}

function StepsCard({ x, play }: { x: Mcseg; play: Play }) {
  const fps = x.refFps
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>Steps <span className="count">{x.classes.length}</span></h3>
          <p className="card-sub">
            {x.classes.length
              ? 'One range each. Everything unmarked is background.'
              : 'Nothing marked yet.'}
          </p>
        </div>
      </div>
      {x.classes.length > 0 && (
        <ul className="cliplist">
          {x.classes.map((c, i) => (
            <li key={c.uid} className="cliprow">
              <button type="button" className="clipmain" onClick={() => x.setRefFrame(c.start)} title="Go to its start">
                <img src={api.frameUrl(c.clip.upload_id, Math.floor((c.end - c.start) / 2))} alt="" loading="lazy" />
                <span className="cliptext">
                  <span className="clipname"><Chip i={i} /> {label(c, i)}</span>
                  <span className="segmeta">
                    {clock(c.start, fps, true)} – {clock(c.end + 1, fps, true)} · {((c.end - c.start + 1) / fps).toFixed(1)} s
                    {c.naming === 'vlm' && !c.name ? ' · the VLM names it' : ''}
                  </span>
                </span>
              </button>
              <button
                type="button"
                className={`btn icon${play.playing === i ? ' on' : ''}`}
                onClick={() => play.setPlaying(play.playing === i ? null : i)}
                aria-label={play.playing === i ? 'Stop' : `Play step ${i + 1}`}
              >
                {play.playing === i ? '■' : '▶'}
              </button>
              <button type="button" className="btn icon danger" onClick={() => x.removeClass(c.uid)} aria-label={`Remove step ${i + 1}`}>
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

// -- step 2 ------------------------------------------------------------------ //
function NamePanel({ x }: { x: Mcseg }) {
  const job = x.describeJob
  return (
    <>
      <header className="xhead">
        <h2>Name the steps</h2>
        <p className="muted small">
          Name each step yourself, or let Qwen3-VL watch its clip and name it. The VLM also describes
          every step you leave without a description. Names and descriptions become the VLM's options
          when it classifies each clip and, if ticked, go into the embeddings. Correct them where they
          are wrong: the VLM describes what it sees and can mistake the part.
        </p>
      </header>

      {(job.busy || job.status === 'error') && (
        <section className="card">
          <Progress job={job} idleLabel="Waiting for the GPU" onCancel={job.busy ? job.cancel : undefined} />
        </section>
      )}

      {x.classes.map((c, i) => (
        <section className="card mcclass" key={c.uid} style={{ borderLeftColor: color(i) }}>
          <div className="card-head">
            <h3><Chip i={i} /> Step {i + 1}</h3>
            <div className="switch" role="radiogroup" aria-label={`How step ${i + 1} is named`}>
              <button
                type="button"
                role="radio"
                aria-checked={c.naming === 'mine'}
                className={c.naming === 'mine' ? 'on' : ''}
                onClick={() => x.updateClass(c.uid, { naming: 'mine' })}
              >
                My name
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={c.naming === 'vlm'}
                className={c.naming === 'vlm' ? 'on' : ''}
                onClick={() => x.updateClass(c.uid, { naming: 'vlm' })}
              >
                VLM names it
              </button>
            </div>
          </div>
          <label className="field">
            Name
            <input
              type="text"
              value={c.name}
              placeholder={c.naming === 'vlm' ? (job.busy ? 'The VLM is watching the clip…' : 'The VLM names it') : 'e.g. Place the manual'}
              onChange={(e) => x.updateClass(c.uid, { name: e.target.value, naming: 'mine' })}
            />
          </label>
          <label className="field">
            What it looks like {c.descBy === 'vlm' && <span className="badge">by the VLM</span>}
            <textarea
              className="xtext"
              rows={4}
              value={c.description}
              placeholder={job.busy ? '' : 'optional: the part, where the hands work, what visibly changes…'}
              onChange={(e) => x.updateClass(c.uid, { description: e.target.value, descBy: 'mine' })}
            />
          </label>
        </section>
      ))}

      <section className="card">
        <div className="row gap">
          <button type="button" className="btn" disabled={job.busy || !x.undescribed} onClick={() => x.describe()}>
            {x.undescribed ? `Have the VLM fill in ${x.undescribed} step${x.undescribed > 1 ? 's' : ''}` : 'Every step is named and described'}
          </button>
          <button type="button" className="btn-text" disabled={job.busy} onClick={() => x.describe(true)}>
            Describe all again
          </button>
        </div>
        <label className="check">
          <input type="checkbox" checked={x.useDescription} onChange={(e) => x.setUseDescription(e.target.checked)} />
          <span>
            Use the descriptions in the embeddings too
            <em>All steps go into one embedding instruction, so each video is embedded once</em>
          </span>
        </label>
      </section>

      <div className="xnav">
        <button type="button" className="btn ghost" onClick={() => x.goto('reference')}>← Back to marking</button>
        <button type="button" className="btn primary big" disabled={job.busy} onClick={() => x.goto('search')}>
          {job.busy ? 'Describing…' : 'Next: search videos →'}
        </button>
      </div>
    </>
  )
}

// =========================================================================== //
// Step 3: the active video to search on the left
// =========================================================================== //
function SearchView({ x, qwen }: { x: Mcseg; qwen?: QwenStatus }) {
  const [playing, setPlaying] = useState<number | null>(null)
  const play: Play = { playing, setPlaying }
  const t = x.target
  const names = x.result?.classes ?? x.classes.map(label)
  const nCls = names.length
  const lab = x.sel?.label[x.frame] ?? -1
  const segIdx = x.segments.findIndex((s) => x.frame >= s.start && x.frame <= s.end)
  const pOf = (k: number) => (x.sel ? x.sel.perFrame[x.frame * x.sel.letters + k] : 0)

  // a new tab: stop what the old one was playing
  const { active } = x
  useEffect(() => setPlaying(null), [active])

  return (
    <div className="page extract">
      <VideoViewer
        key={x.active}
        video={t.video}
        local={t.local}
        onLocalFail={t.localFail}
        empty={
          <div className="xempty">
            <Dropzone
              accept="video/*"
              title={`Drop video ${x.active + 1} to search`}
              hint={`Up to ${MAX_TARGETS} recordings of the station, searched together. MP4, MOV, AVI, MKV or WEBM, up to 500 MB`}
              busy={t.busy}
              onFile={(f) => void x.loadTarget(x.active, f)}
            />
            {t.error && <p className="note danger">{t.error}</p>}
          </div>
        }
        frame={x.frame}
        setFrame={x.setFrame}
        lastFrame={x.lastFrame}
        fps={x.fps}
        segments={x.segments}
        playbackUrl={t.playbackUrl}
        onPlaybackFail={t.playbackFail}
        playbackError={t.playbackError}
        play={play}
        marked={[]}
        badges={
          x.sel && (
            <>
              {segIdx >= 0 ? (
                <span className="xbadge mcbadge" style={{ background: color(x.segments[segIdx].bestRef), borderColor: color(x.segments[segIdx].bestRef) }}>
                  {x.segments[segIdx].bestRef + 1}. {names[x.segments[segIdx].bestRef]}
                </span>
              ) : (
                <span className="xbadge">{lab < 0 ? 'Not checked by the VLM' : 'No step'}</span>
              )}
              {lab >= 0 && (
                <span className="xbadge">
                  {lab < nCls ? `P(${lab + 1}) ${pOf(lab).toFixed(2)}` : `P(anything else) ${pOf(nCls).toFixed(2)}`}
                </span>
              )}
            </>
          )
        }
        timeline={(seek) =>
          x.sel && x.shown ? (
            <McTimeline
              curves={x.sel.curves}
              names={names}
              colors={names.map((_, i) => color(i))}
              stride={x.shown.stride}
              total={x.lastFrame + 1}
              fps={x.fps}
              threshold={x.threshold}
              segments={x.segments}
              bands={x.shown.region}
              excluded={x.excluded}
              frame={x.frame}
              onSeek={seek}
            />
          ) : (
            <div className="timeline placeholder">
              {x.searchJob.busy
                ? 'Searching… one lane per step appears here when the VLM is done.'
                : x.result && t.info && !x.shown
                  ? 'This video was not part of the last search. Run it again to include it.'
                  : 'Run the search to see where each step happens. Shaded stretches are the ones the VLM checked.'}
            </div>
          )
        }
      />

      <aside className="panel">
        <header className="xhead">
          <h2>Search videos</h2>
          <p className="muted small">
            Add up to {MAX_TARGETS} recordings; they are searched together. For each step, the frames most
            like its marked range are checked clip by clip by the vision-language model, which picks one
            of the steps or “anything else”.
          </p>
        </header>
        <QueryCard x={x} names={names} />
        <TargetsCard x={x} />
        <RunCard x={x} qwen={qwen} />
        {x.sel && <ResultsCard x={x} play={play} names={names} />}
        {x.sel && x.segments.length > 0 && <ExportCard x={x} />}
        <AdvancedCard x={x} />
      </aside>
    </div>
  )
}

function QueryCard({ x, names }: { x: Mcseg; names: string[] }) {
  return (
    <section className="card query">
      <div className="card-head">
        <div>
          <h3>{x.classes.length} step{x.classes.length === 1 ? '' : 's'} + background</h3>
          <ul className="mclegend">
            {names.map((n, i) => (
              <li key={i}><Chip i={i} /> {n}</li>
            ))}
            <li><Chip i={0} bg /> anything else</li>
          </ul>
        </div>
        <span className="row gap-sm">
          <button type="button" className="btn-text" onClick={() => x.goto('reference')}>Ranges</button>
          <button type="button" className="btn-text" onClick={() => x.goto('describe')}>Names</button>
        </span>
      </div>
    </section>
  )
}

function TargetsCard({ x }: { x: Mcseg }) {
  // the filled slots, then one empty one to add to
  const shown = x.slots.map((s, i) => ({ s, i })).filter(({ s, i }) =>
    s.info || i === x.active || i === x.slots.findIndex((z) => !z.info))
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>Videos <span className="count">{x.loaded}/{MAX_TARGETS}</span></h3>
          <p className="card-sub">Click one to view it. A search covers all of them.</p>
        </div>
      </div>
      <ul className="mctargets">
        {shown.map(({ s, i }) => {
          const v = s.info
          const n = s.video ? x.counts[s.video.upload_id] : undefined
          return (
            <li key={i} className={`mctarget${i === x.active ? ' on' : ''}`}>
              <button type="button" className="mctarget-main" onClick={() => x.setActive(i)}>
                <b>{i + 1}.</b>{' '}
                {v ? (
                  <span className="mctarget-text">
                    <span className="clipname" title={v.name}>{v.name}</span>
                    <span className="segmeta">
                      {clock(v.frames, v.fps)}
                      {s.uploadPct !== null ? ` · uploading ${Math.round(s.uploadPct * 100)}%` : ''}
                      {s.error && !s.video ? ' · upload failed' : ''}
                      {n !== undefined ? ` · ${n} segment${n === 1 ? '' : 's'}` : ''}
                    </span>
                  </span>
                ) : (
                  <span className="muted">{i === x.active ? 'drop a video on the left' : 'add a video'}</span>
                )}
              </button>
              <label className="btn-text xreplace">
                {v ? 'Replace' : 'Choose'}
                <input
                  type="file"
                  accept="video/*"
                  hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) void x.loadTarget(i, f)
                    e.target.value = ''
                  }}
                />
              </label>
              {s.error && !s.video && v && (
                <button type="button" className="btn-text" onClick={s.retryUpload}>Retry</button>
              )}
            </li>
          )
        })}
      </ul>
    </section>
  )
}

function RunCard({ x, qwen }: { x: Mcseg; qwen?: QwenStatus }) {
  const job = x.searchJob
  if (job.busy) {
    return (
      <section className="card accent">
        <Progress job={job} idleLabel="Waiting for the GPU" onCancel={job.cancel} />
        <p className="muted small">
          Runs on GPU {qwen?.gpu ?? 0}: the embedding model, then the VLM for each video. Object tracking
          on GPU 1 carries on as normal.
        </p>
      </section>
    )
  }
  const knn = x.candidates === 'knn'
  const frames = x.slots.reduce((a, s) => a + (s.info?.frames ?? 0), 0)
  const refFrames = x.ref.info?.frames ?? 0
  const share = knn ? 0.3 : Math.min(1, x.coverage * 1.5 * Math.max(1, x.classes.length * 0.7))
  const est = 2 * LOAD_S + (frames + refFrames) / x.stride / EMBED_FPS + (share * frames) / x.stride / CLIPS_PER_S
  const ready = !!x.ref.video && x.classes.length > 0 && x.loaded > 0 && !x.uploadingTargets && !x.describeJob.busy
  return (
    <section className={`card${x.result && !x.stale ? '' : ' accent'}`}>
      {job.status === 'error' && <Progress job={job} />}
      {x.stale && <p className="note warn">The steps, videos or settings changed since this result. Run again to update it.</p>}
      <div className="field">
        <span>What the VLM checks</span>
        <div className="switch" role="radiogroup" aria-label="Candidate stretches">
          <button
            type="button"
            role="radio"
            aria-checked={knn}
            className={knn ? 'on' : ''}
            onClick={() => x.setCandidates('knn')}
            title="A kNN vote over the steps plus the background of the reference"
          >
            kNN, {x.classes.length + 1} groups
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={!knn}
            className={knn ? '' : 'on'}
            onClick={() => x.setCandidates('similarity')}
            title="Per step, the stretches most similar to its marked range (top share set under Advanced)"
          >
            Most similar
          </button>
        </div>
      </div>
      <p className="muted small">
        {knn
          ? `Each marked range is a group and every other frame of the reference is the background group. Each frame of the videos takes the vote of its ${x.knnK} most similar reference frames (balanced for how short each group is); the stretches where a step's share reaches ${x.knnThreshold} go to the VLM.`
          : `For each step, the ${Math.round(x.coverage * 100)}% of each video most like its marked range (plus 2 s either side) goes to the VLM.`}
      </p>
      {x.step === 'search' && x.error && <p className="note danger">{x.error}</p>}
      <button type="button" className="btn primary big" disabled={!ready} onClick={x.run}>
        {!x.classes.length
          ? 'Mark the steps first'
          : !x.ref.video
            ? x.ref.uploadPct !== null
              ? `Uploading the reference ${Math.round(x.ref.uploadPct * 100)}%…`
              : 'The reference is not uploaded'
            : !x.loaded
              ? 'Add a video to search'
              : x.uploadingTargets
                ? 'Waiting for the uploads…'
                : x.describeJob.busy
                  ? 'Waiting for the descriptions…'
                  : x.result
                    ? `Search ${x.loaded} video${x.loaded > 1 ? 's' : ''} again`
                    : `Find the steps in ${x.loaded} video${x.loaded > 1 ? 's' : ''}`}
      </button>
      {x.loaded > 0 && (
        <p className="muted small">
          Up to about {duration(est)} (less when the videos are already embedded): embed every {x.stride}th
          frame of the reference and the videos, pick the candidates, then have the VLM check a 2-second
          clip every {x.stride} frames there.
        </p>
      )}
    </section>
  )
}

function ResultsCard({ x, play, names }: { x: Mcseg; play: Play; names: string[] }) {
  const { segments, fps } = x
  const byClass = names.map((_, c) => segments.map((s, i) => ({ s, i })).filter(({ s }) => s.bestRef === c))
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>Segments <span className="count">{segments.length}</span></h3>
          <p className="card-sub">
            In this video · the VLM checked {Math.round((x.shown?.coverage ?? 0) * 100)}% of it
          </p>
        </div>
      </div>
      <Slider
        label="Step probability threshold"
        help="A frame counts as a step when that step is the VLM's most likely answer and its smoothed probability reaches this."
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
          <button type="button" className="btn-text" onClick={x.clearExcluded}>Put them back</button>
        </p>
      )}
      {byClass.map((list, c) => (
        <div key={c} className="mcgroup">
          <h4><Chip i={c} /> {names[c]} <span className="count">{list.length}</span></h4>
          {list.length ? (
            <ol className="seglist">
              {list.map(({ s, i }) => {
                const here = x.frame >= s.start && x.frame <= s.end
                return (
                  <li key={i} className={`segitem${here ? ' here' : ''}`}>
                    <button type="button" className="segmain" onClick={() => { play.setPlaying(null); x.setFrame(s.start) }}>
                      <span className="segtext">
                        <span className="segtime">
                          {clock(s.start, fps, true)} – {clock(s.end + 1, fps, true)}
                          <em>{((s.end - s.start + 1) / fps).toFixed(1)} s</em>
                        </span>
                        <span className="segmeta">peak P {s.peak.toFixed(2)} · mean {s.mean.toFixed(2)}</span>
                      </span>
                    </button>
                    <button
                      type="button"
                      className={`btn icon${play.playing === i ? ' on' : ''}`}
                      onClick={() => play.setPlaying(play.playing === i ? null : i)}
                      aria-label={play.playing === i ? 'Stop' : 'Play the segment'}
                    >
                      {play.playing === i ? '■' : '▶'}
                    </button>
                    <button type="button" className="btn icon" onClick={() => x.exclude(s)} aria-label="Leave this segment out" title="Leave it out">
                      ⊘
                    </button>
                  </li>
                )
              })}
            </ol>
          ) : (
            <p className="empty small">Not found above the threshold.</p>
          )}
        </div>
      ))}
    </section>
  )
}

function AdvancedCard({ x }: { x: Mcseg }) {
  return (
    <details className="card options">
      <summary>Advanced</summary>
      {x.candidates === 'knn' ? (
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
            help="The balanced share of a step's votes a stretch needs (after ~1 s smoothing) to go to the VLM. With several steps and the background splitting the vote, shares stay low: 0.25 is the default (on a one-cycle reference, 0.4 let only 1-4 of 9 steps through); lower sends more of the video."
            value={x.knnThreshold}
            min={0.1}
            max={0.9}
            step={0.05}
            format={(v) => v.toFixed(2) + (v === 0.25 ? ' (default)' : '')}
            onChange={x.setKnnThreshold}
          />
        </>
      ) : (
        <Slider
          label="VLM coverage, per step"
          help="For each step, the share of each video, ranked by similarity to its marked range, that the VLM checks (plus 2 s either side). The VLM checks the union over the steps."
          value={x.coverage}
          min={0.05}
          max={1}
          step={0.05}
          format={(v) => (v >= 1 ? 'the whole video' : `${Math.round(v * 100)}%`)}
          onChange={x.setCoverage}
        />
      )}
      <Slider
        label="Smoothing"
        help="Each step's probability is averaged over this window before the frames are assigned."
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
        help="Two stretches of the same step separated by less than this become one segment."
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
            <option key={s} value={s}>every {s}th frame{s === 5 ? ' (tested default)' : ''}</option>
          ))}
        </select>
      </label>
      <p className="muted small">
        The candidates, stride and descriptions change what runs on the GPU; the rest applies instantly.
      </p>
    </details>
  )
}
