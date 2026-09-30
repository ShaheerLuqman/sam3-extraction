import { useState, type ReactNode } from 'react'
import { api, type HealthConfig, type QwenStatus, type UploadInfo } from '../api/client'
import { Dropzone } from '../components/Dropzone'
import { Progress } from '../components/Progress'
import {
  DEFAULT_OTHER, FAWAD_SETTINGS, useFawadSeg, type FawadSeg, type FileLink, type Metrics,
} from '../lib/fawadseg'
import { clock } from './format'
import { Timeline } from './Timeline'
import { VideoViewer, type Play } from './VideoViewer'
import './extract.css'

const letter = (i: number) => String.fromCharCode(65 + i)

export function FawadSegPage({ cfg, qwen }: { cfg: HealthConfig; qwen?: QwenStatus }) {
  const x = useFawadSeg(cfg)
  const [playing, setPlaying] = useState<number | null>(null)
  const play: Play = { playing, setPlaying }

  if (!cfg.extract?.available) {
    return (
      <main>
        <div className="page narrow">
          <p className="note danger">
            This needs the Qwen vLLM environment, and the backend could not find it. Point{' '}
            <code>SAM3_QWEN_PYTHON</code> at its python and restart the backend.
          </p>
        </div>
      </main>
    )
  }

  const r = x.result
  const segIdx = x.segments.findIndex((s) => x.frame >= s.start && x.frame <= s.end)
  const p = x.scored?.perFrame[x.frame]
  const t = x.target

  return (
    <main>
      <div className="page extract">
        <VideoViewer
          video={t.video}
          local={t.local}
          onLocalFail={t.localFail}
          empty={
            <div className="xempty">
              <Dropzone
                accept="video/*"
                title="Drop the video to search"
                hint="The target recording, to find the step in. MP4, MOV, AVI, MKV or WEBM, up to 500 MB"
                busy={t.busy}
                onFile={(f) => void t.load(f)}
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
            r && (
              <>
                {segIdx >= 0 ? (
                  <span className="xbadge in">Segment {segIdx + 1}</span>
                ) : (
                  <span className="xbadge">Not selected</span>
                )}
                <span className="xbadge">
                  {p === undefined || Number.isNaN(p) ? 'not a candidate' : `P(step) ${p.toFixed(2)}`}
                </span>
              </>
            )
          }
          timeline={(seek) =>
            r && x.scored ? (
              <Timeline
                score={x.scored.score}
                stride={r.stride}
                total={x.lastFrame + 1}
                fps={x.fps}
                threshold={0.5}
                range={[0, 1]}
                bands={r.candidate_segments}
                scoreLabel="P(step), smoothed"
                segments={x.segments}
                excluded={[]}
                refFrames={[]}
                frame={x.frame}
                onSeek={seek}
                onThreshold={() => {}}
              />
            ) : (
              <div className="timeline placeholder">
                {x.job.busy
                  ? 'Running… P(step) appears here when the VLM is done.'
                  : 'Run the pipeline to see the selection. Shaded stretches are the kNN candidates the VLM checked; the threshold is the script’s fixed 0.5.'}
              </div>
            )
          }
        />

        <aside className="panel">
          <header className="xhead">
            <h2>Frame extraction fawad segment</h2>
            <p className="muted small">
              The research’s <code>class_N_desc_vlm_hints</code> pipeline, run unchanged: frames →
              Qwen3-VL embeddings → class-balanced kNN candidates from a labelled reference video →
              Qwen3-VL-8B classifies ~2 s clips among the step, its look-alikes and “other”. Same
              inputs, same output files.
            </p>
          </header>
          <ReferenceCard x={x} />
          <StepsCard x={x} />
          <PromptCard x={x} />
          <TargetCard x={x} />
          {x.error && <p className="note danger">{x.error}</p>}
          <RunCard x={x} qwen={qwen} />
          {r && <ResultCard x={x} play={play} />}
          {r && <FilesCard x={x} />}
          <details className="card options">
            <summary>Method (fixed, as in the research)</summary>
            <dl className="ffacts">
              {FAWAD_SETTINGS.map(([k, v]) => (
                <div key={k}>
                  <dt>{k}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
            </dl>
            {r && (
              <>
                <p className="muted small">Commands this run executed:</p>
                <pre className="fcmds">{r.commands.join('\n\n')}</pre>
              </>
            )}
          </details>
        </aside>
      </div>
    </main>
  )
}

// --------------------------------------------------------------------------- //
// inputs
// --------------------------------------------------------------------------- //
function FilePick({ accept, label, onFile }: { accept: string; label: string; onFile: (f: File) => void }) {
  return (
    <label className="btn small">
      {label}
      <input
        type="file"
        accept={accept}
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
          e.target.value = ''
        }}
      />
    </label>
  )
}

function Loaded({ ok, children }: { ok: boolean; children: ReactNode }) {
  return <p className={`note ${ok ? 'ok' : 'info'} small`}>{children}</p>
}

function VideoFacts({ v }: { v: UploadInfo }) {
  return (
    <p className="card-sub">
      {v.width} × {v.height} · {v.frames} frames · {v.fps} fps · {clock(v.frames ?? 0, v.fps || 20)}
    </p>
  )
}

function ReferenceCard({ x }: { x: FawadSeg }) {
  const v = x.ref.video
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>1. Reference video, labelled</h3>
          <p className="card-sub">
            A recording of the station with its per-frame step labels (<code>preds.json</code>). The
            kNN vote and the VLM’s example clips all come from it.
          </p>
        </div>
      </div>
      {v ? (
        <div className="row gap spread">
          <div>
            <h3 className="xtitle" title={v.name}>
              {v.name || 'Reference video'}
            </h3>
            <VideoFacts v={v} />
          </div>
          <label className="btn-text xreplace">
            Replace
            <input
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void x.ref.load(f)
                e.target.value = ''
              }}
            />
          </label>
        </div>
      ) : (
        <Dropzone
          compact
          accept="video/*"
          title="Drop the reference video"
          hint="e.g. 2026-08-20 07_27_09.mp4"
          busy={x.ref.busy}
          onFile={(f) => void x.ref.load(f)}
        />
      )}
      {x.ref.error && <p className="note danger">{x.ref.error}</p>}
      <div className="row gap">
        <FilePick accept=".json,application/json" label={x.refPreds ? 'Replace preds.json' : 'Its preds.json…'} onFile={x.loadRefPreds} />
        {x.refPreds && (
          <span className="muted small" title={x.refPreds.name}>
            {x.refPreds.name} · {x.refPreds.labelled} labelled frames
          </span>
        )}
      </div>
    </section>
  )
}

function StepsCard({ x }: { x: FawadSeg }) {
  const steps = x.detector?.steps ?? []
  const addable = steps.filter((s) => s.id !== x.cls && !x.confusers.includes(s.id))
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>2. The step</h3>
          <p className="card-sub">
            Step names come from <code>detector.json</code>. The confusable steps are the VLM’s other
            named options (the research used 3 and 7 for step 5).
          </p>
        </div>
      </div>
      <div className="row gap">
        <FilePick accept=".json,application/json" label={x.detector ? 'Replace detector.json' : 'detector.json…'} onFile={x.loadDetector} />
        {x.detector && <span className="muted small">{x.detector.name} · {steps.length} steps</span>}
      </div>
      {x.detector && (
        <>
          <label className="field">
            Step to find (<code>--cls</code>)
            <select
              value={x.cls ?? ''}
              onChange={(e) => x.setCls(e.target.value === '' ? null : Number(e.target.value))}
            >
              <option value="">Choose a step…</option>
              {steps.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id}: {s.name}
                </option>
              ))}
            </select>
          </label>
          <div className="field">
            Confusable steps (<code>--confusers</code>)
            <div className="chips">
              {x.confusers.map((c) => (
                <span key={c} className="chip" title={x.stepName(c)}>
                  {c}: {x.stepName(c).slice(0, 38)}
                  {x.stepName(c).length > 38 ? '…' : ''}
                  <button
                    type="button"
                    className="btn-text"
                    aria-label={`Remove step ${c}`}
                    onClick={() => x.setConfusers(x.confusers.filter((y) => y !== c))}
                  >
                    ✕
                  </button>
                </span>
              ))}
            </div>
            <select
              value=""
              onChange={(e) => e.target.value !== '' && x.setConfusers([...x.confusers, Number(e.target.value)])}
            >
              <option value="">Add a confusable step…</option>
              {addable.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id}: {s.name}
                </option>
              ))}
            </select>
          </div>
        </>
      )}
    </section>
  )
}

function PromptCard({ x }: { x: FawadSeg }) {
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>3. Description and hints</h3>
          <p className="card-sub">
            The description goes into the embedding instruction (<code>--describe</code>); the hints
            are appended to each option of the VLM’s question (<code>--hints</code>).
          </p>
        </div>
      </div>
      <div className="field">
        <span className="row gap spread">
          <span>Step description</span>
          <FilePick accept=".txt,text/plain" label="Load .txt…" onFile={x.loadDescription} />
        </span>
        <textarea
          className="xtext"
          rows={5}
          value={x.description}
          placeholder="e.g. class5_description.txt. Leave empty for the plain instruction (the research's class_N_vlm_hints variant)."
          onChange={(e) => x.setDescription(e.target.value)}
        />
        {x.descFile && <span className="muted small">from {x.descFile}</span>}
      </div>
      <div className="field">
        <span className="row gap spread">
          <span>VLM options</span>
          <FilePick accept=".json,application/json" label="Load step_hints.json…" onFile={x.loadHints} />
        </span>
        {x.hintsFile && <span className="muted small">from {x.hintsFile}</span>}
        {!x.optionSteps.length && <span className="muted small">Choose the step first.</span>}
        {x.optionSteps.map((c, i) => (
          <label key={c} className="fopt">
            <span className="fopt-head">
              <b>{letter(i)}</b> = step {c}: “{x.stepName(c)}”{c === x.cls && <em> — the step to find</em>}
            </span>
            <textarea
              className="xtext"
              rows={3}
              value={x.hints[String(c)] ?? ''}
              placeholder="Visual hint (optional): the state of the workpiece, the tool, where the hands work"
              onChange={(e) => x.setHint(String(c), e.target.value)}
            />
          </label>
        ))}
        {x.optionSteps.length > 0 && (
          <label className="fopt">
            <span className="fopt-head">
              <b>{letter(x.optionSteps.length)}</b> = anything else
            </span>
            <textarea
              className="xtext"
              rows={3}
              value={x.hints.other ?? ''}
              placeholder={DEFAULT_OTHER}
              onChange={(e) => x.setHint('other', e.target.value)}
            />
          </label>
        )}
      </div>
    </section>
  )
}

function TargetCard({ x }: { x: FawadSeg }) {
  const v = x.target.video
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>4. Video to search</h3>
          {v ? (
            <>
              <h3 className="xtitle" title={v.name}>
                {v.name}
              </h3>
              <VideoFacts v={v} />
            </>
          ) : (
            <p className="card-sub">Drop it on the left.</p>
          )}
        </div>
        {v && (
          <label className="btn-text xreplace">
            Search another
            <input
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void x.target.load(f)
                e.target.value = ''
              }}
            />
          </label>
        )}
      </div>
      <div className="row gap">
        <FilePick
          accept=".json,application/json"
          label={x.tgtPreds ? 'Replace its preds.json' : 'Its preds.json (optional)…'}
          onFile={x.loadTgtPreds}
        />
        {x.tgtPreds && (
          <button type="button" className="btn-text" onClick={x.clearTgtPreds}>
            Remove
          </button>
        )}
      </div>
      <Loaded ok={!!x.tgtPreds}>
        {x.tgtPreds
          ? `${x.tgtPreds.name}: used only to score the result (metrics.json, TP/FP in the video), never for selection.`
          : 'Without it there are no metrics, and the selected-frames video carries no GT / TP / FP labels.'}
      </Loaded>
    </section>
  )
}

function RunCard({ x, qwen }: { x: FawadSeg; qwen?: QwenStatus }) {
  const { job } = x
  if (job.busy) {
    return (
      <section className="card accent">
        <Progress job={job} idleLabel="Waiting for the GPU" onCancel={job.cancel} />
        <p className="muted small">
          On GPU {qwen?.gpu ?? 0}: decoding both videos, the embedding model, then the VLM. A new
          video costs a few minutes; one already seen skips straight to the kNN and the VLM.
        </p>
      </section>
    )
  }
  return (
    <section className={`card${x.result && !x.stale ? '' : ' accent'}`}>
      {job.status === 'error' && <Progress job={job} />}
      {x.stale && <p className="note warn">The inputs changed since this result. Run again to update it.</p>}
      <button type="button" className="btn primary big" disabled={x.missing.length > 0} onClick={x.run}>
        {x.missing.length ? `Needs ${x.missing[0]}` : x.result ? 'Run again' : 'Run the pipeline'}
      </button>
      {x.missing.length > 1 && <p className="muted small">Still needed: {x.missing.join(', ')}.</p>}
    </section>
  )
}

// --------------------------------------------------------------------------- //
// outputs
// --------------------------------------------------------------------------- //
const pct = (v: number) => v.toFixed(3)

function MetricRow({ name, m, ap }: { name: string; m?: Metrics; ap?: number }) {
  if (!m) return null
  return (
    <tr>
      <th>{name}</th>
      <td>{pct(m.precision)}</td>
      <td>{pct(m.recall)}</td>
      <td>
        <b>{pct(m.f1)}</b>
      </td>
      <td>{pct(m.iou)}</td>
      <td>
        {m.tp}/{m.fp}/{m.fn}
      </td>
      <td>{ap === undefined ? '' : pct(ap)}</td>
    </tr>
  )
}

function ResultCard({ x, play }: { x: FawadSeg; play: Play }) {
  const r = x.result!
  const frames = x.segments.reduce((a, s) => a + s.end - s.start + 1, 0)
  const cand = r.candidate_segments.reduce((a, [s, e]) => a + e - s + 1, 0)
  const fps = x.fps
  const m = r.metrics
  const file = (name: string) => r.files.final.find((f) => f.name === name)
  const video = r.files.final.find((f) => f.name.endsWith('.mp4'))
  return (
    <>
      <section className="card">
        <div className="card-head">
          <div>
            <h3>
              Segments <span className="count">{x.segments.length}</span>
            </h3>
            <p className="card-sub">
              Step {r.class_id}: {r.class_name} · {frames} frames ({(frames / fps).toFixed(1)} s) selected out of{' '}
              {cand} candidate frames, {m?.n_queries ?? r.scores.rows.length} VLM clips · {Math.round(r.seconds)} s
              {r.target.decoded_frames < r.target.header_frames &&
                ` · only ${r.target.decoded_frames} of ${r.target.header_frames} frames decode`}
            </p>
          </div>
        </div>
        {x.segments.length ? (
          <ol className="seglist">
            {x.segments.map((s, i) => (
              <li key={s.start} className={`segitem${x.frame >= s.start && x.frame <= s.end ? ' here' : ''}`}>
                <button
                  type="button"
                  className="segmain"
                  onClick={() => {
                    play.setPlaying(null)
                    x.setFrame(s.start)
                  }}
                >
                  <span className="segidx">{i + 1}</span>
                  <span className="segtext">
                    <span className="segtime">
                      {clock(s.start, fps, true)} – {clock(s.end + 1, fps, true)}
                      <em>{((s.end - s.start + 1) / fps).toFixed(1)} s</em>
                    </span>
                    <span className="segmeta">
                      #{s.start}–{s.end} · mean P {s.mean.toFixed(2)} · peak {s.peak.toFixed(2)}
                    </span>
                  </span>
                </button>
                <span className="segactions">
                  <button
                    type="button"
                    className={`btn icon${play.playing === i ? ' on' : ''}`}
                    onClick={() => play.setPlaying(play.playing === i ? null : i)}
                    aria-label={play.playing === i ? 'Stop' : `Play segment ${i + 1}`}
                  >
                    {play.playing === i ? '■' : '▶'}
                  </button>
                </span>
              </li>
            ))}
          </ol>
        ) : (
          <p className="empty">The VLM accepted none of the candidates.</p>
        )}
      </section>

      {m && (
        <section className="card">
          <div className="card-head">
            <div>
              <h3>Accuracy against the target labels</h3>
              <p className="card-sub">Per frame, step {r.class_id} vs the rest (metrics.json).</p>
            </div>
          </div>
          <table className="ftable">
            <thead>
              <tr>
                <th />
                <th>P</th>
                <th>R</th>
                <th>F1</th>
                <th>IoU</th>
                <th>TP/FP/FN</th>
                <th>AP</th>
              </tr>
            </thead>
            <tbody>
              <MetricRow name="kNN candidates" m={m.candidates} ap={r.candidate_metrics?.ap_knn} />
              <MetricRow name="VLM, final" m={m.final} ap={m.ap_vlm} />
              <MetricRow name="VLM raw, per clip" m={m.vlm_raw_on_queries} />
            </tbody>
          </table>
        </section>
      )}

      <section className="card">
        <div className="card-head">
          <div>
            <h3>The script’s own pictures</h3>
          </div>
        </div>
        {file('timeline.png') && (
          <a href={api.fileUrl(file('timeline.png')!.url)} target="_blank" rel="noreferrer">
            <img className="fimg" src={api.fileUrl(file('timeline.png')!.url)} alt="timeline.png: GT, selection and score over the video" />
          </a>
        )}
        {file('vlm_examples.jpg') && (
          <a href={api.fileUrl(file('vlm_examples.jpg')!.url)} target="_blank" rel="noreferrer">
            <img className="fimg" src={api.fileUrl(file('vlm_examples.jpg')!.url)} alt="vlm_examples.jpg: the example clips shown to the VLM" />
          </a>
        )}
        {video && (
          <div className="result-video">
            <video src={api.fileUrl(video.url)} controls preload="metadata" />
          </div>
        )}
        <details>
          <summary className="muted small">The VLM’s options</summary>
          <ul className="fopts">
            {Object.entries(r.options).map(([k, v]) => (
              <li key={k}>
                <b>{k}</b> = {v}
                {r.examples[k] && <span className="muted"> (examples: ref frames {r.examples[k].join(', ')})</span>}
              </li>
            ))}
          </ul>
        </details>
      </section>
    </>
  )
}

function FilesCard({ x }: { x: FawadSeg }) {
  const r = x.result!
  const size = (b: number) => (b > 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} kB`)
  const group = (title: string, files: FileLink[]) =>
    files.length > 0 && (
      <div className="ffiles">
        <span className="ffiles-head">{title}/</span>
        {files.map((f) => (
          <a key={f.name} href={api.fileUrl(f.url)} target="_blank" rel="noreferrer" download={f.name}>
            {f.name} <em>{size(f.bytes)}</em>
          </a>
        ))}
      </div>
    )
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>Output files</h3>
          <p className="card-sub">As the research wrote them, one folder per stage.</p>
        </div>
        <a className="btn small" href={api.fileUrl(r.zip_url)} download>
          All as ZIP
        </a>
      </div>
      {group(r.folders.final, r.files.final)}
      {group(r.folders.candidates, r.files.candidates)}
      {group('inputs', r.files.inputs)}
      {group('run', r.files.run)}
    </section>
  )
}
