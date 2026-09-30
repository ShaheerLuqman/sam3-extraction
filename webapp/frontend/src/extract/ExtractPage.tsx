import { useMemo, useState } from 'react'
import type { HealthConfig, QwenStatus } from '../api/client'
import { Dropzone } from '../components/Dropzone'
import { Progress } from '../components/Progress'
import { Slider } from '../components/Slider'
import { DEFAULTS, useExtraction, type Extraction } from '../lib/extraction'
import { RefGrid } from './RefGrid'
import { SegmentList } from './SegmentList'
import { clock } from './format'
import { Timeline } from './Timeline'
import { ExportCard } from './ExportCard'
import { VideoViewer, type Play } from './VideoViewer'
import './extract.css'

// measured on the research box: ~45 s to bring vLLM up, ~9.8 frames/s after
const STARTUP_S = 50
const FRAMES_PER_S = 9.8

function minutes(s: number): string {
  return s < 90 ? `${Math.max(1, Math.round(s / 10) * 10)} s` : `${Math.round(s / 60)} min`
}

export function ExtractPage({ cfg, qwen }: { cfg: HealthConfig; qwen?: QwenStatus }) {
  const x = useExtraction(cfg)
  const [playing, setPlaying] = useState<number | null>(null)
  const play: Play = { playing, setPlaying }

  if (!x.available) {
    return (
      <div className="page narrow">
        <p className="note danger">
          Frame extraction needs the Qwen vLLM environment, and the backend could not find it.
          Point <code>SAM3_QWEN_PYTHON</code> at its python and restart the backend.
        </p>
      </div>
    )
  }

  return (
    <div className="page extract">
      <Viewer x={x} play={play} />
      <aside className="panel">
        <header className="xhead">
          <h2>Frame extraction</h2>
          <p className="muted small">
            Give a few images of what you are looking for. Every stretch of the video that looks
            like them comes back as a segment you can review, correct and export.
          </p>
        </header>
        <VideoCard x={x} />
        <RefsCard x={x} />
        <RunCard x={x} qwen={qwen} />
        {x.scored && <ResultsCard x={x} play={play} />}
        {x.scored && x.segments.length > 0 && <ExportCard x={x} />}
        <AdvancedCard x={x} />
      </aside>
    </div>
  )
}

// --------------------------------------------------------------------------- //
// left: the frame, the scrubber, the timeline
// --------------------------------------------------------------------------- //
function Viewer({ x, play }: { x: Extraction; play: Play }) {
  const { frame, segments } = x
  const refFrames = x.refs.flatMap((r) => (r.kind === 'frame' ? [r.frame] : []))
  const isRef = refFrames.includes(frame)
  const segIdx = segments.findIndex((s) => frame >= s.start && frame <= s.end)
  const score = x.scored?.perFrame[frame]
  return (
    <VideoViewer
      video={x.video}
      local={x.src.local}
      onLocalFail={x.src.localFail}
      empty={
        <div className="xempty">
          <Dropzone
            accept="video/*"
            title="Drop the video to search"
            hint="or click to browse. MP4, MOV, AVI, MKV or WEBM, up to 500 MB"
            busy={x.videoBusy}
            onFile={(f) => void x.loadVideo(f)}
          />
        </div>
      }
      frame={frame}
      setFrame={x.setFrame}
      lastFrame={x.lastFrame}
      fps={x.fps}
      segments={segments}
      playbackUrl={x.playbackUrl}
      onPlaybackFail={x.playbackFail}
      playbackError={x.playbackError}
      play={play}
      marked={refFrames}
      badges={
        <>
          {segIdx >= 0 ? (
            <span className="xbadge in">Segment {segIdx + 1}</span>
          ) : x.scored ? (
            <span className="xbadge">Not selected</span>
          ) : null}
          {score !== undefined && <span className="xbadge">score {score.toFixed(2)}</span>}
          {isRef && <span className="xbadge ref">Reference</span>}
        </>
      }
      tools={({ frameOk }) => (
        <button
          type="button"
          className="btn ghost"
          onClick={() => x.addFrameRef(frame)}
          disabled={isRef || !frameOk}
          title="Use what is on screen as another reference"
        >
          ＋ Use this frame as a reference
        </button>
      )}
      timeline={(seek) =>
        x.scored ? (
          <Timeline
            score={x.scored.score}
            stride={x.embStride}
            total={x.lastFrame + 1}
            fps={x.fps}
            threshold={x.threshold}
            segments={segments}
            excluded={x.excluded}
            refFrames={refFrames}
            frame={frame}
            onSeek={seek}
            onThreshold={(v) => x.setThreshold(Math.max(-1, Math.min(6, v)))}
          />
        ) : (
          <div className="timeline placeholder">
            {x.embedJob.busy
              ? 'Embedding the video…'
              : x.embedded
                ? 'Switch on at least one reference to see where it matches.'
                : 'Run the search to see where the references match across the video.'}
          </div>
        )
      }
    />
  )
}

// --------------------------------------------------------------------------- //
// right: inputs, run, results, export
// --------------------------------------------------------------------------- //
function VideoCard({ x }: { x: Extraction }) {
  const { video } = x
  if (!video) return null
  const n = x.lastFrame + 1
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3 className="xtitle" title={video.name}>
            {video.name || 'Video'}
          </h3>
          <p className="card-sub">
            {video.width} × {video.height} · {n} frames · {video.fps} fps · {clock(n, x.fps)}
            {x.embedded && video.frames && x.embedded.decoded < video.frames && (
              <> · only {x.embedded.decoded} of {video.frames} frames decode</>
            )}
          </p>
        </div>
        <label className="btn-text xreplace">
          Replace
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
      {x.error && <p className="note danger">{x.error}</p>}
    </section>
  )
}

function RefsCard({ x }: { x: Extraction }) {
  const n = x.refs.length
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>
            References <span className="count">{n}</span>
          </h3>
          <p className="card-sub">
            3 to 10 images of the step or scene you want, ideally from the same camera. Click a
            thumbnail to leave it out without deleting it.
          </p>
        </div>
      </div>
      {n > 0 && (
        <RefGrid
          refs={x.refs}
          videoId={x.video?.upload_id}
          scored={(r) => x.rows.has(r.uid)}
          onToggle={x.toggleRef}
          onRemove={x.removeRef}
          onShow={x.setFrame}
        />
      )}
      <Dropzone
        compact
        multiple
        accept="image/*"
        title={n ? 'Add more images' : 'Drop reference images'}
        hint="JPG or PNG, several at once"
        busy={x.refsUploading > 0}
        busyLabel={`Uploading ${x.refsUploading}…`}
        onFiles={(fs) => void x.addImages(fs)}
      />
      {!x.video && x.error && <p className="note danger">{x.error}</p>}
      {x.video && (
        <p className="muted small">
          Frames of this video work as references too: find a good one and press{' '}
          <em>Use this frame as a reference</em>.
        </p>
      )}
    </section>
  )
}

function RunCard({ x, qwen }: { x: Extraction; qwen?: QwenStatus }) {
  const { pending, embedJob, video } = x
  const dedicated = qwen?.dedicated ?? true
  const gpuNote = dedicated
    ? `Runs on GPU ${qwen?.gpu ?? 0}; object tracking on GPU 1 carries on as normal.`
    : 'Shares GPU 1 with SAM 3, which is parked in RAM for the run, so object tracking waits.'
  const warm = qwen?.state === 'ready' || qwen?.state === 'busy'
  if (embedJob.busy) {
    return (
      <section className="card accent">
        <Progress job={embedJob} idleLabel="Waiting for the GPU" onCancel={embedJob.cancel} />
        <p className="muted small">{gpuNote}</p>
      </section>
    )
  }
  if (!video) {
    const up = x.src.uploadPct
    if (x.src.local && up !== null)
      return (
        <p className="muted small">
          Uploading the video for the GPU: {Math.round(up * 100)}%. You can already play and scrub it.
        </p>
      )
    if (x.src.local && x.src.error)
      return (
        <p className="note danger">
          {x.src.error}{' '}
          <button type="button" className="btn-text" onClick={x.src.retryUpload}>
            Retry upload
          </button>
        </p>
      )
    return <p className="muted small">Upload a video on the left to start.</p>
  }
  const hasRefs = x.refs.length > 0
  const nFrames = Math.ceil((x.lastFrame + 1) / x.stride)
  const est = (warm ? 2 : STARTUP_S) + (pending?.video ? nFrames / FRAMES_PER_S : 0)

  return (
    <section className={`card${pending ? ' accent' : ''}`}>
      {embedJob.status === 'error' && <Progress job={embedJob} />}
      {pending ? (
        <>
          <button
            type="button"
            className="btn primary big"
            disabled={!hasRefs}
            onClick={x.run}
          >
            {!hasRefs
              ? 'Add a reference first'
              : pending.video
                ? x.embedded || embedJob.status === 'done'
                  ? 'Re-run with the new settings'
                  : 'Find matching segments'
                : `Embed ${pending.images} new image${pending.images > 1 ? 's' : ''}`}
          </button>
          <p className="muted small">
            {pending.video
              ? `Embeds every ${x.stride}th frame (${nFrames} frames) with ${x.model.split('/').pop()}, about ${minutes(est)}. `
              : warm
                ? 'A second or two: the model is already loaded. '
                : `About ${minutes(est)}, most of it loading the model. `}
            {gpuNote} After this, thresholds and references taken from the video apply instantly.
          </p>
        </>
      ) : (
        <p className="note ok">
          Everything is embedded. Changes to the threshold and the references apply instantly.
        </p>
      )}
      {x.scoreError && <p className="note danger">{x.scoreError}</p>}
    </section>
  )
}

function ResultsCard({ x, play }: { x: Extraction; play: Play }) {
  const { segments, fps } = x
  const frames = segments.reduce((a, s) => a + s.end - s.start + 1, 0)
  const total = x.lastFrame + 1
  const refIndex = useMemo(() => {
    const m = new Map(x.refs.map((r, i) => [r.uid, i]))
    return (r: { uid: string }) => m.get(r.uid) ?? 0
  }, [x.refs])

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>
            Segments <span className="count">{segments.length}</span>
          </h3>
          <p className="card-sub">
            {frames} frames · {(frames / fps).toFixed(1)} s · {((100 * frames) / total).toFixed(1)}%
            of the video
          </p>
        </div>
      </div>

      <Slider
        label="Match threshold"
        help="How much more similar than the video's typical frame a frame must be. Drag the dashed line on the timeline to do the same."
        value={x.threshold}
        min={-0.5}
        max={5}
        step={0.05}
        format={(v) => v.toFixed(2) + (v === DEFAULTS.threshold ? ' (default)' : '')}
        minLabel="More segments"
        maxLabel="Only the closest"
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
        active={x.active}
        refIndex={refIndex}
        videoId={x.video?.upload_id}
        fps={fps}
        frame={x.frame}
        playing={play.playing}
        onSeek={x.setFrame}
        onPlay={(i) => play.setPlaying(play.playing === i ? null : i)}
        onUseAsRef={(s) => x.addFrameRef(s.peakFrame)}
        onExclude={x.exclude}
      />
    </section>
  )
}

function AdvancedCard({ x }: { x: Extraction }) {
  const fps = x.fps
  return (
    <details className="card options">
      <summary>Advanced</summary>
      <Slider
        label="Smoothing"
        help="Scores are averaged over this window before thresholding, which removes one-frame flickers."
        value={x.smoothSec}
        min={0}
        max={5}
        step={0.05}
        format={(v) => (v === 0 ? 'off' : `${v.toFixed(2)} s`)}
        onChange={x.setSmoothSec}
      />
      <Slider
        label="Shortest segment"
        help="Matches shorter than this are dropped."
        value={x.minSegSec}
        min={0}
        max={10}
        step={0.1}
        format={(v) => `${v.toFixed(1)} s · ${Math.round(v * fps)} frames`}
        onChange={x.setMinSegSec}
      />
      <Slider
        label="Merge gaps up to"
        help="Two matches separated by less than this become one segment."
        value={x.gapSec}
        min={0}
        max={20}
        step={0.25}
        format={(v) => `${v.toFixed(2)} s · ${Math.round(v * fps)} frames`}
        onChange={x.setGapSec}
      />
      <label className="field">
        Embed every Nth frame
        <select value={x.stride} onChange={(e) => x.setStride(Number(e.target.value))}>
          {[1, 2, 3, 5, 10, 15].map((s) => (
            <option key={s} value={s}>
              every {s === 1 ? '' : `${s}th `}frame{s === x.embedded?.stride ? ' (embedded)' : ''}
              {s === 5 ? ', tested default' : ''}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        Embedding instruction
        <textarea
          rows={4}
          value={x.instruction}
          onChange={(e) => x.setInstruction(e.target.value)}
        />
        <span className="muted small">
          Tells the model what to pay attention to. The default was written for overhead
          assembly-station footage.{' '}
          {x.instruction.trim() !== x.defaultInstruction && (
            <button type="button" className="btn-text" onClick={() => x.setInstruction(x.defaultInstruction)}>
              Reset to default
            </button>
          )}
        </span>
      </label>
      <p className="muted small">
        Changing either of the last two means embedding the video again. Everything else applies
        instantly.
      </p>
    </details>
  )
}
