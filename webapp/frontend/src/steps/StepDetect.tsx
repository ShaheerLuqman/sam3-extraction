import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import { BoxCanvas, type Candidate, type MaskPreview, type Shape } from '../components/BoxCanvas'
import { FrameBar } from '../components/FrameBar'
import { ObjectList } from '../components/ObjectList'
import { Progress } from '../components/Progress'
import { Slider } from '../components/Slider'
import { useJob } from '../hooks/useJob'
import { iou, type Rect } from '../lib/coords'
import {
  className,
  colorFor,
  emptySeed,
  newInstance,
  objectReady,
  seedOn,
  toPayload,
  type Instance,
  type LabeledPoint,
  type Seed,
} from '../lib/objects'
import type { Workspace } from '../lib/workspace'

/** The three prompt types, in the order they are offered. */
type Tool = 'click' | 'box' | 'words'

const TOOLS: { id: Tool; icon: string; title: string; blurb: string }[] = [
  {
    id: 'click',
    icon: '✛',
    title: 'Click to segment',
    blurb: 'Click the object; SAM 3 segments it. Add positive/negative clicks to refine the mask.',
  },
  {
    id: 'box',
    icon: '▭',
    title: 'Bounding box',
    blurb: 'Click one corner, then the opposite corner. Seeded as a rectangular mask.',
  },
  {
    id: 'words',
    icon: '🔍',
    title: 'Text prompt',
    blurb: 'Describe the object, e.g. "red helmet", and detect every match.',
  },
]

type Detections = {
  frame: number
  candidates: Candidate[]
  /** which prompt produced them, so the panel can label itself */
  from: 'words' | 'similar'
  subject: string
}
type Preview = MaskPreview & { frame: number; message?: string }
type Draft = { points: LabeledPoint[]; box: Rect | null }
const EMPTY_DRAFT: Draft = { points: [], box: null }

export function StepDetect({ ws, offline }: { ws: Workspace; offline: boolean }) {
  const { cfg, info, objects, classes, frame, frameCap } = ws
  const previewJob = useJob(cfg.image_poll_ms)
  const searchJob = useJob(cfg.image_poll_ms)

  const [tool, setTool] = useState<Tool>('click')
  const [clickLabel, setClickLabel] = useState<0 | 1>(1)
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT)
  const [preview, setPreview] = useState<Preview | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  /** when set, committing re-seeds this instance instead of creating a new one */
  const [reseedId, setReseedId] = useState<string | null>(null)
  /** the class the next commit is labelled with; sticky, so a run of one class is fast */
  const [draftClass, setDraftClass] = useState<number | null>(null)
  const [status, setStatus] = useState('')

  const [words, setWords] = useState('')
  const [matchStyle, setMatchStyle] = useState<'sam3' | 'yoloe'>('sam3')
  const [detections, setDetections] = useState<Detections | null>(null)
  const [minScore, setMinScore] = useState(0.3)

  const reseed = objects.find((o) => o.id === reseedId) ?? null
  const busy = previewJob.busy || searchJob.busy

  // never let a background frame embedding queue in front of something the user
  // is waiting on
  const { notifyBusy } = ws
  useEffect(() => {
    notifyBusy(busy)
    return () => notifyBusy(false)
  }, [busy, notifyBusy])

  const clearDraft = () => {
    setDraft(EMPTY_DRAFT)
    setPreview(null)
    previewJob.reset()
  }

  const clearCandidates = () => {
    setDetections(null)
    searchJob.reset()
  }

  // ---- mask preview for the current clicks ----------------------------- #
  // `submit` supersedes the job in flight, so clicking faster than the model
  // replies is fine.
  const previewKey = useRef('')
  useEffect(() => {
    if (!info || tool !== 'click') return
    const key = JSON.stringify([frame, draft])
    if (key === previewKey.current) return
    previewKey.current = key
    if (!draft.points.length) {
      previewJob.reset()
      setPreview(null)
      return
    }
    setPreview(null)
    void previewJob.submit('click-preview', {
      upload_id: info.upload_id,
      frame,
      points: draft.points.map((p) => [p.x, p.y]),
      labels: draft.points.map((p) => p.label),
      box: draft.box ? [draft.box.x1, draft.box.y1, draft.box.x2, draft.box.y2] : null,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft, tool, frame, info])

  useEffect(() => {
    if (previewJob.status === 'done' && previewJob.result) {
      const r = previewJob.result
      const b = r.box as number[] | null | undefined
      setPreview({
        frame: Number(r.frame ?? frame),
        polygons: (r.polygons ?? []) as number[][][],
        box: b ? { x1: b[0], y1: b[1], x2: b[2], y2: b[3] } : null,
        message: typeof r.message === 'string' ? r.message : undefined,
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewJob.status])

  // ---- detection results ------------------------------------------------ #
  const pendingSearch = useRef<{
    from: 'words' | 'similar'
    subject: string
    /** the exemplar's own box — it always matches itself, so it is dropped */
    exemplar?: Rect | null
  } | null>(null)
  useEffect(() => {
    if (searchJob.status !== 'done' || !searchJob.result) return
    const raw = (searchJob.result.candidates ?? []) as {
      box: number[]
      score: number
      polygons?: number[][][]
    }[]
    const suggested = Number(searchJob.result.suggest_threshold)
    if (Number.isFinite(suggested)) setMinScore(suggested)
    const asked = pendingSearch.current ?? { from: 'words' as const, subject: '' }
    const found: Candidate[] = raw.map((c) => ({
      box: { x1: c.box[0], y1: c.box[1], x2: c.box[2], y2: c.box[3] } as Rect,
      score: c.score,
      polygons: c.polygons,
    }))
    setDetections({
      frame: Number(searchJob.result.frame ?? frame),
      candidates: flagExemplar(found, asked.exemplar),
      from: asked.from,
      subject: asked.subject,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchJob.status])

  // ---- saving ----------------------------------------------------------- #
  const showPreview = preview && preview.frame === frame ? preview : null
  const hasDraft = draft.points.length > 0 || draft.box !== null
  const canCommit = draft.points.length
    ? !!showPreview?.polygons.length && !previewJob.busy
    : draft.box !== null

  const commitDraft = () => {
    if (!canCommit) return
    const seed: Seed = {
      frame,
      box: draft.box,
      points: draft.points,
      polygons: draft.points.length ? (showPreview?.polygons ?? []) : [],
    }
    if (reseed) {
      ws.patch(reseed.id, {
        seeds: [...reseed.seeds.filter((s) => s.frame !== seed.frame), seed].sort(
          (a, b) => a.frame - b.frame,
        ),
      })
      setSelectedId(reseed.id)
      setReseedId(null)
      setStatus(`${reseed.name} re-seeded on frame ${frame}.`)
    } else {
      const made = newInstance('box', draftClass, classes, objects, seed)
      ws.addObjects(() => [made])
      setSelectedId(made.id)
      setStatus(`Saved as ${made.name}.`)
    }
    ws.ensureTracked(frame)
    clearDraft()
  }

  const candidates =
    detections && detections.frame === frame ? detections.candidates : undefined
  /** candidates are on screen, so they own the canvas for now */
  const showingCandidates = !!candidates?.length
  const addable = (c: Candidate) => !c.existing && c.score >= minScore
  const keptCount = candidates?.filter(addable).length ?? 0
  const alreadyThere = candidates?.some((c) => c.existing) ?? false

  const commitCandidates = () => {
    if (!candidates) return
    const picked = candidates.filter(addable)
    if (!picked.length) return
    ws.addObjects((existing) => {
      const added: Instance[] = []
      for (const cand of picked)
        added.push(
          newInstance('box', draftClass, classes, [...existing, ...added], {
            ...emptySeed(frame),
            box: cand.box,
            polygons: cand.polygons ?? [],
          }),
        )
      return added
    })
    ws.ensureTracked(frame)
    setStatus(`Added ${picked.length} instance${picked.length > 1 ? 's' : ''}.`)
    clearCandidates()
  }

  const dropCandidate = (i: number) =>
    setDetections((s) =>
      s ? { ...s, candidates: s.candidates.filter((_, idx) => idx !== i) } : s,
    )

  // ---- searching -------------------------------------------------------- #
  const detectFromText = () => {
    if (!info || !words.trim()) return
    clearCandidates()
    pendingSearch.current = { from: 'words', subject: words.trim() }
    void searchJob.submit('exemplar', {
      upload_id: info.upload_id,
      frame,
      text: words.trim(),
    })
  }

  const addPhraseObject = () => {
    if (!words.trim()) return
    const phrase = words.trim()
    const made = {
      ...newInstance('text', draftClass, classes, objects),
      phrase,
      promptFrame: frame,
    }
    ws.addObjects(() => [made])
    setSelectedId(made.id)
    setWords('')
    setStatus(`Concept "${phrase}" will be detected and tracked across the clip.`)
  }

  /** Exemplar search: use a committed instance's own mask as the visual exemplar. */
  const findSimilar = (o: Instance) => {
    if (!info || o.kind !== 'box' || !o.seeds.length) return
    // candidates only render on the frame that is on screen, so jump to the
    // seed being used as the exemplar
    const seed = seedOn(o, frame) ?? o.seeds[0]
    if (seed.frame !== frame) ws.setFrame(seed.frame)
    clearCandidates()
    setSelectedId(o.id)
    const box = seed.box ?? boxOfPolygons(seed.polygons)
    pendingSearch.current = { from: 'similar', subject: o.name, exemplar: box }
    void searchJob.submit('exemplar', {
      upload_id: info.upload_id,
      frame: seed.frame,
      box: box ? [box.x1, box.y1, box.x2, box.y2] : null,
      polygons: seed.polygons?.length ? seed.polygons : undefined,
      method: matchStyle,
    })
  }

  // ---- canvas ----------------------------------------------------------- #
  const shapes: Shape[] = useMemo(
    () =>
      objects.flatMap((o) => {
        if (o.kind !== 'box') return []
        const s = seedOn(o, frame)
        if (!s) return []
        return [
          {
            id: o.id,
            label: o.name,
            color: o.color,
            box: s.box,
            polygons: s.polygons,
            selected: o.id === selectedId,
          },
        ]
      }),
    [objects, frame, selectedId],
  )

  const markedFrames = useMemo(
    () => [...new Set(objects.flatMap((o) => o.seeds.map((s) => s.frame)))].sort((a, b) => a - b),
    [objects],
  )

  const canvasMode = showingCandidates
    ? 'view'
    : tool === 'click'
      ? 'points'
      : tool === 'box'
        ? 'box'
        : 'view'

  // ---- running ---------------------------------------------------------- #
  const totalFrames = info?.frames ?? frameCap + 1
  /** seeds the current tracking window would miss (only reachable by lowering it) */
  const outsideWindow = objects.flatMap((o) =>
    o.seeds.filter((s) => s.frame >= ws.maxFrames).map((s) => s.frame),
  )
  const readyCount = objects.filter(objectReady).length
  const canRun = !!info && objects.length > 0 && readyCount === objects.length && !offline && !ws.track.busy

  const run = () => {
    if (!info || !canRun) return
    void ws.track.submit('video-track', {
      upload_id: info.upload_id,
      max_frames: ws.maxFrames,
      threshold: ws.threshold,
      bidirectional: ws.bidirectional,
      objects: objects.map(toPayload),
    })
  }

  if (!info) return null

  const nameSelect = (
    <label className="field">
      <span>Class</span>
      <select
        value={draftClass ?? ''}
        onChange={(e) => setDraftClass(e.target.value === '' ? null : Number(e.target.value))}
      >
        <option value="">— unclassified —</option>
        {classes.map((c) => (
          <option key={c.id} value={c.id}>
            {c.id}: {c.name}
          </option>
        ))}
      </select>
    </label>
  )

  return (
    <div className="page detect">
      <div className="viewer">
        <div className="viewer-canvas">
          <BoxCanvas
            src={api.frameUrl(info.upload_id, frame)}
            natW={info.width}
            natH={info.height}
            mode={canvasMode}
            shapes={showingCandidates ? [] : shapes}
            onPickShape={setSelectedId}
            onRemoveShape={showingCandidates ? undefined : ws.removeObject}
            pendingBox={tool === 'box' || tool === 'click' ? draft.box : null}
            onBox={(r) => setDraft((d) => ({ ...d, box: r }))}
            onClearBox={draft.box ? () => setDraft((d) => ({ ...d, box: null })) : undefined}
            points={tool === 'click' ? draft.points : []}
            onPointsChange={(points) => setDraft((d) => ({ ...d, points }))}
            nextLabel={clickLabel}
            preview={tool === 'click' ? showPreview : null}
            onStatus={setStatus}
            candidates={candidates}
            threshold={minScore}
            onRemoveCandidate={dropCandidate}
            accent={colorFor(draftClass, objects.length)}
          />
        </div>
        <FrameBar
          frame={frame}
          last={frameCap}
          fps={info.fps}
          marked={markedFrames}
          onChange={ws.setFrame}
        />
        <p className="viewer-status">
          {status ||
            (showingCandidates
              ? 'Click the ✕ on a detection to discard it, then commit the rest.'
              : shapes.length
                ? 'Click an instance to select it, or its ✕ to remove it.'
                : TOOLS.find((t) => t.id === tool)?.blurb)}
        </p>
      </div>

      <aside className="panel">
        <section className="card">
          <h3>Prompt type</h3>
          <div className="tools">
            {TOOLS.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`tool${tool === t.id ? ' on' : ''}`}
                title={t.blurb}
                onClick={() => {
                  setTool(t.id)
                  clearDraft()
                  clearCandidates()
                  setStatus('')
                }}
              >
                <span className="tool-icon" aria-hidden>
                  {t.icon}
                </span>
                <span className="tool-title">{t.title}</span>
              </button>
            ))}
          </div>

          {reseed && (
            <p className="note info">
              The next commit re-seeds <b>{reseed.name}</b> on frame {frame}.{' '}
              <button type="button" className="btn-text" onClick={() => setReseedId(null)}>
                Cancel
              </button>
            </p>
          )}

          {tool === 'click' && (
            <div className="toolbody">
              <div className="switch">
                <button
                  type="button"
                  className={clickLabel === 1 ? 'on' : ''}
                  onClick={() => setClickLabel(1)}
                >
                  ＋ Positive
                </button>
                <button
                  type="button"
                  className={clickLabel === 0 ? 'on' : ''}
                  onClick={() => setClickLabel(0)}
                >
                  − Negative
                </button>
              </div>
              <p className="muted small">
                Right-click (or alt-click) drops the opposite label; clicking a point removes it.
                The preview runs the tracker itself, so the mask you commit is the one that gets
                propagated.
              </p>
              <Progress job={previewJob} idleLabel="Segmenting" />
              {draft.points.length > 0 && (
                <button
                  type="button"
                  className="btn-text"
                  onClick={() => setDraft((d) => ({ ...d, points: d.points.slice(0, -1) }))}
                >
                  Undo last click
                </button>
              )}
            </div>
          )}

          {tool === 'box' && (
            <div className="toolbody">
              <p className="muted small">
                Click two opposite corners. <kbd>Esc</kbd> cancels. The box is seeded as a
                rectangular mask, so the tracked shape starts as that rectangle.
              </p>
              {draft.box && <p className="note ok">Box drawn — commit it below.</p>}
            </div>
          )}

          {tool === 'words' && (
            <div className="toolbody">
              <input
                type="text"
                placeholder="e.g. yellow forklift, pallet, helmet"
                value={words}
                onChange={(e) => setWords(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && words.trim() && !busy) detectFromText()
                }}
              />
              <div className="row gap wrap">
                <button
                  type="button"
                  className="btn primary"
                  disabled={!words.trim() || busy || offline}
                  onClick={detectFromText}
                >
                  Detect on this frame
                </button>
                <button
                  type="button"
                  className="btn ghost"
                  disabled={!words.trim()}
                  onClick={addPhraseObject}
                >
                  Add as phrase object
                </button>
              </div>
              <p className="muted small">
                <b>Detect on this frame</b> segments every match here and lets you pick which to
                keep as instances. <b>Add as phrase object</b> sends the concept to the detector
                instead, which finds and tracks every match across the clip under one row.
              </p>
              <Progress job={searchJob} idleLabel="Running text prompt" />
              <Slider
                label="Detection threshold"
                value={ws.threshold}
                min={0.05}
                max={0.9}
                step={0.05}
                minLabel="0.05 — higher recall"
                maxLabel="0.90 — higher precision"
                format={(v) => v.toFixed(2)}
                help="Score cut-off the detector uses for phrase objects at track time. Click, box and exemplar instances ignore it."
                onChange={ws.setThreshold}
              />
            </div>
          )}
        </section>

        {/* save whatever is being marked right now */}
        {hasDraft && tool !== 'words' && (
          <section className="card accent">
            {reseed ? (
              <p className="muted">
                Re-seeding <b>{reseed.name}</b> — it keeps its own class.
              </p>
            ) : (
              classes.length > 0 && nameSelect
            )}
            <div className="row gap">
              <button type="button" className="btn primary" disabled={!canCommit} onClick={commitDraft}>
                {reseed
                  ? `Re-seed ${reseed.name} on frame ${frame}`
                  : !canCommit
                    ? 'Segmenting…'
                    : `Save ${draft.points.length ? 'mask' : 'box'} as ${className(draftClass, classes)}`}
              </button>
              <button type="button" className="btn-text" onClick={clearDraft}>
                Discard
              </button>
            </div>
            {showPreview?.message && <p className="muted small">{showPreview.message}</p>}
          </section>
        )}

        {/* what a search turned up */}
        {candidates && (
          <section className="card accent">
            <div className="card-head">
              <h3>
                {detections?.from === 'similar'
                  ? `Objects similar to ${detections.subject}`
                  : `Detections for "${detections?.subject}"`}
              </h3>
              <button type="button" className="btn icon" onClick={clearCandidates} title="Close">
                ✕
              </button>
            </div>

            {candidates.length === 0 ? (
              <p className="muted">
                No detections on this frame. Try another frame, or a different phrase.
              </p>
            ) : (
              <>
                <p className="muted">
                  <b>{keptCount}</b> of {candidates.length - (alreadyThere ? 1 : 0)} candidates
                  above threshold. Included ones are solid, excluded are dashed grey; the ✕ on the
                  canvas discards one outright.
                </p>
                {alreadyThere && (
                  <p className="muted small">
                    The object you searched from is drawn in grey as{' '}
                    <b>already added</b> — it stays visible so you can compare, but it won't be
                    added again.
                  </p>
                )}
                <Slider
                  label="Confidence threshold"
                  value={minScore}
                  min={0}
                  max={1}
                  step={0.01}
                  minLabel="0.00 — higher recall"
                  maxLabel="1.00 — higher precision"
                  format={(v) => v.toFixed(2)}
                  onChange={setMinScore}
                />
                {classes.length > 0 && nameSelect}
                <div className="row gap">
                  <button
                    type="button"
                    className="btn primary"
                    disabled={!keptCount}
                    onClick={commitCandidates}
                  >
                    Add {keptCount} instance{keptCount === 1 ? '' : 's'} as{' '}
                    {className(draftClass, classes)}
                  </button>
                  <button type="button" className="btn-text" onClick={clearCandidates}>
                    Discard all
                  </button>
                </div>
              </>
            )}
          </section>
        )}

        {/* every instance defined so far */}
        <section className="card">
          <div className="card-head">
            <h3>
              Instances {objects.length > 0 && <span className="count">{objects.length}</span>}
            </h3>
          </div>
          <ObjectList
            objects={objects}
            classes={classes}
            selectedId={selectedId}
            frame={frame}
            busy={busy || offline}
            onSelect={(id) => setSelectedId(id === selectedId ? null : id)}
            onRemove={(id) => {
              ws.removeObject(id)
              if (selectedId === id) setSelectedId(null)
              if (reseedId === id) setReseedId(null)
            }}
            onRename={(id, name) => ws.patch(id, { name })}
            onRelabel={ws.relabel}
            onFindMore={findSimilar}
            matchStyle={matchStyle}
            onMatchStyle={setMatchStyle}
            onGoToFrame={ws.setFrame}
            onReseed={(o) => {
              setReseedId(o.id)
              setTool('click')
              clearDraft()
              clearCandidates()
            }}
          />
        </section>

        <details className="card options">
          <summary>Tracking settings</summary>
          <Slider
            label="Max frames"
            value={Math.min(ws.maxFrames, totalFrames)}
            min={Math.min(cfg.min_frames, totalFrames)}
            max={totalFrames}
            step={1}
            format={(v) =>
              v >= totalFrames
                ? `whole clip · ${totalFrames} frames`
                : info.fps
                  ? `${v} frames · ${(v / info.fps).toFixed(1)}s`
                  : `${v} frames`
            }
            help="The clip is trimmed to this many frames from the start before tracking. Longer takes proportionally longer."
            onChange={ws.setMaxFrames}
          />
          <label className="check">
            <input
              type="checkbox"
              checked={ws.bidirectional}
              onChange={(e) => ws.setBidirectional(e.target.checked)}
            />
            <span>
              Bidirectional — also propagate <b>backwards</b> from each seed frame
              <em>Use when an object is already on screen before the frame you seeded it on.</em>
            </span>
          </label>
        </details>

        <div className="panel-foot">
          {outsideWindow.length > 0 && (
            <p className="note warn">
              {outsideWindow.length} prompt{outsideWindow.length > 1 ? 's sit' : ' sits'} past
              frame {ws.maxFrames - 1} and won't be tracked.{' '}
              <button
                type="button"
                className="btn-text"
                onClick={() => ws.setMaxFrames(Math.max(...outsideWindow) + 1)}
              >
                Extend to frame {Math.max(...outsideWindow)}
              </button>
            </p>
          )}
          {objects.length > 0 && readyCount < objects.length && (
            <p className="note warn">
              {objects.length - readyCount} instance
              {objects.length - readyCount > 1 ? 's' : ''} still need a prompt.
            </p>
          )}
          <Progress job={ws.track} idleLabel="Queued" onCancel={ws.track.cancel} />
          <button type="button" className="btn primary big" disabled={!canRun} onClick={run}>
            {ws.track.busy
              ? 'Tracking…'
              : objects.length
                ? `Track ${objects.length} instance${objects.length > 1 ? 's' : ''}`
                : 'Detect an object first'}
          </button>
          <button type="button" className="btn-text" onClick={() => ws.setStep('video')}>
            ← Back to upload
          </button>
        </div>
      </aside>
    </div>
  )
}

/**
 * Mark the exemplar in its own search results.
 *
 * The object you searched from always comes back as the strongest match. It
 * stays on screen — seeing it matched is how you judge the rest — but it is
 * already an instance, so it is flagged `existing` and left out of the count
 * and the commit rather than being added a second time.
 *
 * The bar is deliberately low: a hand-drawn exemplar box is looser than the
 * tight box SAM 3 returns for the same object, so the self-match can land well
 * under a half overlap. Only the single best-overlapping candidate is ever
 * flagged, which caps the damage at one — and anything overlapping the exemplar
 * that much is sitting on top of an instance that already exists.
 */
const SELF_IOU = 0.3

function flagExemplar(found: Candidate[], exemplar?: Rect | null): Candidate[] {
  if (!exemplar) return found
  let best = -1
  let bestIoU = SELF_IOU
  found.forEach((c, i) => {
    const overlap = iou(c.box, exemplar)
    if (overlap >= bestIoU) {
      bestIoU = overlap
      best = i
    }
  })
  return best < 0 ? found : found.map((c, i) => (i === best ? { ...c, existing: true } : c))
}

function boxOfPolygons(polys?: number[][][]): Rect | null {
  if (!polys?.length) return null
  let x1 = Infinity,
    y1 = Infinity,
    x2 = -Infinity,
    y2 = -Infinity
  for (const poly of polys)
    for (const [x, y] of poly) {
      if (x < x1) x1 = x
      if (y < y1) y1 = y
      if (x > x2) x2 = x
      if (y > y2) y2 = y
    }
  return Number.isFinite(x1) ? { x1, y1, x2, y2 } : null
}
