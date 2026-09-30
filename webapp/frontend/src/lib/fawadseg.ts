// State for the "frame extraction fawad segment" page: the research's
// class_N_desc_vlm_hints pipeline (qwen_vl/extraction_project), run as is on the
// server (backend/fawadseg.py). Its inputs are the research's inputs — a labelled
// reference video (preds.json), detector.json, the step, its confusable steps, a
// description and per-step hints — and its outputs are the research's files. The
// selection is the script's, not recomputed here, so there are no live sliders.
import { useCallback, useMemo, useState } from 'react'
import type { HealthConfig } from '../api/client'
import { useJob } from '../hooks/useJob'
import { useVideo } from './segx'
import type { Scored, Segment } from './segments'

/** A JSON input file, kept as its text: it is sent verbatim. */
export type JsonFile = { name: string; text: string }
export type Step = { id: number; name: string }
export type LoadedPreds = JsonFile & { labelled: number }
export type LoadedDetector = JsonFile & { steps: Step[] }

export type Metrics = {
  tp: number
  fp: number
  fn: number
  tn: number
  precision: number
  recall: number
  f1: number
  iou: number
  accuracy: number
}
export type FileLink = { name: string; url: string; bytes: number }

export type FawadResult = {
  run_id: string
  has_gt: boolean
  class_id: number
  class_name: string
  confusers: number[]
  folders: { candidates: string; final: string }
  target: { name: string; stem: string; decoded_frames: number; header_frames: number; fps: number }
  reference: { name: string; stem: string; decoded_frames: number; header_frames: number }
  stride: number
  segments: [number, number][]
  candidate_segments: [number, number][]
  options: Record<string, string>
  examples: Record<string, number[]>
  /** vlm_scores.csv: frame, p_A.., p_target_smoothed */
  scores: { columns: string[]; rows: number[][] }
  metrics: {
    candidates: Metrics
    final: Metrics
    vlm_raw_on_queries?: Metrics
    ap_vlm?: number
    n_queries: number
  } | null
  candidate_metrics: {
    final: Metrics
    raw_knn: Metrics
    ap_knn: number
    ref_selfcheck: Metrics
  } | null
  files: { candidates: FileLink[]; final: FileLink[]; inputs: FileLink[]; run: FileLink[] }
  zip_url: string
  commands: string[]
  seconds: number
}

/** vlm_classify.py's argparse default for --confusers */
const DEFAULT_CONFUSERS = [3, 7]
/** vlm_classify.option_text's "other" option when the hints give none */
export const DEFAULT_OTHER =
  'anything else: idle, positioning or handling the panel, using the rivnut gun to crimp, fetching parts, or any other step'

/** The research's fixed settings, as the scripts run them. */
export const FAWAD_SETTINGS: [string, string][] = [
  ['Embedding', 'Qwen3-VL-Embedding-8B, every 5th frame, the step description in the instruction'],
  ['Candidates', 'class-balanced kNN vote, k = 15, over the reference labels; 17-frame smoothing, ≥ 0.5, gaps ≤ 35 frames filled, runs < 18 frames dropped'],
  ['VLM', 'Qwen3-VL-8B-Instruct; one 8-frame clip (every 5th frame, ~2 s) every 5th candidate frame'],
  ['Examples', '2 clips per option from the reference (k-means medoids of clean stretches); "other" from the look-alikes of the step'],
  ['Selection', 'P(step) from the first answer token, smoothed over 17 frames within each run, ≥ 0.5, gaps ≤ 35 filled, runs < 18 dropped'],
]

async function readJson(file: File): Promise<{ text: string; data: Record<string, unknown> }> {
  const text = await file.text()
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch {
    throw new Error(`${file.name} is not valid JSON`)
  }
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error(`${file.name} is not a JSON object`)
  return { text, data: data as Record<string, unknown> }
}

async function readPreds(file: File): Promise<LoadedPreds> {
  const { text, data } = await readJson(file)
  const preds = data.preds
  if (!preds || typeof preds !== 'object') throw new Error(`${file.name} has no "preds" — is it a preds.json?`)
  return { name: file.name, text, labelled: Object.keys(preds).length }
}

export function useFawadSeg(cfg: HealthConfig) {
  const ref = useVideo(cfg)
  const target = useVideo(cfg)
  const [refPreds, setRefPreds] = useState<LoadedPreds | null>(null)
  const [tgtPreds, setTgtPreds] = useState<LoadedPreds | null>(null)
  const [detector, setDetector] = useState<LoadedDetector | null>(null)
  const [cls, setClsRaw] = useState<number | null>(null)
  const [confusers, setConfusers] = useState<number[]>([])
  const [description, setDescription] = useState('')
  const [descFile, setDescFile] = useState<string | null>(null)
  const [hints, setHints] = useState<Record<string, string>>({})
  const [hintsFile, setHintsFile] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const guard = useCallback(async (fn: () => Promise<void>) => {
    setError(null)
    try {
      await fn()
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  const loadRefPreds = (f: File) => guard(async () => setRefPreds(await readPreds(f)))
  const loadTgtPreds = (f: File) => guard(async () => setTgtPreds(await readPreds(f)))
  const loadDetector = (f: File) =>
    guard(async () => {
      const { text, data } = await readJson(f)
      const cs = data.cycle_steps as Record<string, string | number> | undefined
      if (!cs || typeof cs !== 'object') throw new Error(`${f.name} has no cycle_steps — is it a detector.json?`)
      const steps = Object.entries(cs)
        .map(([name, id]) => ({ id: Number(id), name: name.split(/\s+/).join(' ') }))
        .sort((a, b) => a.id - b.id)
      setDetector({ name: f.name, text, steps })
      const ids = new Set(steps.map((s) => s.id))
      setClsRaw((c) => (c !== null && ids.has(c) ? c : null))
      setConfusers(DEFAULT_CONFUSERS.filter((c) => ids.has(c)))
    })
  const loadDescription = (f: File) =>
    guard(async () => {
      setDescription(await f.text())
      setDescFile(f.name)
    })
  const loadHints = (f: File) =>
    guard(async () => {
      const { data } = await readJson(f)
      const h: Record<string, string> = {}
      for (const [k, v] of Object.entries(data)) if (!k.startsWith('_') && typeof v === 'string') h[k] = v
      setHints(h)
      setHintsFile(f.name)
    })

  const setCls = useCallback((c: number | null) => {
    setClsRaw(c)
    if (c !== null) setConfusers((cs) => cs.filter((x) => x !== c))
  }, [])

  // the options as vlm_classify numbers them: the named steps sorted, then "other"
  const optionSteps = useMemo(
    () => (cls === null ? [] : [...new Set([cls, ...confusers])].sort((a, b) => a - b)),
    [cls, confusers],
  )
  const stepName = useCallback(
    (id: number) => detector?.steps.find((s) => s.id === id)?.name ?? `step ${id}`,
    [detector],
  )

  // -- the run ---------------------------------------------------------------- //
  const job = useJob(cfg.video_poll_ms)
  const { submit } = job
  // only the hints of the options go in, and blank ones not at all (the script
  // appends " " + hint for any key it finds, so an empty one would change the prompt)
  const sentHints = useMemo(() => {
    const h: Record<string, string> = {}
    for (const k of [...optionSteps.map(String), 'other']) if (hints[k]?.trim()) h[k] = hints[k].trim()
    return h
  }, [hints, optionSteps])

  const missing = [
    !ref.video && 'the reference video',
    !refPreds && "the reference's preds.json",
    !detector && 'detector.json',
    cls === null && 'the step to find',
    cls !== null && !confusers.length && 'at least one confusable step',
    !target.video && 'the video to search',
  ].filter(Boolean) as string[]

  const signature = JSON.stringify([ref.video?.upload_id, target.video?.upload_id, refPreds?.text.length,
    refPreds?.name, tgtPreds?.name, detector?.name, cls, confusers, description, sentHints])
  const [ranWith, setRanWith] = useState<string | null>(null)
  const [ranTarget, setRanTarget] = useState<string | null>(null)
  const run = useCallback(() => {
    if (!ref.video || !target.video || !refPreds || !detector || cls === null) return
    setRanWith(signature)
    setRanTarget(target.video.upload_id)
    void submit('fawadseg-run', {
      ref_upload_id: ref.video.upload_id,
      upload_id: target.video.upload_id,
      ref_preds: refPreds.text,
      detector: detector.text,
      tgt_preds: tgtPreds?.text ?? '',
      cls,
      confusers,
      description,
      hints: sentHints,
    })
  }, [ref.video, target.video, refPreds, detector, tgtPreds, cls, confusers, description, sentHints,
    signature, submit])

  const done = job.status === 'done' ? (job.result as unknown as FawadResult | null) : null
  // a result for another video (the target was replaced) is not shown on it
  const result = done && ranTarget === target.video?.upload_id ? done : null
  const stale = !!result && ranWith !== signature

  const fps = target.info?.fps || result?.target.fps || 20
  const lastFrame = Math.max(0, (result?.target.decoded_frames ?? target.info?.frames ?? 1) - 1)
  const { setFrameRaw } = target
  const setFrame = useCallback(
    (f: number) => setFrameRaw(Math.max(0, Math.min(lastFrame, Math.round(f)))),
    [lastFrame, setFrameRaw],
  )

  // -- the script's scores, for the timeline and the frame badges ------------- //
  const scored: Scored | null = useMemo(() => {
    if (!result) return null
    const n = result.target.decoded_frames
    const st = result.stride
    const col = result.scores.columns.indexOf('p_target_smoothed')
    const perFrame = new Float32Array(n).fill(NaN)
    const score = new Float32Array(Math.ceil(n / st)).fill(NaN)
    const rows = col < 0 ? [] : result.scores.rows
    // as vlm_classify.py: every candidate frame takes the score of the nearest query frame
    const inCand = new Uint8Array(n)
    for (const [a, b] of result.candidate_segments) inCand.fill(1, a, Math.min(n, b + 1))
    let j = 0
    for (let f = 0; f < n && rows.length; f++) {
      while (j + 1 < rows.length && rows[j][0] < f) j++
      if (inCand[f]) perFrame[f] = rows[j][col]
    }
    for (const r of rows) score[Math.round(r[0] / st)] = r[col]
    return { score, perFrame, best: new Int16Array(score.length) }
  }, [result])

  const segments: Segment[] = useMemo(() => {
    if (!result) return []
    return result.segments.map(([start, end]) => {
      let peak = -Infinity
      let peakFrame = start
      let sum = 0
      let k = 0
      for (let f = start; f <= end; f++) {
        const v = scored?.perFrame[f]
        if (v === undefined || Number.isNaN(v)) continue
        sum += v
        k++
        if (v > peak) {
          peak = v
          peakFrame = f
        }
      }
      return { start, end, peak: k ? peak : 0, peakFrame, mean: k ? sum / k : 0, bestRef: 0 }
    })
  }, [result, scored])

  return {
    ref, target, refPreds, tgtPreds, detector, loadRefPreds, loadTgtPreds, loadDetector,
    clearTgtPreds: () => setTgtPreds(null),
    cls, setCls, confusers, setConfusers, optionSteps, stepName,
    description, setDescription: (d: string) => {
      setDescription(d)
      setDescFile(null)
    },
    descFile, loadDescription, hints, setHint: (k: string, v: string) => setHints((h) => ({ ...h, [k]: v })),
    hintsFile, loadHints, sentHints,
    error, missing, run, job, result, stale, scored, segments,
    fps, lastFrame, frame: target.frame, setFrame,
  }
}

export type FawadSeg = ReturnType<typeof useFawadSeg>
