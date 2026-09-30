// State for the multiple-class segmentation page: segment extraction (lib/segx.ts)
// for several steps at once, in three steps:
//
//   1. reference — mark one range per step (class) on a reference video. Each range is
//      cut into a clip on the server; the video itself is uploaded too, since the
//      frames nobody marked are the background class of the search.
//   2. describe  — name each step yourself, or have the VLM name it; the VLM describes
//      every step that has no description of the user's.
//   3. search    — up to three videos. The server embeds, picks candidates per class
//      (a kNN vote over the classes + background, or segx's similarity ranking) and
//      has the VLM choose between the steps and "anything else" clip by clip
//      (backend/mcseg.py). Smoothing, the threshold and cleanup run here, so are live.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, type HealthConfig, type UploadInfo } from '../api/client'
import { useJob } from '../hooks/useJob'
import { grabFrames } from './localVideo'
import { clean, runs, smooth, type Segment } from './segments'
import { SEGX_DEFAULTS, useVideo } from './segx'

export type McStep = 'reference' | 'describe' | 'search'

export const MC_STEPS: { id: McStep; title: string; blurb: string }[] = [
  { id: 'reference', title: 'Mark the steps', blurb: 'One range per step on a reference video' },
  { id: 'describe', title: 'Name them', blurb: 'Yourself, or the VLM names each step' },
  { id: 'search', title: 'Search videos', blurb: 'Find every step in up to three recordings' },
]

export const MAX_CLASSES = 8
export const MAX_TARGETS = 3
/** one per class, in order; the background is grey */
export const MC_COLORS = ['#2563eb', '#d9480f', '#0f7b5a', '#9c36b5', '#c2255c', '#0b7285', '#5c940d', '#b08900']
export const BG_COLOR = '#8a94a6'

/** A step (class): its one marked range on the reference and the clip cut from it. */
export type McClass = {
  uid: string
  start: number
  end: number
  clip: UploadInfo
  /** "mine": the user names it; "vlm": the VLM's name is used */
  naming: 'mine' | 'vlm'
  name: string
  description: string
  /** who wrote the description: a VLM one is replaced when described again */
  descBy: 'vlm' | 'mine' | null
}

export type McTarget = {
  upload_id: string
  name: string
  decoded_frames: number
  fps: number
  stride: number
  /** per class, per embedded frame: the candidate score (kNN share or z-scored similarity) */
  scores: number[][]
  /** per class, the stretches its candidate stage kept */
  regions: [number, number][][]
  /** what the VLM checked: the union over classes */
  region: [number, number][]
  coverage: number
  centers: number[]
  /** per centre, P per letter (the classes in order, then "anything else") */
  p: (number | null)[][]
}

export type McResult = {
  letters: string[]
  options: Record<string, string>
  classes: string[]
  candidates: 'knn' | 'similarity'
  knn: { k: number; threshold: number } | null
  coverage: number | null
  reference_frames: { per_class: number[]; background: number }
  targets: McTarget[]
}

/** A segment of one class: Segment.bestRef is the class index. */
export type McSelection = {
  /** per class, per embedded frame, the smoothed P (NaN where the VLM did not look) */
  curves: Float32Array[]
  /** per frame: the class it went to, n_cls for anything else, -1 not checked */
  label: Int8Array
  /** per frame, per letter, smoothed P (flattened: f * letters + k) */
  perFrame: Float32Array
  letters: number
  segments: Segment[]
}

/** The browser half of the method (scripts/mcseg_eval.py runs the same): P per letter
 *  smoothed within each run of consecutive checked clips, each frame to its most likely
 *  letter, kept for a class when that P reaches the threshold, then per class short
 *  gaps filled and short runs dropped. */
export function selectMulti(
  t: McTarget, nCls: number, threshold: number, smoothSec: number, minSegSec: number,
  gapSec: number, excluded: [number, number][],
): McSelection {
  const { centers, stride: st, decoded_frames: n, fps } = t
  const L = nCls + 1
  const sm = centers.map(() => new Float32Array(L))
  const win = Math.max(1, Math.round((smoothSec * fps) / st))
  for (let a = 0; a < centers.length; ) {
    let b = a
    while (b + 1 < centers.length && centers[b + 1] - centers[b] === st) b++
    for (let k = 0; k < L; k++) {
      const col = Float32Array.from(t.p.slice(a, b + 1), (row) => row[k] ?? 0)
      smooth(col, win).forEach((v, i) => (sm[a + i][k] = v))
    }
    a = b + 1
  }
  const inRegion = new Uint8Array(n)
  for (const [a, b] of t.region) inRegion.fill(1, a, Math.min(n, b + 1))
  const perFrame = new Float32Array(n * L)
  const label = new Int8Array(n).fill(-1)
  let j = 0
  for (let f = 0; f < n && centers.length; f++) {
    if (!inRegion[f]) continue
    while (j + 1 < centers.length && Math.abs(centers[j + 1] - f) <= Math.abs(centers[j] - f)) j++
    let best = 0
    for (let k = 0; k < L; k++) {
      perFrame[f * L + k] = sm[j][k]
      if (sm[j][k] > sm[j][best]) best = k
    }
    label[f] = best
  }
  const curves = Array.from({ length: nCls }, (_, c) => {
    const out = new Float32Array(Math.ceil(n / st))
    for (let i = 0; i < out.length; i++) {
      const f = Math.min(n - 1, i * st)
      out[i] = inRegion[f] ? perFrame[f * L + c] : NaN
    }
    return out
  })
  const segments: Segment[] = []
  const minSeg = Math.max(1, Math.round(minSegSec * fps))
  const gap = Math.max(0, Math.round(gapSec * fps))
  for (let c = 0; c < nCls; c++) {
    const mask = new Uint8Array(n)
    for (let f = 0; f < n; f++) mask[f] = label[f] === c && perFrame[f * L + c] >= threshold ? 1 : 0
    for (const [a, b] of excluded) mask.fill(0, a, b + 1)
    const cleaned = clean(mask, minSeg, gap)
    for (const [a, b] of excluded) cleaned.fill(0, a, b + 1)
    for (const [start, end] of runs(cleaned)) {
      let peak = -Infinity
      let peakFrame = start
      let sum = 0
      for (let f = start; f <= end; f++) {
        const v = perFrame[f * L + c]
        sum += v
        if (v > peak) {
          peak = v
          peakFrame = f
        }
      }
      segments.push({ start, end, peak, peakFrame, mean: sum / (end - start + 1), bestRef: c })
    }
  }
  segments.sort((a, b) => a.start - b.start)
  return { curves, label, perFrame, letters: L, segments }
}

let seq = 0
const uid = () => `m${++seq}`

export function useMcseg(cfg: HealthConfig) {
  const [step, setStep] = useState<McStep>('reference')
  const [error, setError] = useState<string | null>(null)
  const [classes, setClasses] = useState<McClass[]>([])

  // -- 1. reference: one range per class ----------------------------------- //
  // uploaded in the background as soon as it is open: its unmarked frames are the
  // background class, so the server needs all of it
  const ref = useVideo(cfg, { upload: 'always' })
  const refLast = Math.max(0, (ref.info?.frames ?? 1) - 1)
  const { setFrameRaw: setRefFrameRaw, load: loadRefVideo } = ref
  const setRefFrame = useCallback(
    (f: number) => setRefFrameRaw(Math.max(0, Math.min(refLast, Math.round(f)))),
    [refLast, setRefFrameRaw],
  )
  const [markIn, setMarkIn] = useState<number | null>(null)
  const [markOut, setMarkOut] = useState<number | null>(null)
  const [cutting, setCutting] = useState<{ done: number; total: number } | null>(null)
  const [newName, setNewName] = useState('')

  // the classes belong to the reference they were marked on: a new one starts over
  const loadRef = useCallback(
    async (file: File) => {
      setMarkIn(null)
      setMarkOut(null)
      setClasses([])
      await loadRefVideo(file)
    },
    [loadRefVideo],
  )

  const selection = useMemo<[number, number] | null>(
    () => (markIn !== null && markOut !== null ? [Math.min(markIn, markOut), Math.max(markIn, markOut)] : null),
    [markIn, markOut],
  )
  /** the class whose range the selection overlaps, other than `except` */
  const overlapping = useCallback(
    (a: number, b: number, except?: string) => classes.find((c) => c.uid !== except && c.start <= b && c.end >= a),
    [classes],
  )

  const { local: refLocal, video: refVideo, info: refInfo } = ref
  const cut = useCallback(
    async (start: number, end: number): Promise<UploadInfo | null> => {
      if (!refInfo) return null
      setCutting({ done: 0, total: end - start + 1 })
      try {
        if (refLocal) {
          const frames = await grabFrames(refLocal, start, end, (done, total) => setCutting({ done, total }))
          const stem = refLocal.name.replace(/\.[^.]+$/, '')
          return await api.segxClipFrames(frames, refLocal.fps, `${stem} [${start}-${end}]`)
        }
        if (refVideo) return await api.segxCut({ upload_id: refVideo.upload_id, start, end })
        return null
      } finally {
        setCutting(null)
      }
    },
    [refLocal, refVideo, refInfo],
  )

  /** the selection as a new class, or as the new range of `replace` */
  const markClass = useCallback(
    async (replace?: string) => {
      if (!selection) return
      const [start, end] = selection
      const hit = overlapping(start, end, replace)
      if (hit) return setError(`That range overlaps "${hit.name || 'another step'}". Each frame can belong to one step only.`)
      if (!replace && classes.length >= MAX_CLASSES) return setError(`At most ${MAX_CLASSES} steps.`)
      setError(null)
      try {
        const clip = await cut(start, end)
        if (!clip) return
        if (replace) {
          setClasses((cs) => cs.map((c) => (c.uid === replace ? { ...c, start, end, clip } : c)))
        } else {
          const name = newName.trim()
          setClasses((cs) => [...cs, {
            uid: uid(), start, end, clip, naming: name ? 'mine' : 'vlm', name, description: '', descBy: null,
          }])
          setNewName('')
        }
        setMarkIn(null)
        setMarkOut(null)
      } catch (e) {
        setError((e as Error).message)
      }
    },
    [selection, overlapping, classes.length, cut, newName],
  )

  const removeClass = useCallback((id: string) => setClasses((cs) => cs.filter((c) => c.uid !== id)), [])
  const updateClass = useCallback(
    (id: string, patch: Partial<McClass>) => setClasses((cs) => cs.map((c) => (c.uid === id ? { ...c, ...patch } : c))),
    [],
  )

  // -- 2. describe ------------------------------------------------------------ //
  const describeJob = useJob(cfg.image_poll_ms)
  const { submit: submitDescribe } = describeJob
  // which classes (and which clip of each) the running / last description is for
  const describedRef = useRef<{ uid: string; clip: string }[]>([])
  const needsVlm = (c: McClass) => c.naming === 'vlm' || !c.description.trim() || c.descBy === 'vlm'
  const describe = useCallback(
    (all = false) => {
      const todo = classes.filter((c) => all || needsVlm(c))
      if (!todo.length) return
      describedRef.current = todo.map((c) => ({ uid: c.uid, clip: c.clip.upload_id }))
      void submitDescribe('mcseg-describe', { classes: todo.map((c) => [c.clip.upload_id]) })
    },
    [classes, submitDescribe],
  )
  // a finished description goes into the classes it was for, unless the user has
  // since written their own (a name of theirs is kept; so is a description of theirs)
  useEffect(() => {
    if (describeJob.status !== 'done') return
    const got = (describeJob.result as { classes?: { name: string; description: string }[] } | null)?.classes ?? []
    const map = new Map(describedRef.current.map((d, i) => [d.uid, { ...d, ...got[i] }]))
    setClasses((cs) => cs.map((c) => {
      const d = map.get(c.uid)
      if (!d || d.clip !== c.clip.upload_id) return c
      return {
        ...c,
        name: c.naming === 'vlm' ? d.name || c.name : c.name,
        description: c.descBy === 'mine' && c.description.trim() ? c.description : d.description || c.description,
        descBy: c.descBy === 'mine' && c.description.trim() ? 'mine' : 'vlm',
      }
    }))
    describedRef.current = []
  }, [describeJob.status, describeJob.result])

  // classes whose clip changed since their VLM description, or that have none yet
  const undescribed = classes.filter((c) => (c.naming === 'vlm' && !c.name) || !c.description.trim()).length

  const goto = useCallback(
    (s: McStep) => {
      setStep(s)
      if (s === 'describe' && !describeJob.busy && classes.some((c) => (c.naming === 'vlm' && !c.name) || !c.description.trim()))
        describe()
    },
    [describeJob.busy, classes, describe],
  )

  // -- 3. search ------------------------------------------------------------- //
  const t0 = useVideo(cfg)
  const t1 = useVideo(cfg)
  const t2 = useVideo(cfg)
  const slots = useMemo(() => [t0, t1, t2], [t0, t1, t2])
  const [active, setActive] = useState(0)
  const target = slots[active]
  const [useDescription, setUseDescription] = useState(true)
  const [candidates, setCandidates] = useState<'knn' | 'similarity'>('similarity')
  const [knnK, setKnnK] = useState(15)
  const [knnThreshold, setKnnThreshold] = useState(0.25)
  const [coverage, setCoverage] = useState(0.25)
  const [stride, setStride] = useState(5)
  const [threshold, setThreshold] = useState(SEGX_DEFAULTS.threshold)
  const [smoothSec, setSmoothSec] = useState(SEGX_DEFAULTS.smoothSec)
  const [minSegSec, setMinSegSec] = useState(SEGX_DEFAULTS.minSegSec)
  const [gapSec, setGapSec] = useState(SEGX_DEFAULTS.gapSec)
  // per target upload, the ranges the user left out
  const [excluded, setExcluded] = useState<Record<string, [number, number][]>>({})

  const searchJob = useJob(cfg.video_poll_ms)
  const exportJob = useJob(cfg.image_poll_ms)
  const { submit: submitSearch } = searchJob
  const { submit: submitExport, reset: resetExport } = exportJob

  const loadTarget = useCallback(
    async (i: number, file: File) => {
      resetExport()
      setActive(i)
      await slots[i].load(file)
    },
    [slots, resetExport],
  )
  const targetIds = slots.map((s) => s.video?.upload_id ?? null)
  const loaded = slots.filter((s) => s.info).length
  const uploadingTargets = slots.some((s) => s.info && !s.video)

  const classSig = classes.map((c) => [c.start, c.end, c.clip.upload_id, c.name, c.description])
  const signature = JSON.stringify([ref.video?.upload_id, classSig, targetIds, useDescription, candidates,
    knnK, knnThreshold, coverage, stride])
  const [searched, setSearched] = useState<string | null>(null)
  const run = useCallback(() => {
    const ids = targetIds.filter((i): i is string => !!i)
    if (!ref.video || !classes.length || !ids.length) return
    setError(null)
    setSearched(signature)
    setExcluded({})
    resetExport()
    void submitSearch('mcseg-search', {
      ref_upload_id: ref.video.upload_id,
      classes: classes.map((c, i) => ({
        name: c.name.trim() || `step ${i + 1}`, description: c.description, start: c.start, end: c.end,
        clip_id: c.clip.upload_id,
      })),
      target_ids: ids,
      candidates, knn_k: knnK, knn_threshold: knnThreshold, coverage, use_description: useDescription, stride,
    })
  }, [ref.video, classes, targetIds, signature, candidates, knnK, knnThreshold, coverage, useDescription,
    stride, submitSearch, resetExport])

  const result = searchJob.status === 'done' ? (searchJob.result as unknown as McResult | null) : null
  const stale = !!result && searched !== signature
  const shown = result?.targets.find((t) => t.upload_id === target.video?.upload_id) ?? null
  const nCls = result?.classes.length ?? classes.length
  const shownExcluded = useMemo(() => (shown ? excluded[shown.upload_id] ?? [] : []), [shown, excluded])

  const sel = useMemo(
    () => (shown ? selectMulti(shown, nCls, threshold, smoothSec, minSegSec, gapSec, shownExcluded) : null),
    [shown, nCls, threshold, smoothSec, minSegSec, gapSec, shownExcluded],
  )
  // the segment counts of every target, for the tabs
  const counts = useMemo(
    () => Object.fromEntries((result?.targets ?? []).map((t) => [t.upload_id,
      selectMulti(t, nCls, threshold, smoothSec, minSegSec, gapSec, excluded[t.upload_id] ?? []).segments.length])),
    [result, nCls, threshold, smoothSec, minSegSec, gapSec, excluded],
  )
  const segments = useMemo(() => sel?.segments ?? [], [sel])

  const fps = target.info?.fps || shown?.fps || 20
  const lastFrame = Math.max(0, (shown?.decoded_frames ?? target.info?.frames ?? 1) - 1)
  const { setFrameRaw: setTargetFrameRaw } = target
  const setFrame = useCallback(
    (f: number) => setTargetFrameRaw(Math.max(0, Math.min(lastFrame, Math.round(f)))),
    [lastFrame, setTargetFrameRaw],
  )

  // -- export (CPU): the active target's segments, every class --------------- //
  const doExport = useCallback(
    (opts: { video: boolean; zipEvery: number }) => {
      if (!shown || !segments.length || !result) return
      const cname = (i: number) => result.classes[i] ?? `step ${i + 1}`
      void submitExport('extract-export', {
        upload_id: shown.upload_id,
        segments: segments.map((s) => ({ start: s.start, end: s.end, peak: round(s.peak), mean: round(s.mean), best_ref: s.bestRef })),
        video: opts.video,
        zip_every: opts.zipEvery,
        decoded_frames: shown.decoded_frames,
        settings: {
          task: 'multiple class segmentation',
          model: 'Qwen3-VL-Embedding-8B candidates + Qwen3-VL-8B-Instruct multiple-choice clip classification',
          score: 'P(step) per ~2 s clip from the VLM, smoothed; best_ref is the index into classes',
          classes: result.classes,
          segments_by_class: Object.fromEntries(result.classes.map((n, i) => [n,
            segments.filter((s) => s.bestRef === i).map((s) => [s.start, s.end])])),
          vlm_options: result.options,
          class_labels: segments.map((s) => cname(s.bestRef)),
          threshold, smoothing_s: smoothSec, min_segment_s: minSegSec, merge_gap_s: gapSec,
          candidates: result.candidates,
          ...(result.knn ? { knn: result.knn } : { coverage: result.coverage }),
          stride: shown.stride,
          excluded_ranges: shownExcluded,
        },
        references: classes.map((c) => ({ kind: 'step clip', name: c.name, source: ref.info?.name, start: c.start, end: c.end })),
      })
    },
    [shown, segments, result, submitExport, threshold, smoothSec, minSegSec, gapSec, shownExcluded, classes, ref.info],
  )

  return {
    step, goto, reached: (classes.length ? 'search' : 'reference') as McStep, error, setError,
    // 1
    ref, refFrame: ref.frame, setRefFrame, refLast, refFps: ref.info?.fps || 20, loadRef,
    markIn, markOut, setMarkIn, setMarkOut, selection, cutting, markClass, newName, setNewName,
    classes, removeClass, updateClass, overlapping,
    // 2
    describeJob, describe, undescribed,
    // 3
    slots, active, setActive, target, loadTarget, loaded, uploadingTargets,
    useDescription, setUseDescription, candidates, setCandidates, knnK, setKnnK, knnThreshold, setKnnThreshold,
    coverage, setCoverage, stride, setStride,
    threshold, setThreshold, smoothSec, setSmoothSec, minSegSec, setMinSegSec, gapSec, setGapSec,
    run, searchJob, result, stale, shown, sel, segments, counts,
    excluded: shownExcluded,
    exclude: (s: Segment) => shown && setExcluded((x) => ({ ...x, [shown.upload_id]: [...(x[shown.upload_id] ?? []), [s.start, s.end]] })),
    clearExcluded: () => shown && setExcluded((x) => ({ ...x, [shown.upload_id]: [] })),
    frame: target.frame, setFrame, lastFrame, fps, embStride: shown?.stride ?? stride,
    exportJob, doExport,
  }
}

export type Mcseg = ReturnType<typeof useMcseg>

const round = (v: number) => Math.round(v * 1000) / 1000
