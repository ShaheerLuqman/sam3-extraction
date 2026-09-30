// State for the segment-extraction page, in three steps:
//
//   1. reference — mark where the step starts and ends on a reference video (or
//      several). Each marked range is cut, frame-exact, into a clip on the server.
//      Ranges of other steps can be marked too, as counter-examples.
//   2. describe  — the VLM watches the step clips and names and describes the step.
//   3. search    — upload another video; the server does the heavy part in one job
//      (embed -> candidates -> VLM clip classification, see backend/segx.py) and
//      returns P(step) for every `stride`-th candidate frame. Smoothing, the
//      threshold and segment cleanup run here, on those numbers, so they are live.
import { useCallback, useMemo, useRef, useState } from 'react'
import { api, type HealthConfig, type UploadInfo } from '../api/client'
import { useJob } from '../hooks/useJob'
import { usePlayback } from '../hooks/usePlayback'
import { closeLocal, grabFrames, openLocal, type LocalVideo } from './localVideo'
import { select, smooth, type Scored, type Segment } from './segments'

export type SegxStep = 'reference' | 'describe' | 'search'

export const SEGX_STEPS: { id: SegxStep; title: string; blurb: string }[] = [
  { id: 'reference', title: 'Mark the step', blurb: 'Its start and end on a reference video' },
  { id: 'describe', title: 'Describe it', blurb: 'The VLM names the step in words' },
  { id: 'search', title: 'Search a video', blurb: 'Find the step in another recording' },
]

export type ClipKind = 'step' | 'other'

/** A clip of the step (or of another step). `src` is set when it was marked on a
 *  reference video here, rather than uploaded ready-cut. */
export type Clip = {
  uid: string
  up: UploadInfo
  src?: { id: string; name: string; start: number; end: number }
}

export type SearchResult = {
  decoded_frames: number
  fps: number
  stride: number
  instruction: string
  options: { A: string; B: string }
  region: [number, number][]
  centers: number[]
  p: (number | null)[]
  coverage: number
  candidates?: 'similarity' | 'knn'
  knn?: { k: number; threshold: number; step_frames: number; other_frames: number } | null
  examples: { A: { clip: number; frame: number }[]; B: { clip: number; frame: number }[] }
}

// The research's post-processing, in seconds so it carries across frame rates:
// a 17-frame (~1 s) smoothing window, runs under ~1 s dropped, gaps under ~2 s
// filled, P(step) >= 0.5.
export const SEGX_DEFAULTS = { threshold: 0.5, smoothSec: 0.95, minSegSec: 1.0, gapSec: 2.0 }

let seq = 0
const uid = () => `c${++seq}`

/** What the pages show about a video, whichever copy they have. */
export type VideoInfo = { id: string; name: string; width: number; height: number; frames: number; fps: number }

/** A video with a frame cursor, as the viewers need it.
 *
 *  The user's file is opened in the browser first (lib/localVideo): playback and
 *  scrubbing then never touch the network. It goes to the server only when the
 *  server has to process it — `upload: 'always'` (a video to search) uploads it
 *  in the background while it is already on screen; `'fallback'` (a reference
 *  video, of which only marked frames are sent) uploads only if the browser
 *  cannot play it itself, and then the server's frames and playback copy are used. */
export function useVideo(cfg: HealthConfig, opts: { upload: 'always' | 'fallback' } = { upload: 'always' }) {
  const [video, setVideo] = useState<UploadInfo | null>(null)
  const [local, setLocal] = useState<LocalVideo | null>(null)
  const [opening, setOpening] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  // background upload progress, 0..1; null when none is running
  const [uploadPct, setUploadPct] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [frame, setFrameRaw] = useState(0)
  const playback = usePlayback(cfg.image_poll_ms)
  const { request: requestPlayback } = playback
  const fileRef = useRef<File | null>(null)
  const seqRef = useRef(0)
  const mode = opts.upload

  const upload = useCallback(
    async (file: File, seq: number, withPlayback: boolean): Promise<UploadInfo | null> => {
      setUploadPct(0)
      setError(null)
      try {
        const up = await api.uploadWithProgress(file, (f) => seq === seqRef.current && setUploadPct(f))
        if (seq !== seqRef.current) return null
        if (up.kind !== 'video') throw new Error(`expected a video, got a ${up.kind}`)
        const v = { ...up, name: up.name || file.name }
        setVideo(v)
        if (withPlayback) requestPlayback(up.upload_id)
        return v
      } catch (e) {
        if (seq === seqRef.current) setError((e as Error).message)
        return null
      } finally {
        if (seq === seqRef.current) setUploadPct(null)
      }
    },
    [requestPlayback],
  )

  const load = useCallback(
    async (file: File): Promise<UploadInfo | null> => {
      const seq = ++seqRef.current
      fileRef.current = file
      setFile(file)
      setOpening(true)
      setError(null)
      setVideo(null)
      setFrameRaw(0)
      const lv = await openLocal(file)
      if (seq !== seqRef.current) {
        closeLocal(lv)
        return null
      }
      setLocal((old) => {
        closeLocal(old)
        return lv
      })
      setOpening(false)
      if (!lv || mode === 'always') return upload(file, seq, !lv)
      return null
    },
    [mode, upload],
  )

  /** the browser could not play its own copy after all: use the server's */
  const localFail = useCallback(() => {
    setLocal((old) => {
      closeLocal(old)
      return null
    })
    if (video) requestPlayback(video.upload_id)
    else if (fileRef.current) void upload(fileRef.current, seqRef.current, true)
  }, [video, requestPlayback, upload])

  const retryUpload = useCallback(() => {
    if (fileRef.current) void upload(fileRef.current, seqRef.current, !local)
  }, [local, upload])

  const info: VideoInfo | null = local
    ? { id: local.id, name: local.name, width: local.width, height: local.height, frames: local.frames, fps: local.fps }
    : video
      ? {
          id: video.upload_id, name: video.name || 'video', width: video.width, height: video.height,
          frames: video.frames ?? 0, fps: video.fps || 20,
        }
      : null

  return {
    video, local, info, file, error, load, localFail, retryUpload, frame, setFrameRaw,
    // "busy" = nothing to show yet: opening the file, or uploading one the browser cannot play
    busy: opening || (!local && uploadPct !== null),
    uploadPct,
    playbackUrl: playback.url,
    playbackError: playback.error,
    playbackFail: playback.fail,
  }
}

export function useSegx(cfg: HealthConfig) {
  const [step, setStep] = useState<SegxStep>('reference')
  const [error, setError] = useState<string | null>(null)
  const [steps, setSteps] = useState<Clip[]>([])
  const [others, setOthers] = useState<Clip[]>([])
  const [uploading, setUploading] = useState(0)

  // -- 1. reference: mark the step on a video ------------------------------- //
  // only its marked frames go to the server, so it is not uploaded unless the
  // browser cannot play it
  const ref = useVideo(cfg, { upload: 'fallback' })
  const refLast = Math.max(0, (ref.info?.frames ?? 1) - 1)
  const { setFrameRaw: setRefFrameRaw } = ref
  const setRefFrame = useCallback(
    (f: number) => setRefFrameRaw(Math.max(0, Math.min(refLast, Math.round(f)))),
    [refLast, setRefFrameRaw],
  )
  const [markIn, setMarkIn] = useState<number | null>(null)
  const [markOut, setMarkOut] = useState<number | null>(null)
  // cutting a marked range: frames grabbed in the browser so far, of how many
  const [cutting, setCutting] = useState<{ done: number; total: number } | null>(null)

  const { load: loadRefVideo } = ref
  const loadRef = useCallback(
    async (file: File) => {
      setMarkIn(null)
      setMarkOut(null)
      await loadRefVideo(file)
    },
    [loadRefVideo],
  )

  // the selection, whichever way round the ends were set
  const selection = useMemo<[number, number] | null>(
    () => (markIn !== null && markOut !== null ? [Math.min(markIn, markOut), Math.max(markIn, markOut)] : null),
    [markIn, markOut],
  )

  const { local: refLocal, video: refVideo, info: refInfo, file: refFile } = ref
  // every reference video something was marked on, by the id its clips carry: its
  // file, so the kNN candidates can upload it (the default never does), and its upload
  const refSources = useRef(new Map<string, { file: File | null; name: string; uploadId: string | null }>())
  const addMarked = useCallback(
    async (kind: ClipKind) => {
      if (!refInfo || !selection) return
      const [start, end] = selection
      setCutting({ done: 0, total: end - start + 1 })
      setError(null)
      try {
        let up: UploadInfo
        if (refLocal) {
          // the frames come from the user's own copy; only they are uploaded
          const frames = await grabFrames(refLocal, start, end, (done, total) => setCutting({ done, total }))
          const stem = refLocal.name.replace(/\.[^.]+$/, '')
          up = await api.segxClipFrames(frames, refLocal.fps, `${stem} [${start}-${end}]`)
        } else if (refVideo) {
          up = await api.segxCut({ upload_id: refVideo.upload_id, start, end })
        } else return
        const clip: Clip = { uid: uid(), up, src: { id: refInfo.id, name: refInfo.name, start, end } }
        if (!refSources.current.has(refInfo.id))
          refSources.current.set(refInfo.id, { file: refFile, name: refInfo.name, uploadId: refVideo?.upload_id ?? null })
        ;(kind === 'step' ? setSteps : setOthers)((cs) => [...cs, clip])
        setMarkIn(null)
        setMarkOut(null)
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setCutting(null)
      }
    },
    [refLocal, refVideo, refInfo, refFile, selection],
  )

  const addClips = useCallback(async (files: File[], to: ClipKind) => {
    setError(null)
    setUploading((n) => n + files.length)
    for (const file of files) {
      try {
        const up = await api.upload(file)
        if (up.kind !== 'video') throw new Error(`${file.name} is not a video clip`)
        const clip = { uid: uid(), up: { ...up, name: up.name || file.name } }
        ;(to === 'step' ? setSteps : setOthers)((cs) => [...cs, clip])
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setUploading((n) => n - 1)
      }
    }
  }, [])

  const removeClip = useCallback((id: string) => {
    setSteps((cs) => cs.filter((c) => c.uid !== id))
    setOthers((cs) => cs.filter((c) => c.uid !== id))
  }, [])

  // -- 2. describe: the VLM names the step from the clips ------------------ //
  const describeJob = useJob(cfg.image_poll_ms)
  const { submit: submitDescribe } = describeJob
  const [nameEdit, setNameEdit] = useState<string | null>(null)
  const [descEdit, setDescEdit] = useState<string | null>(null)
  // which set of step clips the current description is of
  const [describedFor, setDescribedFor] = useState<string | null>(null)
  const stepIds = steps.map((c) => c.up.upload_id).join(',')
  const described =
    describeJob.status === 'done' ? (describeJob.result as { name?: string; description?: string } | null) : null
  const name = nameEdit ?? described?.name ?? ''
  const description = descEdit ?? described?.description ?? ''
  const edited = nameEdit !== null || descEdit !== null
  const describeStale = !!describedFor && describedFor !== stepIds

  const requestDescription = useCallback(() => {
    if (!stepIds) return
    setDescribedFor(stepIds)
    void submitDescribe('segx-describe', { clip_ids: stepIds.split(',') })
  }, [stepIds, submitDescribe])
  const describe = useCallback(() => {
    setNameEdit(null)
    setDescEdit(null)
    requestDescription()
  }, [requestDescription])

  // arriving at step 2 describes the step clips, unless that is already done
  // (or under way) for these clips, or the user has written their own
  const goto = useCallback(
    (s: SegxStep) => {
      setStep(s)
      if (s === 'describe' && !edited && describedFor !== stepIds) requestDescription()
    },
    [edited, describedFor, stepIds, requestDescription],
  )

  // -- 3. search ------------------------------------------------------------ //
  const target = useVideo(cfg)
  const [useDescription, setUseDescription] = useState(true)
  const [coverage, setCoverage] = useState(0.25)
  const [stride, setStride] = useState(5)
  // how the stretches the VLM checks are chosen: the similarity ranking, or the
  // research's two-group kNN vote (marked = step, the rest of the reference = not)
  const [candidates, setCandidates] = useState<'similarity' | 'knn'>('similarity')
  const [knnK, setKnnK] = useState(15)
  const [knnThreshold, setKnnThreshold] = useState(0.5)
  // uploading the reference videos for the kNN candidates: which of how many, 0..1
  const [refUpload, setRefUpload] = useState<{ i: number; n: number; pct: number } | null>(null)
  const [threshold, setThreshold] = useState(SEGX_DEFAULTS.threshold)
  const [smoothSec, setSmoothSec] = useState(SEGX_DEFAULTS.smoothSec)
  const [minSegSec, setMinSegSec] = useState(SEGX_DEFAULTS.minSegSec)
  const [gapSec, setGapSec] = useState(SEGX_DEFAULTS.gapSec)
  const [excluded, setExcluded] = useState<[number, number][]>([])

  const searchJob = useJob(cfg.video_poll_ms)
  const exportJob = useJob(cfg.image_poll_ms)
  const { submit: submitSearch } = searchJob
  const { submit: submitExport, reset: resetExport } = exportJob

  const video = target.video
  const { load: loadTarget } = target
  const loadVideo = useCallback(
    async (file: File) => {
      setExcluded([])
      resetExport()
      await loadTarget(file)
    },
    [loadTarget, resetExport],
  )

  const signature = JSON.stringify([video?.upload_id, stepIds, others.map((c) => c.up.upload_id),
    name, description, useDescription, coverage, stride, candidates, knnK, knnThreshold])
  const [searched, setSearched] = useState<string | null>(null)
  const run = useCallback(async () => {
    if (!video || !steps.length) return
    setError(null)
    let knn = {}
    if (candidates === 'knn') {
      // the reference videos the marks came from, each with its marked ranges
      const ids = [...new Set([...steps, ...others].flatMap((c) => (c.src ? [c.src.id] : [])))]
      const references = []
      for (const [i, id] of ids.entries()) {
        const src = refSources.current.get(id)
        if (!src) continue
        if (!src.uploadId) {
          if (!src.file) return setError(`The reference video ${src.name} is no longer open: load it again.`)
          setRefUpload({ i: i + 1, n: ids.length, pct: 0 })
          try {
            const up = await api.uploadWithProgress(src.file, (pct) => setRefUpload({ i: i + 1, n: ids.length, pct }))
            src.uploadId = up.upload_id
          } catch (e) {
            setRefUpload(null)
            return setError(`Uploading the reference video ${src.name}: ${(e as Error).message}`)
          }
        }
        const ranges = (cs: Clip[]) => cs.filter((c) => c.src?.id === id).map((c) => [c.src!.start, c.src!.end])
        references.push({ upload_id: src.uploadId, steps: ranges(steps), others: ranges(others) })
      }
      setRefUpload(null)
      knn = {
        candidates: 'knn',
        references,
        // ready-cut clips (no reference video) join the vote on their side
        knn_step_ids: steps.filter((c) => !c.src).map((c) => c.up.upload_id),
        knn_other_ids: others.filter((c) => !c.src).map((c) => c.up.upload_id),
        knn_k: knnK,
        knn_threshold: knnThreshold,
      }
    }
    setSearched(signature)
    setExcluded([])
    resetExport()
    void submitSearch('segx-search', {
      upload_id: video.upload_id,
      step_ids: steps.map((c) => c.up.upload_id),
      other_ids: others.map((c) => c.up.upload_id),
      name,
      description,
      use_description: useDescription,
      coverage,
      stride,
      ...knn,
    })
  }, [video, steps, others, name, description, useDescription, coverage, stride, candidates, knnK,
    knnThreshold, signature, submitSearch, resetExport])

  const result =
    searchJob.status === 'done' ? (searchJob.result as unknown as SearchResult | null) : null
  // a result for another video (the target was replaced) is not shown at all
  const shown = result && searched && JSON.parse(searched)[0] === video?.upload_id ? result : null
  const stale = !!shown && searched !== signature

  const fps = target.info?.fps || shown?.fps || 20
  const lastFrame = Math.max(0, (shown?.decoded_frames ?? target.info?.frames ?? 1) - 1)
  const { setFrameRaw: setTargetFrameRaw } = target
  const setFrame = useCallback(
    (f: number) => setTargetFrameRaw(Math.max(0, Math.min(lastFrame, Math.round(f)))),
    [lastFrame, setTargetFrameRaw],
  )

  // -- selection (browser, live) -------------------------------------------- //
  const scored: Scored | null = useMemo(() => {
    if (!shown || !shown.centers.length) return null
    const { centers, stride: st, decoded_frames: n } = shown
    // smooth P(step) within each run of consecutive candidate clips
    const p = Float32Array.from(shown.p, (v) => (v == null ? 0 : v))
    const sm = new Float32Array(p.length)
    const win = Math.max(1, Math.round((smoothSec * fps) / st))
    for (let a = 0; a < centers.length; ) {
      let b = a
      while (b + 1 < centers.length && centers[b + 1] - centers[b] === st) b++
      sm.set(smooth(p.slice(a, b + 1), win), a)
      a = b + 1
    }
    const inRegion = new Uint8Array(n)
    for (const [a, b] of shown.region) inRegion.fill(1, a, b + 1)
    const perFrame = new Float32Array(n)
    let j = 0
    for (let f = 0; f < n; f++) {
      if (!inRegion[f]) continue
      while (j + 1 < centers.length && Math.abs(centers[j + 1] - f) <= Math.abs(centers[j] - f)) j++
      perFrame[f] = sm[j]
    }
    // the timeline's curve: one value per `st` frames, a gap where the VLM did not look
    const score = new Float32Array(Math.ceil(n / st))
    for (let i = 0; i < score.length; i++) {
      const f = Math.min(n - 1, i * st)
      score[i] = inRegion[f] ? perFrame[f] : NaN
    }
    return { score, perFrame, best: new Int16Array(score.length) }
  }, [shown, smoothSec, fps])

  const segments: Segment[] = useMemo(() => {
    if (!scored || !shown) return []
    return select(scored, threshold, Math.max(1, Math.round(minSegSec * fps)),
      Math.max(0, Math.round(gapSec * fps)), excluded, shown.stride)
  }, [scored, shown, threshold, minSegSec, gapSec, fps, excluded])

  // -- export (CPU) ---------------------------------------------------------- //
  const doExport = useCallback(
    (opts: { video: boolean; zipEvery: number }) => {
      if (!video || !segments.length || !shown) return
      const clipRef = (kind: string) => (c: Clip) => ({
        kind, name: c.up.name, frames: c.up.frames,
        ...(c.src ? { source: c.src.name, start: c.src.start, end: c.src.end } : {}),
      })
      void submitExport('extract-export', {
        upload_id: video.upload_id,
        segments: segments.map((s) => ({ start: s.start, end: s.end, peak: round(s.peak), mean: round(s.mean) })),
        video: opts.video,
        zip_every: opts.zipEvery,
        decoded_frames: shown.decoded_frames,
        settings: {
          task: 'segment extraction',
          model: 'Qwen3-VL-Embedding-8B candidates + Qwen3-VL-8B-Instruct clip classification',
          score: 'P(step) per ~2 s clip from the VLM, smoothed; peak/mean are that probability',
          step_name: name,
          step_description: description,
          description_in_embedding: useDescription,
          vlm_options: shown.options,
          threshold,
          smoothing_s: smoothSec,
          min_segment_s: minSegSec,
          merge_gap_s: gapSec,
          candidates: shown.candidates ?? 'similarity',
          ...(shown.knn ? { knn: shown.knn } : { coverage }),
          stride: shown.stride,
          excluded_ranges: excluded,
        },
        references: [...steps.map(clipRef('step clip')), ...others.map(clipRef('other clip'))],
      })
    },
    [video, segments, shown, submitExport, name, description, useDescription, threshold,
      smoothSec, minSegSec, gapSec, coverage, excluded, steps, others],
  )

  return {
    step, goto, reached: (steps.length ? 'search' : 'reference') as SegxStep,
    error, steps, others, uploading, addClips, removeClip,
    // 1
    ref, refFrame: ref.frame, setRefFrame, refLast, refFps: ref.info?.fps || 20, loadRef,
    markIn, markOut, setMarkIn, setMarkOut, selection, cutting, addMarked,
    // 2
    name, setName: setNameEdit, description, setDescription: setDescEdit, edited, describe,
    describeJob, describeStale,
    // 3
    video, videoBusy: target.busy, videoError: target.error, loadVideo, target,
    useDescription, setUseDescription, coverage, setCoverage, stride, setStride,
    candidates, setCandidates, knnK, setKnnK, knnThreshold, setKnnThreshold, refUpload,
    threshold, setThreshold, smoothSec, setSmoothSec, minSegSec, setMinSegSec, gapSec, setGapSec,
    run, searchJob, result: shown, stale, scored, segments,
    excluded, exclude: (s: Segment) => setExcluded((x) => [...x, [s.start, s.end]]),
    clearExcluded: () => setExcluded([]),
    frame: target.frame, setFrame, lastFrame, fps, embStride: shown?.stride ?? stride,
    exportJob, doExport,
    playbackUrl: target.playbackUrl,
    playbackError: target.playbackError,
    playbackFail: target.playbackFail,
  }
}

export type Segx = ReturnType<typeof useSegx>

const round = (v: number) => Math.round(v * 1000) / 1000

/** The marked clips of `video`, as segments the viewer can play and step through. */
export function markedSegments(clips: Clip[], videoId: string | undefined): Segment[] {
  return clips
    .filter((c) => c.src && c.src.id === videoId)
    .map((c) => ({ start: c.src!.start, end: c.src!.end, peak: 1, peakFrame: c.src!.start, mean: 1, bestRef: 0 }))
}
