// State for the frame-extraction page: a video, a handful of references, and
// the segments of the video that look like them.
//
// GPU work happens only in `run()` (embedding, on the server's inference worker).
// A reference picked from the video itself needs only a score call, which is CPU
// and milliseconds, and threshold/smoothing changes never leave the browser.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, type ExtractRefBody, type HealthConfig } from '../api/client'
import { useJob } from '../hooks/useJob'
import { useVideo } from './segx'
import { scoreRows, select, type Segment } from './segments'

export type Ref =
  | { uid: string; kind: 'image'; uploadId: string; name: string; thumb: string; enabled: boolean }
  | { uid: string; kind: 'frame'; frame: number; enabled: boolean }

/** What the video was last embedded with — scores are only valid against this. */
type Embedded = {
  videoId: string
  key: string
  stride: number
  instruction: string
  decoded: number
  images: Set<string>
}

// The research tuned these at ~17.5 fps: a 17-frame smoothing window, runs under
// 18 frames dropped, gaps under 35 frames filled. In seconds they carry over to
// other frame rates. z >= 1.5 was the best fixed threshold on every step tested.
export const DEFAULTS = { threshold: 1.5, smoothSec: 0.95, minSegSec: 1.0, gapSec: 2.0 }

let uidSeq = 0
const uid = () => `r${++uidSeq}`

export function useExtraction(cfg: HealthConfig) {
  const ex = cfg.extract
  // played from the user's own file; uploaded in the background for embedding
  const src = useVideo(cfg)
  const video = src.video
  const [error, setError] = useState<string | null>(null)
  const [refs, setRefs] = useState<Ref[]>([])
  const [refsUploading, setRefsUploading] = useState(0)

  const [stride, setStride] = useState(ex?.default_stride ?? 5)
  const [instruction, setInstruction] = useState(ex?.default_instruction ?? '')
  const [threshold, setThreshold] = useState(DEFAULTS.threshold)
  const [smoothSec, setSmoothSec] = useState(DEFAULTS.smoothSec)
  const [minSegSec, setMinSegSec] = useState(DEFAULTS.minSegSec)
  const [gapSec, setGapSec] = useState(DEFAULTS.gapSec)

  const [embedded, setEmbedded] = useState<Embedded | null>(null)
  const [frames, setFrames] = useState<number[] | null>(null)
  const [rows, setRows] = useState<Map<string, number[]>>(new Map())
  const [excluded, setExcluded] = useState<[number, number][]>([])
  const [frame, setFrameRaw] = useState(0)

  const embedJob = useJob(cfg.video_poll_ms)
  const exportJob = useJob(cfg.image_poll_ms)
  const submitted = useRef<Omit<Embedded, 'decoded'> | null>(null)
  const { submit: submitEmbed } = embedJob
  const { submit: submitExport, reset: resetExport } = exportJob

  const lastFrame = Math.max(0, (embedded?.decoded ?? src.info?.frames ?? 1) - 1)
  const setFrame = useCallback(
    (f: number) => setFrameRaw(Math.max(0, Math.min(lastFrame, Math.round(f)))),
    [lastFrame],
  )

  // -- inputs ------------------------------------------------------------- //
  const { load: loadSrc } = src
  const loadVideo = useCallback(async (file: File) => {
    setError(null)
    // frame references belong to the old video; images carry over
    setRefs((rs) => rs.filter((r) => r.kind === 'image'))
    setEmbedded(null)
    setFrames(null)
    setRows(new Map())
    setExcluded([])
    setFrameRaw(0)
    resetExport()
    await loadSrc(file)
  }, [resetExport, loadSrc])

  const addImages = useCallback(async (files: File[]) => {
    setError(null)
    setRefsUploading((n) => n + files.length)
    for (const file of files) {
      try {
        const up = await api.upload(file)
        if (up.kind !== 'image') throw new Error(`${file.name} is not an image`)
        const ref: Ref = {
          uid: uid(),
          kind: 'image',
          uploadId: up.upload_id,
          name: file.name,
          thumb: URL.createObjectURL(file),
          enabled: true,
        }
        setRefs((rs) => [...rs, ref])
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setRefsUploading((n) => n - 1)
      }
    }
  }, [])

  const addFrameRef = useCallback((f: number) => {
    setRefs((rs) =>
      rs.some((r) => r.kind === 'frame' && r.frame === f)
        ? rs
        : [...rs, { uid: uid(), kind: 'frame', frame: f, enabled: true }],
    )
  }, [])

  const removeRef = useCallback((id: string) => {
    setRefs((rs) => {
      const r = rs.find((x) => x.uid === id)
      if (r?.kind === 'image') URL.revokeObjectURL(r.thumb)
      return rs.filter((x) => x.uid !== id)
    })
  }, [])

  const toggleRef = useCallback((id: string) => {
    setRefs((rs) => rs.map((r) => (r.uid === id ? { ...r, enabled: !r.enabled } : r)))
  }, [])

  // -- embedding (GPU) ---------------------------------------------------- //
  const instr = instruction.trim() || ex?.default_instruction || ''
  const current =
    !!video &&
    !!embedded &&
    embedded.videoId === video.upload_id &&
    embedded.stride === stride &&
    embedded.instruction === instr
  const newImages = refs.filter(
    (r) => r.kind === 'image' && !(current && embedded!.images.has(r.uploadId)),
  ).length
  /** what the next run() would have to embed, or null when nothing */
  const pending: null | { video: boolean; images: number } =
    !video ? null : !current ? { video: true, images: newImages } : newImages ? { video: false, images: newImages } : null

  const run = useCallback(() => {
    if (!video) return
    const imageIds = refs.flatMap((r) => (r.kind === 'image' ? [r.uploadId] : []))
    const reuse = current ? embedded!.images : new Set<string>()
    submitted.current = {
      videoId: video.upload_id,
      key: '',
      stride,
      instruction: instr,
      images: new Set([...reuse, ...imageIds]),
    }
    void submitEmbed('extract-embed', {
      upload_id: video.upload_id,
      image_ids: imageIds,
      stride,
      instruction: instr,
    })
  }, [video, refs, current, embedded, stride, instr, submitEmbed])

  useEffect(() => {
    const r = embedJob.result as { key?: string; decoded_frames?: number } | null
    if (embedJob.status !== 'done' || !r?.key || !submitted.current) return
    const s = submitted.current
    submitted.current = null
    // a new key (or video) invalidates every row; the same key keeps them
    if (!embedded || embedded.key !== r.key || embedded.videoId !== s.videoId) setRows(new Map())
    setEmbedded({ ...s, key: r.key, decoded: r.decoded_frames ?? 0 })
  }, [embedJob.status, embedJob.result, embedded])

  // -- scores (CPU): fetch a row for every reference that lacks one ------- //
  const [scoreError, setScoreError] = useState<string | null>(null)
  const inflight = useRef(new Set<string>())
  // a reply that lands after a re-embed belongs to the old key: drop it
  const liveKey = useRef<string | null>(null)
  useEffect(() => {
    liveKey.current = current && embedded ? `${embedded.videoId}/${embedded.key}` : null
  })
  useEffect(() => {
    if (!video || !current || !embedded) return
    const need = refs.filter(
      (r) =>
        !rows.has(r.uid) &&
        !inflight.current.has(r.uid) &&
        (r.kind === 'frame' || embedded.images.has(r.uploadId)),
    )
    if (!need.length) return
    need.forEach((r) => inflight.current.add(r.uid))
    const body: ExtractRefBody[] = need.map((r) =>
      r.kind === 'image' ? { kind: 'image', id: r.uploadId } : { kind: 'frame', frame: r.frame },
    )
    const key = embedded.key
    const tag = `${embedded.videoId}/${key}`
    api
      .extractScore({ upload_id: video.upload_id, key, refs: body })
      .then((res) => {
        if (liveKey.current !== tag) return
        setFrames(res.frames)
        setRows((prev) => {
          const next = new Map(prev)
          need.forEach((r, i) => next.set(r.uid, res.rows[i]))
          return next
        })
        setScoreError(null)
      })
      .catch((e) => setScoreError((e as Error).message))
      .finally(() => need.forEach((r) => inflight.current.delete(r.uid)))
  }, [video, current, embedded, refs, rows])

  // -- selection (browser, live) ------------------------------------------ //
  const fps = video?.fps || 20
  const embStride = embedded?.stride ?? stride
  const active = useMemo(
    () => (current ? refs.filter((r) => r.enabled && rows.has(r.uid)) : []),
    [current, refs, rows],
  )
  const scored = useMemo(() => {
    if (!active.length || !embedded || !frames) return null
    const win = Math.max(1, Math.round((smoothSec * fps) / embStride))
    return scoreRows(active.map((r) => rows.get(r.uid)!), win, embStride, embedded.decoded)
  }, [active, rows, embedded, frames, smoothSec, fps, embStride])

  const segments: Segment[] = useMemo(() => {
    if (!scored) return []
    const minSeg = Math.max(1, Math.round(minSegSec * fps))
    const gap = Math.max(0, Math.round(gapSec * fps))
    return select(scored, threshold, minSeg, gap, excluded, embStride)
  }, [scored, threshold, minSegSec, gapSec, fps, excluded, embStride])

  const exclude = useCallback((s: Segment) => {
    setExcluded((x) => [...x, [s.start, s.end]])
  }, [])

  // -- export (CPU) -------------------------------------------------------- //
  const doExport = useCallback(
    (opts: { video: boolean; zipEvery: number }) => {
      if (!video || !segments.length) return
      void submitExport('extract-export', {
        upload_id: video.upload_id,
        segments: segments.map((s) => ({
          start: s.start,
          end: s.end,
          peak: round(s.peak),
          mean: round(s.mean),
          best_ref: refs.indexOf(active[s.bestRef]),
        })),
        video: opts.video,
        zip_every: opts.zipEvery,
        decoded_frames: embedded?.decoded,
        settings: {
          threshold,
          smoothing_s: smoothSec,
          min_segment_s: minSegSec,
          merge_gap_s: gapSec,
          stride: embStride,
          instruction: embedded?.instruction,
          excluded_ranges: excluded,
        },
        references: refs.map((r, i) => ({
          index: i,
          enabled: r.enabled,
          ...(r.kind === 'image' ? { kind: 'image', name: r.name } : { kind: 'frame', frame: r.frame }),
        })),
      })
    },
    [video, segments, submitExport, refs, active, embedded, threshold, smoothSec, minSegSec, gapSec, embStride, excluded],
  )

  return {
    available: !!ex?.available,
    model: ex?.model ?? 'Qwen3-VL-Embedding',
    defaultInstruction: ex?.default_instruction ?? '',
    video, videoBusy: src.busy, loadVideo, error: error ?? src.error,
    refs, refsUploading, addImages, addFrameRef, removeRef, toggleRef,
    stride, setStride, instruction, setInstruction,
    threshold, setThreshold, smoothSec, setSmoothSec, minSegSec, setMinSegSec, gapSec, setGapSec,
    embedJob, run, pending, embedded: current ? embedded : null,
    scored, scoreError, active, rows, segments,
    excluded, exclude, clearExcluded: () => setExcluded([]),
    frame, setFrame, lastFrame, fps, embStride,
    exportJob, doExport,
    playbackUrl: src.playbackUrl,
    playbackError: src.playbackError,
    playbackFail: src.playbackFail,
    src,
  }
}

export type Extraction = ReturnType<typeof useExtraction>

const round = (v: number) => Math.round(v * 1000) / 1000
