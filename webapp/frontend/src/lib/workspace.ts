import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type HealthConfig } from '../api/client'
import { useJob } from '../hooks/useJob'
import { useUpload } from '../hooks/useUpload'
import { colorFor, fromPayload, instanceName, type ClassDef, type Instance } from './objects'

export type Step = 'video' | 'detect' | 'result'

export const STEPS: { id: Step; title: string; blurb: string }[] = [
  { id: 'video', title: 'Upload video', blurb: 'The clip to process, plus an optional classes.txt' },
  { id: 'detect', title: 'Detect objects', blurb: 'Segment, box, or text-prompt objects on a frame' },
  { id: 'result', title: 'Track & export', blurb: 'Run object tracking, then download the results' },
]

/** Wait this long after the frame stops changing before embedding it. */
const WARM_DELAY_MS = 500
/** ...and this long after the tracking window stops changing before preparing it. */
const PREP_DELAY_MS = 1000

/**
 * Everything that has to outlive a step change: the upload, the class list, the
 * instances defined so far, and the tracking job. Whatever only matters during
 * detection — the active prompt type, an uncommitted mask — stays in that step.
 */
export function useWorkspace(cfg: HealthConfig) {
  const upload = useUpload('video')
  /** frame embeddings, computed ahead of the first prompt so detection is instant */
  const warmup = useJob(cfg.image_poll_ms)
  /** a tracking run's CPU preamble, done while the user annotates. Runs on the
   *  backend's CPU worker, so it never competes with `warmup` for the GPU. */
  const clipPrep = useJob(cfg.video_poll_ms)
  const track = useJob(cfg.video_poll_ms)

  const [step, setStep] = useState<Step>('video')
  const [classes, setClasses] = useState<ClassDef[]>([])
  const [classSource, setClassSource] = useState<string | null>(null)
  const [objects, setObjects] = useState<Instance[]>([])
  const [frame, setFrame] = useState(0)
  const [maxFrames, setMaxFrames] = useState(cfg.default_max_frames)
  const [threshold, setThreshold] = useState(0.5)
  const [bidirectional, setBidirectional] = useState(false)

  const info = upload.info
  /** the last frame that exists — deliberately not tied to `maxFrames`, so you
   *  can scrub anywhere in the clip to place a prompt */
  const frameCap = Math.max(0, (info?.frames ?? 1) - 1)
  const curFrame = Math.min(frame, frameCap)

  // the label list lives on the server, so a reload keeps it
  useEffect(() => {
    api
      .classes()
      .then((r) => {
        setClasses(r.classes)
        setClassSource(r.source)
      })
      .catch(() => {})
  }, [])

  // -- frame embeddings --------------------------------------------------- #
  const warmSubmit = warmup.submit
  const warmReset = warmup.reset
  const warmedRef = useRef('')
  const warmFrame = useCallback(
    (uploadId: string, idx: number) => {
      const key = `${uploadId}:${idx}`
      if (warmedRef.current === key) return
      warmedRef.current = key
      void warmSubmit('prepare', { upload_id: uploadId, frame: idx })
    },
    [warmSubmit],
  )

  /** Set while the model is doing something the user is waiting on, so a
   *  background embedding never queues ahead of it. */
  const busyRef = useRef(false)
  const notifyBusy = useCallback((busy: boolean) => {
    busyRef.current = busy
  }, [])

  useEffect(() => {
    if (!info || step !== 'detect') return
    const t = setTimeout(() => {
      if (!busyRef.current) warmFrame(info.upload_id, curFrame)
    }, WARM_DELAY_MS)
    return () => clearTimeout(t)
  }, [info, curFrame, step, warmFrame])

  // -- preparing the clip for tracking ------------------------------------ #
  const prepSubmit = clipPrep.submit
  const prepReset = clipPrep.reset
  /** how many frames the backend has been asked to prepare so far */
  const preppedRef = useRef(0)
  const prepClip = useCallback(
    (uploadId: string, frames: number) => {
      if (frames <= preppedRef.current) return
      preppedRef.current = frames
      void prepSubmit('prep-clip', { upload_id: uploadId, frames })
    },
    [prepSubmit],
  )

  // widening the window means there is more to prepare; debounced so dragging
  // the slider doesn't start a job per step
  useEffect(() => {
    if (!info || maxFrames <= preppedRef.current) return
    const t = setTimeout(() => prepClip(info.upload_id, maxFrames), PREP_DELAY_MS)
    return () => clearTimeout(t)
  }, [info, maxFrames, prepClip])

  // -- the video --------------------------------------------------------- #
  const doUpload = upload.upload
  const trackReset = track.reset
  const loadVideo = useCallback(
    async (file: File) => {
      setObjects([])
      setFrame(0)
      trackReset()
      warmReset()
      prepReset()
      warmedRef.current = ''
      preppedRef.current = 0
      const res = await doUpload(file)
      if (!res) return
      const total = res.frames ?? cfg.default_max_frames
      const window = Math.max(cfg.min_frames, Math.min(cfg.default_max_frames, total))
      setMaxFrames(window)
      // the whole point of doing this now: by the time anyone reaches step 2,
      // frame 0 is embedded and the tracking window is being prepared in the
      // background, so nothing is waiting on either
      warmFrame(res.upload_id, 0)
      prepClip(res.upload_id, window)
    },
    [cfg.default_max_frames, cfg.min_frames, doUpload, prepClip, prepReset, trackReset,
     warmFrame, warmReset],
  )

  // -- the marked objects ------------------------------------------------ #
  const patch = useCallback(
    (id: string, p: Partial<Instance>) =>
      setObjects((os) => os.map((o) => (o.id === id ? { ...o, ...p } : o))),
    [],
  )

  /** `make` sees the list as it stands, so new names don't collide. */
  const addObjects = useCallback(
    (make: (existing: Instance[]) => Instance[]) => setObjects((os) => [...os, ...make(os)]),
    [],
  )

  const removeObject = useCallback(
    (id: string) => setObjects((os) => os.filter((o) => o.id !== id)),
    [],
  )

  /** Marking something past the tracking window would silently not be tracked,
   *  so committing a prompt there widens the window to include it. */
  const ensureTracked = useCallback(
    (f: number) => setMaxFrames((m) => Math.max(m, f + 1)),
    [],
  )

  /** Changing an instance's class re-labels and re-colours it to match. */
  const relabel = useCallback(
    (id: string, cls: number | null) =>
      setObjects((os) => {
        const i = os.findIndex((o) => o.id === id)
        if (i < 0) return os
        const others = os.filter((o) => o.id !== id)
        return os.map((o) =>
          o.id === id
            ? { ...o, cls, color: colorFor(cls, i), name: instanceName(cls, classes, others) }
            : o,
        )
      }),
    [classes],
  )

  /** Replay a stored run: its settings always, its instances when they belong
   *  to the upload that is open (otherwise their coordinates mean nothing). */
  const restoreRun = useCallback(
    (run: { settings: { max_frames: number; threshold: number; bidirectional: boolean }
            objects?: unknown[] },
     withObjects: boolean) => {
      setMaxFrames(run.settings.max_frames)
      setThreshold(run.settings.threshold)
      setBidirectional(run.settings.bidirectional)
      if (withObjects) {
        const replayed = (run.objects ?? [])
          .map((o, i) => fromPayload(o, i))
          .filter((o): o is Instance => o !== null)
        setObjects(replayed)
        setStep('detect')
      }
    },
    [],
  )

  const uploadClear = upload.clear
  const startOver = useCallback(() => {
    setObjects([])
    setFrame(0)
    trackReset()
    warmReset()
    prepReset()
    warmedRef.current = ''
    preppedRef.current = 0
    uploadClear()
    setStep('video')
  }, [prepReset, trackReset, uploadClear, warmReset])

  return {
    cfg,
    step,
    setStep,
    upload,
    info,
    warmup,
    clipPrep,
    track,
    classes,
    classSource,
    setClassList: (c: ClassDef[], src: string | null) => {
      setClasses(c)
      setClassSource(src)
    },
    objects,
    addObjects,
    patch,
    removeObject,
    relabel,
    frame: curFrame,
    setFrame,
    frameCap,
    maxFrames,
    setMaxFrames,
    threshold,
    setThreshold,
    bidirectional,
    setBidirectional,
    loadVideo,
    restoreRun,
    ensureTracked,
    notifyBusy,
    startOver,
  }
}

export type Workspace = ReturnType<typeof useWorkspace>
