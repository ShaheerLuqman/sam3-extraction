// Empty base -> Vite proxy in dev / same-origin when FastAPI serves dist.
// Set VITE_API_BASE to the tunnel URL when the frontend moves to a static host.
export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? ''

export type HealthConfig = {
  min_frames: number
  default_max_frames: number
  image_poll_ms: number
  video_poll_ms: number
  upload_max_bytes: number
  extract?: {
    available: boolean
    default_stride: number
    default_instruction: string
    model: string
    /** physical GPU Qwen runs on */
    gpu: string
    /** false = it shares SAM 3's GPU and parks SAM 3 for each run */
    dedicated_gpu: boolean
    keep_warm_s: number
  }
}

export type Health = {
  /** busy = SAM 3 is parked in RAM while frame extraction has the GPU */
  status: 'loading' | 'ready' | 'busy' | 'error'
  models: { video: boolean }
  gpu: { name?: string; visible_devices?: string; mem_free_mb?: number; mem_total_mb?: number }
  load_error: string | null
  ffmpeg: boolean
  /** frame extraction's embedding worker */
  qwen?: QwenStatus
  config: HealthConfig
}

export type QwenStatus = {
  gpu: string
  dedicated: boolean
  state: 'off' | 'starting' | 'ready' | 'busy'
  keep_warm_s: number
  idle_s: number | null
}

export type UploadInfo = {
  upload_id: string
  kind: 'image' | 'video'
  /** the file name it was uploaded under */
  name?: string
  width: number
  height: number
  frames?: number
  fps?: number
  duration?: number
}

export type ClassesResponse = {
  classes: { id: number; name: string }[]
  count: number
  source: string | null
  warning?: string
}

export type JobResult = Record<string, unknown> & {
  message?: string
  tracked_video_url?: string
  json_url?: string
  mime?: string
  codec_warning?: string | null
  objects?: number
  frames?: number
}

export type RunSettings = {
  max_frames: number
  threshold: number
  bidirectional: boolean
}

/** One tracking run as the history lists it. */
export type RunSummary = {
  id: string
  status: 'running' | 'done' | 'error'
  created_at: number
  finished_at: number | null
  error: string | null
  upload_id: string
  source: string
  width: number
  height: number
  source_frames?: number | null
  fps?: number | null
  settings: RunSettings
  class_names?: Record<string, string>
  result?: JobResult | null
  /** false once the sweeper has removed the video/JSON this points at */
  outputs_present: boolean
  object_count: number
}

/** ...and in full, with the instances that were prompted. */
export type RunDetail = Omit<RunSummary, 'object_count'> & {
  objects: unknown[]
}

/** z-scored similarity rows, one per reference, over the embedded frames */
export type ExtractScores = {
  frames: number[]
  decoded_frames: number
  rows: number[][]
}

export type ExtractRefBody = { kind: 'image'; id: string } | { kind: 'frame'; frame: number }

export type JobSnapshot = {
  id: string
  kind: string
  status: 'queued' | 'running' | 'done' | 'error'
  progress: number
  stage: string
  queued_ahead: number
  result: JobResult | null
  error: string | null
}

/** An HTTP error from the backend. A failure to reach it at all is a plain
 *  TypeError from fetch (or a TimeoutError) instead. */
export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

/** True when the backend could not be reached or did not answer in time — worth
 *  retrying — as opposed to it answering with an error. 502-504 are the dev
 *  proxy's way of saying the backend is not answering. */
export function isUnreachable(e: unknown): boolean {
  if (e instanceof ApiError) return e.status >= 502 && e.status <= 504
  return true
}

async function req<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const { timeoutMs, ...rest } = init ?? {}
  const signal = timeoutMs ? AbortSignal.timeout(timeoutMs) : rest.signal
  const res = await fetch(API_BASE + path, { ...rest, signal })
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      /* keep statusText */
    }
    throw new ApiError(typeof detail === 'string' ? detail : JSON.stringify(detail), res.status)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => req<Health>('/api/health', { timeoutMs: 6000 }),

  upload: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return req<UploadInfo>('/api/uploads', { method: 'POST', body: fd })
  },

  /** upload with progress (0..1), which fetch cannot report */
  uploadWithProgress: (file: File, onProgress: (f: number) => void) =>
    new Promise<UploadInfo>((resolve, reject) => {
      const x = new XMLHttpRequest()
      x.open('POST', API_BASE + '/api/uploads')
      x.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
      x.onload = () => {
        let body: { detail?: string } & Partial<UploadInfo> = {}
        try {
          body = JSON.parse(x.responseText)
        } catch {
          /* not JSON */
        }
        if (x.status >= 200 && x.status < 300) resolve(body as UploadInfo)
        else reject(new ApiError(body.detail ?? `upload failed (${x.status})`, x.status))
      }
      x.onerror = () => reject(new Error('the upload was cut off: the backend could not be reached'))
      const fd = new FormData()
      fd.append('file', file)
      x.send(fd)
    }),

  /** a clip marked in the browser, as its frames (JPEG, in order) */
  segxClipFrames: (frames: Blob[], fps: number, name: string) => {
    const fd = new FormData()
    frames.forEach((b, i) => fd.append('frames', b, `${String(i).padStart(6, '0')}.jpg`))
    fd.append('fps', String(fps))
    fd.append('name', name)
    return req<UploadInfo>('/api/segx/clip-frames', { method: 'POST', body: fd })
  },

  classes: () => req<ClassesResponse>('/api/classes'),

  uploadClasses: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return req<ClassesResponse>('/api/classes', { method: 'POST', body: fd })
  },

  clearClasses: () => req<ClassesResponse>('/api/classes', { method: 'DELETE' }),

  frameUrl: (uploadId: string, idx: number) =>
    `${API_BASE}/api/uploads/${uploadId}/frame/${idx}.jpg`,

  fileUrl: (url: string) => API_BASE + url,

  startJob: (kind: string, body: unknown) =>
    req<{ job_id: string }>(`/api/jobs/${kind}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }),

  job: (id: string) => req<JobSnapshot>(`/api/jobs/${id}`, { timeoutMs: 15000 }),

  cancelJob: (id: string) =>
    req<{ cancelled: boolean }>(`/api/jobs/${id}`, { method: 'DELETE' }),

  extractScore: (body: { upload_id: string; key: string; refs: ExtractRefBody[] }) =>
    req<ExtractScores>('/api/extract/score', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }),

  /** frames start..end (inclusive) of a video, cut into an upload of its own */
  segxCut: (body: { upload_id: string; start: number; end: number }) =>
    req<UploadInfo & { source_id: string; start: number; end: number }>('/api/segx/cut', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }),

  runs: () => req<{ runs: RunSummary[]; count: number }>('/api/runs'),

  run: (id: string) => req<RunDetail>(`/api/runs/${id}`),

  deleteRun: (id: string) =>
    req<{ deleted: boolean }>(`/api/runs/${id}`, { method: 'DELETE' }),
}
