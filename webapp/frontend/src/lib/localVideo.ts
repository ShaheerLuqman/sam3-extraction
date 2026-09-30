// The user's own copy of a video, played and scrubbed in the browser.
//
// The file the user picked is already on their machine, so there is no reason to
// pull frames or a playback copy of it back over the (often slow, VS Code
// forwarded) connection to the backend. The catch is frame numbers: the server
// counts frames the way cv2 decodes them, and these sources' timestamps drift,
// so "frame N" cannot be computed as N / fps. Instead the MP4's own sample table
// (read with mp4box.js, from the moov box only) gives every frame's presentation
// time, and seeking and playback go through that table both ways — frame-exact
// with the server.
//
// MP4/MOV only, and only codecs this browser decodes. Anything else returns null
// and the page falls back to the server (upload, JPEG frames, playback copy).
import { createFile } from 'mp4box'

export type LocalVideo = {
  /** stands in for an upload id until (or instead of) the upload */
  id: string
  file: File
  /** object URL of `file` */
  url: string
  name: string
  width: number
  height: number
  fps: number
  frames: number
  /** presentation time (s) of each frame in display order, on the <video>'s clock */
  times: Float64Array
}

let seq = 0

export async function openLocal(file: File): Promise<LocalVideo | null> {
  if (!/\.(mp4|m4v|mov)$/i.test(file.name) && !/mp4|quicktime/.test(file.type)) return null
  try {
    const boxes = await topLevelBoxes(file)
    const moov = boxes.find((b) => b.type === 'moov')
    if (!moov || moov.size > 256 * 1024 * 1024) return null
    const mp4 = createFile()
    let info: Awaited<ReturnType<typeof ready>> | null = null
    const ready = () =>
      new Promise<Parameters<NonNullable<typeof mp4.onReady>>[0]>((resolve, reject) => {
        mp4.onReady = resolve
        mp4.onError = (_m: string, e: string) => reject(new Error(e))
      })
    const got = ready()
    // ftyp + moov back to back, as if the file were just those: the sample table
    // (times) is all that is needed, and it does not care where the mdat is
    const parts = boxes.filter((x) => x.type === 'ftyp' || x.type === 'moov')
    const buf = (await new Blob(parts.map((b) => file.slice(b.offset, b.offset + b.size))).arrayBuffer()) as
      ArrayBuffer & { fileStart: number }
    buf.fileStart = 0
    mp4.appendBuffer(buf as never)
    mp4.flush()
    info = await Promise.race([got, new Promise<null>((r) => setTimeout(() => r(null), 3000))])
    const track = info?.videoTracks?.[0]
    if (!track) return null
    if (!document.createElement('video').canPlayType(`video/mp4; codecs="${track.codec}"`)) return null

    const samples = mp4.getTrackSamplesInfo(track.id)
    if (!samples?.length) return null
    const cts = samples.map((s) => s.cts).sort((a, b) => a - b)
    // the edit list decides where the media starts on the presentation clock:
    // empty edits (media_time -1) delay it, the first real one says which media
    // time plays at that point
    let offset = cts[0]
    let delay = 0
    for (const e of track.edits ?? []) {
      if (e.media_time === -1) delay += e.segment_duration / track.movie_timescale
      else {
        offset = e.media_time
        break
      }
    }
    const times = Float64Array.from(cts, (c) => (c - offset) / track.timescale + delay)
    const span = times[times.length - 1] - times[0]
    const fps = times.length > 1 && span > 0 ? (times.length - 1) / span : 20
    return {
      id: `local-${++seq}`,
      file,
      url: URL.createObjectURL(file),
      name: file.name,
      width: track.video?.width ?? track.track_width,
      height: track.video?.height ?? track.track_height,
      fps: Math.round(fps * 1000) / 1000,
      frames: times.length,
      times,
    }
  } catch {
    return null
  }
}

export function closeLocal(v: LocalVideo | null) {
  if (v) URL.revokeObjectURL(v.url)
}

/** The frame on screen at `t`: the last one whose presentation time is <= t. */
export function frameAt(v: LocalVideo, t: number): number {
  const ts = v.times
  let lo = 0
  let hi = ts.length - 1
  if (t <= ts[0]) return 0
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (ts[mid] <= t + 1e-6) lo = mid
    else hi = mid - 1
  }
  return lo
}

/** A time that shows frame `f`: a third of the way into its display interval, so
 *  neither rounding nor a decoder's idea of "nearest" can land a frame off. */
export function timeOf(v: LocalVideo, f: number): number {
  const ts = v.times
  const i = Math.max(0, Math.min(ts.length - 1, f))
  const next = i + 1 < ts.length ? ts[i + 1] : ts[i] + 1 / v.fps
  return ts[i] + (next - ts[i]) / 3
}

/** Frames start..end of the local video as JPEGs (longest side at most `maxSide`),
 *  exactly those frames: each is seeked to on a detached <video> and drawn. */
export async function grabFrames(
  v: LocalVideo,
  start: number,
  end: number,
  onProgress?: (done: number, total: number) => void,
  maxSide = 1280,
): Promise<Blob[]> {
  const el = document.createElement('video')
  el.muted = true
  el.preload = 'auto'
  el.src = v.url
  await new Promise<void>((resolve, reject) => {
    el.onloadeddata = () => resolve()
    el.onerror = () => reject(new Error('the browser could not decode this video'))
  })
  const scale = Math.min(1, maxSide / Math.max(el.videoWidth, el.videoHeight))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round((el.videoWidth * scale) / 2) * 2
  canvas.height = Math.round((el.videoHeight * scale) / 2) * 2
  const g = canvas.getContext('2d')!
  const out: Blob[] = []
  try {
    for (let f = start; f <= end; f++) {
      await seekTo(el, timeOf(v, f))
      g.drawImage(el, 0, 0, canvas.width, canvas.height)
      out.push(
        await new Promise<Blob>((resolve, reject) =>
          canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('could not encode a frame'))), 'image/jpeg', 0.92),
        ),
      )
      onProgress?.(f - start + 1, end - start + 1)
    }
  } finally {
    el.removeAttribute('src')
    el.load()
  }
  return out
}

function seekTo(el: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve) => {
    // 'seeked' fires once the frame at t is decoded and current
    el.addEventListener('seeked', () => resolve(), { once: true })
    el.currentTime = t
  })
}

/** Top-level MP4 boxes, from their headers alone (the mdat is skipped, not read). */
async function topLevelBoxes(file: File) {
  const boxes: { type: string; offset: number; size: number }[] = []
  let off = 0
  for (let n = 0; off + 8 <= file.size && n < 64; n++) {
    const h = new DataView(await file.slice(off, off + 16).arrayBuffer())
    let size = h.getUint32(0)
    const type = String.fromCharCode(h.getUint8(4), h.getUint8(5), h.getUint8(6), h.getUint8(7))
    if (size === 1) size = Number(h.getBigUint64(8))
    else if (size === 0) size = file.size - off
    if (size < 8) break
    boxes.push({ type, offset: off, size })
    if (type === 'moov') break
    off += size
  }
  return boxes
}
