import { api } from '../api/client'
import type { Ref } from '../lib/extraction'

/** m:ss (or m:ss.s) for a frame index at `fps`. */
export function clock(frame: number, fps: number, tenths = false): string {
  const s = frame / (fps || 1)
  const m = Math.floor(s / 60)
  const sec = s - m * 60
  return `${m}:${(tenths ? sec.toFixed(1) : String(Math.floor(sec))).padStart(tenths ? 4 : 2, '0')}`
}

export function refThumb(r: Ref, videoId?: string): string {
  if (r.kind === 'image') return r.thumb
  return videoId ? api.frameUrl(videoId, r.frame) : ''
}

export function refLabel(r: Ref): string {
  return r.kind === 'image' ? r.name : `Frame ${r.frame}`
}
