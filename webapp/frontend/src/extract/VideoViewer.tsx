import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, type UploadInfo } from '../api/client'
import { FrameBar } from '../components/FrameBar'
import { frameAt, timeOf, type LocalVideo } from '../lib/localVideo'
import type { Segment } from '../lib/segments'

export type Play = { playing: number | null; setPlaying: (i: number | null) => void }

type Props = {
  video: UploadInfo | null
  /** the user's own copy, played in the browser: when set, no frames or playback
   *  copy come from the server at all */
  local?: LocalVideo | null
  /** the browser could not play `local` after all (the page falls back to the server) */
  onLocalFail?: () => void
  /** what the left side shows before there is a video */
  empty: ReactNode
  frame: number
  setFrame: (f: number) => void
  lastFrame: number
  fps: number
  segments: Segment[]
  playbackUrl: string | null
  playbackError: string | null
  /** the browser could not play `playbackUrl` (the next format down is tried) */
  onPlaybackFail?: () => void
  play: Play
  /** ticks on the frame bar */
  marked: number[]
  /** pills over the frame's top-left corner */
  badges: ReactNode
  /** extra buttons left of the segment controls; `frameOk` is false when the frame failed to decode */
  tools?: (ctx: { frameOk: boolean }) => ReactNode
  /** the timeline under the frame; `seek` also stops playback */
  timeline: (seek: (f: number) => void) => ReactNode
}

const RATES = [0.25, 0.5, 1, 2, 4]

/** The frame, a frame bar, play/pause, segment navigation and playback, and a
 *  timeline — the left half of the extraction pages. */
export function VideoViewer(p: Props) {
  const { video, local, frame, setFrame, segments, fps, playbackUrl, lastFrame } = p
  const { playing, setPlaying } = p.play
  const vref = useRef<HTMLVideoElement>(null)
  const rootRef = useRef<HTMLElement>(null)
  // plain play/pause: playing from this frame on, to the end of the video
  const [free, setFree] = useState<number | null>(null)
  const [rate, setRate] = useState(1)
  const rateRef = useRef(rate)
  // the frame playback last stopped on: the <video> stays on screen there until
  // the still underneath has loaded, so stopping never flashes an old frame
  const [heldAt, setHeldAt] = useState<number | null>(null)
  // whether the <video> has painted a frame of the current playback yet: until it
  // has, the still stays on top, so a slow start (or no decoder) is never black
  const [painted, setPainted] = useState(false)
  // a copy the browser could not play; without a playable one, playback steps through JPEGs
  const [failedUrl, setFailedUrl] = useState<string | null>(null)
  const smooth = !!local || (!!playbackUrl && failedUrl !== playbackUrl)
  // the video's clock <-> frame numbers: the local file's own frame times, or the
  // server's copy, re-timed so frame N is at exactly N / fps
  const toFrame = useCallback(
    (t: number) => (local ? frameAt(local, t) : Math.round(t * fps)),
    [local, fps],
  )
  const toTime = useCallback(
    (f: number) => (local ? timeOf(local, f) : (f + 0.5) / fps), // mid-frame, so rounding cannot land a frame early
    [local, fps],
  )

  // what is playing: a segment, or the video from where play was pressed
  const range = useMemo(() => {
    if (playing !== null) return segments[playing] ?? null
    return free !== null ? { start: free, end: lastFrame } : null
  }, [playing, segments, free, lastFrame])
  const active = range !== null

  const stop = useCallback(() => {
    setPlaying(null)
    setFree(null)
  }, [setPlaying])

  // no JPEG requests while the video is playing — it would be 17 a second for
  // nothing — and none at all for a local file
  const url = video && !local && (!active || !smooth) ? api.frameUrl(video.upload_id, frame) : ''
  const shown = useLoadedImage(url)

  const { onPlaybackFail, onLocalFail } = p
  const fail = useCallback(() => {
    if (local) return onLocalFail?.()
    if (!playbackUrl) return
    setFailedUrl(playbackUrl)
    onPlaybackFail?.()
  }, [local, playbackUrl, onPlaybackFail, onLocalFail])

  // a local file is also the still: while not playing, it is seeked to the frame
  useEffect(() => {
    const v = vref.current
    if (!v || !local || active) return
    if (v.seeking || toFrame(v.currentTime) !== frame) v.currentTime = toTime(frame)
  }, [local, active, frame, toFrame, toTime])

  useEffect(() => {
    rateRef.current = rate
    if (vref.current) vref.current.playbackRate = rate
  }, [rate])

  // Smooth playback runs in the browser's own decoder on the playback copy,
  // which is re-timed so frame N is at exactly N / fps. Each presented frame
  // reports its index back, so the timeline, badges and list follow along.
  useEffect(() => {
    const v = vref.current
    if (!v || !range || !smooth) return
    let handle = 0
    let alive = true
    let last = range.start
    let frames = 0
    const stopAt = (f: number) => {
      last = f
      setFrame(f)
      if (++frames === 2) setPainted(true) // the first may still be the pre-seek one
      if (f >= range.end) stop()
    }
    const hasRvfc = typeof v.requestVideoFrameCallback === 'function'
    const onFrame = (_: number, meta: VideoFrameCallbackMetadata) => {
      if (!alive) return
      stopAt(toFrame(meta.mediaTime))
      handle = v.requestVideoFrameCallback(onFrame)
    }
    const onTime = () => alive && stopAt(toFrame(v.currentTime))
    const onEnded = () => alive && stop()
    setPainted(false)
    v.playbackRate = rateRef.current
    v.currentTime = toTime(range.start)
    if (hasRvfc) handle = v.requestVideoFrameCallback(onFrame)
    else v.addEventListener('timeupdate', onTime)
    v.addEventListener('ended', onEnded)
    v.play().catch((e: Error) => {
      // AbortError = stopped before it started; anything else = it cannot play this
      if (alive && e.name !== 'AbortError') fail()
      else if (alive) stop()
    })
    // playing but nothing presented in 8 s: treat it as unplayable (JPEGs take over)
    const watchdog = window.setTimeout(() => alive && frames === 0 && fail(), 8000)
    return () => {
      alive = false
      clearTimeout(watchdog)
      v.pause()
      setHeldAt(last)
      v.removeEventListener('ended', onEnded)
      if (hasRvfc) v.cancelVideoFrameCallback(handle)
      else v.removeEventListener('timeupdate', onTime)
    }
  }, [range, smooth, toFrame, toTime, setFrame, stop, fail])

  // Without a playable copy (still being made, or none this browser can decode):
  // step through the frames as JPEGs against the clock, skipping frames when
  // loading falls behind, so playback still runs at the right speed.
  const videoId = video?.upload_id
  useEffect(() => {
    if (!range || smooth || !videoId) return
    let alive = true
    const run = async () => {
      let f = range.start
      let clock = f // where playback should be, in (fractional) frames
      let t = performance.now()
      while (alive) {
        await loadImage(api.frameUrl(videoId, f))
        if (!alive) return
        setFrame(f)
        if (f >= range.end) return stop()
        const now = performance.now()
        clock += ((now - t) * fps * rateRef.current) / 1000
        t = now
        const next = Math.min(range.end, Math.max(f + 1, Math.floor(clock)))
        if (next > clock) {
          await new Promise((r) => setTimeout(r, ((next - clock) * 1000) / (fps * rateRef.current)))
          clock = next
          t = performance.now()
        }
        f = next
      }
    }
    void run()
    return () => {
      alive = false
    }
  }, [range, smooth, videoId, fps, setFrame, stop])

  const videoOn = !!local || (smooth && ((active && painted) || (heldAt === frame && shown.src !== url)))

  // a threshold change can reshuffle the segments under a playing one
  useEffect(() => setPlaying(null), [segments.length, setPlaying])

  // anything the user does to move the playhead ends playback
  const seek = (f: number) => {
    stop()
    setFrame(f)
  }
  const togglePlay = () => {
    if (active) return stop()
    setPlaying(null)
    setFree(frame >= lastFrame ? 0 : frame) // at the end: from the top
  }

  // space plays / pauses, while this viewer is on screen and not typing
  const toggleRef = useRef(togglePlay)
  useEffect(() => {
    toggleRef.current = togglePlay
  })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== ' ' || !rootRef.current?.offsetParent || e.ctrlKey || e.metaKey || e.altKey) return
      const t = e.target as HTMLElement
      if (t.closest('input, textarea, select, button, [contenteditable]')) return
      e.preventDefault()
      toggleRef.current()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  if (!video && !local) return <section className="viewer">{p.empty}</section>

  const segIdx = segments.findIndex((s) => frame >= s.start && frame <= s.end)
  const goSeg = (dir: 1 | -1) => {
    const next =
      dir > 0 ? segments.find((s) => s.start > frame) : [...segments].reverse().find((s) => s.end < frame)
    if (next) seek(next.start)
  }

  return (
    <section className="viewer" ref={rootRef}>
      <div className="viewer-canvas">
        <div className="boxcanvas xframe">
          {shown.src && <img src={shown.src} alt={`Frame ${frame}`} />}
          {smooth && (
            <video
              ref={vref}
              className={videoOn ? '' : 'xhidden'}
              src={local ? local.url : api.fileUrl(playbackUrl!)}
              muted
              playsInline
              preload="auto"
              onError={fail}
            />
          )}
          {shown.error && !local && (!active || !smooth) && (
            <p className="xframe-error">
              Frame {frame} could not be decoded. The file may be truncated past this point.
            </p>
          )}
          <div className="xbadges">{p.badges}</div>
        </div>
      </div>

      <FrameBar frame={frame} last={lastFrame} fps={fps} marked={p.marked} onChange={seek} />

      <div className="xtools">
        <span className="row gap">
          <button
            type="button"
            className={`btn xplay${free !== null ? ' on' : ''}`}
            onClick={togglePlay}
            title={active ? 'Pause (space)' : 'Play from this frame (space)'}
            aria-label={active ? 'Pause' : 'Play'}
          >
            {active ? '❚❚ Pause' : '▶ Play'}
          </button>
          <select
            className="xrate"
            value={rate}
            onChange={(e) => setRate(Number(e.target.value))}
            aria-label="Playback speed"
            title="Playback speed"
          >
            {RATES.map((r) => (
              <option key={r} value={r}>
                {r}×
              </option>
            ))}
          </select>
          {!smooth && (
            <span
              className="xplaynote"
              title={p.playbackError ?? 'A smooth-playback copy of the video is being made; until then playback steps through frames'}
            >
              {p.playbackError ? 'frame-by-frame playback' : 'preparing smooth playback…'}
            </span>
          )}
          {p.tools?.({ frameOk: !shown.error })}
        </span>
        {segments.length > 0 && (
          <span className="row gap">
            <button type="button" className="btn ghost" onClick={() => goSeg(-1)}>
              ‹ Previous segment
            </button>
            <button
              type="button"
              className="btn ghost"
              onClick={() => {
                if (playing !== null) return stop()
                const i = segIdx >= 0 ? segIdx : segments.findIndex((s) => s.start > frame)
                if (i >= 0) {
                  setFree(null)
                  setPlaying(i)
                }
              }}
            >
              {playing !== null ? '■ Stop' : '▶ Play segment'}
            </button>
            <button type="button" className="btn ghost" onClick={() => goSeg(1)}>
              Next segment ›
            </button>
          </span>
        )}
      </div>

      {p.timeline(seek)}
    </section>
  )
}

/** Keep the previous frame on screen until the next one has loaded — no flashing. */
function useLoadedImage(url: string) {
  const [state, setState] = useState<{ src: string; error: boolean }>({ src: '', error: false })
  useEffect(() => {
    if (!url) return
    let alive = true
    const img = new Image()
    img.onload = () => alive && setState({ src: url, error: false })
    img.onerror = () => alive && setState((s) => ({ src: s.src, error: true }))
    img.src = url
    return () => {
      alive = false
    }
  }, [url])
  return state
}

/** Resolves when the image has loaded (or failed: playback just moves on). */
function loadImage(src: string): Promise<void> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = img.onerror = () => resolve()
    img.src = src
  })
}
