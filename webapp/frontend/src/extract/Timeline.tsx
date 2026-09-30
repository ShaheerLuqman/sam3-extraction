import { useEffect, useRef, useState } from 'react'
import type { Segment } from '../lib/segments'
import { clock } from './format'

type Props = {
  /** smoothed score per embedded frame (every `stride`-th frame) */
  score: Float32Array | null
  stride: number
  /** frames in the video (the decodable ones) */
  total: number
  fps: number
  threshold: number
  segments: Segment[]
  excluded: [number, number][]
  refFrames: number[]
  frame: number
  onSeek: (f: number) => void
  onThreshold: (v: number) => void
  /** fixed y range (e.g. [0, 1] for a probability); default fits the score */
  range?: [number, number]
  /** shaded stretches, e.g. what the VLM looked at */
  bands?: [number, number][]
  /** what the curve is, for the hover readout */
  scoreLabel?: string
}

const PAD = { l: 38, r: 10, t: 12, b: 22 }
const H = 168
// the app's light tokens (index.css); a canvas cannot read them without a lookup
const C = {
  grid: '#e2e7f0',
  text: '#74819a',
  curve: '#1d4ed8',
  seg: 'rgba(37, 99, 235, 0.13)',
  segBar: '#2563eb',
  excl: 'rgba(116, 129, 154, 0.16)',
  band: 'rgba(154, 91, 6, 0.07)',
  thr: '#9a5b06',
  ref: '#0f7b5a',
  cursor: '#16202e',
}

/** Score over the whole video, the threshold, and what it selects. Click or drag to
 *  seek; drag the dashed line to move the threshold. */
export function Timeline(p: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [width, setWidth] = useState(800)
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)
  const drag = useRef<'seek' | 'thr' | null>(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(200, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // y range: always show 0 and the threshold, with a little headroom
  let lo = -1
  let hi = p.threshold + 1
  if (p.range) {
    ;[lo, hi] = p.range
  } else {
    if (p.score) {
      for (const v of p.score) {
        if (v < lo) lo = v
        if (v > hi) hi = v
      }
    }
    lo = Math.floor(lo)
    hi = Math.ceil(hi + 0.2)
  }
  const plotW = width - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const last = Math.max(1, p.total - 1)
  const xOf = (f: number) => PAD.l + (f / last) * plotW
  const yOf = (v: number) => PAD.t + (1 - (v - lo) / (hi - lo)) * plotH
  const fOf = (x: number) => Math.round(Math.max(0, Math.min(1, (x - PAD.l) / plotW)) * last)
  const vOf = (y: number) => lo + (1 - (y - PAD.t) / plotH) * (hi - lo)

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv) return
    const dpr = window.devicePixelRatio || 1
    cv.width = Math.round(width * dpr)
    cv.height = Math.round(H * dpr)
    cv.style.width = `${width}px`
    cv.style.height = `${H}px`
    const g = cv.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, width, H)
    g.font = '11px system-ui, sans-serif'

    // horizontal grid: whole values, or quarters on a 0-1 scale
    const step = hi - lo <= 1.5 ? 0.25 : hi - lo > 8 ? 2 : 1
    g.strokeStyle = C.grid
    g.fillStyle = C.text
    g.lineWidth = 1
    g.textAlign = 'right'
    g.textBaseline = 'middle'
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      const y = Math.round(yOf(v)) + 0.5
      g.beginPath()
      g.moveTo(PAD.l, y)
      g.lineTo(width - PAD.r, y)
      g.stroke()
      g.fillText(`${Number(v.toFixed(2))}`, PAD.l - 6, y)
    }

    // time ticks
    const secs = p.total / (p.fps || 1)
    const nice = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
    const every = nice.find((n) => secs / n <= Math.max(3, plotW / 90)) ?? 3600
    g.textAlign = 'center'
    g.textBaseline = 'alphabetic'
    for (let s = 0; s <= secs; s += every) {
      const x = xOf(s * p.fps)
      g.fillText(clock(s * p.fps, p.fps), x, H - 6)
    }

    // shaded bands, excluded ranges, then the selected segments
    g.fillStyle = C.band
    for (const [a, b] of p.bands ?? []) g.fillRect(xOf(a), PAD.t, Math.max(2, xOf(b) - xOf(a)), plotH)
    g.fillStyle = C.excl
    for (const [a, b] of p.excluded) g.fillRect(xOf(a), PAD.t, Math.max(2, xOf(b) - xOf(a)), plotH)
    for (const s of p.segments) {
      const x = xOf(s.start)
      const w = Math.max(2, xOf(s.end) - x)
      g.fillStyle = C.seg
      g.fillRect(x, PAD.t, w, plotH)
      g.fillStyle = C.segBar
      g.fillRect(x, PAD.t, w, 4)
    }

    // score curve
    if (p.score && p.score.length) {
      g.strokeStyle = C.curve
      g.lineWidth = 1.3
      g.lineJoin = 'round'
      g.beginPath()
      let pen = false // NaN = no score there (e.g. the VLM did not look): a gap
      p.score.forEach((v, i) => {
        if (!Number.isFinite(v)) {
          pen = false
          return
        }
        const x = xOf(Math.min(last, i * p.stride))
        const y = yOf(v)
        if (pen) g.lineTo(x, y)
        else g.moveTo(x, y)
        pen = true
      })
      g.stroke()
    }

    // threshold
    const ty = Math.round(yOf(p.threshold)) + 0.5
    g.strokeStyle = C.thr
    g.lineWidth = 1.5
    g.setLineDash([6, 4])
    g.beginPath()
    g.moveTo(PAD.l, ty)
    g.lineTo(width - PAD.r, ty)
    g.stroke()
    g.setLineDash([])
    // its label sits on a plate at the left, clear of the segment bands' top bars
    const label = `threshold ${p.threshold.toFixed(2)}`
    g.font = '600 11px system-ui, sans-serif'
    const lw = g.measureText(label).width + 10
    g.fillStyle = 'rgba(255, 255, 255, 0.92)'
    g.fillRect(PAD.l + 4, ty - 17, lw, 15)
    g.fillStyle = C.thr
    g.textAlign = 'left'
    g.textBaseline = 'middle'
    g.fillText(label, PAD.l + 9, ty - 9.5)

    // reference frames taken from this video
    g.fillStyle = C.ref
    for (const f of p.refFrames) {
      const x = xOf(f)
      const y = PAD.t + plotH
      g.beginPath()
      g.moveTo(x, y - 7)
      g.lineTo(x - 5, y)
      g.lineTo(x + 5, y)
      g.closePath()
      g.fill()
    }

    // playhead
    const cx = Math.round(xOf(p.frame)) + 0.5
    g.strokeStyle = C.cursor
    g.lineWidth = 1.5
    g.beginPath()
    g.moveTo(cx, PAD.t - 4)
    g.lineTo(cx, PAD.t + plotH)
    g.stroke()

    // hover readout
    if (hover && !drag.current) {
      const f = fOf(hover.x)
      const i = Math.min((p.score?.length ?? 1) - 1, Math.round(f / p.stride))
      const v = p.score?.[i]
      const text = `${clock(f, p.fps, true)} · frame ${f}${
        v !== undefined && Number.isFinite(v) ? ` · ${p.scoreLabel ?? 'score'} ${v.toFixed(2)}` : ''
      }`
      g.font = '12px system-ui, sans-serif'
      const tw = g.measureText(text).width + 12
      const bx = Math.min(width - PAD.r - tw, Math.max(PAD.l, hover.x - tw / 2))
      g.strokeStyle = 'rgba(22, 32, 46, 0.35)'
      g.lineWidth = 1
      g.beginPath()
      g.moveTo(hover.x + 0.5, PAD.t)
      g.lineTo(hover.x + 0.5, PAD.t + plotH)
      g.stroke()
      g.fillStyle = 'rgba(255, 255, 255, 0.95)'
      g.fillRect(bx, PAD.t + 6, tw, 20)
      g.strokeRect(bx + 0.5, PAD.t + 6.5, tw - 1, 19)
      g.fillStyle = C.cursor
      g.textAlign = 'left'
      g.textBaseline = 'middle'
      g.fillText(text, bx + 6, PAD.t + 16)
    }
  })

  const pos = (e: React.PointerEvent) => {
    const r = canvasRef.current!.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }
  const nearThreshold = (y: number) => Math.abs(y - yOf(p.threshold)) <= 7

  return (
    <div className="timeline" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        role="slider"
        aria-label="Video timeline — click to seek"
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={p.frame}
        tabIndex={0}
        style={{ cursor: hover && nearThreshold(hover.y) ? 'ns-resize' : 'pointer' }}
        onPointerDown={(e) => {
          const { x, y } = pos(e)
          e.currentTarget.setPointerCapture(e.pointerId)
          drag.current = nearThreshold(y) ? 'thr' : 'seek'
          if (drag.current === 'seek') p.onSeek(fOf(x))
        }}
        onPointerMove={(e) => {
          const { x, y } = pos(e)
          setHover({ x, y })
          if (drag.current === 'seek') p.onSeek(fOf(x))
          if (drag.current === 'thr') p.onThreshold(Math.round(vOf(y) * 20) / 20)
        }}
        onPointerUp={() => {
          drag.current = null
        }}
        onPointerLeave={() => setHover(null)}
        onKeyDown={(e) => {
          const big = e.shiftKey ? 10 : 1
          if (e.key === 'ArrowRight') p.onSeek(p.frame + p.stride * big)
          else if (e.key === 'ArrowLeft') p.onSeek(p.frame - p.stride * big)
          else return
          e.preventDefault()
        }}
      />
    </div>
  )
}
