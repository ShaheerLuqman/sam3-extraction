import { useEffect, useRef, useState } from 'react'
import type { Segment } from '../lib/segments'
import { clock } from './format'

type Props = {
  /** per class, the smoothed P per embedded frame (every `stride`-th); NaN = not checked */
  curves: Float32Array[]
  names: string[]
  colors: string[]
  stride: number
  total: number
  fps: number
  threshold: number
  /** Segment.bestRef is the class */
  segments: Segment[]
  /** what the VLM checked */
  bands: [number, number][]
  excluded: [number, number][]
  frame: number
  onSeek: (f: number) => void
}

const PAD = { l: 30, r: 10, t: 6, b: 22 }
const LANE = 40
const GAP = 4
const C = {
  grid: '#e2e7f0',
  text: '#74819a',
  band: 'rgba(154, 91, 6, 0.07)',
  excl: 'rgba(116, 129, 154, 0.16)',
  thr: '#9a5b06',
  cursor: '#16202e',
}

/** One lane per class: its P(step) from the VLM, the threshold, and the segments it
 *  won. Click or drag to seek. */
export function McTimeline(p: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [width, setWidth] = useState(800)
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)
  const dragging = useRef(false)
  const n = p.curves.length
  const H = PAD.t + n * (LANE + GAP) + PAD.b

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(200, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const plotW = width - PAD.l - PAD.r
  const last = Math.max(1, p.total - 1)
  const xOf = (f: number) => PAD.l + (f / last) * plotW
  const fOf = (x: number) => Math.round(Math.max(0, Math.min(1, (x - PAD.l) / plotW)) * last)
  const laneTop = (c: number) => PAD.t + c * (LANE + GAP)
  const yOf = (c: number, v: number) => laneTop(c) + (1 - v) * LANE

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

    for (let c = 0; c < n; c++) {
      const top = laneTop(c)
      const color = p.colors[c]
      g.strokeStyle = C.grid
      g.lineWidth = 1
      g.strokeRect(PAD.l + 0.5, top + 0.5, plotW - 1, LANE - 1)
      // the class's number in its colour, in the gutter
      g.fillStyle = color
      g.fillRect(4, top + LANE / 2 - 9, 20, 18)
      g.fillStyle = '#fff'
      g.textAlign = 'center'
      g.textBaseline = 'middle'
      g.font = '600 11px system-ui, sans-serif'
      g.fillText(String(c + 1), 14, top + LANE / 2)
      g.font = '11px system-ui, sans-serif'

      g.fillStyle = C.band
      for (const [a, b] of p.bands) g.fillRect(xOf(a), top, Math.max(2, xOf(b) - xOf(a)), LANE)
      g.fillStyle = C.excl
      for (const [a, b] of p.excluded) g.fillRect(xOf(a), top, Math.max(2, xOf(b) - xOf(a)), LANE)
      for (const s of p.segments) {
        if (s.bestRef !== c) continue
        const x = xOf(s.start)
        const w = Math.max(2, xOf(s.end) - x)
        g.globalAlpha = 0.18
        g.fillStyle = color
        g.fillRect(x, top, w, LANE)
        g.globalAlpha = 1
        g.fillRect(x, top, w, 4)
      }

      // threshold
      const ty = Math.round(yOf(c, p.threshold)) + 0.5
      g.strokeStyle = C.thr
      g.setLineDash([4, 4])
      g.beginPath()
      g.moveTo(PAD.l, ty)
      g.lineTo(width - PAD.r, ty)
      g.stroke()
      g.setLineDash([])

      // P curve
      const curve = p.curves[c]
      g.strokeStyle = color
      g.lineWidth = 1.3
      g.lineJoin = 'round'
      g.beginPath()
      let pen = false
      curve.forEach((v, i) => {
        if (!Number.isFinite(v)) {
          pen = false
          return
        }
        const x = xOf(Math.min(last, i * p.stride))
        const y = yOf(c, v)
        if (pen) g.lineTo(x, y)
        else g.moveTo(x, y)
        pen = true
      })
      g.stroke()
    }

    // time ticks
    g.fillStyle = C.text
    g.textAlign = 'center'
    g.textBaseline = 'alphabetic'
    const secs = p.total / (p.fps || 1)
    const nice = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
    const every = nice.find((k) => secs / k <= Math.max(3, plotW / 90)) ?? 3600
    for (let s = 0; s <= secs; s += every) g.fillText(clock(s * p.fps, p.fps), xOf(s * p.fps), H - 6)

    // playhead
    const cx = Math.round(xOf(p.frame)) + 0.5
    g.strokeStyle = C.cursor
    g.lineWidth = 1.5
    g.beginPath()
    g.moveTo(cx, PAD.t - 2)
    g.lineTo(cx, H - PAD.b + 2)
    g.stroke()

    // hover readout: the time and the lane's class and P
    if (hover && !dragging.current) {
      const f = fOf(hover.x)
      const c = Math.floor((hover.y - PAD.t) / (LANE + GAP))
      const i = Math.round(f / p.stride)
      const v = c >= 0 && c < n ? p.curves[c][Math.min(p.curves[c].length - 1, i)] : undefined
      const text = `${clock(f, p.fps, true)} · frame ${f}${
        c >= 0 && c < n ? ` · ${p.names[c]}${v !== undefined && Number.isFinite(v) ? ` P ${v.toFixed(2)}` : ' · not checked'}` : ''
      }`
      g.font = '12px system-ui, sans-serif'
      const tw = g.measureText(text).width + 12
      const bx = Math.min(width - PAD.r - tw, Math.max(PAD.l, hover.x - tw / 2))
      const by = Math.max(0, Math.min(H - PAD.b - 20, hover.y - 28))
      g.fillStyle = 'rgba(255, 255, 255, 0.95)'
      g.fillRect(bx, by, tw, 20)
      g.strokeStyle = 'rgba(22, 32, 46, 0.35)'
      g.lineWidth = 1
      g.strokeRect(bx + 0.5, by + 0.5, tw - 1, 19)
      g.fillStyle = C.cursor
      g.textAlign = 'left'
      g.textBaseline = 'middle'
      g.fillText(text, bx + 6, by + 10)
    }
  })

  const pos = (e: React.PointerEvent) => {
    const r = canvasRef.current!.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  return (
    <div className="timeline" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        role="slider"
        aria-label="Video timeline, one lane per step — click to seek"
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={p.frame}
        tabIndex={0}
        style={{ cursor: 'pointer' }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId)
          dragging.current = true
          p.onSeek(fOf(pos(e).x))
        }}
        onPointerMove={(e) => {
          const q = pos(e)
          setHover(q)
          if (dragging.current) p.onSeek(fOf(q.x))
        }}
        onPointerUp={() => {
          dragging.current = false
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
