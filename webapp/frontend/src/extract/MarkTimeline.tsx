import { useEffect, useRef, useState } from 'react'
import { clock } from './format'

type Props = {
  /** frames in the video */
  total: number
  fps: number
  frame: number
  /** ranges already marked on this video */
  steps: [number, number][]
  others: [number, number][]
  /** optional ranges in a colour of their own (multiple class segmentation's steps) */
  colored?: { start: number; end: number; color: string }[]
  markIn: number | null
  markOut: number | null
  onSeek: (f: number) => void
  onMarkIn: (f: number) => void
  onMarkOut: (f: number) => void
}

const PAD = { l: 12, r: 12, t: 22, b: 22 }
const H = 96
const C = {
  track: '#eef1f6',
  text: '#74819a',
  step: '#2563eb',
  stepSoft: 'rgba(37, 99, 235, 0.2)',
  other: '#7c5cc4',
  otherSoft: 'rgba(124, 92, 196, 0.2)',
  sel: '#9a5b06',
  selSoft: 'rgba(245, 158, 11, 0.24)',
  cursor: '#16202e',
}

/** The whole reference video as one lane: what is marked, the start/end being set
 *  (drag either handle to move it), and the playhead. Click or drag to seek. */
export function MarkTimeline(p: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [width, setWidth] = useState(800)
  const [hover, setHover] = useState<number | null>(null)
  const drag = useRef<'seek' | 'in' | 'out' | null>(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(200, e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const plotW = width - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const last = Math.max(1, p.total - 1)
  const xOf = (f: number) => PAD.l + (f / last) * plotW
  const fOf = (x: number) => Math.round(Math.max(0, Math.min(1, (x - PAD.l) / plotW)) * last)

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
    g.fillStyle = C.track
    g.fillRect(PAD.l, PAD.t, plotW, plotH)

    // time ticks
    g.font = '11px system-ui, sans-serif'
    g.fillStyle = C.text
    g.textAlign = 'center'
    g.textBaseline = 'alphabetic'
    const secs = p.total / (p.fps || 1)
    const nice = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
    const every = nice.find((n) => secs / n <= Math.max(3, plotW / 90)) ?? 3600
    for (let s = 0; s <= secs; s += every) g.fillText(clock(s * p.fps, p.fps), xOf(s * p.fps), H - 6)

    const range = (a: number, b: number, soft: string, solid: string) => {
      const x = xOf(a)
      const w = Math.max(3, xOf(b) - x)
      g.fillStyle = soft
      g.fillRect(x, PAD.t, w, plotH)
      g.fillStyle = solid
      g.fillRect(x, PAD.t, w, 4)
    }
    for (const [a, b] of p.others) range(a, b, C.otherSoft, C.other)
    for (const [a, b] of p.steps) range(a, b, C.stepSoft, C.step)
    for (const r of p.colored ?? []) range(r.start, r.end, `${r.color}33`, r.color)

    // the selection being set, with a flag on each end that is set
    const { markIn: mi, markOut: mo } = p
    if (mi !== null && mo !== null) range(Math.min(mi, mo), Math.max(mi, mo), C.selSoft, C.sel)
    const flag = (f: number, label: string) => {
      const x = Math.round(xOf(f)) + 0.5
      g.strokeStyle = C.sel
      g.lineWidth = 2
      g.beginPath()
      g.moveTo(x, PAD.t - 6)
      g.lineTo(x, PAD.t + plotH)
      g.stroke()
      g.font = '600 11px system-ui, sans-serif'
      const w = g.measureText(label).width + 10
      const bx = Math.min(width - w, Math.max(0, x - w / 2))
      g.fillStyle = C.sel
      g.fillRect(bx, 1, w, 16)
      g.fillStyle = '#fff'
      g.textAlign = 'left'
      g.textBaseline = 'middle'
      g.fillText(label, bx + 5, 9.5)
    }
    if (mi !== null) flag(mi, mo !== null && mo < mi ? 'end' : 'start')
    if (mo !== null) flag(mo, mi !== null && mo < mi ? 'start' : 'end')

    // playhead
    const cx = Math.round(xOf(p.frame)) + 0.5
    g.strokeStyle = C.cursor
    g.lineWidth = 1.5
    g.beginPath()
    g.moveTo(cx, PAD.t - 2)
    g.lineTo(cx, PAD.t + plotH + 2)
    g.stroke()

    if (hover !== null && !drag.current) {
      const f = fOf(hover)
      const text = `${clock(f, p.fps, true)} · frame ${f}`
      g.font = '12px system-ui, sans-serif'
      const tw = g.measureText(text).width + 12
      const bx = Math.min(width - PAD.r - tw, Math.max(PAD.l, hover - tw / 2))
      g.fillStyle = 'rgba(255, 255, 255, 0.95)'
      g.fillRect(bx, PAD.t + plotH / 2 - 10, tw, 20)
      g.strokeStyle = 'rgba(22, 32, 46, 0.35)'
      g.lineWidth = 1
      g.strokeRect(bx + 0.5, PAD.t + plotH / 2 - 9.5, tw - 1, 19)
      g.fillStyle = C.cursor
      g.textAlign = 'left'
      g.textBaseline = 'middle'
      g.fillText(text, bx + 6, PAD.t + plotH / 2)
    }
  })

  const xAt = (e: React.PointerEvent) => e.clientX - canvasRef.current!.getBoundingClientRect().left
  const handleAt = (x: number): 'in' | 'out' | null => {
    const near = (f: number | null) => f !== null && Math.abs(xOf(f) - x) <= 6
    return near(p.markOut) ? 'out' : near(p.markIn) ? 'in' : null
  }

  return (
    <div className="timeline marktl" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        role="slider"
        aria-label="Reference video — click to seek, drag the start and end flags to adjust them"
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={p.frame}
        tabIndex={0}
        style={{ cursor: hover !== null && handleAt(hover) ? 'ew-resize' : 'pointer' }}
        onPointerDown={(e) => {
          const x = xAt(e)
          e.currentTarget.setPointerCapture(e.pointerId)
          drag.current = handleAt(x) ?? 'seek'
          if (drag.current === 'seek') p.onSeek(fOf(x))
        }}
        onPointerMove={(e) => {
          const x = xAt(e)
          setHover(x)
          const f = fOf(x)
          if (drag.current === 'seek') p.onSeek(f)
          else if (drag.current === 'in') {
            p.onMarkIn(f)
            p.onSeek(f)
          } else if (drag.current === 'out') {
            p.onMarkOut(f)
            p.onSeek(f)
          }
        }}
        onPointerUp={() => {
          drag.current = null
        }}
        onPointerLeave={() => setHover(null)}
        onKeyDown={(e) => {
          const big = e.shiftKey ? 10 : 1
          if (e.key === 'ArrowRight') p.onSeek(Math.min(last, p.frame + big))
          else if (e.key === 'ArrowLeft') p.onSeek(Math.max(0, p.frame - big))
          else return
          e.preventDefault()
        }}
      />
    </div>
  )
}
