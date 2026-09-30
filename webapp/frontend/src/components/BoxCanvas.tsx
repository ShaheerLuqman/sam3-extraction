import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { containFit, normRect, toCanvasCoords, toImageCoords, type Rect } from '../lib/coords'
import type { LabeledPoint } from '../lib/objects'

export type Candidate = {
  box: Rect
  score: number
  polygons?: number[][][]
  /** the instance this search started from — shown, but never added again */
  existing?: boolean
}
/** A mask outline in image pixels, from the backend's click preview. */
export type MaskPreview = { polygons: number[][][]; box?: Rect | null }

/** A committed instance as it appears on this frame. */
export type Shape = {
  id: string
  label: string
  color: [number, number, number]
  box: Rect | null
  polygons: number[][][]
  selected?: boolean
}

type Props = {
  src: string
  natW: number
  natH: number
  /** 'box' drags out a rectangle, 'points' places click prompts, 'exemplar' is
   *  one throwaway box for the exemplar search, 'view' is read-only. */
  mode: 'box' | 'points' | 'exemplar' | 'view'
  /** committed instances on this frame, drawn under the live prompt */
  shapes?: Shape[]
  onPickShape?: (id: string) => void
  /** clicking the ✕ on a committed instance removes it */
  onRemoveShape?: (id: string) => void
  /** the prompt being built right now */
  pendingBox?: Rect | null
  onBox?: (r: Rect) => void
  onClearBox?: () => void
  points?: LabeledPoint[]
  onPointsChange?: (points: LabeledPoint[]) => void
  nextLabel?: 0 | 1
  preview?: MaskPreview | null
  onStatus?: (msg: string) => void
  /** candidates from the exemplar or text search */
  candidates?: Candidate[]
  threshold?: number
  onRemoveCandidate?: (index: number) => void
  accent?: [number, number, number]
}

const POS = '#16a34a'
const NEG = '#dc2626'
const PENDING = '#d97706'
/** the "already added" exemplar — deliberately not the accent colour */
const EXISTING: [number, number, number] = [100, 116, 139]
/** click within this many screen px of a point to remove it instead of adding one */
const HIT_PX = 10
/** the ✕ discs drawn on each committed instance and kept candidate */
const BADGE_R = 10
const BADGE_HIT = 14

type Badge =
  | { kind: 'shape'; id: string; x: number; y: number }
  | { kind: 'candidate'; index: number; x: number; y: number }
  | { kind: 'pending'; x: number; y: number }

export function BoxCanvas({
  src,
  natW,
  natH,
  mode,
  shapes = [],
  onPickShape,
  onRemoveShape,
  pendingBox = null,
  onBox,
  onClearBox,
  points = [],
  onPointsChange,
  nextLabel = 1,
  preview,
  onStatus,
  candidates,
  threshold = 0.5,
  onRemoveCandidate,
  accent = [37, 99, 235],
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  /** where the ✕ discs ended up this paint, in element-local px */
  const badgesRef = useRef<Badge[]>([])
  const [size, setSize] = useState({ w: 0, h: 0 })
  const [corner, setCorner] = useState<{ x: number; y: number } | null>(null)
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null)
  const [overBadge, setOverBadge] = useState(false)

  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    setSize({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    setCorner(null)
    setHover(null)
  }, [src])

  useEffect(() => {
    const cv = canvasRef.current
    if (!cv || !size.w || !size.h) return
    const dpr = window.devicePixelRatio || 1
    cv.width = size.w * dpr
    cv.height = size.h * dpr
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size.w, size.h)
    const fit = containFit(natW, natH, size.w, size.h)
    const acc = `rgb(${accent[0]}, ${accent[1]}, ${accent[2]})`
    const badges: Badge[] = []

    const drawRect = (b: Rect, color: string, dashed = false, width = 2) => {
      const a = toCanvasCoords(b.x1, b.y1, fit)
      const c = toCanvasCoords(b.x2, b.y2, fit)
      ctx.lineWidth = width
      ctx.strokeStyle = color
      ctx.setLineDash(dashed ? [6, 4] : [])
      ctx.strokeRect(a.x, a.y, c.x - a.x, c.y - a.y)
      ctx.setLineDash([])
    }

    const tracePolys = (polys: number[][][]) => {
      ctx.beginPath()
      for (const poly of polys) {
        poly.forEach(([x, y], i) => {
          const p = toCanvasCoords(x, y, fit)
          if (i === 0) ctx.moveTo(p.x, p.y)
          else ctx.lineTo(p.x, p.y)
        })
        ctx.closePath()
      }
    }

    const drawMask = (
      polys: number[][][],
      rgb: [number, number, number],
      alpha: number,
      width = 2,
    ) => {
      if (!polys.length) return
      tracePolys(polys)
      ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`
      ctx.fill('evenodd')
      ctx.strokeStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
      ctx.lineWidth = width
      ctx.stroke()
    }

    /** A name tag that stays readable on any background. */
    const drawTag = (at: Rect, text: string, rgb: [number, number, number]) => {
      const p = toCanvasCoords(at.x1, at.y1, fit)
      ctx.font = '600 12px system-ui, sans-serif'
      const w = ctx.measureText(text).width + 12
      const h = 19
      const y = p.y - h - 3 >= 0 ? p.y - h - 3 : p.y + 3
      roundRect(ctx, p.x, y, w, h, 5)
      ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.96)`
      ctx.fill()
      ctx.fillStyle = '#ffffff'
      ctx.fillText(text, p.x + 6, y + 13.5)
    }

    /** The ✕ disc that deletes whatever it sits on. */
    const drawClose = (cx: number, cy: number) => {
      ctx.beginPath()
      ctx.arc(cx, cy, BADGE_R, 0, Math.PI * 2)
      ctx.fillStyle = '#ffffff'
      ctx.fill()
      ctx.lineWidth = 1.5
      ctx.strokeStyle = 'rgba(16, 32, 51, 0.35)'
      ctx.stroke()
      ctx.strokeStyle = '#b42318'
      ctx.lineWidth = 2
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.moveTo(cx - 3.5, cy - 3.5)
      ctx.lineTo(cx + 3.5, cy + 3.5)
      ctx.moveTo(cx + 3.5, cy - 3.5)
      ctx.lineTo(cx - 3.5, cy + 3.5)
      ctx.stroke()
      ctx.lineCap = 'butt'
    }

    // committed instances first, so the live prompt always sits on top
    for (const sh of shapes) {
      const a = sh.selected ? 0.4 : 0.22
      if (sh.polygons.length) drawMask(sh.polygons, sh.color, a, sh.selected ? 3 : 2)
      if (sh.box) {
        drawRect(
          sh.box,
          `rgb(${sh.color[0]}, ${sh.color[1]}, ${sh.color[2]})`,
          sh.polygons.length > 0,
          sh.selected ? 3 : 2,
        )
        if (!sh.polygons.length) {
          const q1 = toCanvasCoords(sh.box.x1, sh.box.y1, fit)
          const q2 = toCanvasCoords(sh.box.x2, sh.box.y2, fit)
          ctx.fillStyle = `rgba(${sh.color[0]}, ${sh.color[1]}, ${sh.color[2]}, ${a * 0.6})`
          ctx.fillRect(q1.x, q1.y, q2.x - q1.x, q2.y - q1.y)
        }
      }
      const anchor = sh.box ?? bounds(sh.polygons)
      if (!anchor) continue
      drawTag(anchor, sh.label, sh.color)
      if (onRemoveShape) {
        const c = toCanvasCoords(anchor.x2, anchor.y1, fit)
        const cx = Math.min(size.w - BADGE_R - 1, c.x)
        const cy = Math.max(BADGE_R + 1, c.y)
        drawClose(cx, cy)
        badges.push({ kind: 'shape', id: sh.id, x: cx, y: cy })
      }
    }

    // the mask the current clicks select
    if (preview?.polygons?.length) drawMask(preview.polygons, accent, 0.35)

    // exemplar / text-search candidates
    if (candidates) {
      candidates.forEach((cand, i) => {
        // the object the search started from is drawn so you can see it matched,
        // but it is already an instance, so it gets no score, no ✕ and no count
        if (cand.existing) {
          if (cand.polygons?.length) drawMask(cand.polygons, EXISTING, 0.22, 2)
          drawRect(cand.box, `rgb(${EXISTING[0]}, ${EXISTING[1]}, ${EXISTING[2]})`, false, 2)
          drawTag(cand.box, 'already added', EXISTING)
          return
        }
        const on = cand.score >= threshold
        const cColor: [number, number, number] = on ? accent : [148, 163, 184]
        if (cand.polygons && cand.polygons.length > 0) {
          drawMask(cand.polygons, cColor, on ? 0.3 : 0.08, on ? 2 : 1)
        }
        drawRect(cand.box, on ? acc : 'rgba(148,163,184,0.6)', !on, on ? 2 : 1)
        if (!on) return
        drawTag(cand.box, `${Math.round(cand.score * 100)}%`, accent)
        if (onRemoveCandidate) {
          const c = toCanvasCoords(cand.box.x2, cand.box.y1, fit)
          const cx = Math.min(size.w - BADGE_R - 1, c.x)
          const cy = Math.max(BADGE_R + 1, c.y)
          drawClose(cx, cy)
          badges.push({ kind: 'candidate', index: i, x: cx, y: cy })
        }
      })
    }

    if (pendingBox) {
      drawRect(pendingBox, PENDING, true, 2.5)
      if (onClearBox) {
        const c = toCanvasCoords(pendingBox.x2, pendingBox.y1, fit)
        const cx = Math.min(size.w - BADGE_R - 1, c.x)
        const cy = Math.max(BADGE_R + 1, c.y)
        drawClose(cx, cy)
        badges.push({ kind: 'pending', x: cx, y: cy })
      }
    }

    for (const pt of points) {
      const p = toCanvasCoords(pt.x, pt.y, fit)
      ctx.beginPath()
      ctx.arc(p.x, p.y, 7, 0, Math.PI * 2)
      ctx.fillStyle = pt.label ? POS : NEG
      ctx.fill()
      ctx.lineWidth = 2.5
      ctx.strokeStyle = '#ffffff'
      ctx.stroke()
      // a plus / minus bar, so the two kinds stay apart without relying on colour
      ctx.beginPath()
      ctx.moveTo(p.x - 3.5, p.y)
      ctx.lineTo(p.x + 3.5, p.y)
      if (pt.label) {
        ctx.moveTo(p.x, p.y - 3.5)
        ctx.lineTo(p.x, p.y + 3.5)
      }
      ctx.lineWidth = 2
      ctx.strokeStyle = '#fff'
      ctx.stroke()
    }

    if (corner && mode !== 'points') {
      const p = toCanvasCoords(corner.x, corner.y, fit)
      ctx.strokeStyle = PENDING
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(p.x - 10, p.y)
      ctx.lineTo(p.x + 10, p.y)
      ctx.moveTo(p.x, p.y - 10)
      ctx.lineTo(p.x, p.y + 10)
      ctx.stroke()
      if (hover) drawRect(normRect(corner, hover), PENDING, true, 2.5)
    }

    badgesRef.current = badges
  }, [
    shapes, points, preview, pendingBox, corner, hover, size, natW, natH, mode,
    candidates, threshold, accent, onClearBox, onRemoveShape, onRemoveCandidate,
  ])

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (corner) {
        setCorner(null)
        onStatus?.('Cancelled.')
      } else if (pendingBox && onClearBox) {
        onClearBox()
        onStatus?.('Box removed.')
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [corner, pendingBox, onClearBox, onStatus])

  const localPos = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top, rect }
  }

  const evtToImg = (e: React.MouseEvent) => {
    const { x, y, rect } = localPos(e)
    return toImageCoords(x, y, containFit(natW, natH, rect.width, rect.height), natW, natH)
  }

  const badgeAt = (x: number, y: number): Badge | null => {
    for (let i = badgesRef.current.length - 1; i >= 0; i--) {
      const b = badgesRef.current[i]
      if (Math.hypot(x - b.x, y - b.y) <= BADGE_HIT) return b
    }
    return null
  }

  const addPoint = (e: React.MouseEvent, label: 0 | 1) => {
    if (!onPointsChange) return
    const p = evtToImg(e)
    const { rect } = localPos(e)
    const { scale } = containFit(natW, natH, rect.width, rect.height)
    const hitIdx = points.findIndex((q) => Math.hypot(q.x - p.x, q.y - p.y) * scale <= HIT_PX)
    if (hitIdx >= 0) {
      onPointsChange(points.filter((_, i) => i !== hitIdx))
      onStatus?.('Click removed.')
      return
    }
    onPointsChange([...points, { ...p, label }])
    onStatus?.('')
  }

  const onClick = (e: React.MouseEvent) => {
    const { x, y } = localPos(e)
    // a ✕ always wins over whatever the current tool would do
    const badge = badgeAt(x, y)
    if (badge) {
      if (badge.kind === 'shape') {
        onRemoveShape?.(badge.id)
        onStatus?.('Instance removed.')
      } else if (badge.kind === 'candidate') {
        onRemoveCandidate?.(badge.index)
        onStatus?.('Candidate discarded.')
      } else {
        onClearBox?.()
        onStatus?.('Box removed.')
      }
      return
    }

    if (mode === 'points') {
      // alt-click flips the kind, for trackpads where right-click is awkward
      addPoint(e, e.altKey ? (nextLabel ? 0 : 1) : nextLabel)
      return
    }

    const p = evtToImg(e)
    // clicking a committed instance selects it rather than starting a box
    const hit = onPickShape && !corner ? topShapeAt(shapes, p) : null
    if (hit) {
      onPickShape!(hit)
      return
    }
    if (mode === 'view') return
    if (!corner) {
      setCorner(p)
      onStatus?.('Click the opposite corner.')
      return
    }
    const r = normRect(corner, p)
    setCorner(null)
    if (r.x2 - r.x1 < 3 || r.y2 - r.y1 < 3) {
      onStatus?.('Box too small — try again.')
      return
    }
    onBox?.(r)
    onStatus?.('')
  }

  const onContextMenu = (e: React.MouseEvent) => {
    if (mode !== 'points') return
    e.preventDefault()
    addPoint(e, 0)
  }

  const cursor = overBadge ? 'pointer' : mode === 'view' ? 'default' : 'crosshair'

  return (
    <div className="boxcanvas" ref={wrapRef}>
      <img src={src} alt="Video frame" draggable={false} />
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h, cursor }}
        onMouseMove={(e) => {
          const { x, y } = localPos(e)
          setOverBadge(badgeAt(x, y) !== null)
          setHover(evtToImg(e))
        }}
        onMouseLeave={() => {
          setHover(null)
          setOverBadge(false)
        }}
        onClick={onClick}
        onContextMenu={onContextMenu}
      />
      {corner && (
        <div className="canvas-tip">
          Click the opposite corner · <kbd>Esc</kbd> to cancel
        </div>
      )}
    </div>
  )
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.arcTo(x + w, y, x + w, y + h, r)
  ctx.arcTo(x + w, y + h, x, y + h, r)
  ctx.arcTo(x, y + h, x, y, r)
  ctx.arcTo(x, y, x + w, y, r)
  ctx.closePath()
}

function bounds(polys: number[][][]): Rect | null {
  let x1 = Infinity,
    y1 = Infinity,
    x2 = -Infinity,
    y2 = -Infinity
  for (const poly of polys)
    for (const [x, y] of poly) {
      if (x < x1) x1 = x
      if (y < y1) y1 = y
      if (x > x2) x2 = x
      if (y > y2) y2 = y
    }
  return Number.isFinite(x1) ? { x1, y1, x2, y2 } : null
}

/** Smallest instance whose extent covers this point — the last drawn wins ties. */
function topShapeAt(shapes: Shape[], p: { x: number; y: number }): string | null {
  let best: { id: string; area: number } | null = null
  for (const sh of shapes) {
    const b = sh.box ?? bounds(sh.polygons)
    if (!b || p.x < b.x1 || p.x > b.x2 || p.y < b.y1 || p.y > b.y2) continue
    const area = (b.x2 - b.x1) * (b.y2 - b.y1)
    if (!best || area <= best.area) best = { id: sh.id, area }
  }
  return best?.id ?? null
}
