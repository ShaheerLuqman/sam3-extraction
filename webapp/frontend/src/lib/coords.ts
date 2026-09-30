// The displayed image uses object-fit: contain, so there is a uniform scale and
// symmetric letterbox padding between the rendered pixels and the natural image.

export type Rect = { x1: number; y1: number; x2: number; y2: number }

export type FitBox = { offX: number; offY: number; scale: number; drawW: number; drawH: number }

/** How the natural image sits inside an element of size (elW, elH) under `contain`. */
export function containFit(natW: number, natH: number, elW: number, elH: number): FitBox {
  const scale = Math.min(elW / natW, elH / natH)
  const drawW = natW * scale
  const drawH = natH * scale
  return { offX: (elW - drawW) / 2, offY: (elH - drawH) / 2, scale, drawW, drawH }
}

/** Pointer position (element-local px) -> natural image px, clamped to bounds. */
export function toImageCoords(
  px: number,
  py: number,
  fit: FitBox,
  natW: number,
  natH: number,
): { x: number; y: number } {
  const x = (px - fit.offX) / fit.scale
  const y = (py - fit.offY) / fit.scale
  return {
    x: Math.max(0, Math.min(natW, Math.round(x))),
    y: Math.max(0, Math.min(natH, Math.round(y))),
  }
}

/** Natural image px -> element-local px (for drawing overlays). */
export function toCanvasCoords(x: number, y: number, fit: FitBox): { x: number; y: number } {
  return { x: fit.offX + x * fit.scale, y: fit.offY + y * fit.scale }
}

/** Intersection over union of two boxes, 0 when they don't overlap. */
export function iou(a: Rect, b: Rect): number {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)
  const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1)
  if (w <= 0 || h <= 0) return 0
  const inter = w * h
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
  return union > 0 ? inter / union : 0
}

export function normRect(a: { x: number; y: number }, b: { x: number; y: number }): Rect {
  return {
    x1: Math.min(a.x, b.x),
    y1: Math.min(a.y, b.y),
    x2: Math.max(a.x, b.x),
    y2: Math.max(a.y, b.y),
  }
}
