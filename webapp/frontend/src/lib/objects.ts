import type { Rect } from './coords'

/** A click prompt: 1 = "this is the object", 0 = "this is not". */
export type LabeledPoint = { x: number; y: number; label: 0 | 1 }

/**
 * One instance's prompt on one frame — a drawn rectangle, or a committed click
 * segment (`points`, plus the outline the model returned for them and the box
 * they refined, if any). One prompt per frame: an instance is a single tracked
 * identity, so several segments on a frame are several instances.
 */
export type Seed = {
  frame: number
  box: Rect | null
  points: LabeledPoint[]
  /** the segmented outline in image pixels — display only, the model re-derives
   *  it from `points` at track time so what you accepted is what gets tracked */
  polygons: number[][][]
}

/** One tracked identity: a class, a colour, and its prompt on each seeded frame. */
export type Instance = {
  id: string
  name: string
  cls: number | null
  color: [number, number, number]
  kind: 'box' | 'text'
  seeds: Seed[] // box kind
  phrase: string // text kind
  promptFrame: number // text kind
}

export type ClassDef = { id: number; name: string }

export const isSegment = (s: Seed): boolean => s.points.length > 0

// distinct, evenly-spread hues
const HUES = [210, 20, 140, 280, 45, 320, 170, 95, 255, 5]

export function paletteColor(i: number): [number, number, number] {
  const h = HUES[Math.abs(i) % HUES.length] / 360
  const s = 0.7
  const v = 1
  const f = (n: number) => {
    const k = (n + h * 6) % 6
    return Math.round(255 * (v - v * s * Math.max(0, Math.min(k, 4 - k, 1))))
  }
  return [f(5), f(3), f(1)]
}

/** Colour follows the class, so every instance of a class reads the same. */
export const colorFor = (cls: number | null, fallbackIndex: number): [number, number, number] =>
  paletteColor(cls === null ? fallbackIndex : cls)

export function rgbCss([r, g, b]: [number, number, number]) {
  return `rgb(${r}, ${g}, ${b})`
}

export const className = (cls: number | null, classes: ClassDef[]): string =>
  cls === null ? 'unclassified' : (classes.find((c) => c.id === cls)?.name ?? `class ${cls}`)

let counter = 0
const nextId = () => `i${Date.now()}_${(counter += 1)}`

/** Label an instance "<class> N", numbering within its class.
 *  Takes the lowest free number rather than a count, so deleting one and adding
 *  another can't produce two instances with the same label. */
export function instanceName(
  cls: number | null,
  classes: ClassDef[],
  existing: Instance[],
): string {
  const base = className(cls, classes)
  const taken = new Set(existing.filter((o) => o.cls === cls).map((o) => o.name))
  let n = 1
  while (taken.has(`${base} ${n}`)) n += 1
  return `${base} ${n}`
}

export function newInstance(
  kind: 'box' | 'text',
  cls: number | null,
  classes: ClassDef[],
  existing: Instance[],
  seed?: Seed,
): Instance {
  return {
    id: nextId(),
    name: instanceName(cls, classes, existing),
    cls,
    color: colorFor(cls, existing.length),
    kind,
    seeds: seed ? [seed] : [],
    phrase: '',
    promptFrame: 0,
  }
}

export const emptySeed = (frame: number): Seed => ({
  frame,
  box: null,
  points: [],
  polygons: [],
})

/** Why this seed can't be sent, or null. Mirrors backend/schemas.py. */
export function seedIssue(s: Seed): string | null {
  if (!s.box && !s.points.length && !(s.polygons && s.polygons.length))
    return `frame ${s.frame}: no box, click segment, or mask`
  if (s.points.length && !s.box && !s.points.some((p) => p.label === 1))
    return `frame ${s.frame}: needs at least one positive click`
  return null
}

/** This object's prompt on a given frame, if it has one. */
export const seedOn = (o: Instance, frame: number): Seed | undefined =>
  o.seeds.find((s) => s.frame === frame)

export function objectReady(o: Instance): boolean {
  if (o.kind === 'text') return o.phrase.trim() !== ''
  return o.seeds.length > 0 && o.seeds.every((s) => seedIssue(s) === null)
}

/** The inverse of `toPayload`, for replaying a stored run's instances. */
export function fromPayload(raw: unknown, index: number): Instance | null {
  const o = raw as Record<string, unknown>
  if (!o || (o.kind !== 'box' && o.kind !== 'text')) return null
  const cls = typeof o.cls === 'number' ? o.cls : null
  const color = Array.isArray(o.color) && o.color.length === 3
    ? (o.color as [number, number, number])
    : colorFor(cls, index)
  const seeds: Seed[] = Array.isArray(o.seeds)
    ? (o.seeds as Record<string, unknown>[]).map((s) => {
        const b = s.box as number[] | null | undefined
        const pts = (s.points as number[][] | undefined) ?? []
        const labels = (s.labels as number[] | undefined) ?? []
        return {
          frame: Number(s.frame ?? 0),
          box: b && b.length === 4 ? { x1: b[0], y1: b[1], x2: b[2], y2: b[3] } : null,
          points: pts.map((p, i) => ({ x: p[0], y: p[1], label: (labels[i] ? 1 : 0) as 0 | 1 })),
          polygons: (s.polygons as number[][][] | undefined) ?? [],
        }
      })
    : []
  return {
    id: nextId(),
    name: typeof o.name === 'string' ? o.name : `object ${index + 1}`,
    cls,
    color,
    kind: o.kind,
    seeds,
    phrase: typeof o.phrase === 'string' ? o.phrase : '',
    promptFrame: Number(o.prompt_frame ?? 0),
  }
}

export function toPayload(o: Instance) {
  const common = { name: o.name, color: o.color, cls: o.cls }
  if (o.kind === 'box') {
    return {
      ...common,
      kind: 'box' as const,
      seeds: o.seeds.map((s) => ({
        frame: s.frame,
        box: s.box ? [s.box.x1, s.box.y1, s.box.x2, s.box.y2] : null,
        points: s.points.map((p) => [p.x, p.y]),
        labels: s.points.map((p) => p.label),
        polygons: s.polygons ?? [],
      })),
    }
  }
  return {
    ...common,
    kind: 'text' as const,
    phrase: o.phrase.trim(),
    prompt_frame: o.promptFrame,
  }
}
