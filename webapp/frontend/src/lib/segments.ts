// Frame-extraction scoring, done in the browser so the threshold is live.
//
// A port of smooth / clean / segments from
// qwen_vl/extraction_project/scripts/match_frames.py, which is where the method
// was measured. The server sends one z-scored similarity row per reference over
// the embedded (every `stride`-th) frames; everything below is cheap and O(n).

export type Segment = {
  start: number
  end: number
  /** highest smoothed score inside the segment */
  peak: number
  /** the frame the peak is on */
  peakFrame: number
  mean: number
  /** index (into the enabled rows) of the reference that matched best */
  bestRef: number
}

export type Scored = {
  /** smoothed score per embedded frame — what the timeline draws */
  score: Float32Array
  /** which enabled row gave each embedded frame its (unsmoothed) max */
  best: Int16Array
  /** per video frame, the score of its nearest embedded frame */
  perFrame: Float32Array
}

/** Max over rows, then a centred moving average with edge padding. */
export function scoreRows(rows: number[][], win: number, stride: number, decoded: number): Scored {
  const n = rows[0]?.length ?? 0
  const raw = new Float32Array(n).fill(-Infinity)
  const best = new Int16Array(n)
  rows.forEach((row, r) => {
    for (let i = 0; i < n; i++) {
      if (row[i] > raw[i]) {
        raw[i] = row[i]
        best[i] = r
      }
    }
  })
  const score = smooth(raw, win)
  const perFrame = new Float32Array(decoded)
  for (let f = 0; f < decoded; f++) {
    perFrame[f] = n ? score[Math.min(n - 1, Math.max(0, Math.round(f / stride)))] : 0
  }
  return { score, best, perFrame }
}

export function smooth(x: Float32Array, win: number): Float32Array {
  if (win <= 1 || x.length === 0) return x
  const pad = Math.floor(win / 2)
  const at = (i: number) => x[Math.min(x.length - 1, Math.max(0, i))]
  const out = new Float32Array(x.length)
  let sum = 0
  for (let j = -pad; j < win - pad; j++) sum += at(j)
  for (let i = 0; i < x.length; i++) {
    out[i] = sum / win
    sum += at(i + win - pad) - at(i - pad)
  }
  return out
}

/** Runs of true, as inclusive [start, end] pairs. */
export function runs(mask: Uint8Array): [number, number][] {
  const out: [number, number][] = []
  let start = -1
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] && start < 0) start = i
    if (!mask[i] && start >= 0) {
      out.push([start, i - 1])
      start = -1
    }
  }
  if (start >= 0) out.push([start, mask.length - 1])
  return out
}

/** Fill gaps of at most `maxGap` frames, then drop runs shorter than `minSeg`. */
export function clean(mask: Uint8Array, minSeg: number, maxGap: number): Uint8Array {
  const m = mask.slice()
  const segs = runs(m)
  for (let i = 1; i < segs.length; i++) {
    const a1 = segs[i - 1][1]
    const b0 = segs[i][0]
    if (b0 - a1 - 1 <= maxGap) m.fill(1, a1 + 1, b0)
  }
  for (const [a, b] of runs(m)) if (b - a + 1 < minSeg) m.fill(0, a, b + 1)
  return m
}

/** Frames at or above `threshold`, cleaned, minus the ranges the user excluded. */
export function select(
  s: Scored,
  threshold: number,
  minSeg: number,
  maxGap: number,
  excluded: [number, number][],
  stride: number,
): Segment[] {
  const { perFrame, best } = s
  const mask = new Uint8Array(perFrame.length)
  for (let f = 0; f < perFrame.length; f++) mask[f] = perFrame[f] >= threshold ? 1 : 0
  // exclusions go in before cleaning, so the gap they leave is never re-filled
  for (const [a, b] of excluded) mask.fill(0, a, b + 1)
  const cleaned = clean(mask, minSeg, maxGap)
  for (const [a, b] of excluded) cleaned.fill(0, a, b + 1)

  return runs(cleaned).map(([start, end]) => {
    let peak = -Infinity
    let peakFrame = start
    let sum = 0
    for (let f = start; f <= end; f++) {
      sum += perFrame[f]
      if (perFrame[f] > peak) {
        peak = perFrame[f]
        peakFrame = f
      }
    }
    const i = Math.min(best.length - 1, Math.round(peakFrame / stride))
    return { start, end, peak, peakFrame, mean: sum / (end - start + 1), bestRef: best[i] ?? 0 }
  })
}
