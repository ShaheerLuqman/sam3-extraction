type Props = {
  frame: number
  last: number
  fps?: number
  /** frames that already carry a prompt, marked as ticks on the scrubber */
  marked: number[]
  onChange: (f: number) => void
}

function clock(frame: number, fps?: number): string {
  if (!fps) return ''
  const s = frame / fps
  const m = Math.floor(s / 60)
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`
}

/** Frame scrubber: step one frame at a time or drag to any frame. */
export function FrameBar({ frame, last, fps, marked, onChange }: Props) {
  const go = (f: number) => onChange(Math.max(0, Math.min(last, f)))
  const time = clock(frame, fps)

  return (
    <div className="framebar">
      <button type="button" className="btn icon" onClick={() => go(frame - 1)} disabled={frame <= 0}>
        ‹
      </button>

      <div className="framebar-track">
        <input
          type="range"
          min={0}
          max={Math.max(1, last)}
          step={1}
          value={frame}
          onChange={(e) => go(Number(e.target.value))}
          aria-label="Frame"
        />
        {marked.length > 0 && last > 0 && (
          <div className="framebar-ticks" aria-hidden>
            {marked.map((f) => (
              <span key={f} className="framebar-tick" style={{ left: `${(f / last) * 100}%` }} />
            ))}
          </div>
        )}
      </div>

      <button
        type="button"
        className="btn icon"
        onClick={() => go(frame + 1)}
        disabled={frame >= last}
      >
        ›
      </button>

      <span className="framebar-read">
        Frame {frame} / {last}
        {time && <span className="framebar-time">{time}</span>}
      </span>
    </div>
  )
}
