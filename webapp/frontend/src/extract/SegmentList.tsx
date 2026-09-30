import type { Ref } from '../lib/extraction'
import type { Segment } from '../lib/segments'
import { clock, refLabel, refThumb } from './format'

type Props = {
  segments: Segment[]
  /** the enabled, scored references, in the order Segment.bestRef indexes */
  active: Ref[]
  /** position of each reference in the full list, for its number badge */
  refIndex: (r: Ref) => number
  videoId?: string
  fps: number
  frame: number
  playing: number | null
  onSeek: (f: number) => void
  onPlay: (i: number) => void
  /** optional: add the segment as a reference */
  onUseAsRef?: (s: Segment) => void
  onExclude: (s: Segment) => void
}

export function SegmentList(p: Props) {
  if (!p.segments.length) {
    return <p className="empty">Nothing is above the threshold. Lower it, or add references.</p>
  }
  return (
    <ol className="seglist">
      {p.segments.map((s, i) => {
        const here = p.frame >= s.start && p.frame <= s.end
        const ref = p.active[s.bestRef]
        return (
          <li key={`${s.start}-${s.end}`} className={`segitem${here ? ' here' : ''}`}>
            <button type="button" className="segmain" onClick={() => p.onSeek(s.start)}>
              <span className="segidx">{i + 1}</span>
              <span className="segtext">
                <span className="segtime">
                  {clock(s.start, p.fps, true)} – {clock(s.end + 1, p.fps, true)}
                  <em>{((s.end - s.start + 1) / p.fps).toFixed(1)} s</em>
                </span>
                <span className="segmeta" title={`frames ${s.start}–${s.end}, peak score ${s.peak.toFixed(2)}`}>
                  #{s.start}–{s.end} · peak {s.peak.toFixed(2)}
                </span>
              </span>
              {ref && (
                <img
                  className="segref"
                  src={refThumb(ref, p.videoId)}
                  alt=""
                  title={`Closest to reference ${p.refIndex(ref) + 1}: ${refLabel(ref)}`}
                />
              )}
            </button>
            <span className="segactions">
              <button
                type="button"
                className={`btn icon${p.playing === i ? ' on' : ''}`}
                onClick={() => p.onPlay(i)}
                title={p.playing === i ? 'Stop' : 'Play this segment'}
                aria-label={p.playing === i ? 'Stop' : `Play segment ${i + 1}`}
              >
                {p.playing === i ? '■' : '▶'}
              </button>
              {p.onUseAsRef && (
                <button
                  type="button"
                  className="btn icon"
                  onClick={() => p.onUseAsRef?.(s)}
                  title="Correct match: add its best frame as a reference, to find more like it"
                  aria-label={`Add segment ${i + 1} as a reference`}
                >
                  ＋
                </button>
              )}
              <button
                type="button"
                className="btn icon danger"
                onClick={() => p.onExclude(s)}
                title="Wrong match: leave these frames out"
                aria-label={`Exclude segment ${i + 1}`}
              >
                ✕
              </button>
            </span>
          </li>
        )
      })}
    </ol>
  )
}
