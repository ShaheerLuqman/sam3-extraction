import type { Ref } from '../lib/extraction'
import { refLabel, refThumb } from './format'

type Props = {
  refs: Ref[]
  videoId?: string
  /** refs that have a score row — the others are waiting on an embed or a score call */
  scored: (r: Ref) => boolean
  onToggle: (uid: string) => void
  onRemove: (uid: string) => void
  onShow: (frame: number) => void
}

/** The references as a grid of thumbnails. Click one to switch it off and on
 *  without losing it; the ✕ removes it. */
export function RefGrid({ refs, videoId, scored, onToggle, onRemove, onShow }: Props) {
  return (
    <ul className="refgrid">
      {refs.map((r, i) => (
        <li key={r.uid} className={`refcell${r.enabled ? '' : ' off'}`}>
          <button
            type="button"
            className="refthumb"
            onClick={() => onToggle(r.uid)}
            title={r.enabled ? 'In use — click to leave it out' : 'Left out — click to use it again'}
            aria-pressed={r.enabled}
          >
            <img src={refThumb(r, videoId)} alt={refLabel(r)} loading="lazy" />
            <span className="refnum">{i + 1}</span>
            {!scored(r) && r.enabled && <span className="refwait" title="Not embedded yet">…</span>}
          </button>
          <span className="refmeta">
            {r.kind === 'frame' ? (
              <button type="button" className="btn-text" onClick={() => onShow(r.frame)}>
                Frame {r.frame}
              </button>
            ) : (
              <span className="refname" title={r.name}>
                {r.name}
              </span>
            )}
            <button
              type="button"
              className="btn icon danger refx"
              onClick={() => onRemove(r.uid)}
              aria-label={`Remove ${refLabel(r)}`}
            >
              ✕
            </button>
          </span>
        </li>
      ))}
    </ul>
  )
}
