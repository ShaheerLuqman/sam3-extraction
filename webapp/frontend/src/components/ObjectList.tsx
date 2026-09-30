import { objectReady, rgbCss, seedOn, type ClassDef, type Instance } from '../lib/objects'

type Props = {
  objects: Instance[]
  classes: ClassDef[]
  selectedId: string | null
  frame: number
  busy: boolean
  onSelect: (id: string) => void
  onRemove: (id: string) => void
  onRename: (id: string, name: string) => void
  onRelabel: (id: string, cls: number | null) => void
  onFindMore: (o: Instance) => void
  /** which model backs the similar-object search — it sits beside the button */
  matchStyle: 'sam3' | 'yoloe'
  onMatchStyle: (m: 'sam3' | 'yoloe') => void
  onGoToFrame: (f: number) => void
  onReseed: (o: Instance) => void
}

export function ObjectList({
  objects,
  classes,
  selectedId,
  frame,
  busy,
  onSelect,
  onRemove,
  onRename,
  onRelabel,
  onFindMore,
  matchStyle,
  onMatchStyle,
  onGoToFrame,
  onReseed,
}: Props) {
  if (objects.length === 0) {
    return (
      <p className="empty">
        No instances yet. Use a click, box, or text prompt above to detect your first object.
      </p>
    )
  }

  return (
    <ul className="objlist">
      {objects.map((o) => {
        const open = o.id === selectedId
        const here = o.kind === 'box' && !!seedOn(o, frame)
        const firstFrame = o.seeds[0]?.frame
        return (
          <li key={o.id} className={`objitem${open ? ' open' : ''}`}>
            <div className="objrow" onClick={() => onSelect(o.id)}>
              <span className="swatch" style={{ background: rgbCss(o.color) }} />
              <input
                className="objname"
                value={o.name}
                aria-label="Name"
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => onRename(o.id, e.target.value)}
              />
              <span className="objwhere">
                {o.kind === 'text'
                  ? 'text prompt'
                  : here
                    ? `seeded here · ${o.seeds.length} seed${o.seeds.length > 1 ? 's' : ''}`
                    : `frame ${firstFrame ?? 0}`}
              </span>
              {!objectReady(o) && (
                <span className="badge warn" title="Needs a mask, box, or phrase">
                  no prompt
                </span>
              )}
              <button
                type="button"
                className="btn icon danger"
                title={`Remove ${o.name}`}
                aria-label={`Remove ${o.name}`}
                onClick={(e) => {
                  e.stopPropagation()
                  onRemove(o.id)
                }}
              >
                ✕
              </button>
            </div>

            {open && (
              <div className="objdetail">
                {classes.length > 0 && (
                  <label className="field">
                    <span>Class</span>
                    <select
                      value={o.cls ?? ''}
                      onChange={(e) =>
                        onRelabel(o.id, e.target.value === '' ? null : Number(e.target.value))
                      }
                    >
                      <option value="">— unclassified —</option>
                      {classes.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.id}: {c.name}
                        </option>
                      ))}
                    </select>
                  </label>
                )}

                {o.kind === 'box' && (
                  <>
                    <div className="objframes">
                      {o.seeds.map((s) => (
                        <button
                          key={s.frame}
                          type="button"
                          className={`pill${s.frame === frame ? ' on' : ''}`}
                          onClick={() => onGoToFrame(s.frame)}
                        >
                          frame {s.frame}
                        </button>
                      ))}
                    </div>
                    <div className="findsim">
                      <button
                        type="button"
                        className="btn ghost"
                        disabled={busy}
                        onClick={() => onFindMore(o)}
                      >
                        Find similar objects
                      </button>
                      <select
                        aria-label="Similar-object search model"
                        title="SAM 3 exemplar: precise masks, noisier in clutter. YOLOE: cleaner in clutter, looser boxes."
                        value={matchStyle}
                        onChange={(e) => onMatchStyle(e.target.value as 'sam3' | 'yoloe')}
                      >
                        <option value="sam3">SAM 3 exemplar</option>
                        <option value="yoloe">YOLOE</option>
                      </select>
                    </div>
                    <p className="muted small">
                      Uses this instance's own mask as a visual exemplar and proposes every
                      matching object on the frame — the exemplar itself is left out.
                    </p>
                    <button
                      type="button"
                      className="btn-text"
                      onClick={() => onReseed(o)}
                      title="Add a second prompt for this same instance on the frame you're on, to re-anchor a track that has drifted"
                    >
                      Re-seed on frame {frame}
                    </button>
                  </>
                )}
              </div>
            )}
          </li>
        )
      })}
    </ul>
  )
}
