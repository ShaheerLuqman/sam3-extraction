import { useRef, useState } from 'react'
import { api } from '../api/client'
import type { ClassDef } from '../lib/objects'

type Props = {
  classes: ClassDef[]
  source: string | null
  onLoaded: (classes: ClassDef[], source: string | null) => void
}

/**
 * The optional class list. Assigning a class to a particular instance happens
 * at save time during detection, not here.
 */
export function LabelsCard({ classes, source, onLoaded }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)

  const send = async (file: File) => {
    setBusy(true)
    setError(null)
    setNote(null)
    try {
      const r = await api.uploadClasses(file)
      onLoaded(r.classes, r.source ?? file.name)
      setNote(r.warning ?? null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const clear = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.clearClasses()
      onLoaded([], null)
      setNote(null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>Class labels</h3>
          <p className="card-sub">Optional — a classes.txt, one class name per line.</p>
        </div>
        {classes.length > 0 && (
          <span className="badge ok">
            {classes.length} class{classes.length > 1 ? 'es' : ''}
          </span>
        )}
      </div>

      {classes.length === 0 ? (
        <p className="muted">
          Upload a <code>classes.txt</code> to assign a class to each object you detect. Line N
          is class id N (the YOLO convention); the id and the resolved name both land in the
          result JSON. Without one, instances track as <em>unclassified 1</em>,{' '}
          <em>unclassified 2</em>, and so on.
        </p>
      ) : (
        <>
          <div className="chips">
            {classes.slice(0, 12).map((c) => (
              <span key={c.id} className="chip">
                {c.name}
              </span>
            ))}
            {classes.length > 12 && <span className="chip more">+{classes.length - 12} more</span>}
          </div>
          {source && <p className="muted small">Loaded from {source}</p>}
        </>
      )}

      <div className="row gap">
        <button type="button" className="btn ghost" disabled={busy} onClick={() => fileRef.current?.click()}>
          {busy ? 'Uploading…' : classes.length ? 'Replace classes.txt' : 'Upload classes.txt'}
        </button>
        {classes.length > 0 && (
          <button type="button" className="btn-text" disabled={busy} onClick={clear}>
            Clear class list
          </button>
        )}
      </div>

      <input
        ref={fileRef}
        type="file"
        accept=".txt,.names,text/plain"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) void send(f)
        }}
      />
      {error && <p className="note danger">{error}</p>}
      {note && !error && <p className="note warn">{note}</p>}
    </section>
  )
}
