import { useState } from 'react'
import { api } from '../api/client'
import { Progress } from '../components/Progress'
import type { useJob } from '../hooks/useJob'
import type { Segment } from '../lib/segments'

/** What an export card needs from a page. */
export type Exportable = {
  exportJob: ReturnType<typeof useJob>
  /** the ZIP's default "every Nth frame" */
  embStride: number
  segments: Segment[]
  doExport: (opts: { video: boolean; zipEvery: number }) => void
}

/** Export the selection: JSON always, optionally a clip and a ZIP of JPEGs. */
export function ExportCard({ x }: { x: Exportable }) {
  const { exportJob } = x
  const [clip, setClip] = useState(true)
  const [zip, setZip] = useState(false)
  const [every, setEvery] = useState(x.embStride)
  const r = exportJob.result as
    | { video_url?: string; zip_url?: string; json_url?: string; zipped?: number; message?: string }
    | null
  // a new selection makes the last export stale
  const sig = x.segments.map((s) => `${s.start}-${s.end}`).join(',')
  const [exportedSig, setExportedSig] = useState<string | null>(null)
  const stale = exportJob.status === 'done' && exportedSig !== sig

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>Export</h3>
          <p className="card-sub">
            Always a JSON of the segments (frame ranges, times, scores) and the settings behind them.
          </p>
        </div>
      </div>
      <label className="check">
        <input type="checkbox" checked={clip} onChange={(e) => setClip(e.target.checked)} />
        <span>
          A video of just the selected frames
          <em>each frame labelled with its segment and position in the source</em>
        </span>
      </label>
      <label className="check">
        <input type="checkbox" checked={zip} onChange={(e) => setZip(e.target.checked)} />
        <span>
          The frames as JPEGs, in a ZIP
          <em>one folder per segment</em>
        </span>
      </label>
      {zip && (
        <label className="field xevery">
          Keep every
          <input
            type="number"
            min={1}
            max={1000}
            value={every}
            onChange={(e) => setEvery(Math.max(1, Number(e.target.value) || 1))}
          />
          th frame
        </label>
      )}
      <button
        type="button"
        className="btn primary"
        disabled={exportJob.busy}
        onClick={() => {
          setExportedSig(sig)
          x.doExport({ video: clip, zipEvery: zip ? every : 0 })
        }}
      >
        Export {x.segments.length} segment{x.segments.length > 1 ? 's' : ''}
      </button>
      <Progress job={exportJob} onCancel={exportJob.busy ? exportJob.cancel : undefined} />
      {exportJob.status === 'done' && r && (
        <div className="xexport">
          {stale && (
            <p className="note warn">The selection changed since this export. Export again to match it.</p>
          )}
          {r.video_url && (
            <video className="runvideo" src={api.fileUrl(r.video_url)} controls muted playsInline />
          )}
          <div className="row gap wrap">
            {r.video_url && (
              <a className="btn ghost" href={api.fileUrl(r.video_url)} download>
                Download clip
              </a>
            )}
            {r.zip_url && (
              <a className="btn ghost" href={api.fileUrl(r.zip_url)} download>
                Download {r.zipped} frames (ZIP)
              </a>
            )}
            {r.json_url && (
              <a className="btn ghost" href={api.fileUrl(r.json_url)} download>
                Download segments (JSON)
              </a>
            )}
          </div>
          {r.message && <p className="muted small">{r.message}</p>}
        </div>
      )}
    </section>
  )
}
