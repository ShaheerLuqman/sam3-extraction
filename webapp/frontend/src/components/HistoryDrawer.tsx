import { useCallback, useEffect, useState } from 'react'
import { api, type RunDetail, type RunSummary } from '../api/client'

type Props = {
  open: boolean
  onClose: () => void
  /** the upload currently loaded — a run's instances only replay onto its own video */
  currentUploadId?: string
  onRestore: (run: RunDetail, withObjects: boolean) => void
}

function when(ts: number): string {
  const d = new Date(ts * 1000)
  const mins = Math.round((Date.now() - d.getTime()) / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} min ago`
  if (mins < 60 * 24) return `${Math.round(mins / 60)} h ago`
  return d.toLocaleString()
}

function took(run: RunSummary): string {
  if (!run.finished_at) return ''
  const s = Math.round(run.finished_at - run.created_at)
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`
}

export function HistoryDrawer({ open, onClose, currentUploadId, onRestore }: Props) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [openId, setOpenId] = useState<string | null>(null)
  const [detail, setDetail] = useState<RunDetail | null>(null)

  const load = useCallback(async () => {
    try {
      setRuns((await api.runs()).runs)
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  useEffect(() => {
    if (open) void load()
  }, [open, load])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  const expand = async (id: string) => {
    if (openId === id) {
      setOpenId(null)
      setDetail(null)
      return
    }
    setOpenId(id)
    setDetail(null)
    try {
      setDetail(await api.run(id))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  const remove = async (id: string) => {
    try {
      await api.deleteRun(id)
      if (openId === id) {
        setOpenId(null)
        setDetail(null)
      }
      await load()
    } catch (e) {
      setError((e as Error).message)
    }
  }

  if (!open) return null

  return (
    <div className="drawer-scrim" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <header className="drawer-head">
          <div>
            <h2>History</h2>
            <p className="card-sub">
              Every tracking run, newest first. Each time you track is its own run.
            </p>
          </div>
          <button type="button" className="btn icon" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        {error && <p className="note danger">{error}</p>}
        {runs === null && !error && <p className="muted">Loading…</p>}
        {runs?.length === 0 && <p className="empty">No runs yet.</p>}

        <ul className="runlist">
          {runs?.map((r) => (
            <li key={r.id} className={`runitem${openId === r.id ? ' open' : ''}`}>
              <button type="button" className="runrow" onClick={() => void expand(r.id)}>
                <span className={`badge ${r.status === 'done' ? 'ok' : r.status === 'error' ? 'bad' : 'warn'}`}>
                  {r.status}
                </span>
                <span className="runmain">
                  <span className="runsource">{r.source}</span>
                  <span className="runmeta">
                    {r.object_count} instance{r.object_count === 1 ? '' : 's'} ·{' '}
                    {r.settings.max_frames} frames
                    {r.settings.bidirectional ? ' · bidirectional' : ''}
                    {took(r) ? ` · ${took(r)}` : ''}
                  </span>
                </span>
                <span className="runwhen">{when(r.created_at)}</span>
              </button>

              {openId === r.id && (
                <div className="rundetail">
                  {r.status === 'error' && <p className="note danger">{r.error}</p>}

                  {r.status === 'done' && !r.outputs_present && (
                    <p className="note warn">
                      The video and JSON for this run have been cleared out. The settings and
                      instances below are still here.
                    </p>
                  )}

                  {r.outputs_present && r.result?.tracked_video_url && (
                    <>
                      <video
                        className="runvideo"
                        src={api.fileUrl(String(r.result.tracked_video_url))}
                        controls
                        muted
                        playsInline
                      />
                      <div className="row gap wrap">
                        <a
                          className="btn ghost"
                          href={api.fileUrl(String(r.result.tracked_video_url))}
                          download
                        >
                          Download video
                        </a>
                        {r.result.json_url && (
                          <a
                            className="btn ghost"
                            href={api.fileUrl(String(r.result.json_url))}
                            download
                          >
                            Download tracks (JSON)
                          </a>
                        )}
                      </div>
                    </>
                  )}

                  <dl className="runfacts">
                    <div><dt>Source</dt><dd>{r.source}</dd></div>
                    <div><dt>Resolution</dt><dd>{r.width} × {r.height}</dd></div>
                    <div><dt>Max frames</dt><dd>{r.settings.max_frames}</dd></div>
                    <div><dt>Detection threshold</dt><dd>{r.settings.threshold.toFixed(2)}</dd></div>
                    <div><dt>Bidirectional</dt><dd>{r.settings.bidirectional ? 'yes' : 'no'}</dd></div>
                    <div><dt>Started</dt><dd>{new Date(r.created_at * 1000).toLocaleString()}</dd></div>
                  </dl>

                  {detail && (
                    <div className="row gap wrap">
                      <button
                        type="button"
                        className="btn primary"
                        onClick={() => {
                          onRestore(detail, detail.upload_id === currentUploadId)
                          onClose()
                        }}
                      >
                        {detail.upload_id === currentUploadId
                          ? 'Reuse settings and instances'
                          : 'Reuse settings'}
                      </button>
                      <button type="button" className="btn-text" onClick={() => void remove(r.id)}>
                        Delete this run
                      </button>
                    </div>
                  )}
                  {detail && detail.upload_id !== currentUploadId && (
                    <p className="muted small">
                      The instances belong to a different upload, so only the settings can be
                      reused — their coordinates would not line up with the video you have open.
                    </p>
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      </aside>
    </div>
  )
}
