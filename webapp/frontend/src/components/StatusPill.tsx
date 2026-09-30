import { useEffect, useState } from 'react'
import { api, type Health } from '../api/client'

/** Backend + model + GPU state, condensed into one pill in the top bar. */
export function StatusPill({ onHealth }: { onHealth: (h: Health | null) => void }) {
  const [health, setHealth] = useState<Health | null>(null)
  const [err, setErr] = useState<string | null>(null)

  // One check at a time (each times out after 6 s), and "unreachable" only after
  // two misses in a row: a single slow answer is not an outage.
  useEffect(() => {
    let alive = true
    let misses = 0
    let timer = 0
    const poll = async () => {
      try {
        const h = await api.health()
        if (!alive) return
        misses = 0
        setHealth(h)
        setErr(null)
        onHealth(h)
      } catch (e) {
        if (!alive) return
        if (++misses >= 2) {
          const m = (e as Error).message
          setErr(/timeout|signal/i.test(m) ? 'no answer in 6 s' : m)
          onHealth(null)
        }
      }
      if (alive) timer = window.setTimeout(poll, misses ? 1500 : 3000)
    }
    void poll()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [onHealth])

  const state = err ? 'down' : (health?.status ?? 'loading')
  const label = {
    ready: 'Ready',
    loading: 'Loading models',
    busy: 'GPU busy · frame extraction',
    error: 'Model load failed',
    down: 'Backend unreachable · retrying',
  }[state]

  const gpu = health?.gpu
  const q = health?.qwen
  const qwen =
    q && q.dedicated && q.state !== 'off'
      ? ` · Qwen ${q.state === 'ready' ? 'loaded' : q.state} on GPU ${q.gpu}`
      : ''
  const detail =
    state === 'ready'
      ? `SAM 3 on GPU ${gpu?.visible_devices ?? '?'} · ${gpu?.mem_free_mb ?? '?'} MB free` +
        qwen +
        (health?.ffmpeg ? '' : ' · no ffmpeg, falling back to VP8/webm')
      : state === 'loading'
        ? 'First start takes 1–2 min'
        : state === 'busy'
          ? 'SAM 3 is parked in RAM while Qwen embeds; tracking resumes when it is done'
        : err
          ? `${err}. Your work on this page is kept, and running jobs carry on on the server.`
          : (health?.load_error ?? 'see the server log')

  return (
    <div className={`status-pill ${state}`} title={detail}>
      <span className="status-dot" />
      <span className="status-label">{label}</span>
      <span className="status-detail">{detail}</span>
    </div>
  )
}
