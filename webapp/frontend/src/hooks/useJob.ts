import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError, isUnreachable, type JobSnapshot } from '../api/client'

type State = {
  status: 'idle' | JobSnapshot['status']
  progress: number
  stage: string
  queuedAhead: number
  result: JobSnapshot['result']
  error: string | null
}

const IDLE: State = {
  status: 'idle',
  progress: 0,
  stage: '',
  queuedAhead: 0,
  result: null,
  error: null,
}

// how long a job is still waited for while the backend cannot be reached
const GIVE_UP_MS = 15 * 60 * 1000

/** Submit a job and poll it to completion. Re-submitting supersedes the previous
 *  run (the old job is cancelled).
 *
 *  Polls one request at a time, and rides out the backend (or the tunnel to it)
 *  dropping out: the job keeps running on the server, so a failed poll just
 *  retries. It only gives up when the server says the job is gone (a restart
 *  forgets every job) or after 15 minutes without an answer. */
export function useJob(pollMs: number) {
  const [state, setState] = useState<State>(IDLE)
  const jobIdRef = useRef<string | null>(null)
  const timerRef = useRef<number | null>(null)

  const stopTimer = () => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }

  const cancel = useCallback(() => {
    const id = jobIdRef.current
    stopTimer()
    if (id) api.cancelJob(id).catch(() => {})
    jobIdRef.current = null
    // drop straight back to the idle state so the UI leaves "running"
    // (polling has stopped, so it would otherwise never see the job end)
    setState(IDLE)
  }, [])

  const submit = useCallback(
    async (kind: string, body: unknown) => {
      cancel()
      setState({ ...IDLE, status: 'queued', stage: 'submitting' })
      try {
        const { job_id } = await api.startJob(kind, body)
        jobIdRef.current = job_id
        let lostSince = 0
        const tick = async () => {
          if (jobIdRef.current !== job_id) return
          try {
            const s = await api.job(job_id)
            if (jobIdRef.current !== job_id) return
            lostSince = 0
            setState({
              status: s.status,
              progress: s.progress,
              stage: s.stage,
              queuedAhead: s.queued_ahead,
              result: s.result,
              error: s.error,
            })
            if (s.status === 'done' || s.status === 'error') {
              jobIdRef.current = null
              return
            }
          } catch (e) {
            if (jobIdRef.current !== job_id) return
            const gone = e instanceof ApiError && e.status === 404
            lostSince ||= Date.now()
            if (gone || !isUnreachable(e) || Date.now() - lostSince > GIVE_UP_MS) {
              jobIdRef.current = null
              setState((p) => ({
                ...p,
                status: 'error',
                error: gone
                  ? 'The backend restarted and no longer has this job. Run it again.'
                  : (e as Error).message,
              }))
              return
            }
            setState((p) => ({ ...p, stage: 'backend not answering, still waiting for this job…' }))
          }
          timerRef.current = window.setTimeout(tick, lostSince ? Math.max(pollMs, 2000) : pollMs)
        }
        void tick()
      } catch (e) {
        setState({ ...IDLE, status: 'error', error: (e as Error).message })
      }
    },
    [cancel, pollMs],
  )

  const reset = useCallback(() => {
    cancel()
    setState(IDLE)
  }, [cancel])

  useEffect(() => () => cancel(), [cancel])

  return { ...state, submit, cancel, reset, busy: state.status === 'queued' || state.status === 'running' }
}
