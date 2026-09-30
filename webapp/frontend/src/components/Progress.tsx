import type { useJob } from '../hooks/useJob'

type Job = ReturnType<typeof useJob>

/** The backend reports its own stage names; they are the right words already. */
const stageText = (stage: string) =>
  stage ? stage[0].toUpperCase() + stage.slice(1) : 'Working'

type Props = {
  job: Job
  /** what to show while the job is queued or still warming up */
  idleLabel?: string
  onCancel?: () => void
  /** show a tick and this message once the job has finished */
  doneLabel?: string
}

export function Progress({ job, idleLabel, onCancel, doneLabel }: Props) {
  if (job.status === 'idle') return null

  if (job.status === 'error') {
    return (
      <div className="progress error">
        <span className="progress-icon">!</span>
        <span>{job.error ?? 'Job failed'}</span>
      </div>
    )
  }

  if (job.status === 'done') {
    if (!doneLabel) return null
    return (
      <div className="progress done">
        <span className="progress-icon">✓</span>
        <span>{doneLabel}</span>
      </div>
    )
  }

  const warming = job.status === 'running' && job.progress < 0.02
  const unknown = warming || job.status === 'queued'
  const pct = Math.round(job.progress * 100)
  const label =
    job.status === 'queued'
      ? job.queuedAhead > 0
        ? `Queued — ${job.queuedAhead} job${job.queuedAhead > 1 ? 's' : ''} ahead`
        : (idleLabel ?? 'Queued')
      : warming
        ? (idleLabel ?? 'Warming up')
        : stageText(job.stage)

  return (
    <div className="progress">
      <div className="progress-bar">
        <div
          className={`progress-fill${unknown ? ' unknown' : ''}`}
          style={unknown ? undefined : { width: `${pct}%` }}
        />
      </div>
      <div className="progress-row">
        <span>{label}</span>
        <span className="progress-right">
          {!unknown && <span className="progress-pct">{pct}%</span>}
          {onCancel && (
            <button type="button" className="btn-text" onClick={onCancel}>
              Cancel
            </button>
          )}
        </span>
      </div>
    </div>
  )
}
