import { api } from '../api/client'
import { Progress } from '../components/Progress'
import { ResultVideo } from '../components/ResultView'
import type { Workspace } from '../lib/workspace'

export function StepResult({ ws }: { ws: Workspace }) {
  const { track } = ws
  const result = track.result
  const done = track.status === 'done' && !!result?.tracked_video_url

  return (
    <div className="page narrow">
      <header className="page-head">
        <h2>{done ? 'Tracking complete' : 'Running object tracking'}</h2>
        <p className="lead">
          {done
            ? 'Every instance is composited onto the clip with its mask, box and class label.'
            : 'Propagating each instance through the clip. This runs on the GPU worker; you can leave the page open.'}
        </p>
      </header>

      {!done && (
        <section className="card">
          <Progress job={track} idleLabel="Queued" onCancel={track.cancel} />
          {track.status === 'error' && (
            <button type="button" className="btn ghost" onClick={() => ws.setStep('detect')}>
              ← Back to detection
            </button>
          )}
        </section>
      )}

      {done && (
        <>
          <ResultVideo result={result} />

          <section className="card">
            <div className="card-head">
              <div>
                <h3>Output</h3>
                <p className="card-sub">
                  {typeof result?.objects === 'number' ? `${result.objects} instances tracked` : ''}
                  {typeof result?.frames === 'number' ? ` over ${result.frames} frames` : ''}
                </p>
              </div>
            </div>
            {result?.message && <p className="muted">{result.message}</p>}
            {result?.codec_warning && <p className="note warn">{result.codec_warning}</p>}

            <div className="row gap wrap">
              <a className="btn primary" href={api.fileUrl(String(result?.tracked_video_url))} download>
                Download video
              </a>
              {result?.json_url && (
                <a className="btn ghost" href={api.fileUrl(String(result.json_url))} download>
                  Download tracks (JSON)
                </a>
              )}
            </div>
            <p className="muted small">
              The JSON carries the per-frame bounding box for every instance, plus its class id
              and resolved class name.
            </p>
          </section>

          <div className="page-foot row gap wrap">
            <button type="button" className="btn ghost" onClick={() => ws.setStep('detect')}>
              ← Detect more objects
            </button>
            <button type="button" className="btn-text" onClick={ws.startOver}>
              Start over with a new video
            </button>
          </div>
        </>
      )}
    </div>
  )
}
