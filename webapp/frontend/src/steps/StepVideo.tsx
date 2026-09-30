import { api } from '../api/client'
import { Dropzone } from '../components/Dropzone'
import { LabelsCard } from '../components/LabelsCard'
import type { Workspace } from '../lib/workspace'

function duration(frames?: number, fps?: number): string {
  if (!frames || !fps) return ''
  const s = frames / fps
  if (s < 60) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
}

export function StepVideo({ ws }: { ws: Workspace }) {
  const { info, upload } = ws

  return (
    <div className="page narrow">
      <header className="page-head">
        <h2>Upload a video</h2>
        <p className="lead">
          The clip to run detection and tracking on.
        </p>
      </header>

      <section className="card">
        {!info ? (
          <Dropzone
            accept="video/*"
            title="Drop a video here"
            hint="or click to browse — MP4, MOV, AVI, MKV or WEBM, up to 500 MB"
            busy={upload.busy}
            busyLabel="Uploading…"
            onFile={(f) => void ws.loadVideo(f)}
          />
        ) : (
          <div className="videocard">
            <img
              className="videothumb"
              src={api.frameUrl(info.upload_id, 0)}
              alt="Frame 0"
            />
            <div className="videofacts">
              <h3>Uploaded</h3>
              <p className="muted">
                {info.width} × {info.height} · {info.frames} frames · {info.fps} fps
                {duration(info.frames, info.fps) ? ` · ${duration(info.frames, info.fps)}` : ''}
              </p>
              <button
                type="button"
                className="btn-text"
                onClick={ws.startOver}
                disabled={upload.busy}
              >
                Replace video
              </button>
            </div>
          </div>
        )}

        {upload.error && <p className="note danger">{upload.error}</p>}
      </section>

      <LabelsCard classes={ws.classes} source={ws.classSource} onLoaded={ws.setClassList} />

      <div className="page-foot">
        <button
          type="button"
          className="btn primary big"
          disabled={!info}
          onClick={() => ws.setStep('detect')}
        >
          Next: detect objects →
        </button>
        {!info && <p className="muted small">Upload a video to continue.</p>}
      </div>
    </div>
  )
}
