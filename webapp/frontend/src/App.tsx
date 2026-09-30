import { useCallback, useEffect, useState } from 'react'
import './App.css'
import { ExtractPage } from './extract/ExtractPage'
import { FawadSegPage } from './extract/FawadSegPage'
import { SegxPage } from './extract/SegxPage'
import { HistoryDrawer } from './components/HistoryDrawer'
import { StatusPill } from './components/StatusPill'
import { Stepper } from './components/Stepper'
import { StepDetect } from './steps/StepDetect'
import { StepResult } from './steps/StepResult'
import { StepVideo } from './steps/StepVideo'
import { useWorkspace, type Step } from './lib/workspace'
import type { Health, HealthConfig } from './api/client'

type Mode = 'track' | 'extract' | 'segment' | 'fawad'

function savedMode(): Mode {
  try {
    const m = localStorage.getItem('sam3.mode')
    return m === 'extract' || m === 'segment' || m === 'fawad' ? m : 'track'
  } catch {
    return 'track'
  }
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  // The config from the last health check that got through. Kept when the backend
  // drops out: unmounting the pages would throw away every upload, mark and
  // result, and cancel their running jobs. Replaced only when it really changes.
  const [cfg, setCfg] = useState<HealthConfig | null>(null)
  const onHealth = useCallback((h: Health | null) => {
    setHealth(h)
    if (h) setCfg((c) => (c && JSON.stringify(c) === JSON.stringify(h.config) ? c : h.config))
  }, [])
  const [mode, setModeRaw] = useState<Mode>(savedMode)
  const setMode = (m: Mode) => {
    setModeRaw(m)
    try {
      localStorage.setItem('sam3.mode', m)
    } catch {
      /* private window: the choice just isn't remembered */
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            ◎
          </span>
          <span className="brand-text">
            <b>SAM 3</b>
            <span>
              {mode === 'track'
                ? 'Object detection and tracking in video'
                : mode === 'extract'
                  ? 'Find the frames that match reference images'
                  : mode === 'segment'
                    ? 'Find where a step happens, from clips of it in other videos'
                    : 'Find a step, from a labelled reference video (the research pipeline, as is)'}
            </span>
          </span>
        </div>
        <nav className="switch modeswitch" aria-label="Mode">
          <button type="button" className={mode === 'track' ? 'on' : ''} onClick={() => setMode('track')}>
            Object tracking
          </button>
          <button type="button" className={mode === 'extract' ? 'on' : ''} onClick={() => setMode('extract')}>
            Frame extraction
          </button>
          <button type="button" className={mode === 'segment' ? 'on' : ''} onClick={() => setMode('segment')}>
            Segment extraction
          </button>
          <button type="button" className={mode === 'fawad' ? 'on' : ''} onClick={() => setMode('fawad')}>
            Frame extraction fawad segment
          </button>
        </nav>
        <StatusPill onHealth={onHealth} />
      </header>

      {cfg ? (
        // all stay mounted, so switching never loses prompts or results
        <>
          <div className="modepane" hidden={mode !== 'track'}>
            <Workspace cfg={cfg} offline={health?.status !== 'ready'} />
          </div>
          <div className="modepane" hidden={mode !== 'extract'}>
            <main>
              <ExtractPage cfg={cfg} qwen={health?.qwen} />
            </main>
          </div>
          <div className="modepane" hidden={mode !== 'segment'}>
            <SegxPage cfg={cfg} qwen={health?.qwen} />
          </div>
          <div className="modepane" hidden={mode !== 'fawad'}>
            <FawadSegPage cfg={cfg} qwen={health?.qwen} />
          </div>
        </>
      ) : (
        <div className="booting">
          <span className="spinner big" />
          <p>Connecting…</p>
        </div>
      )}
    </div>
  )
}

function Workspace({ cfg, offline }: { cfg: HealthConfig; offline: boolean }) {
  const ws = useWorkspace(cfg)
  const { step, setStep, info, track } = ws
  const [historyOpen, setHistoryOpen] = useState(false)

  // the furthest step that is open: you need an upload to detect on, and a
  // tracking job to have results
  const reached: Step = track.status !== 'idle' ? 'result' : info ? 'detect' : 'video'

  // a finished tracking job is the whole point of step 2 — go straight to it
  useEffect(() => {
    if (track.status === 'done' && track.result?.tracked_video_url) setStep('result')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [track.status])

  return (
    <>
      <div className="steprow">
        <Stepper current={step} reached={reached} onGo={setStep} />
        <button type="button" className="btn ghost history-btn" onClick={() => setHistoryOpen(true)}>
          History
        </button>
      </div>
      <main>
        {step === 'video' && <StepVideo ws={ws} />}
        {step === 'detect' && <StepDetect ws={ws} offline={offline} />}
        {step === 'result' && <StepResult ws={ws} />}
      </main>
      <HistoryDrawer
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
        currentUploadId={info?.upload_id}
        onRestore={ws.restoreRun}
      />
    </>
  )
}
