import { useCallback, useRef, useState } from 'react'
import { useJob } from './useJob'

type Format = 'mp4' | 'webm'

/** H.264 if this browser can decode it (quickest for the server to make), else VP9.
 *  VS Code's built-in browser and Chromium builds without the proprietary codecs
 *  answer "" for H.264 — their <video> would stay black. */
function preferredFormat(): Format {
  try {
    return document.createElement('video').canPlayType('video/mp4; codecs="avc1.64001E"') ? 'mp4' : 'webm'
  } catch {
    return 'mp4'
  }
}

/** The smooth-playback copy of a video (backend `extract-playback`), with a way
 *  down when the browser turns out not to play it: H.264 -> VP9 -> none, and with
 *  none the viewer plays frame by frame from JPEGs. */
export function usePlayback(pollMs: number) {
  const job = useJob(pollMs)
  const { submit } = job
  const current = useRef<{ id: string; format: Format } | null>(null)
  const [dead, setDead] = useState(false)

  const request = useCallback(
    (uploadId: string, format: Format = preferredFormat()) => {
      current.current = { id: uploadId, format }
      setDead(false)
      void submit('extract-playback', { upload_id: uploadId, format })
    },
    [submit],
  )

  /** the browser could not play what it was given: try the next format down */
  const fail = useCallback(() => {
    const c = current.current
    if (c && c.format === 'mp4') request(c.id, 'webm')
    else setDead(true)
  }, [request])

  const url =
    !dead && job.status === 'done' ? ((job.result as { url?: string } | null)?.url ?? null) : null
  return {
    request,
    fail,
    url,
    error: dead
      ? 'This browser cannot play the video, so segments play frame by frame.'
      : job.status === 'error'
        ? job.error
        : null,
  }
}
