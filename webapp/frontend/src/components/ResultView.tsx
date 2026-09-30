import { api, type JobResult } from '../api/client'

export function ResultVideo({ result }: { result: JobResult | null }) {
  if (!result?.tracked_video_url) return null
  return (
    <div className="result-video">
      <video
        key={result.tracked_video_url}
        src={api.fileUrl(result.tracked_video_url)}
        controls
        autoPlay
        loop
        muted
        playsInline
      />
    </div>
  )
}
