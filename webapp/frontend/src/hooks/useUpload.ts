import { useCallback, useState } from 'react'
import { api, type UploadInfo } from '../api/client'

export function useUpload(expect: 'image' | 'video') {
  const [info, setInfo] = useState<UploadInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const upload = useCallback(
    async (file: File) => {
      setBusy(true)
      setError(null)
      setInfo(null)
      try {
        const res = await api.upload(file)
        if (res.kind !== expect) {
          throw new Error(`expected ${expect}, got a ${res.kind}`)
        }
        setInfo(res)
        return res
      } catch (e) {
        setError((e as Error).message)
        return null
      } finally {
        setBusy(false)
      }
    },
    [expect],
  )

  const clear = useCallback(() => {
    setInfo(null)
    setError(null)
  }, [])

  return { info, busy, error, upload, clear }
}
