import { useRef, useState } from 'react'

type Props = {
  accept: string
  title: string
  hint: string
  busy?: boolean
  busyLabel?: string
  /** accept several files at once; they arrive together in onFiles */
  multiple?: boolean
  onFile?: (file: File) => void
  onFiles?: (files: File[]) => void
  compact?: boolean
}

export function Dropzone({
  accept,
  title,
  hint,
  busy,
  busyLabel = 'Uploading…',
  multiple,
  onFile,
  onFiles,
  compact,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const take = (list: FileList | null | undefined) => {
    const files = Array.from(list ?? [])
    if (!files.length) return
    if (onFiles) onFiles(multiple ? files : files.slice(0, 1))
    else onFile?.(files[0])
  }

  return (
    <button
      type="button"
      className={`dropzone${over ? ' over' : ''}${compact ? ' compact' : ''}`}
      disabled={busy}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        take(e.dataTransfer.files)
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        hidden
        onChange={(e) => {
          take(e.target.files)
          e.target.value = ''
        }}
      />
      <span className="dz-icon" aria-hidden>
        {busy ? <span className="spinner" /> : '＋'}
      </span>
      <span className="dz-title">{busy ? busyLabel : title}</span>
      {!busy && <span className="dz-hint">{hint}</span>}
    </button>
  )
}
