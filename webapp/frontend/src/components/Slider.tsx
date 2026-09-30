type Props = {
  label: string
  /** one line telling the user what moving this actually does */
  help?: string
  value: number
  min: number
  max: number
  step: number
  /** how the number reads to a person, e.g. "120 frames" or "Fairly sure" */
  format?: (v: number) => string
  minLabel?: string
  maxLabel?: string
  onChange: (v: number) => void
  disabled?: boolean
}

export function Slider({
  label,
  help,
  value,
  min,
  max,
  step,
  format,
  minLabel,
  maxLabel,
  onChange,
  disabled,
}: Props) {
  return (
    <label className="slider">
      <span className="slider-head">
        <span className="slider-label">{label}</span>
        <span className="slider-value">{format ? format(value) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {(minLabel || maxLabel) && (
        <span className="slider-ends">
          <span>{minLabel}</span>
          <span>{maxLabel}</span>
        </span>
      )}
      {help && <span className="slider-help">{help}</span>}
    </label>
  )
}
