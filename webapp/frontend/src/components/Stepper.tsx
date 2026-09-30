import { STEPS } from '../lib/workspace'

export type StepDef<S extends string> = { id: S; title: string; blurb: string }

type Props<S extends string> = {
  current: S
  /** the furthest step that has been unlocked — later ones are not clickable yet */
  reached: S
  onGo: (s: S) => void
  /** the steps, in order; the tracking workspace's by default */
  steps?: StepDef<S>[]
}

export function Stepper<S extends string>({ current, reached, onGo, steps }: Props<S>) {
  const list = steps ?? (STEPS as unknown as StepDef<S>[])
  const index = (s: S) => list.findIndex((x) => x.id === s)
  const at = index(current)
  const max = index(reached)

  return (
    <nav className="stepper" aria-label="Progress">
      {list.map((s, i) => {
        const state = i < at ? 'done' : i === at ? 'now' : 'todo'
        const open = i <= max
        return (
          <button
            key={s.id}
            type="button"
            className={`step ${state}`}
            disabled={!open || i === at}
            aria-current={i === at ? 'step' : undefined}
            onClick={() => onGo(s.id)}
          >
            <span className="step-num">{i < at ? '✓' : i + 1}</span>
            <span className="step-text">
              <span className="step-title">{s.title}</span>
              <span className="step-blurb">{s.blurb}</span>
            </span>
          </button>
        )
      })}
    </nav>
  )
}
