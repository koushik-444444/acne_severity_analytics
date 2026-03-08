import { useState } from 'react'

import { AdvancedImageViewer } from './AdvancedImageViewer'
import type { ViewerState } from './types'

export function SplitCompareViewer({
  beforeSrc,
  afterSrc,
  state,
  onChange,
}: {
  beforeSrc: string
  afterSrc: string
  state: ViewerState
  onChange: (next: ViewerState) => void
}) {
  const [divider, setDivider] = useState(50)

  return (
    <div className="relative h-[480px] overflow-hidden rounded-[1.25rem] border border-white/5 bg-black">
      <AdvancedImageViewer src={beforeSrc} alt="Previous compare" state={state} onChange={onChange} heightClass="h-[480px]" />
      <div className="pointer-events-none absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - divider}% 0 0)` }}>
        <AdvancedImageViewer src={afterSrc} alt="Current compare" state={state} onChange={onChange} heightClass="h-[480px]" />
      </div>
      <div className="absolute inset-y-0 z-20 w-px bg-cyan-400/70" style={{ left: `${divider}%` }} />
      <div className="pointer-events-none absolute left-4 top-4 z-30 rounded-full border border-white/10 bg-black/60 px-3 py-1 text-[10px] uppercase tracking-[0.25em] text-zinc-300 backdrop-blur">
        prior / current compare
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={divider}
        onChange={(event) => setDivider(Number(event.target.value))}
        aria-label="Split compare divider position"
        className="absolute bottom-4 left-1/2 z-30 w-64 -translate-x-1/2 accent-cyan-400"
      />
    </div>
  )
}
