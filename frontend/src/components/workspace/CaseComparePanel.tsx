import { Sparkles } from 'lucide-react'

import {
  describeDelta,
  formatSignedDelta,
  getClinicalDeltaLabel,
  getDeltaTone,
} from '../../lib/clinical-utils'
import type { ComparePayload, SessionSummary } from '../../types/api'

type CompareRegionDelta = NonNullable<ComparePayload>['regions'][string]

export function CaseComparePanel({
  compare,
  baselineSession,
  isPinnedBaselineCompare,
  isViewingPinnedBaseline,
  regionDeltaCards,
  onClearBaseline,
}: {
  compare: ComparePayload
  baselineSession: SessionSummary | null
  isPinnedBaselineCompare: boolean
  isViewingPinnedBaseline: boolean
  regionDeltaCards: [string, CompareRegionDelta][]
  onClearBaseline: () => void
}) {
  const compareNarrative = compare
    ? `${describeDelta(compare.lesion_delta, 'Lesion burden')} and ${describeDelta(compare.gags_delta, 'GAGS', true)} versus ${isPinnedBaselineCompare ? 'the pinned baseline' : 'the previous archived session'}.`
    : null

  return (
    <div className="holographic-panel rounded-[1.75rem] p-6">
      <div className="mb-4 flex items-center gap-2">
        <Sparkles aria-hidden="true" className="h-4 w-4 text-cyan-400" />
        <h3 className="terminal-text text-[10px] text-cyan-400/80">CASE COMPARE</h3>
      </div>
      {compare ? (
        <>
          <div className="mb-4 rounded-[1.25rem] border border-cyan-400/15 bg-cyan-400/6 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="terminal-text text-[9px] text-cyan-400/80">
                  {isPinnedBaselineCompare ? 'PINNED BASELINE DELTA' : 'LONGITUDINAL DELTA'}
                </div>
                <p className="mt-2 text-sm text-zinc-300">
                  {compareNarrative}
                </p>
                <div className="mt-2 text-xs text-zinc-500">
                  Comparing current session {compare.current_session_id.slice(0, 8)} against {compare.previous_session_id.slice(0, 8)}.
                </div>
              </div>
              {baselineSession ? (
                <button type="button" onClick={onClearBaseline} className="rounded-full border border-cyan-400/30 px-3 py-1 text-[10px] text-cyan-100 hover:bg-cyan-400/10">
                  Clear baseline
                </button>
              ) : null}
            </div>

            <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-4">
              <div className={`rounded-xl border px-3 py-3 ${getDeltaTone(compare.lesion_delta, true)}`}>
                <div className="terminal-text text-[8px] text-current/70">LESION DELTA</div>
                <div className="mt-2 text-lg font-semibold">{formatSignedDelta(compare.lesion_delta)}</div>
                <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-current/75">{getClinicalDeltaLabel(compare.lesion_delta, true)}</div>
              </div>
              <div className={`rounded-xl border px-3 py-3 ${getDeltaTone(compare.gags_delta, true)}`}>
                <div className="terminal-text text-[8px] text-current/70">GAGS DELTA</div>
                <div className="mt-2 text-lg font-semibold">{formatSignedDelta(compare.gags_delta)}</div>
                <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-current/75">{getClinicalDeltaLabel(compare.gags_delta, true)}</div>
              </div>
              <div className={`rounded-xl border px-3 py-3 ${getDeltaTone(compare.symmetry_delta_change, true)}`}>
                <div className="terminal-text text-[8px] text-current/70">SYMMETRY SHIFT</div>
                <div className="mt-2 text-lg font-semibold">{formatSignedDelta(compare.symmetry_delta_change, '%')}</div>
                <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-current/75">{getClinicalDeltaLabel(compare.symmetry_delta_change, true)}</div>
              </div>
              <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-3 text-zinc-200">
                <div className="terminal-text text-[8px] text-zinc-500">SEVERITY BAND</div>
                <div className="mt-2 text-sm font-semibold text-white">
                  {compare.severity_change.from} <span aria-label="to">{'\u2192'}</span> {compare.severity_change.to}
                </div>
              </div>
            </div>

            {regionDeltaCards.length > 0 ? (
              <div className="mt-4 flex flex-wrap gap-2 text-xs text-zinc-300">
                {regionDeltaCards.slice(0, 3).map(([region, values]) => (
                  <span key={`baseline-chip-${region}`} className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                    {region.replaceAll('_', ' ')} {formatSignedDelta(values.count_delta)} lesions
                  </span>
                ))}
              </div>
            ) : null}
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {regionDeltaCards.map(([region, values]) => (
              <div key={region} className="rounded-xl border border-white/5 bg-white/3 p-4">
                <div className="terminal-text mb-2 text-[9px] text-zinc-500">{region}</div>
                <div className="text-sm text-white">count delta {values.count_delta >= 0 ? '+' : ''}{values.count_delta}</div>
                <div className="text-xs text-zinc-500">lpi delta {values.lpi_delta >= 0 ? '+' : ''}{values.lpi_delta}</div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="space-y-3 text-sm text-zinc-500">
          {isViewingPinnedBaseline ? (
            <div className="rounded-xl border border-cyan-400/15 bg-cyan-400/8 px-4 py-3 text-cyan-100">
              This session is the pinned baseline. Open another session to generate a baseline delta callout.
            </div>
          ) : null}
          <p>No previous session is available for comparison yet.</p>
        </div>
      )}
    </div>
  )
}
