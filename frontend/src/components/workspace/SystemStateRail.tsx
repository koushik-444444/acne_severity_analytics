import type { SessionStatus } from '../../types/api'
import { formatDate } from '../../lib/clinical-utils'

export function SystemStateRail({
  status,
  rightRailWidth,
  onRightRailWidthChange,
}: {
  status: SessionStatus | null
  rightRailWidth: number
  onRightRailWidthChange: (width: number) => void
}) {
  return (
    <aside className="space-y-6">
      <div className="holographic-panel rounded-[2rem] p-6">
        <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">SYSTEM STATE</div>
        <div className="space-y-3 text-sm text-zinc-400">
          <div>Stage: {status?.stage ?? 'idle'}</div>
          <div>Detail: {status?.detail ?? 'Awaiting workflow'}</div>
          <div>Progress: {status?.progress ?? 0}%</div>
          <div>Updated: {status?.updated_at ? formatDate(status.updated_at) : 'n/a'}</div>
        </div>
      </div>

      <div className="holographic-panel rounded-[2rem] p-6">
        <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">RECOMMENDED NEXT ACTIONS</div>
        <ol className="space-y-3 text-sm text-zinc-400">
          <li>1. Compare this case against the previous session.</li>
          <li>2. Review region-level GAGS contribution before exporting.</li>
          <li>3. Use privacy purge for sensitive test sessions.</li>
        </ol>
      </div>
      <div className="holographic-panel rounded-[2rem] p-6">
        <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">WORKSPACE SHORTCUTS</div>
        <div className="space-y-2 text-xs text-zinc-500">
          <div>`+` zoom in</div>
          <div>`-` zoom out</div>
          <div>`R` reset viewer</div>
          <div>`O` toggle overlay</div>
          <div>`C` toggle compare mode</div>
          <div>`F` fullscreen</div>
        </div>
        <input
          type="range"
          min={320}
          max={520}
          value={rightRailWidth}
          onChange={(e) => onRightRailWidthChange(Number(e.target.value))}
          aria-label="Right panel width"
          className="mt-4 w-full accent-cyan-400"
        />
      </div>
    </aside>
  )
}
