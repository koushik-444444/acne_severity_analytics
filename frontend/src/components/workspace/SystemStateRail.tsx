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
    <aside aria-label="System state and shortcuts" className="space-y-6">
      <div className="holographic-panel rounded-[2rem] p-6">
        <h3 className="terminal-text mb-3 text-[10px] text-cyan-400/80">SYSTEM STATE</h3>
        <div className="space-y-3 text-sm text-zinc-400">
          <div>Stage: {status?.stage ?? 'idle'}</div>
          <div>Detail: {status?.detail ?? 'Awaiting workflow'}</div>
          <div>Progress: {status?.progress ?? 0}%</div>
          <div>Updated: {status?.updated_at ? formatDate(status.updated_at) : 'n/a'}</div>
        </div>
      </div>

      <div className="holographic-panel rounded-[2rem] p-6">
        <h3 className="terminal-text mb-3 text-[10px] text-cyan-400/80">RECOMMENDED NEXT ACTIONS</h3>
        <ol className="space-y-3 text-sm text-zinc-400">
          <li>Compare this case against the previous session.</li>
          <li>Review region-level GAGS contribution before exporting.</li>
          <li>Use privacy purge for sensitive test sessions.</li>
        </ol>
      </div>
      <div className="holographic-panel rounded-[2rem] p-6">
        <h3 className="terminal-text mb-3 text-[10px] text-cyan-400/80">WORKSPACE SHORTCUTS</h3>
        <div className="space-y-2 text-xs text-zinc-500">
          <div><kbd className="rounded border border-white/10 bg-white/5 px-1 py-0.5 font-mono text-zinc-300">+</kbd> zoom in</div>
          <div><kbd className="rounded border border-white/10 bg-white/5 px-1 py-0.5 font-mono text-zinc-300">-</kbd> zoom out</div>
          <div><kbd className="rounded border border-white/10 bg-white/5 px-1 py-0.5 font-mono text-zinc-300">R</kbd> reset viewer</div>
          <div><kbd className="rounded border border-white/10 bg-white/5 px-1 py-0.5 font-mono text-zinc-300">O</kbd> toggle overlay</div>
          <div><kbd className="rounded border border-white/10 bg-white/5 px-1 py-0.5 font-mono text-zinc-300">C</kbd> toggle compare mode</div>
          <div><kbd className="rounded border border-white/10 bg-white/5 px-1 py-0.5 font-mono text-zinc-300">F</kbd> fullscreen</div>
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
