import { Shield } from 'lucide-react'

import type { ExportPreset } from '../../types/api'

export function PrivacyExportPanel({
  privacyMode,
  retentionHours,
  sessionId,
  exportPreset,
  onExportPresetChange,
  onExport,
  onNewCase,
  isExporting,
}: {
  privacyMode: boolean
  retentionHours: number
  sessionId: string
  exportPreset: ExportPreset
  onExportPresetChange: (preset: ExportPreset) => void
  onExport: () => void
  onNewCase: () => void
  isExporting: boolean
}) {
  return (
    <div className="holographic-panel rounded-[1.75rem] p-6">
      <div className="mb-4 flex items-center gap-2">
        <Shield aria-hidden="true" className="h-4 w-4 text-cyan-400" />
        <h3 className="terminal-text text-[10px] text-cyan-400/80">PRIVACY + EXPORT</h3>
      </div>
      <div className="space-y-3 text-sm text-zinc-400">
        <div>Privacy mode: {privacyMode ? 'enabled' : 'disabled'}</div>
        <div>Retention: {retentionHours} hours</div>
        <div>Current session: {sessionId}</div>
      </div>
      <div className="mt-4 space-y-2">
        <h4 className="terminal-text text-[9px] text-zinc-500">EXPORT PRESET</h4>
        <select
          value={exportPreset}
          onChange={(e) => onExportPresetChange(e.target.value as ExportPreset)}
          aria-label="Export preset"
          className="w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
        >
          <option value="clinical">Clinical</option>
          <option value="compact">Compact</option>
          <option value="presentation">Presentation</option>
        </select>
      </div>
      <div className="mt-6 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={onExport}
          disabled={isExporting}
          className="terminal-text rounded-full border border-white/10 px-4 py-2 text-[10px] text-white transition-colors hover:border-cyan-400/20 hover:text-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isExporting ? 'EXPORTING...' : `EXPORT ${exportPreset.toUpperCase()}`}
        </button>
        <button
          type="button"
          onClick={onNewCase}
          className="terminal-text rounded-full bg-cyan-400 px-4 py-2 text-[10px] text-black"
        >
          NEW CASE
        </button>
      </div>
    </div>
  )
}
