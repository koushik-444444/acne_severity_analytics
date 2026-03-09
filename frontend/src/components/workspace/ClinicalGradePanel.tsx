import {
  getAcneTypeColor,
  getAcneTypeLabel,
  getSeverityTone,
} from '../../lib/clinical-utils'
import type { AnalyzeResponse, ConsensusLesion, RegionStats } from '../../types/api'

export function ClinicalGradePanel({
  active,
  displayGags,
  displaySeverity,
  severityTone,
  regionRows,
  consensusLesions,
}: {
  active: AnalyzeResponse
  displayGags: number
  displaySeverity: string
  severityTone: string
  regionRows: [string, RegionStats][]
  consensusLesions: ConsensusLesion[]
}) {
  return (
    <div className="space-y-6">
      <div className="holographic-panel rounded-[1.75rem] p-6">
        <h3 className="terminal-text mb-3 text-[10px] text-cyan-400/80">CLINICAL GRADE</h3>
        <div className="space-y-3 text-sm text-zinc-300">
          <div className="flex items-center justify-between rounded-xl border border-white/5 bg-white/3 px-4 py-3">
            <span className="text-zinc-500">GAGS total</span>
            <span className="text-xl font-semibold text-cyan-400">{displayGags}</span>
          </div>
          <div className={`flex items-center justify-between rounded-xl border px-4 py-3 ${severityTone}`}>
            <span className="text-zinc-500">Severity band</span>
            <span className="font-semibold text-white">{displaySeverity}</span>
          </div>
          <div className="text-xs leading-relaxed text-zinc-500">
            GAGS is a computed clinical grade derived from region-weighted lesion burden, not a separately detected object.
          </div>
        </div>
      </div>

      <GagsBreakdownPanel regionRows={regionRows} consensusLesions={consensusLesions} />
      <ConsensusInspectorPanel active={active} />
      <SeverityExplainerPanel active={active} />
      <SourceStreamDebuggerPanel active={active} />
    </div>
  )
}

function GagsBreakdownPanel({
  regionRows,
  consensusLesions,
}: {
  regionRows: [string, RegionStats][]
  consensusLesions: ConsensusLesion[]
}) {
  return (
    <div className="holographic-panel rounded-[1.75rem] p-6">
      <h3 className="terminal-text mb-3 text-[10px] text-cyan-400/80">HOW GAGS WAS CALCULATED</h3>
      <div className="space-y-3 text-sm text-zinc-400">
        <div className="rounded-xl border border-white/5 bg-white/3 px-4 py-3">
          Regional GAGS = region weight x highest severity grade in that zone. Typed detections now drive real grades (1-4) instead of a flat default.
        </div>
        <table className="w-full text-sm text-zinc-400">
          <thead>
            <tr className="rounded-xl border border-white/5 bg-black/20 text-xs uppercase tracking-[0.2em] text-zinc-500">
              <th scope="col" className="px-3 py-2 text-left font-normal">Region</th>
              <th scope="col" className="px-3 py-2 text-right font-normal">Lesions</th>
              <th scope="col" className="px-3 py-2 text-right font-normal">Max Grade</th>
              <th scope="col" className="px-3 py-2 text-right font-normal">GAGS</th>
            </tr>
          </thead>
          <tbody>
            {regionRows.map(([region, data]) => {
              const regionData = data as RegionStats
              const regionLesions = consensusLesions.filter((l) => l.region === region)
              const maxGrade = regionLesions.length > 0
                ? Math.max(...regionLesions.map((l) => l.severity_grade ?? 2))
                : 0
              return (
                <tr key={`gags-${region}`} className="border-b border-white/5">
                  <td className="px-3 py-3 text-zinc-300">{region.replaceAll('_', ' ')}</td>
                  <td className="px-3 py-3 text-right text-zinc-500">{regionData.count}</td>
                  <td className="px-3 py-3 text-right text-zinc-400">{maxGrade > 0 ? maxGrade : '-'}</td>
                  <td className="px-3 py-3 text-right font-semibold text-cyan-300">{regionData.gags_score ?? 0}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ConsensusInspectorPanel({ active }: { active: AnalyzeResponse }) {
  return (
    <div className="holographic-panel rounded-[1.75rem] p-6">
      <h3 className="terminal-text mb-3 text-[10px] text-cyan-400/80">CONSENSUS INSPECTOR</h3>
      <p className="mb-4 text-sm text-zinc-500">{active.results?.consensus_summary?.summary ?? 'No consensus data available'}</p>
      <div className="space-y-2 text-sm text-zinc-300">
        <div>Verified lesions: {active.results?.consensus_summary?.verified_lesions ?? 0}</div>
        <div>Average confidence: {active.results?.consensus_summary?.average_confidence ?? 0}</div>
        <div>Top regions: {(active.results?.consensus_summary?.top_regions ?? []).map((item) => item.region).join(', ') || 'n/a'}</div>
      </div>
      {active.results?.consensus_summary?.type_counts && Object.keys(active.results.consensus_summary.type_counts).length > 0 && (
        <div className="mt-4">
          <div className="mb-2 text-[10px] uppercase tracking-[0.15em] text-zinc-500">Type distribution</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(active.results.consensus_summary.type_counts)
              .sort(([, a], [, b]) => b - a)
              .map(([cls, n]) => (
                <span key={cls} className={`rounded-full border px-2.5 py-1 text-xs font-medium ${getAcneTypeColor(cls)}`}>
                  {getAcneTypeLabel(cls)} {n}
                </span>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}

function SeverityExplainerPanel({ active }: { active: AnalyzeResponse }) {
  return (
    <div className="holographic-panel rounded-[1.75rem] p-6">
      <h3 className="terminal-text mb-3 text-[10px] text-cyan-400/80">WHY THIS SEVERITY?</h3>
      <div className="space-y-3">
        {Object.entries(active.results?.clinical_analysis?.regions ?? {}).map(([region, data]) => {
          const regionData = data as RegionStats
          return (
            <div key={region} className="flex items-center justify-between rounded-xl border border-white/5 bg-white/3 px-4 py-3 text-sm">
              <span className="text-zinc-400">{region.replaceAll('_', ' ')}</span>
              <span className="font-medium text-white">{regionData.count} lesions · GAGS {regionData.gags_score ?? 0}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function SourceStreamDebuggerPanel({ active }: { active: AnalyzeResponse }) {
  return (
    <div className="holographic-panel rounded-[1.75rem] p-6">
      <h3 className="terminal-text mb-3 text-[10px] text-cyan-400/80">SOURCE STREAM DEBUGGER</h3>
      <div className="space-y-3 text-sm text-zinc-400">
        <div>Strongest stream: {String(active.results?.source_stream_provenance?.strongest_stream ?? 'n/a')}</div>
        <div>Total source proposals: {String(active.results?.source_stream_provenance?.stream_total ?? 0)}</div>
        <div className="grid grid-cols-1 gap-2">
          {Object.entries(active.results?.source_stream_provenance?.streams ?? {}).map(([name, count]) => {
            const classDist = active.results?.source_stream_provenance?.stream_classes?.[name]
            return (
              <div key={name} className="rounded-xl border border-white/5 bg-white/3 px-3 py-2">
                <div className="flex items-center justify-between">
                  <span>{name}</span>
                  <span className="font-medium text-white">{count}</span>
                </div>
                {classDist && Object.keys(classDist).length > 0 && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {Object.entries(classDist).sort(([, a], [, b]) => (b as number) - (a as number)).map(([cls, n]) => (
                      <span key={cls} className={`rounded-full border px-1.5 py-0.5 text-[8px] ${getAcneTypeColor(cls)}`}>
                        {getAcneTypeLabel(cls)} {n}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
