import { motion } from 'framer-motion'

import {
  getAcneTypeColor,
  getAcneTypeLabel,
  getConfidenceTextTone,
  getSeverityGradeLabel,
} from '../../lib/clinical-utils'
import type { ConsensusLesion } from '../../types/api'

type LesionWithKey = ConsensusLesion & { key: string }

export function LesionTable({
  lesions,
  filteredLesions,
  lesionRegions,
  lesionTypes,
  lesionRegionFilter,
  lesionConfidenceFilter,
  lesionTypeFilter,
  onRegionFilterChange,
  onConfidenceFilterChange,
  onTypeFilterChange,
  activeLesionKey,
  onLesionHover,
}: {
  lesions: LesionWithKey[]
  filteredLesions: LesionWithKey[]
  lesionRegions: string[]
  lesionTypes: string[]
  lesionRegionFilter: string
  lesionConfidenceFilter: 'all' | 'high' | 'medium' | 'low'
  lesionTypeFilter: string
  onRegionFilterChange: (value: string) => void
  onConfidenceFilterChange: (value: 'all' | 'high' | 'medium' | 'low') => void
  onTypeFilterChange: (value: string) => void
  activeLesionKey: string | null
  onLesionHover: (key: string | null) => void
}) {
  if (lesions.length === 0) return null

  return (
    <div className="holographic-panel rounded-[1.75rem] p-6">
      <div className="terminal-text mb-4 text-[10px] text-cyan-400/80">CONSENSUS LESION TABLE</div>
      <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-3">
        <select
          value={lesionRegionFilter}
          onChange={(e) => onRegionFilterChange(e.target.value)}
          aria-label="Filter by lesion region"
          className="rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
        >
          <option value="all">All regions</option>
          {lesionRegions.map((region) => (
            <option key={region} value={region}>{region}</option>
          ))}
        </select>
        <select
          value={lesionConfidenceFilter}
          onChange={(e) => onConfidenceFilterChange(e.target.value as 'all' | 'high' | 'medium' | 'low')}
          aria-label="Filter by confidence tier"
          className="rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
        >
          <option value="all">All confidence tiers</option>
          <option value="high">High confidence</option>
          <option value="medium">Medium confidence</option>
          <option value="low">Low / review</option>
        </select>
        <select
          value={lesionTypeFilter}
          onChange={(e) => onTypeFilterChange(e.target.value)}
          aria-label="Filter by acne type"
          className="rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
        >
          <option value="all">All acne types</option>
          {lesionTypes.map((t) => (
            <option key={t} value={t}>{getAcneTypeLabel(t)}</option>
          ))}
        </select>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {filteredLesions.slice(0, 12).map((lesion, idx) => (
          <motion.button
            key={lesion.key}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.03, duration: 0.25 }}
            onMouseEnter={() => onLesionHover(lesion.key)}
            onMouseLeave={() => onLesionHover(null)}
            className={`rounded-xl border bg-white/3 p-4 text-left text-sm transition-all ${activeLesionKey === lesion.key ? 'border-cyan-400/40 bg-cyan-400/8 shadow-[0_0_25px_rgba(0,242,255,0.12)]' : 'border-white/5'}`}
          >
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="font-medium text-white">{lesion.region}</span>
              <span className={`rounded-full border px-2 py-0.5 text-[9px] font-medium ${getAcneTypeColor(lesion.class_name)}`}>{getAcneTypeLabel(lesion.class_name)}</span>
              <span className={`terminal-text text-[8px] ${getConfidenceTextTone(lesion.confidence)}`}>{lesion.confidence_level}</span>
            </div>
            <div className="text-zinc-400">confidence {lesion.confidence} · {getSeverityGradeLabel(lesion.severity_grade)}</div>
            <div className="text-zinc-500">votes {lesion.votes} · reliability {lesion.reliability_score}</div>
          </motion.button>
        ))}
      </div>
    </div>
  )
}
