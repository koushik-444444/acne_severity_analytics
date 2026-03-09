import { Activity, Loader2, Trash2 } from 'lucide-react'

import type { ProfileSummary, SessionSummary } from '../../types/api'
import { HistoryCardSkeleton } from './Skeleton'

const DEFAULT_PROFILE_ID = 'default-profile'

export function SessionArchiveRail({
  activeProfileId,
  onProfileChange,
  profiles,
  history,
  baselineSession,
  onSelectHistory,
  onPurge,
  onPinBaseline,
  leftRailWidth,
  onLeftRailWidthChange,
  hasMore,
  isLoadingMore,
  onLoadMore,
}: {
  activeProfileId: string
  onProfileChange: (profileId: string) => void
  profiles: ProfileSummary[]
  history: SessionSummary[]
  baselineSession: SessionSummary | null
  onSelectHistory: (item: SessionSummary) => void
  onPurge: (item: SessionSummary) => void
  onPinBaseline: (item: SessionSummary) => void
  leftRailWidth: number
  onLeftRailWidthChange: (width: number) => void
  hasMore?: boolean
  isLoadingMore?: boolean
  onLoadMore?: () => void
}) {
  return (
    <aside className="holographic-panel relative rounded-[2rem] p-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <div className="terminal-text text-[10px] text-cyan-400/80">SESSION ARCHIVE</div>
          <div className="text-sm text-zinc-500">Longitudinal timeline</div>
        </div>
        <Activity className="h-4 w-4 text-cyan-400" />
      </div>

      <div className="mb-4 space-y-2">
        <div className="terminal-text text-[9px] text-zinc-500">PATIENT / PROFILE</div>
        <select
          value={activeProfileId}
          onChange={(e) => onProfileChange(e.target.value)}
          aria-label="Patient profile"
          className="w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
        >
          {(profiles.length ? profiles : [{ profile_id: DEFAULT_PROFILE_ID, sessions: 0, latest_timestamp: '', latest_severity: null }]).map((profile) => (
            <option key={profile.profile_id} value={profile.profile_id}>
              {profile.profile_id} · {profile.sessions} sessions
            </option>
          ))}
        </select>
      </div>

      <div className="max-h-[760px] space-y-3 overflow-y-auto pr-2">
        {history.length === 0 ? (
          <EmptyHistoryState />
        ) : (
          history.map((item) => (
            <SessionCard
              key={item.session_id}
              item={item}
              isBaseline={baselineSession?.session_id === item.session_id}
              onSelect={onSelectHistory}
              onPurge={onPurge}
              onPinBaseline={onPinBaseline}
            />
          ))
        )}
        {isLoadingMore && (
          <div className="space-y-3">
            <HistoryCardSkeleton />
            <HistoryCardSkeleton />
          </div>
        )}
        {hasMore && !isLoadingMore && (
          <button
            type="button"
            onClick={onLoadMore}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 py-2 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
          >
            <Loader2 className="h-3 w-3" />
            Load older sessions
          </button>
        )}
      </div>
      <input
        type="range"
        min={260}
        max={420}
        value={leftRailWidth}
        onChange={(e) => onLeftRailWidthChange(Number(e.target.value))}
        aria-label="Left panel width"
        className="mt-4 w-full accent-cyan-400"
      />
    </aside>
  )
}

function EmptyHistoryState() {
  return (
    <div className="flex flex-col items-center gap-3 py-12 text-center">
      <Activity className="h-8 w-8 text-zinc-700" />
      <div className="terminal-text text-[10px] text-zinc-600">NO SESSIONS YET</div>
      <p className="max-w-[200px] text-xs text-zinc-600">
        Run your first analysis to see session history here.
      </p>
    </div>
  )
}

function SessionCard({
  item,
  isBaseline,
  onSelect,
  onPurge,
  onPinBaseline,
}: {
  item: SessionSummary
  isBaseline: boolean
  onSelect: (item: SessionSummary) => void
  onPurge: (item: SessionSummary) => void
  onPinBaseline: (item: SessionSummary) => void
}) {
  return (
    <button
      onClick={() => onSelect(item)}
      className="w-full rounded-2xl border border-white/5 bg-white/3 p-4 text-left transition-all hover:border-cyan-400/25 hover:bg-cyan-400/5"
    >
      <div className="mb-2 flex items-start justify-between gap-3">
        <span className="terminal-text text-[9px] text-cyan-400/80">{item.severity ?? 'Unknown'}</span>
        <span className="terminal-text text-[8px] text-zinc-600">{new Date(item.timestamp).toLocaleDateString()}</span>
      </div>
      <div className="text-sm font-semibold tracking-tight">GAGS {item.gags_score ?? 0}</div>
      <div className="mt-1 text-xs text-zinc-500">{item.lesion_count ?? 0} lesions · symmetry {item.symmetry_delta ?? 0}%</div>
      <div className="mt-2 flex flex-wrap gap-2">
        {item.note ? <span className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-2 py-1 text-[10px] text-cyan-200">noted</span> : null}
        {item.status?.failed ? <span className="rounded-full border border-red-400/20 bg-red-400/10 px-2 py-1 text-[10px] text-red-200">failed</span> : null}
        {item.status?.completed ? <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-1 text-[10px] text-emerald-200">complete</span> : null}
      </div>
      <div className="mt-3 flex items-center justify-between">
        <span className="terminal-text text-[8px] text-zinc-700">#{item.session_id.slice(0, 8)}</span>
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation()
            onPurge(item)
          }}
          className="inline-flex items-center gap-1 text-[10px] text-zinc-500 hover:text-red-300"
        >
          <Trash2 className="h-3 w-3" /> purge
        </button>
      </div>
      <div className="mt-2 flex items-center justify-between">
        <button
          onClick={(event) => {
            event.stopPropagation()
            onPinBaseline(item)
          }}
          className={`rounded-full px-2 py-1 text-[10px] ${isBaseline ? 'bg-cyan-400 text-black' : 'border border-white/10 text-zinc-400 hover:border-cyan-400/20 hover:text-white'}`}
        >
          {isBaseline ? 'Baseline pinned' : 'Pin baseline'}
        </button>
      </div>
    </button>
  )
}
