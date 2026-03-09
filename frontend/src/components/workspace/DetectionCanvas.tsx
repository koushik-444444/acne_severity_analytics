import { Eye, EyeOff, Maximize2, Minimize2, Search, SearchX } from 'lucide-react'

import { clamp } from '../../lib/clinical-utils'
import type { ConsensusLesion } from '../../types/api'
import { AdvancedImageViewer } from './AdvancedImageViewer'
import type { ViewerState } from './types'

type LesionWithKey = ConsensusLesion & { key: string }

export function DetectionCanvas({
  activeImage,
  displayGags,
  displaySeverity,
  severityTone,
  showDetectionOverlay,
  onToggleOverlay,
  mainViewer,
  onMainViewerChange,
  onResetMainViewer,
  isFullscreen,
  onToggleFullscreen,
  lesionOverlayItems,
  activeLesionKey,
  onLesionHover,
}: {
  activeImage: string
  displayGags: number
  displaySeverity: string
  severityTone: string
  showDetectionOverlay: boolean
  onToggleOverlay: () => void
  mainViewer: ViewerState
  onMainViewerChange: (state: ViewerState | ((prev: ViewerState) => ViewerState)) => void
  onResetMainViewer: () => void
  isFullscreen: boolean
  onToggleFullscreen: () => void
  lesionOverlayItems: LesionWithKey[]
  activeLesionKey: string | null
  onLesionHover: (key: string | null) => void
}) {
  return (
    <div className="rounded-[1.75rem] border border-white/5 bg-black/30 p-4">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="terminal-text text-[10px] text-cyan-400/80">ACNE DETECTION CANVAS</h3>
        <div className="flex flex-wrap items-center gap-2">
          <button
            aria-pressed={showDetectionOverlay}
            onClick={onToggleOverlay}
            className="inline-flex items-center gap-2 rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
          >
            {showDetectionOverlay ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            {showDetectionOverlay ? 'Detection overlay' : 'Original image'}
          </button>
          <button aria-label="Zoom in" onClick={() => onMainViewerChange((state) => ({ ...state, scale: clamp(state.scale + 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
            <Search className="h-4 w-4" />
          </button>
          <button aria-label="Zoom out" onClick={() => onMainViewerChange((state) => ({ ...state, scale: clamp(state.scale - 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
            <SearchX className="h-4 w-4" />
          </button>
          <button aria-label="Reset viewer zoom and position" onClick={onResetMainViewer} className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
            Reset
          </button>
          <button aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'} onClick={onToggleFullscreen} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
        </div>
      </div>
      <div className="mb-4 flex flex-wrap gap-3">
        <div className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-300">
          GAGS SCORE: <span className="font-semibold text-white">{displayGags}</span>
        </div>
        <div className={`rounded-full border px-4 py-2 text-sm ${severityTone}`}>
          SEVERITY: <span className="font-semibold text-white">{displaySeverity}</span>
        </div>
        <div className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-zinc-300">
          VIEW: <span className="font-semibold text-white">{showDetectionOverlay ? 'Overlay' : 'Original'}</span>
        </div>
      </div>
      <AdvancedImageViewer
        src={activeImage}
        alt="Acne detection visual"
        state={mainViewer}
        onChange={onMainViewerChange}
        heightClass="h-[520px]"
        lesions={showDetectionOverlay ? lesionOverlayItems : []}
        activeLesionKey={activeLesionKey}
        onLesionHover={onLesionHover}
      />
    </div>
  )
}
