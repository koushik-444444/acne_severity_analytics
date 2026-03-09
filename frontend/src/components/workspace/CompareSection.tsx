import { Search, SearchX } from 'lucide-react'
import { motion } from 'framer-motion'

import { clamp } from '../../lib/clinical-utils'
import type { ConsensusLesion } from '../../types/api'
import { AdvancedImageViewer } from './AdvancedImageViewer'
import { SplitCompareViewer } from './SplitCompareViewer'
import type { ViewerMode, ViewerState } from './types'

type LesionWithKey = ConsensusLesion & { key: string }

export function CompareSection({
  compareViewerMode,
  compareFullscreen,
  activeImage,
  previousImage,
  showCompareOverlay,
  onToggleCompareOverlay,
  compareViewer,
  onCompareViewerChange,
  onResetCompareViewer,
  onSetCompareViewerMode,
  onSetCompareFullscreen,
  lesionOverlayItems,
  activeLesionKey,
  onLesionHover,
  activeDiagnosticImage,
  activeOriginalImage,
  previousDiagnosticImage,
  previousOriginalImage,
}: {
  compareViewerMode: ViewerMode
  compareFullscreen: boolean
  activeImage: string
  previousImage: string
  showCompareOverlay: boolean
  onToggleCompareOverlay: () => void
  compareViewer: ViewerState
  onCompareViewerChange: (state: ViewerState | ((prev: ViewerState) => ViewerState)) => void
  onResetCompareViewer: () => void
  onSetCompareViewerMode: (mode: ViewerMode) => void
  onSetCompareFullscreen: (value: boolean) => void
  lesionOverlayItems: LesionWithKey[]
  activeLesionKey: string | null
  onLesionHover: (key: string | null) => void
  activeDiagnosticImage: string | null
  activeOriginalImage: string | null
  previousDiagnosticImage: string | null
  previousOriginalImage: string | null
}) {
  const currentCompareImage = showCompareOverlay
    ? activeDiagnosticImage ?? activeOriginalImage ?? ''
    : activeOriginalImage ?? activeDiagnosticImage ?? ''
  const previousCompareImage = showCompareOverlay
    ? previousDiagnosticImage ?? previousOriginalImage ?? ''
    : previousOriginalImage ?? previousDiagnosticImage ?? ''

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      {compareViewerMode === 'single' && !compareFullscreen && (
        <SingleCompare
          currentImage={currentCompareImage}
          previousImage={previousCompareImage}
          showCompareOverlay={showCompareOverlay}
          onToggleCompareOverlay={onToggleCompareOverlay}
          compareViewer={compareViewer}
          onCompareViewerChange={onCompareViewerChange}
          onResetCompareViewer={onResetCompareViewer}
          onSetCompareViewerMode={onSetCompareViewerMode}
          onSetCompareFullscreen={onSetCompareFullscreen}
          lesionOverlayItems={lesionOverlayItems}
          activeLesionKey={activeLesionKey}
          onLesionHover={onLesionHover}
        />
      )}

      {compareViewerMode === 'split' && previousImage && !compareFullscreen && (
        <SplitCompare
          activeImage={activeImage}
          previousImage={previousImage}
          compareViewer={compareViewer}
          onCompareViewerChange={onCompareViewerChange}
          onResetCompareViewer={onResetCompareViewer}
          onSetCompareViewerMode={onSetCompareViewerMode}
          onSetCompareFullscreen={onSetCompareFullscreen}
        />
      )}

      {compareFullscreen && (
        <FullscreenCompare
          activeImage={activeImage}
          previousImage={previousImage}
          showCompareOverlay={showCompareOverlay}
          compareViewer={compareViewer}
          onCompareViewerChange={onCompareViewerChange}
          onSetCompareFullscreen={onSetCompareFullscreen}
          lesionOverlayItems={lesionOverlayItems}
          activeLesionKey={activeLesionKey}
          onLesionHover={onLesionHover}
        />
      )}
    </motion.div>
  )
}

function SingleCompare({
  currentImage,
  previousImage,
  showCompareOverlay,
  onToggleCompareOverlay,
  compareViewer,
  onCompareViewerChange,
  onResetCompareViewer,
  onSetCompareViewerMode,
  onSetCompareFullscreen,
  lesionOverlayItems,
  activeLesionKey,
  onLesionHover,
}: {
  currentImage: string
  previousImage: string
  showCompareOverlay: boolean
  onToggleCompareOverlay: () => void
  compareViewer: ViewerState
  onCompareViewerChange: (state: ViewerState | ((prev: ViewerState) => ViewerState)) => void
  onResetCompareViewer: () => void
  onSetCompareViewerMode: (mode: ViewerMode) => void
  onSetCompareFullscreen: (value: boolean) => void
  lesionOverlayItems: LesionWithKey[]
  activeLesionKey: string | null
  onLesionHover: (key: string | null) => void
}) {
  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
      <div className="rounded-[1.75rem] border border-white/5 bg-black/30 p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="terminal-text text-[10px] text-cyan-400/80">VISUAL COMPARE · CURRENT</div>
          <div className="flex items-center gap-2">
            <button
              onClick={onToggleCompareOverlay}
              className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
            >
              {showCompareOverlay ? 'Detection pair' : 'Original pair'}
            </button>
            <button aria-label="Zoom in compare" onClick={() => onCompareViewerChange((state) => ({ ...state, scale: clamp(state.scale + 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
              <Search className="h-4 w-4" />
            </button>
            <button aria-label="Zoom out compare" onClick={() => onCompareViewerChange((state) => ({ ...state, scale: clamp(state.scale - 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
              <SearchX className="h-4 w-4" />
            </button>
            <button onClick={onResetCompareViewer} className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
              Reset
            </button>
            <button
              onClick={() => onSetCompareViewerMode('split')}
              className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
            >
              Split view
            </button>
            <button
              onClick={() => onSetCompareFullscreen(true)}
              className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
            >
              Fullscreen compare
            </button>
          </div>
        </div>
        <AdvancedImageViewer
          src={currentImage}
          alt="Current acne detection"
          state={compareViewer}
          onChange={onCompareViewerChange}
          heightClass="h-[420px]"
          lesions={showCompareOverlay ? lesionOverlayItems : []}
          activeLesionKey={activeLesionKey}
          onLesionHover={onLesionHover}
        />
      </div>
      <div className="rounded-[1.75rem] border border-white/5 bg-black/30 p-4">
        <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">VISUAL COMPARE · PRIOR</div>
        {previousImage ? (
          <AdvancedImageViewer
            src={previousImage}
            alt="Previous acne detection"
            state={compareViewer}
            onChange={onCompareViewerChange}
            heightClass="h-[420px]"
          />
        ) : (
          <div className="flex h-[420px] items-center justify-center rounded-[1.25rem] border border-dashed border-white/10 bg-black/20 text-sm text-zinc-500">
            No previous diagnostic image available.
          </div>
        )}
      </div>
    </div>
  )
}

function SplitCompare({
  activeImage,
  previousImage,
  compareViewer,
  onCompareViewerChange,
  onResetCompareViewer,
  onSetCompareViewerMode,
  onSetCompareFullscreen,
}: {
  activeImage: string
  previousImage: string
  compareViewer: ViewerState
  onCompareViewerChange: (state: ViewerState | ((prev: ViewerState) => ViewerState)) => void
  onResetCompareViewer: () => void
  onSetCompareViewerMode: (mode: ViewerMode) => void
  onSetCompareFullscreen: (value: boolean) => void
}) {
  return (
    <div className="rounded-[1.75rem] border border-white/5 bg-black/30 p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="terminal-text text-[10px] text-cyan-400/80">SPLIT COMPARE VIEWER</div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onSetCompareViewerMode('single')}
            className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
          >
            Side by side
          </button>
          <button onClick={onResetCompareViewer} className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
            Reset
          </button>
          <button
            onClick={() => onSetCompareFullscreen(true)}
            className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
          >
            Fullscreen compare
          </button>
        </div>
      </div>
      <SplitCompareViewer
        beforeSrc={previousImage}
        afterSrc={activeImage}
        state={compareViewer}
        onChange={onCompareViewerChange}
      />
    </div>
  )
}

function FullscreenCompare({
  activeImage,
  previousImage,
  showCompareOverlay,
  compareViewer,
  onCompareViewerChange,
  onSetCompareFullscreen,
  lesionOverlayItems,
  activeLesionKey,
  onLesionHover,
}: {
  activeImage: string
  previousImage: string
  showCompareOverlay: boolean
  compareViewer: ViewerState
  onCompareViewerChange: (state: ViewerState | ((prev: ViewerState) => ViewerState)) => void
  onSetCompareFullscreen: (value: boolean) => void
  lesionOverlayItems: LesionWithKey[]
  activeLesionKey: string | null
  onLesionHover: (key: string | null) => void
}) {
  return (
    <div className="rounded-[1.75rem] border border-cyan-400/15 bg-black/40 p-4 shadow-[0_0_40px_rgba(0,242,255,0.08)]">
      <div className="mb-3 flex items-center justify-between">
        <div className="terminal-text text-[10px] text-cyan-400/80">FULLSCREEN COMPARE WORKSPACE</div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onSetCompareFullscreen(false)}
            className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
          >
            Exit compare workspace
          </button>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <AdvancedImageViewer
          src={activeImage}
          alt="Current compare workspace"
          state={compareViewer}
          onChange={onCompareViewerChange}
          heightClass="h-[640px]"
          lesions={showCompareOverlay ? lesionOverlayItems : []}
          activeLesionKey={activeLesionKey}
          onLesionHover={onLesionHover}
        />
        {previousImage ? (
          <AdvancedImageViewer
            src={previousImage}
            alt="Previous compare workspace"
            state={compareViewer}
            onChange={onCompareViewerChange}
            heightClass="h-[640px]"
          />
        ) : (
          <div className="flex h-[640px] items-center justify-center rounded-[1.25rem] border border-dashed border-white/10 bg-black/20 text-sm text-zinc-500">
            No previous image available.
          </div>
        )}
      </div>
    </div>
  )
}
