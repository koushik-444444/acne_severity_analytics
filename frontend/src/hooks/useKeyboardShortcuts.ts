import { useEffect } from 'react'

import { clamp } from '../lib/clinical-utils'
import type { ViewerState } from '../components/workspace/types'

type KeyboardShortcutDeps = {
  active: unknown
  compare: unknown
  setMainViewer: React.Dispatch<React.SetStateAction<ViewerState>>
  resetMainViewer: () => void
  resetCompareViewer: () => void
  setShowDetectionOverlay: React.Dispatch<React.SetStateAction<boolean>>
  setCompareViewerMode: React.Dispatch<React.SetStateAction<'single' | 'split'>>
  toggleFullscreen: () => Promise<void>
}

/**
 * Binds global keyboard shortcuts for the workspace viewer:
 * `+` / `-`  zoom, `R` reset, `O` toggle overlay,
 * `C` toggle compare mode, `F` fullscreen.
 *
 * Shortcuts are suppressed when focus is in an input, textarea,
 * select, or contenteditable element.
 */
export function useKeyboardShortcuts(deps: KeyboardShortcutDeps) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable) return
      if (!deps.active) return
      if (event.key === '+') deps.setMainViewer((state) => ({ ...state, scale: clamp(state.scale + 0.25, 1, 4) }))
      if (event.key === '-') deps.setMainViewer((state) => ({ ...state, scale: clamp(state.scale - 0.25, 1, 4) }))
      if (event.key.toLowerCase() === 'r') {
        deps.resetMainViewer()
        deps.resetCompareViewer()
      }
      if (event.key.toLowerCase() === 'o') deps.setShowDetectionOverlay((value) => !value)
      if (event.key.toLowerCase() === 'c' && deps.compare) deps.setCompareViewerMode((mode) => (mode === 'single' ? 'split' : 'single'))
      if (event.key.toLowerCase() === 'f') {
        event.preventDefault()
        void deps.toggleFullscreen()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [deps.active, deps.compare]) // eslint-disable-line react-hooks/exhaustive-deps
}
