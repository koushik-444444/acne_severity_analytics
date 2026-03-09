import { useEffect, useRef, useState } from 'react'

import {
  getProfileStorageKey,
  readStorageItem,
  removeStorageItem,
  writeStorageItem,
} from '../lib/clinical-utils'
import type { ExportPreset } from '../types/api'
import type { ViewerMode, WorkspaceUiPrefs } from '../components/workspace/types'

const UI_PREFS_KEY = 'clearskin-ui-prefs'

function loadUiPrefs(profileId: string): WorkspaceUiPrefs | null {
  const storageKey = getProfileStorageKey(UI_PREFS_KEY, profileId)
  const raw = readStorageItem(storageKey)
  if (!raw) {
    return null
  }

  try {
    return JSON.parse(raw) as WorkspaceUiPrefs
  } catch {
    removeStorageItem(storageKey)
    return null
  }
}

/**
 * Hydrates per-profile UI preferences from localStorage and
 * debounces persistence with a 300ms save timer.
 */
export function useWorkspacePrefs(activeProfileId: string) {
  const [leftRailWidth, setLeftRailWidth] = useState(320)
  const [rightRailWidth, setRightRailWidth] = useState(380)
  const [compareViewerMode, setCompareViewerMode] = useState<ViewerMode>('single')
  const [showDetectionOverlay, setShowDetectionOverlay] = useState(true)
  const [showCompareOverlay, setShowCompareOverlay] = useState(true)
  const [exportPreset, setExportPreset] = useState<ExportPreset>('clinical')
  const [prefsHydrated, setPrefsHydrated] = useState(false)
  const saveTimerRef = useRef<number | null>(null)

  // Hydrate preferences when profile changes
  useEffect(() => {
    setPrefsHydrated(false)

    const storedPrefs = loadUiPrefs(activeProfileId)
    setLeftRailWidth(storedPrefs?.leftRailWidth ?? 320)
    setRightRailWidth(storedPrefs?.rightRailWidth ?? 380)
    setCompareViewerMode(storedPrefs?.compareViewerMode ?? 'single')
    setShowDetectionOverlay(storedPrefs?.showDetectionOverlay ?? true)
    setShowCompareOverlay(storedPrefs?.showCompareOverlay ?? true)
    setExportPreset(storedPrefs?.exportPreset ?? 'clinical')
    setPrefsHydrated(true)
  }, [activeProfileId])

  // Debounced persistence
  useEffect(() => {
    if (!prefsHydrated) {
      return
    }

    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current)
    }

    saveTimerRef.current = window.setTimeout(() => {
      writeStorageItem(
        getProfileStorageKey(UI_PREFS_KEY, activeProfileId),
        JSON.stringify({
          leftRailWidth,
          rightRailWidth,
          compareViewerMode,
          showDetectionOverlay,
          showCompareOverlay,
          exportPreset,
        } satisfies WorkspaceUiPrefs),
      )
      saveTimerRef.current = null
    }, 300)

    return () => {
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current)
      }
    }
  }, [activeProfileId, prefsHydrated, leftRailWidth, rightRailWidth, compareViewerMode, showDetectionOverlay, showCompareOverlay, exportPreset])

  return {
    leftRailWidth,
    setLeftRailWidth,
    rightRailWidth,
    setRightRailWidth,
    compareViewerMode,
    setCompareViewerMode,
    showDetectionOverlay,
    setShowDetectionOverlay,
    showCompareOverlay,
    setShowCompareOverlay,
    exportPreset,
    setExportPreset,
    prefsHydrated,
  }
}
