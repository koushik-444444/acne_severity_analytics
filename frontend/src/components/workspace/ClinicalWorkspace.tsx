import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react'
import { Activity, Camera, Eye, EyeOff, Loader2, Maximize2, Minimize2, Search, SearchX, Shield, Sparkles, Trash2 } from 'lucide-react'
import { motion } from 'framer-motion'

import { api } from '../../services/api'
import type {
  AnalyzeResponse,
  ComparePayload,
  ConsensusLesion,
  PrivacyConfig,
  ProfileSummary,
  RegionStats,
  SessionDetail,
  SessionStatus,
  SessionSummary,
} from '../../types/api'

type CompareRegionDelta = NonNullable<ComparePayload>['regions'][string]
type ViewerMode = 'single' | 'split'
const UI_PREFS_KEY = 'clearskin-ui-prefs'
const BASELINE_SESSION_KEY = 'clearskin-baseline-session'
const ONBOARDING_SEEN_KEY = `${UI_PREFS_KEY}-onboarding-seen`
const DEFAULT_PROFILE_ID = 'default-profile'

type WorkspaceUiPrefs = {
  leftRailWidth?: number
  rightRailWidth?: number
  compareViewerMode?: ViewerMode
  showDetectionOverlay?: boolean
  showCompareOverlay?: boolean
  exportPreset?: 'clinical' | 'compact' | 'presentation'
}

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

function getSeverityTone(severity: string) {
  const normalized = severity.toLowerCase()
  if (normalized.includes('very severe') || normalized.includes('cystic')) {
    return 'border-red-500/35 bg-red-500/15 text-red-200 shadow-[0_0_30px_rgba(239,68,68,0.12)]'
  }
  if (normalized.includes('severe')) {
    return 'border-orange-500/35 bg-orange-500/15 text-orange-200 shadow-[0_0_30px_rgba(249,115,22,0.12)]'
  }
  if (normalized.includes('moderate')) {
    return 'border-amber-500/35 bg-amber-500/15 text-amber-200 shadow-[0_0_30px_rgba(245,158,11,0.12)]'
  }
  if (normalized.includes('mild')) {
    return 'border-cyan-400/35 bg-cyan-400/10 text-cyan-200 shadow-[0_0_30px_rgba(0,242,255,0.10)]'
  }
  return 'border-white/10 bg-white/5 text-zinc-200'
}

type ViewerState = {
  scale: number
  x: number
  y: number
}

type ActiveCaseProfileGuard = {
  sessionId: string
  sessionProfileId: string
  requestedProfileId: string
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function getProfileStorageKey(baseKey: string, profileId: string) {
  return `${baseKey}:${profileId || DEFAULT_PROFILE_ID}`
}

function formatSignedDelta(value: number, suffix = '') {
  return `${value > 0 ? '+' : ''}${value}${suffix}`
}

function getClinicalDeltaStatus(value: number, betterWhenLower = true): 'improved' | 'stable' | 'worsened' {
  if (value === 0) {
    return 'stable'
  }

  const improved = betterWhenLower ? value < 0 : value > 0
  return improved ? 'improved' : 'worsened'
}

function getClinicalDeltaLabel(value: number, betterWhenLower = true) {
  const status = getClinicalDeltaStatus(value, betterWhenLower)
  if (status === 'stable') {
    return 'Stable'
  }
  return status === 'improved' ? 'Improved' : 'Worsened'
}

function getDeltaTone(value: number, betterWhenLower = true) {
  if (value === 0) {
    return 'border-white/10 bg-white/5 text-zinc-200'
  }

  const improved = betterWhenLower ? value < 0 : value > 0
  return improved
    ? 'border-emerald-400/25 bg-emerald-400/10 text-emerald-100'
    : 'border-rose-400/25 bg-rose-400/10 text-rose-100'
}

function describeDelta(value: number, label: string, betterWhenLower = true) {
  if (value === 0) {
    return `${label} remained stable`
  }

  const status = getClinicalDeltaStatus(value, betterWhenLower)
  const magnitude = Math.abs(value)
  return status === 'improved'
    ? `${label} improved by ${magnitude}`
    : `${label} worsened by ${magnitude}`
}

function readStorageItem(key: string) {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function writeStorageItem(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value)
  } catch {
    // ignore storage write failures
  }
}

function removeStorageItem(key: string) {
  try {
    window.localStorage.removeItem(key)
  } catch {
    // ignore storage removal failures
  }
}

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

export function ClinicalWorkspace() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [status, setStatus] = useState<SessionStatus | null>(null)
  const [history, setHistory] = useState<SessionSummary[]>([])
  const [profiles, setProfiles] = useState<ProfileSummary[]>([])
  const [activeProfileId, setActiveProfileId] = useState<string>(DEFAULT_PROFILE_ID)
  const [active, setActive] = useState<AnalyzeResponse | null>(null)
  const [compare, setCompare] = useState<ComparePayload | null>(null)
  const [previousSession, setPreviousSession] = useState<SessionDetail | null>(null)
  const [baselineSession, setBaselineSession] = useState<SessionSummary | null>(null)
  const [privacy, setPrivacy] = useState<PrivacyConfig | null>(null)
  const [privacyMode, setPrivacyMode] = useState(false)
  const [retentionHours, setRetentionHours] = useState(72)
  const [showDetectionOverlay, setShowDetectionOverlay] = useState(true)
  const [showCompareOverlay, setShowCompareOverlay] = useState(true)
  const [compareViewerMode, setCompareViewerMode] = useState<ViewerMode>('single')
  const [compareFullscreen, setCompareFullscreen] = useState(false)
  const [leftRailWidth, setLeftRailWidth] = useState(320)
  const [rightRailWidth, setRightRailWidth] = useState(380)
  const [mainViewer, setMainViewer] = useState<ViewerState>({ scale: 1, x: 0, y: 0 })
  const [compareViewer, setCompareViewer] = useState<ViewerState>({ scale: 1, x: 0, y: 0 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [activeLesionKey, setActiveLesionKey] = useState<string | null>(null)
  const [lesionRegionFilter, setLesionRegionFilter] = useState<string>('all')
  const [lesionConfidenceFilter, setLesionConfidenceFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all')
  const [noteDraft, setNoteDraft] = useState('')
  const [exportPreset, setExportPreset] = useState<'clinical' | 'compact' | 'presentation'>('clinical')
  const [showOnboarding, setShowOnboarding] = useState(false)
  const [prefsHydrated, setPrefsHydrated] = useState(false)
  const [profileSwitchNotice, setProfileSwitchNotice] = useState<{ from: string; to: string } | null>(null)
  const [profileGuard, setProfileGuard] = useState<ActiveCaseProfileGuard | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isExporting, setIsExporting] = useState(false)
  const [isSavingNotes, setIsSavingNotes] = useState(false)
  const [isLoadingSession, setIsLoadingSession] = useState(false)
  const [isPurging, setIsPurging] = useState(false)
  const workspaceRef = useRef<HTMLDivElement | null>(null)
  const hasHydratedInitialProfileRef = useRef(false)
  const previousProfileRef = useRef(activeProfileId)
  const profileNoticeTimerRef = useRef<number | null>(null)
  const prefsSaveTimerRef = useRef<number | null>(null)

  useEffect(() => {
    setShowOnboarding(readStorageItem(ONBOARDING_SEEN_KEY) !== 'true')

    let activeRequest = true

    void Promise.all([api.getPrivacy(), api.getProfiles()])
      .then(([privacyConfig, profileItems]) => {
        if (!activeRequest) {
          return
        }

        setPrivacy(privacyConfig)
        setProfiles(profileItems)
        setRetentionHours(privacyConfig.default_retention_hours)
      })
      .catch((err) => {
        if (activeRequest) {
          setError(err instanceof Error ? err.message : 'Failed to initialize workspace')
        }
      })

    return () => {
      activeRequest = false
    }
  }, [])

  useEffect(() => {
    setPrefsHydrated(false)
    setBaselineSession(null)

    const storedPrefs = loadUiPrefs(activeProfileId)
    setLeftRailWidth(storedPrefs?.leftRailWidth ?? 320)
    setRightRailWidth(storedPrefs?.rightRailWidth ?? 380)
    setCompareViewerMode(storedPrefs?.compareViewerMode ?? 'single')
    setShowDetectionOverlay(storedPrefs?.showDetectionOverlay ?? true)
    setShowCompareOverlay(storedPrefs?.showCompareOverlay ?? true)
    setExportPreset(storedPrefs?.exportPreset ?? 'clinical')
    setPrefsHydrated(true)

    let activeRequest = true
    const baselineStorageKey = getProfileStorageKey(BASELINE_SESSION_KEY, activeProfileId)

    void Promise.all([api.getHistory(30, activeProfileId), api.getProfiles()])
      .then(([historyItems, profileItems]) => {
        if (!activeRequest) {
          return
        }

        setHistory(historyItems)
        setProfiles(profileItems)

        const storedBaseline = readStorageItem(baselineStorageKey)
        if (!storedBaseline) {
          return
        }

        const match = historyItems.find((item) => item.session_id === storedBaseline)
        if (match) {
          setBaselineSession(match)
          return
        }

        removeStorageItem(baselineStorageKey)
      })
      .catch((err) => {
        if (activeRequest) {
          setError(err instanceof Error ? err.message : 'Failed to load profile workspace')
        }
      })

    return () => {
      activeRequest = false
    }
  }, [activeProfileId])

  useEffect(() => {
    if (!hasHydratedInitialProfileRef.current) {
      hasHydratedInitialProfileRef.current = true
      previousProfileRef.current = activeProfileId
      return
    }

    const previousProfileId = previousProfileRef.current
    if (previousProfileId === activeProfileId) {
      return
    }

    setProfileSwitchNotice({ from: previousProfileId, to: activeProfileId })
    previousProfileRef.current = activeProfileId

    if (profileNoticeTimerRef.current !== null) {
      window.clearTimeout(profileNoticeTimerRef.current)
    }

    profileNoticeTimerRef.current = window.setTimeout(() => {
      setProfileSwitchNotice(null)
      profileNoticeTimerRef.current = null
    }, 2600)
  }, [activeProfileId])

  useEffect(() => {
    return () => {
      if (profileNoticeTimerRef.current !== null) {
        window.clearTimeout(profileNoticeTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!prefsHydrated) {
      return
    }

    if (prefsSaveTimerRef.current !== null) {
      window.clearTimeout(prefsSaveTimerRef.current)
    }

    prefsSaveTimerRef.current = window.setTimeout(() => {
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
      prefsSaveTimerRef.current = null
    }, 300)

    return () => {
      if (prefsSaveTimerRef.current !== null) {
        window.clearTimeout(prefsSaveTimerRef.current)
      }
    }
  }, [activeProfileId, prefsHydrated, leftRailWidth, rightRailWidth, compareViewerMode, showDetectionOverlay, showCompareOverlay, exportPreset])

  useEffect(() => {
    if (!sessionId || !isAnalyzing) return
    let cancelled = false

    const pollStatus = async () => {
      while (!cancelled) {
        try {
          const nextStatus = await api.getStatus(sessionId)
          if (cancelled) {
            return
          }
          setStatus(nextStatus)
          if (nextStatus.completed || nextStatus.failed) {
            return
          }
        } catch {
          if (cancelled) {
            return
          }
        }

        await new Promise((resolve) => window.setTimeout(resolve, 1500))
      }
    }

    void pollStatus()

    return () => {
      cancelled = true
    }
  }, [sessionId, isAnalyzing])

  const regionDeltaCards = useMemo<[string, CompareRegionDelta][]>(() => {
    if (!compare?.regions) return []
    return (Object.entries(compare.regions) as [string, CompareRegionDelta][])
      .sort((a, b) => Math.abs(b[1].count_delta) - Math.abs(a[1].count_delta))
      .slice(0, 6)
  }, [compare])
  const isPinnedBaselineCompare = Boolean(compare && baselineSession?.session_id === compare.previous_session_id)
  const isViewingPinnedBaseline = Boolean(active?.session_id && baselineSession?.session_id === active.session_id)
  const compareNarrative = useMemo(() => {
    if (!compare) {
      return null
    }

    const compareTarget = isPinnedBaselineCompare ? 'the pinned baseline' : 'the previous archived session'
    return `${describeDelta(compare.lesion_delta, 'Lesion burden')} and ${describeDelta(compare.gags_delta, 'GAGS', true)} versus ${compareTarget}.`
  }, [compare, isPinnedBaselineCompare])

  const consensusLesions = active?.results?.consensus_summary?.lesions ?? []
  const clinicalAnalysis = active?.results?.clinical_analysis
  const displaySeverity = active?.severity ?? clinicalAnalysis?.clinical_severity ?? 'Unknown'
  const displayGags = active?.gags_score ?? clinicalAnalysis?.gags_total_score ?? 0
  const displayLesions = active?.lesion_count ?? clinicalAnalysis?.total_lesions ?? 0
  const displaySymmetry = active?.symmetry_delta ?? clinicalAnalysis?.symmetry_delta ?? 0
  const severityTone = getSeverityTone(displaySeverity)
  const regionRows = (Object.entries(clinicalAnalysis?.regions ?? {}) as [string, RegionStats][]).sort(
    (a, b) => (b[1].gags_score ?? 0) - (a[1].gags_score ?? 0),
  )
  const activeImage = showDetectionOverlay ? active?.diagnostic_image ?? active?.original_image ?? '' : active?.original_image ?? active?.diagnostic_image ?? ''
  const previousImage = showCompareOverlay
    ? previousSession?.diagnostic_image ?? previousSession?.original_image ?? ''
    : previousSession?.original_image ?? previousSession?.diagnostic_image ?? ''
  const lesionOverlayItems = consensusLesions.map((lesion, index) => ({
    ...lesion,
    key: `${lesion.region}-${index}`,
  }))
  const filteredLesionOverlayItems = lesionOverlayItems.filter((lesion) => {
    const regionPass = lesionRegionFilter === 'all' || lesion.region === lesionRegionFilter
    const confidencePass =
      lesionConfidenceFilter === 'all'
      || (lesionConfidenceFilter === 'high' && lesion.confidence >= 0.7)
      || (lesionConfidenceFilter === 'medium' && lesion.confidence >= 0.4 && lesion.confidence < 0.7)
      || (lesionConfidenceFilter === 'low' && lesion.confidence < 0.4)
    return regionPass && confidencePass
  })
  const lesionRegions = Array.from(new Set(lesionOverlayItems.map((lesion) => lesion.region))).sort()
  const activeProfileSummary = profiles.find((profile) => profile.profile_id === activeProfileId) ?? null
  const activeSessionProfileId = active?.profile_id
    ?? history.find((item) => item.session_id === active?.session_id)?.profile_id
    ?? DEFAULT_PROFILE_ID

  const refreshHistory = async () => {
    const [items, profileItems] = await Promise.all([api.getHistory(30, activeProfileId), api.getProfiles()])
    setHistory(items)
    setProfiles(profileItems)
  }

  const loadCompareContext = async (
    currentSessionId: string,
    options?: {
      previousSessionId?: string | null
      fallbackCompare?: ComparePayload
    },
  ) => {
    const requestedPreviousSessionId = options?.previousSessionId
    let nextCompare = options?.fallbackCompare ?? null

    if (requestedPreviousSessionId) {
      nextCompare = requestedPreviousSessionId === currentSessionId
        ? null
        : await api.getCompare(currentSessionId, requestedPreviousSessionId)
    } else if (!nextCompare) {
      nextCompare = await api.getCompare(currentSessionId)
    }

    setCompare(nextCompare)

    if (nextCompare?.previous_session_id) {
      const previous = await api.getSession(nextCompare.previous_session_id)
      setPreviousSession(previous)
    } else {
      setPreviousSession(null)
    }

    return nextCompare
  }

  const handleFileChange = (event: import('react').ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null
    setSelectedFile(file)
    setPreviewUrl((oldUrl) => {
      if (oldUrl) URL.revokeObjectURL(oldUrl)
      return file ? URL.createObjectURL(file) : null
    })
    setError(null)
  }

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleStart = async () => {
    if (!selectedFile) return
    try {
      setError(null)
      setIsAnalyzing(true)
      setStatus({
        stage: 'warming_up',
        detail: 'Booting segmentation + cloud engines for first analysis',
        progress: 3,
      })
      const start = await api.startAnalysis({
        profile_id: activeProfileId,
        privacy_mode: privacyMode,
        retention_hours: retentionHours,
      })
      const resolvedProfileId = start.profile_id ?? activeProfileId
      setSessionId(start.session_id)
      if (resolvedProfileId !== activeProfileId) {
        setActiveProfileId(resolvedProfileId)
      }
      setStatus(start.status)

      const form = new FormData()
      form.append('file', selectedFile)
      form.append('session_id', start.session_id)
      form.append('profile_id', resolvedProfileId)
      form.append('privacy_mode', String(privacyMode))
      form.append('retention_hours', String(retentionHours))

      const result = await api.analyze(form)
      setActive(result)
      await loadCompareContext(result.session_id, {
        previousSessionId: baselineSession?.session_id,
        fallbackCompare: result.compare,
      })
      setStatus(result.status)
      setShowDetectionOverlay(true)
      setCompareViewerMode('single')
      setActiveLesionKey(null)
      setProfileGuard(null)
      await refreshHistory()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleSelectHistory = async (item: SessionSummary) => {
    try {
      setIsLoadingSession(true)
      const session = await api.getSession(item.session_id)
      const sessionProfileId = item.profile_id ?? DEFAULT_PROFILE_ID
      if (sessionProfileId !== activeProfileId) {
        setActiveProfileId(sessionProfileId)
      }
      const comparePayload = await loadCompareContext(item.session_id, {
        previousSessionId: baselineSession?.session_id,
      })
      setSessionId(item.session_id)
      setActive({
        session_id: session.session_id,
        status: session.status ?? { stage: 'completed', detail: 'Loaded from archive', progress: 100 },
        severity: session.severity,
        gags_score: session.gags_score,
        lesion_count: session.lesion_count,
        symmetry_delta: session.symmetry_delta,
        results: session.results ?? {},
        compare: comparePayload,
        diagnostic_image: session.diagnostic_image ?? null,
        original_image: session.original_image ?? null,
      })
      setNoteDraft(session.note ?? '')
      setStatus(session.status ?? null)
      setShowDetectionOverlay(true)
      setCompareViewerMode('single')
      setActiveLesionKey(null)
      setProfileGuard(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load session')
    } finally {
      setIsLoadingSession(false)
    }
  }

  const handlePurge = async (item: SessionSummary) => {
    if (!window.confirm('Permanently delete this session? This cannot be undone.')) return
    try {
      setIsPurging(true)
      await api.purgeSession(item.session_id)
      if (active?.session_id === item.session_id) {
        setActive(null)
        setCompare(null)
        setPreviousSession(null)
        setSessionId(null)
        setStatus(null)
      }
      await refreshHistory()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to purge session')
    } finally {
      setIsPurging(false)
    }
  }

  const resetCaseWorkspace = () => {
    setActive(null)
    setCompare(null)
    setPreviousSession(null)
    setSelectedFile(null)
    setPreviewUrl(null)
    setSessionId(null)
    setStatus(null)
    setNoteDraft('')
    setActiveLesionKey(null)
    setProfileGuard(null)
  }

  const handleExport = async () => {
    if (!active?.session_id) return
    try {
      setIsExporting(true)
      const bundle = await api.exportBundle(active.session_id, exportPreset, compare?.previous_session_id)
      if (bundle.pdf_data_uri) {
        window.open(bundle.pdf_data_uri, '_blank')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  const pinBaseline = async (item: SessionSummary) => {
    setBaselineSession(item)
    writeStorageItem(getProfileStorageKey(BASELINE_SESSION_KEY, activeProfileId), item.session_id)

    if (active?.session_id) {
      try {
        await loadCompareContext(active.session_id, { previousSessionId: item.session_id })
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load baseline comparison')
      }
    }
  }

  const clearBaseline = async () => {
    setBaselineSession(null)
    removeStorageItem(getProfileStorageKey(BASELINE_SESSION_KEY, activeProfileId))

    if (active?.session_id) {
      try {
        await loadCompareContext(active.session_id)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to restore default comparison')
      }
    }
  }

  const dismissOnboarding = () => {
    setShowOnboarding(false)
    writeStorageItem(ONBOARDING_SEEN_KEY, 'true')
  }

  const saveNotes = async () => {
    if (!active?.session_id) return
    try {
      setIsSavingNotes(true)
      await api.updateSessionNotes(active.session_id, noteDraft)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save notes')
    } finally {
      setIsSavingNotes(false)
    }
  }

  const resetMainViewer = () => setMainViewer({ scale: 1, x: 0, y: 0 })
  const resetCompareViewer = () => setCompareViewer({ scale: 1, x: 0, y: 0 })

  const toggleFullscreen = async () => {
    if (!workspaceRef.current) return
    if (!document.fullscreenElement) {
      await workspaceRef.current.requestFullscreen()
      setIsFullscreen(true)
    } else {
      await document.exitFullscreen()
      setIsFullscreen(false)
    }
  }

  useEffect(() => {
    const onFullScreenChange = () => setIsFullscreen(Boolean(document.fullscreenElement))
    document.addEventListener('fullscreenchange', onFullScreenChange)
    return () => document.removeEventListener('fullscreenchange', onFullScreenChange)
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable) return
      if (!active) return
      if (event.key === '+') setMainViewer((state) => ({ ...state, scale: clamp(state.scale + 0.25, 1, 4) }))
      if (event.key === '-') setMainViewer((state) => ({ ...state, scale: clamp(state.scale - 0.25, 1, 4) }))
      if (event.key.toLowerCase() === 'r') {
        resetMainViewer()
        resetCompareViewer()
      }
      if (event.key.toLowerCase() === 'o') setShowDetectionOverlay((value) => !value)
      if (event.key.toLowerCase() === 'c' && compare) setCompareViewerMode((mode) => (mode === 'single' ? 'split' : 'single'))
      if (event.key.toLowerCase() === 'f') {
        event.preventDefault()
        void toggleFullscreen()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [active, compare])

  return (
    <section ref={workspaceRef} className="medical-grid relative bg-black py-32">
      <div className="mx-auto max-w-[1600px] px-8">
        <div className="mb-16 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="terminal-text mb-4 text-[10px] tracking-[0.35em] text-cyan-400/70">CLINICAL WORKSPACE</p>
            <h2 className="text-5xl font-bold tracking-tighter md:text-6xl">Neural Archive Workspace</h2>
            <p className="mt-4 max-w-2xl text-lg text-zinc-500">
              Longitudinal diagnosis, consensus inspection, privacy controls, and case comparison in a single clinical shell.
            </p>
            <div className="mt-5 flex flex-wrap items-center gap-3 text-xs">
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/15 bg-cyan-400/8 px-3 py-2 text-cyan-100">
                <span className="terminal-text text-[9px] tracking-[0.24em] text-cyan-400/80">ACTIVE PROFILE</span>
                <span className="font-medium text-white">{activeProfileId}</span>
              </div>
              {activeProfileSummary ? (
                <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-zinc-300">
                  <span>{activeProfileSummary.sessions} archived sessions</span>
                  <span className="text-zinc-500">latest {activeProfileSummary.latest_severity ?? 'pending'}</span>
                </div>
              ) : null}
            </div>
          </div>

          <div className="holographic-panel flex items-center gap-4 rounded-2xl px-6 py-4">
            <Shield className="h-5 w-5 text-cyan-400" />
            <div>
              <div className="terminal-text text-[9px] text-cyan-400/80">RETENTION WINDOW</div>
              <div className="text-sm text-zinc-300">{retentionHours} hours</div>
            </div>
          </div>
        </div>

        {profileSwitchNotice ? (
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="pointer-events-none absolute right-8 top-10 z-20 rounded-2xl border border-cyan-400/15 bg-black/75 px-4 py-3 shadow-[0_0_30px_rgba(0,242,255,0.08)] backdrop-blur"
          >
            <div className="terminal-text text-[9px] tracking-[0.3em] text-cyan-400/80">PROFILE SWITCHED</div>
            <div className="mt-2 text-sm text-zinc-300">
              Workspace moved from <span className="text-zinc-500">{profileSwitchNotice.from}</span> to <span className="text-white">{profileSwitchNotice.to}</span>
            </div>
          </motion.div>
        ) : null}

        {profileGuard ? (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 flex flex-col gap-4 rounded-[1.5rem] border border-amber-400/20 bg-amber-400/8 px-5 py-4 text-sm text-amber-50 shadow-[0_0_30px_rgba(251,191,36,0.08)] lg:flex-row lg:items-center lg:justify-between"
          >
            <div>
              <div className="terminal-text text-[9px] tracking-[0.28em] text-amber-200/80">ACTIVE CASE PROFILE MISMATCH</div>
              <p className="mt-2 max-w-3xl text-amber-50/90">
                Session {profileGuard.sessionId.slice(0, 8)} belongs to profile {profileGuard.sessionProfileId}. You switched the archive to {profileGuard.requestedProfileId}. Reset the current case or return to the matching profile archive.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <button
                onClick={() => {
                  setActiveProfileId(profileGuard.sessionProfileId)
                  setProfileGuard(null)
                }}
                className="rounded-full border border-amber-200/30 px-4 py-2 text-[11px] font-medium text-amber-50 hover:bg-amber-200/10"
              >
                Return to {profileGuard.sessionProfileId}
              </button>
              <button
                onClick={resetCaseWorkspace}
                className="rounded-full bg-amber-300 px-4 py-2 text-[11px] font-semibold text-black"
              >
                Reset active case
              </button>
            </div>
          </motion.div>
        ) : null}

        <div className="grid grid-cols-1 gap-8 xl:grid-cols-[minmax(260px,var(--left-rail))_minmax(0,1fr)_minmax(320px,var(--right-rail))]" style={{ ['--left-rail' as string]: `${leftRailWidth}px`, ['--right-rail' as string]: `${rightRailWidth}px` }}>
          <aside className="holographic-panel rounded-[2rem] p-6 relative">
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
                onChange={(e) => {
                  const nextProfileId = e.target.value
                  setActiveProfileId(nextProfileId)
                  if (active?.session_id && activeSessionProfileId !== nextProfileId) {
                    setProfileGuard({
                      sessionId: active.session_id,
                      sessionProfileId: activeSessionProfileId,
                      requestedProfileId: nextProfileId,
                    })
                  } else {
                    setProfileGuard(null)
                  }
                }}
                className="w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
              >
                {(profiles.length ? profiles : [{ profile_id: DEFAULT_PROFILE_ID, sessions: 0, latest_timestamp: '', latest_severity: null }]).map((profile) => (
                  <option key={profile.profile_id} value={profile.profile_id}>
                    {profile.profile_id} · {profile.sessions} sessions
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-3 overflow-y-auto pr-2 max-h-[760px]">
              {history.map((item) => (
                <button
                  key={item.session_id}
                  onClick={() => void handleSelectHistory(item)}
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
                        void handlePurge(item)
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
                        pinBaseline(item)
                      }}
                      className={`rounded-full px-2 py-1 text-[10px] ${baselineSession?.session_id === item.session_id ? 'bg-cyan-400 text-black' : 'border border-white/10 text-zinc-400 hover:border-cyan-400/20 hover:text-white'}`}
                    >
                      {baselineSession?.session_id === item.session_id ? 'Baseline pinned' : 'Pin baseline'}
                    </button>
                  </div>
                </button>
              ))}
            </div>
            <input
              type="range"
              min={260}
              max={420}
              value={leftRailWidth}
              onChange={(e) => setLeftRailWidth(Number(e.target.value))}
              className="mt-4 w-full accent-cyan-400"
            />
          </aside>

          <main className="holographic-panel rounded-[2rem] p-8">
            <div className="mb-8 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <div className="terminal-text text-[10px] text-cyan-400/80">LIVE DIAGNOSTIC CANVAS</div>
                <div className="text-sm text-zinc-500">
                  {status ? `${status.stage} · ${status.detail}` : 'Awaiting upload'}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <label className="inline-flex items-center gap-2 text-sm text-zinc-400">
                  <input type="checkbox" checked={privacyMode} onChange={(e) => setPrivacyMode(e.target.checked)} />
                  Privacy mode
                </label>
                <label className="inline-flex items-center gap-2 text-sm text-zinc-400">
                  Retention
                  <input
                    type="number"
                    min={1}
                    max={privacy?.max_retention_hours ?? 720}
                    value={retentionHours}
                    onChange={(e) => setRetentionHours(Number(e.target.value))}
                    className="w-20 rounded-lg border border-white/10 bg-black/40 px-2 py-1 text-sm text-white"
                  />
                </label>
              </div>
            </div>

            {!active ? (
              <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1.2fr_0.8fr]">
                <div className="rounded-[2rem] border border-dashed border-white/10 bg-white/3 p-8">
                  <label className="group block cursor-pointer overflow-hidden rounded-[1.5rem] border border-white/5 bg-black/30 p-8 text-center transition-all hover:border-cyan-400/25 hover:bg-cyan-400/5">
                    <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} disabled={isAnalyzing} />
                    {previewUrl ? (
                      <div className="relative overflow-hidden rounded-[1.5rem]">
                        <img src={previewUrl} alt="Preview" className="h-[420px] w-full object-cover opacity-60" />
                        {isAnalyzing && (
                          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0 bg-cyan-400/8 backdrop-blur-[1px]">
                            <div className="scanner-line opacity-60" />
                            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                              <Loader2 className="h-10 w-10 animate-spin text-cyan-400" />
                              <div className="terminal-text text-[10px] tracking-[0.3em] text-cyan-400">{status?.detail ?? 'PROCESSING'}</div>
                              <div className="h-2 w-64 overflow-hidden rounded-full bg-white/10">
                                <div className="h-full bg-cyan-400 transition-all" style={{ width: `${status?.progress ?? 0}%` }} />
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </div>
                    ) : (
                      <div className="flex h-[420px] flex-col items-center justify-center gap-4 rounded-[1.5rem] border border-dashed border-white/10 bg-black/20">
                        <Camera className="h-10 w-10 text-zinc-600 transition-colors group-hover:text-cyan-400" />
                        <div className="terminal-text text-[11px] text-zinc-500">SELECT_CLINICAL_IMAGE</div>
                      </div>
                    )}
                  </label>

                  <div className="mt-6 flex items-center gap-4">
                    <button
                      onClick={() => void handleStart()}
                      disabled={!selectedFile || isAnalyzing}
                      className="terminal-text flex-1 bg-cyan-400 px-6 py-4 text-[10px] font-bold text-black transition-all hover:tracking-[0.2em] disabled:cursor-not-allowed disabled:opacity-30"
                    >
                      {isAnalyzing ? 'PROCESSING...' : 'EXECUTE_DIAGNOSTIC'}
                    </button>
                  </div>

                  <div className="mt-4 rounded-2xl border border-cyan-400/10 bg-cyan-400/5 px-4 py-3 text-sm text-zinc-400">
                    First run may take longer while the clinical engine warms up. Live progress will begin as soon as the backend finishes lazy initialization.
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="holographic-panel rounded-[1.75rem] p-6">
                    <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">PRIVACY CONSOLE</div>
                    <div className="space-y-3 text-sm text-zinc-400">
                      <div>Original uploads: {privacyMode ? 'purged after run' : 'retained for archive compare'}</div>
                      <div>Retention window: {retentionHours} hours</div>
                      <div>Stored fields: {privacy?.stored_fields?.length ?? 0}</div>
                    </div>
                  </div>

                  <div className="holographic-panel rounded-[1.75rem] p-6">
                    <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">EXPLAINABILITY PREVIEW</div>
                    <p className="text-sm leading-relaxed text-zinc-500">
                      Every completed case unlocks per-region lesion burden, consensus confidence, GAGS contributions, and longitudinal delta analysis.
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-8">
                <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
                  <MetricCard label="SEVERITY" value={displaySeverity} accent tone={severityTone} />
                  <MetricCard label="GAGS SCORE" value={String(displayGags)} accent />
                  <MetricCard label="LESIONS" value={String(displayLesions)} />
                  <MetricCard label="SYMMETRY" value={`${displaySymmetry}%`} />
                </div>

                <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.3fr_0.7fr]">
                  <div className="rounded-[1.75rem] border border-white/5 bg-black/30 p-4">
                    <div className="mb-4 flex items-center justify-between">
                      <div className="terminal-text text-[10px] text-cyan-400/80">ACNE DETECTION CANVAS</div>
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          onClick={() => setShowDetectionOverlay((value) => !value)}
                          className="inline-flex items-center gap-2 rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
                        >
                          {showDetectionOverlay ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                          {showDetectionOverlay ? 'Detection overlay' : 'Original image'}
                        </button>
                        <button onClick={() => setMainViewer((state) => ({ ...state, scale: clamp(state.scale + 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                          <Search className="h-4 w-4" />
                        </button>
                        <button onClick={() => setMainViewer((state) => ({ ...state, scale: clamp(state.scale - 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                          <SearchX className="h-4 w-4" />
                        </button>
                        <button onClick={resetMainViewer} className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                          Reset
                        </button>
                        <button onClick={() => void toggleFullscreen()} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
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
                      onChange={setMainViewer}
                      heightClass="h-[520px]"
                      lesions={showDetectionOverlay ? lesionOverlayItems : []}
                      activeLesionKey={activeLesionKey}
                      onLesionHover={setActiveLesionKey}
                    />
                  </div>

                  <div className="space-y-6">
                    <div className="holographic-panel rounded-[1.75rem] p-6">
                      <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">CLINICAL GRADE</div>
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

                    <div className="holographic-panel rounded-[1.75rem] p-6">
                      <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">HOW GAGS WAS CALCULATED</div>
                      <div className="space-y-3 text-sm text-zinc-400">
                        <div className="rounded-xl border border-white/5 bg-white/3 px-4 py-3">
                          Total score is built from region-level lesion burden mapped onto anatomical zones and summed into a clinical severity band.
                        </div>
                        <div className="grid grid-cols-1 gap-2">
                          <div className="flex items-center justify-between rounded-xl border border-white/5 bg-black/20 px-3 py-2 text-xs uppercase tracking-[0.2em] text-zinc-500">
                            <span>Region</span>
                            <span>Lesions</span>
                            <span>Regional GAGS</span>
                          </div>
                          {regionRows.map(([region, data]) => {
                            const regionData = data as RegionStats
                            return (
                              <div key={`gags-${region}`} className="grid grid-cols-[1fr_auto_auto] items-center gap-3 rounded-xl border border-white/5 bg-white/3 px-3 py-3 text-sm">
                                <span className="text-zinc-300">{region.replaceAll('_', ' ')}</span>
                                <span className="text-zinc-500">{regionData.count}</span>
                                <span className="font-semibold text-cyan-300">{regionData.gags_score ?? 0}</span>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    </div>

                    <div className="holographic-panel rounded-[1.75rem] p-6">
                      <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">CONSENSUS INSPECTOR</div>
                      <p className="mb-4 text-sm text-zinc-500">{active.results?.consensus_summary?.summary ?? 'No consensus data available'}</p>
                      <div className="space-y-2 text-sm text-zinc-300">
                        <div>Verified lesions: {active.results?.consensus_summary?.verified_lesions ?? 0}</div>
                        <div>Average confidence: {active.results?.consensus_summary?.average_confidence ?? 0}</div>
                        <div>Top regions: {(active.results?.consensus_summary?.top_regions ?? []).map((item) => item.region).join(', ') || 'n/a'}</div>
                      </div>
                    </div>

                    <div className="holographic-panel rounded-[1.75rem] p-6">
                      <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">WHY THIS SEVERITY?</div>
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

                    <div className="holographic-panel rounded-[1.75rem] p-6">
                      <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">SOURCE STREAM DEBUGGER</div>
                      <div className="space-y-3 text-sm text-zinc-400">
                        <div>Strongest stream: {String((active.results?.source_stream_provenance as { strongest_stream?: string } | undefined)?.strongest_stream ?? 'n/a')}</div>
                        <div>Total source proposals: {String((active.results?.source_stream_provenance as { stream_total?: number } | undefined)?.stream_total ?? 0)}</div>
                        <div className="grid grid-cols-1 gap-2">
                          {Object.entries(((active.results?.source_stream_provenance as { streams?: Record<string, number> } | undefined)?.streams ?? {})).map(([name, count]) => (
                            <div key={name} className="flex items-center justify-between rounded-xl border border-white/5 bg-white/3 px-3 py-2">
                              <span>{name}</span>
                              <span className="font-medium text-white">{count}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {compare && compareViewerMode === 'single' && !compareFullscreen && (
                  <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                    <div className="rounded-[1.75rem] border border-white/5 bg-black/30 p-4">
                      <div className="mb-3 flex items-center justify-between">
                        <div className="terminal-text text-[10px] text-cyan-400/80">VISUAL COMPARE · CURRENT</div>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setShowCompareOverlay((value) => !value)}
                            className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
                          >
                            {showCompareOverlay ? 'Detection pair' : 'Original pair'}
                          </button>
                          <button onClick={() => setCompareViewer((state) => ({ ...state, scale: clamp(state.scale + 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                            <Search className="h-4 w-4" />
                          </button>
                          <button onClick={() => setCompareViewer((state) => ({ ...state, scale: clamp(state.scale - 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                            <SearchX className="h-4 w-4" />
                          </button>
                          <button onClick={resetCompareViewer} className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                            Reset
                          </button>
                          <button
                            onClick={() => setCompareViewerMode('split')}
                            className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
                          >
                            Split view
                          </button>
                          <button
                            onClick={() => setCompareFullscreen(true)}
                            className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
                          >
                            Fullscreen compare
                          </button>
                        </div>
                      </div>
                      <AdvancedImageViewer
                        src={showCompareOverlay ? active.diagnostic_image ?? active.original_image ?? '' : active.original_image ?? active.diagnostic_image ?? ''}
                        alt="Current acne detection"
                        state={compareViewer}
                        onChange={setCompareViewer}
                        heightClass="h-[420px]"
                        lesions={showCompareOverlay ? lesionOverlayItems : []}
                        activeLesionKey={activeLesionKey}
                        onLesionHover={setActiveLesionKey}
                      />
                    </div>
                    <div className="rounded-[1.75rem] border border-white/5 bg-black/30 p-4">
                      <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">VISUAL COMPARE · PRIOR</div>
                      {previousSession?.diagnostic_image || previousSession?.original_image ? (
                        <AdvancedImageViewer
                          src={showCompareOverlay ? previousSession.diagnostic_image ?? previousSession.original_image ?? '' : previousSession.original_image ?? previousSession.diagnostic_image ?? ''}
                          alt="Previous acne detection"
                          state={compareViewer}
                          onChange={setCompareViewer}
                          heightClass="h-[420px]"
                        />
                      ) : (
                        <div className="flex h-[420px] items-center justify-center rounded-[1.25rem] border border-dashed border-white/10 bg-black/20 text-sm text-zinc-500">
                          No previous diagnostic image available.
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {compare && compareViewerMode === 'split' && previousImage && !compareFullscreen && (
                  <div className="rounded-[1.75rem] border border-white/5 bg-black/30 p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <div className="terminal-text text-[10px] text-cyan-400/80">SPLIT COMPARE VIEWER</div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setCompareViewerMode('single')}
                          className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white"
                        >
                          Side by side
                        </button>
                        <button onClick={resetCompareViewer} className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                          Reset
                        </button>
                        <button
                          onClick={() => setCompareFullscreen(true)}
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
                      onChange={setCompareViewer}
                    />
                  </div>
                )}

                {compare && compareFullscreen && (
                  <div className="rounded-[1.75rem] border border-cyan-400/15 bg-black/40 p-4 shadow-[0_0_40px_rgba(0,242,255,0.08)]">
                    <div className="mb-3 flex items-center justify-between">
                      <div className="terminal-text text-[10px] text-cyan-400/80">FULLSCREEN COMPARE WORKSPACE</div>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => setCompareFullscreen(false)}
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
                        onChange={setCompareViewer}
                        heightClass="h-[640px]"
                        lesions={showCompareOverlay ? lesionOverlayItems : []}
                        activeLesionKey={activeLesionKey}
                        onLesionHover={setActiveLesionKey}
                      />
                      {previousImage ? (
                        <AdvancedImageViewer
                          src={previousImage}
                          alt="Previous compare workspace"
                          state={compareViewer}
                          onChange={setCompareViewer}
                          heightClass="h-[640px]"
                        />
                      ) : (
                        <div className="flex h-[640px] items-center justify-center rounded-[1.25rem] border border-dashed border-white/10 bg-black/20 text-sm text-zinc-500">
                          No previous image available.
                        </div>
                      )}
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                  <div className="holographic-panel rounded-[1.75rem] p-6">
                    <div className="mb-4 flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-cyan-400" />
                      <div className="terminal-text text-[10px] text-cyan-400/80">CASE COMPARE</div>
                    </div>
                    {compare ? (
                      <>
                        <div className="mb-4 rounded-[1.25rem] border border-cyan-400/15 bg-cyan-400/6 p-4">
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div>
                              <div className="terminal-text text-[9px] text-cyan-400/80">
                                {isPinnedBaselineCompare ? 'PINNED BASELINE DELTA' : 'LONGITUDINAL DELTA'}
                              </div>
                              <p className="mt-2 text-sm text-zinc-300">
                                {compareNarrative}
                              </p>
                              <div className="mt-2 text-xs text-zinc-500">
                                Comparing current session {compare.current_session_id.slice(0, 8)} against {compare.previous_session_id.slice(0, 8)}.
                              </div>
                            </div>
                            {baselineSession ? (
                              <button onClick={() => void clearBaseline()} className="rounded-full border border-cyan-400/30 px-3 py-1 text-[10px] text-cyan-100 hover:bg-cyan-400/10">
                                Clear baseline
                              </button>
                            ) : null}
                          </div>

                          <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-4">
                            <div className={`rounded-xl border px-3 py-3 ${getDeltaTone(compare.lesion_delta, true)}`}>
                              <div className="terminal-text text-[8px] text-current/70">LESION DELTA</div>
                              <div className="mt-2 text-lg font-semibold">{formatSignedDelta(compare.lesion_delta)}</div>
                              <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-current/75">{getClinicalDeltaLabel(compare.lesion_delta, true)}</div>
                            </div>
                            <div className={`rounded-xl border px-3 py-3 ${getDeltaTone(compare.gags_delta, true)}`}>
                              <div className="terminal-text text-[8px] text-current/70">GAGS DELTA</div>
                              <div className="mt-2 text-lg font-semibold">{formatSignedDelta(compare.gags_delta)}</div>
                              <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-current/75">{getClinicalDeltaLabel(compare.gags_delta, true)}</div>
                            </div>
                            <div className={`rounded-xl border px-3 py-3 ${getDeltaTone(compare.symmetry_delta_change, true)}`}>
                              <div className="terminal-text text-[8px] text-current/70">SYMMETRY SHIFT</div>
                              <div className="mt-2 text-lg font-semibold">{formatSignedDelta(compare.symmetry_delta_change, '%')}</div>
                              <div className="mt-1 text-[11px] uppercase tracking-[0.24em] text-current/75">{getClinicalDeltaLabel(compare.symmetry_delta_change, true)}</div>
                            </div>
                            <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-3 text-zinc-200">
                              <div className="terminal-text text-[8px] text-zinc-500">SEVERITY BAND</div>
                              <div className="mt-2 text-sm font-semibold text-white">
                                {compare.severity_change.from} {'->'} {compare.severity_change.to}
                              </div>
                            </div>
                          </div>

                          {regionDeltaCards.length > 0 ? (
                            <div className="mt-4 flex flex-wrap gap-2 text-xs text-zinc-300">
                              {regionDeltaCards.slice(0, 3).map(([region, values]) => (
                                <span key={`baseline-chip-${region}`} className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                                  {region.replaceAll('_', ' ')} {formatSignedDelta(values.count_delta)} lesions
                                </span>
                              ))}
                            </div>
                          ) : null}
                        </div>
                        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                          {regionDeltaCards.map(([region, values]) => (
                            <div key={region} className="rounded-xl border border-white/5 bg-white/3 p-4">
                              <div className="terminal-text mb-2 text-[9px] text-zinc-500">{region}</div>
                              <div className="text-sm text-white">count delta {values.count_delta >= 0 ? '+' : ''}{values.count_delta}</div>
                              <div className="text-xs text-zinc-500">lpi delta {values.lpi_delta >= 0 ? '+' : ''}{values.lpi_delta}</div>
                            </div>
                          ))}
                        </div>
                      </>
                    ) : (
                      <div className="space-y-3 text-sm text-zinc-500">
                        {isViewingPinnedBaseline ? (
                          <div className="rounded-xl border border-cyan-400/15 bg-cyan-400/8 px-4 py-3 text-cyan-100">
                            This session is the pinned baseline. Open another session to generate a baseline delta callout.
                          </div>
                        ) : null}
                        <p>No previous session is available for comparison yet.</p>
                      </div>
                    )}
                  </div>

                  <div className="holographic-panel rounded-[1.75rem] p-6">
                    <div className="mb-4 flex items-center gap-2">
                      <Shield className="h-4 w-4 text-cyan-400" />
                      <div className="terminal-text text-[10px] text-cyan-400/80">PRIVACY + EXPORT</div>
                    </div>
                    <div className="space-y-3 text-sm text-zinc-400">
                      <div>Privacy mode: {privacyMode ? 'enabled' : 'disabled'}</div>
                      <div>Retention: {retentionHours} hours</div>
                      <div>Current session: {active.session_id}</div>
                    </div>
                    <div className="mt-4 space-y-2">
                      <div className="terminal-text text-[9px] text-zinc-500">EXPORT PRESET</div>
                      <select
                        value={exportPreset}
                        onChange={(e) => setExportPreset(e.target.value as 'clinical' | 'compact' | 'presentation')}
                        className="w-full rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
                      >
                        <option value="clinical">Clinical</option>
                        <option value="compact">Compact</option>
                        <option value="presentation">Presentation</option>
                      </select>
                    </div>
                    <div className="mt-6 flex flex-wrap gap-3">
                      <button onClick={() => void handleExport()} disabled={isExporting} className="terminal-text rounded-full border border-white/10 px-4 py-2 text-[10px] text-white transition-colors hover:border-cyan-400/20 hover:text-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed">
                        {isExporting ? 'EXPORTING...' : `EXPORT ${exportPreset.toUpperCase()}`}
                      </button>
                      <button
                        onClick={resetCaseWorkspace}
                        className="terminal-text rounded-full bg-cyan-400 px-4 py-2 text-[10px] text-black"
                      >
                        NEW CASE
                      </button>
                    </div>
                  </div>
                </div>

                {consensusLesions.length > 0 && (
                  <div className="holographic-panel rounded-[1.75rem] p-6">
                    <div className="terminal-text mb-4 text-[10px] text-cyan-400/80">CONSENSUS LESION TABLE</div>
                    <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-2">
                      <select
                        value={lesionRegionFilter}
                        onChange={(e) => setLesionRegionFilter(e.target.value)}
                        className="rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
                      >
                        <option value="all">All regions</option>
                        {lesionRegions.map((region) => (
                          <option key={region} value={region}>{region}</option>
                        ))}
                      </select>
                      <select
                        value={lesionConfidenceFilter}
                        onChange={(e) => setLesionConfidenceFilter(e.target.value as 'all' | 'high' | 'medium' | 'low')}
                        className="rounded-xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
                      >
                        <option value="all">All confidence tiers</option>
                        <option value="high">High confidence</option>
                        <option value="medium">Medium confidence</option>
                        <option value="low">Low / review</option>
                      </select>
                    </div>
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                      {filteredLesionOverlayItems.slice(0, 12).map((lesion) => (
                        <button
                          key={lesion.key}
                          onMouseEnter={() => setActiveLesionKey(lesion.key)}
                          onMouseLeave={() => setActiveLesionKey(null)}
                          className={`rounded-xl border bg-white/3 p-4 text-left text-sm transition-all ${activeLesionKey === lesion.key ? 'border-cyan-400/40 bg-cyan-400/8 shadow-[0_0_25px_rgba(0,242,255,0.12)]' : 'border-white/5'}`}
                        >
                          <div className="mb-2 flex items-center justify-between">
                            <span className="font-medium text-white">{lesion.region}</span>
                            <span className={`terminal-text text-[8px] ${getConfidenceTextTone(lesion.confidence)}`}>{lesion.confidence_level}</span>
                          </div>
                          <div className="text-zinc-400">confidence {lesion.confidence}</div>
                          <div className="text-zinc-500">votes {lesion.votes} · reliability {lesion.reliability_score}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div className="holographic-panel rounded-[1.75rem] p-6">
                  <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">SESSION NOTES</div>
                  <textarea
                    value={noteDraft}
                    onChange={(e) => setNoteDraft(e.target.value)}
                    placeholder="Add dermatologist notes, case observations, or treatment context..."
                    className="min-h-[140px] w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white outline-none"
                  />
                  <div className="mt-4 flex items-center justify-between">
                    <div className="text-xs text-zinc-500">Notes persist with this session.</div>
                    <button onClick={() => void saveNotes()} disabled={isSavingNotes} className="rounded-full border border-white/10 px-4 py-2 text-xs text-white transition-colors hover:border-cyan-400/20 hover:text-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed">
                      {isSavingNotes ? 'Saving...' : 'Save notes'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {error && <div className="mt-6 rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}

            {showOnboarding && (
              <div className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm">
                <div className="holographic-panel max-w-2xl rounded-[2rem] p-8">
                  <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">WORKSPACE ONBOARDING</div>
                  <h3 className="mb-4 text-3xl font-bold tracking-tight">How to use the clinical workspace</h3>
                  <div className="space-y-3 text-sm leading-relaxed text-zinc-400">
                    <div>1. Upload a case and wait for the first warmup analysis to complete.</div>
                    <div>2. Use `+`, `-`, `R`, `O`, `C`, and `F` for quick viewer controls.</div>
                    <div>3. Pin a baseline session from the archive to compare future sessions against it.</div>
                    <div>4. Hover lesion cards to highlight matching detections in the viewer.</div>
                    <div>5. Save session notes and export a clinical, compact, or presentation report.</div>
                  </div>
                  <div className="mt-6 flex justify-end">
                    <button
                      onClick={dismissOnboarding}
                      className="rounded-full bg-cyan-400 px-5 py-2 text-sm font-semibold text-black"
                    >
                      Got it
                    </button>
                  </div>
                </div>
              </div>
            )}
          </main>

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
                onChange={(e) => setRightRailWidth(Number(e.target.value))}
                className="mt-4 w-full accent-cyan-400"
              />
            </div>
          </aside>
        </div>
      </div>
    </section>
  )
}

function MetricCard({ label, value, accent = false, tone }: { label: string; value: string; accent?: boolean; tone?: string }) {
  return (
    <div className={`rounded-[1.5rem] border border-white/5 bg-white/3 p-6 ${tone ?? ''}`}>
      <div className="terminal-text mb-2 text-[9px] text-zinc-500">{label}</div>
      <div className={accent ? 'text-3xl font-bold text-cyan-400' : 'text-3xl font-bold text-white'}>{value}</div>
    </div>
  )
}

function getConfidenceTextTone(confidence: number) {
  if (confidence >= 0.7) return 'text-cyan-300'
  if (confidence >= 0.4) return 'text-amber-300'
  return 'text-zinc-300'
}

function AdvancedImageViewer({
  src,
  alt,
  state,
  onChange,
  heightClass,
  lesions = [],
  activeLesionKey = null,
  onLesionHover,
}: {
  src: string
  alt: string
  state: ViewerState
  onChange: (next: ViewerState) => void
  heightClass: string
  lesions?: Array<ConsensusLesion & { key: string }>
  activeLesionKey?: string | null
  onLesionHover?: (key: string | null) => void
}) {
  const dragRef = useRef<{ x: number; y: number } | null>(null)
  const minimapDragRef = useRef(false)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [naturalDims, setNaturalDims] = useState<{ w: number; h: number } | null>(null)

  const startDrag = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (state.scale <= 1) return
    dragRef.current = { x: event.clientX - state.x, y: event.clientY - state.y }
  }

  const onMove = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!dragRef.current || state.scale <= 1) return
    onChange({
      ...state,
      x: event.clientX - dragRef.current.x,
      y: event.clientY - dragRef.current.y,
    })
  }

  const stopDrag = () => {
    dragRef.current = null
    minimapDragRef.current = false
  }

  const moveFromMinimap = (event: ReactMouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const relX = (event.clientX - rect.left) / rect.width
    const relY = (event.clientY - rect.top) / rect.height
    onChange({
      ...state,
      x: (0.5 - relX) * 120,
      y: (0.5 - relY) * 120,
    })
  }

  return (
    <div
      className={`relative overflow-hidden rounded-[1.25rem] border border-white/5 bg-black ${heightClass}`}
      onMouseMove={onMove}
      onMouseUp={stopDrag}
      onMouseLeave={stopDrag}
    >
      <div className="pointer-events-none absolute left-4 top-4 z-10 rounded-full border border-white/10 bg-black/50 px-3 py-1 text-[10px] uppercase tracking-[0.25em] text-zinc-400 backdrop-blur">
        zoom {state.scale.toFixed(2)}x
      </div>
      <div className="pointer-events-none absolute left-4 bottom-4 z-10 rounded-full border border-white/10 bg-black/50 px-3 py-1 text-[10px] uppercase tracking-[0.25em] text-zinc-400 backdrop-blur">
        {cursor ? `x ${cursor.x.toFixed(0)} · y ${cursor.y.toFixed(0)}` : 'x -- · y --'}
      </div>
      <div
        className="absolute bottom-4 right-4 z-10 h-24 w-24 overflow-hidden rounded-lg border border-white/10 bg-black/60 backdrop-blur"
        onMouseDown={() => {
          minimapDragRef.current = true
        }}
        onMouseMove={(event) => {
          if (minimapDragRef.current) moveFromMinimap(event)
        }}
        onMouseUp={stopDrag}
      >
        {src ? (
          <>
            <img src={src} alt="navigator" className="h-full w-full object-cover opacity-60" />
            <div
              className="absolute border border-cyan-300/80 bg-cyan-400/10"
              style={{
                left: `${clamp(50 - state.x / 8, 5, 75)}%`,
                top: `${clamp(50 - state.y / 8, 5, 75)}%`,
                width: `${clamp(100 / state.scale, 18, 100)}%`,
                height: `${clamp(100 / state.scale, 18, 100)}%`,
                transform: 'translate(-50%, -50%)',
              }}
            />
          </>
        ) : null}
      </div>
      {src ? (
        <div
          role="presentation"
          onMouseDown={startDrag}
          onDoubleClick={() => onChange({ scale: 1, x: 0, y: 0 })}
          className="flex h-full w-full cursor-grab items-center justify-center active:cursor-grabbing"
          onMouseMove={(event) => {
            onMove(event)
            const rect = event.currentTarget.getBoundingClientRect()
            setCursor({ x: event.clientX - rect.left, y: event.clientY - rect.top })
          }}
        >
          <div className="relative max-h-full max-w-full">
            <img
              src={src}
              alt={alt}
              ref={imgRef}
              className="max-h-full max-w-full object-contain select-none transition-transform duration-200"
              style={{ transform: `translate(${state.x}px, ${state.y}px) scale(${state.scale})` }}
              draggable={false}
              onLoad={(e) => {
                const img = e.currentTarget
                setNaturalDims({ w: img.naturalWidth, h: img.naturalHeight })
              }}
            />
            {lesions.length > 0 && naturalDims && (
              <div className="pointer-events-none absolute inset-0">
                {lesions.map((lesion) => {
                  const [x1, y1, x2, y2] = lesion.bbox
                  const pctLeft = (x1 / naturalDims.w) * 100
                  const pctTop = (y1 / naturalDims.h) * 100
                  const pctWidth = (Math.max(0, x2 - x1) / naturalDims.w) * 100
                  const pctHeight = (Math.max(0, y2 - y1) / naturalDims.h) * 100
                  const isActive = activeLesionKey === lesion.key
                  const tone = lesion.confidence >= 0.7
                    ? 'border-cyan-300/90 shadow-[0_0_20px_rgba(0,242,255,0.25)]'
                    : lesion.confidence >= 0.4
                      ? 'border-amber-300/90 shadow-[0_0_18px_rgba(245,158,11,0.18)]'
                      : 'border-white/70 shadow-[0_0_12px_rgba(255,255,255,0.10)]'

                  return (
                    <button
                      key={lesion.key}
                      type="button"
                      onMouseEnter={() => onLesionHover?.(lesion.key)}
                      onMouseLeave={() => onLesionHover?.(null)}
                      className={`pointer-events-auto absolute border transition-all ${tone} ${isActive ? 'scale-[1.02] bg-cyan-400/10' : 'bg-transparent'}`}
                      style={{
                        left: `${pctLeft}%`,
                        top: `${pctTop}%`,
                        width: `${pctWidth}%`,
                        height: `${pctHeight}%`,
                        transform: `translate(${state.x}px, ${state.y}px) scale(${state.scale})`,
                        transformOrigin: 'top left',
                      }}
                    >
                      <span className={`absolute -top-5 left-0 rounded bg-black/70 px-1 py-0.5 text-[8px] text-white backdrop-blur ${isActive ? 'opacity-100' : 'opacity-0'}`}>
                        {lesion.region} {lesion.confidence.toFixed(2)}
                      </span>
                    </button>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex h-full items-center justify-center text-sm text-zinc-500">No image available</div>
      )}
    </div>
  )
}

function SplitCompareViewer({
  beforeSrc,
  afterSrc,
  state,
  onChange,
}: {
  beforeSrc: string
  afterSrc: string
  state: ViewerState
  onChange: (next: ViewerState) => void
}) {
  const [divider, setDivider] = useState(50)

  return (
    <div className="relative h-[480px] overflow-hidden rounded-[1.25rem] border border-white/5 bg-black">
      <AdvancedImageViewer src={beforeSrc} alt="Previous compare" state={state} onChange={onChange} heightClass="h-[480px]" />
      <div className="pointer-events-none absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - divider}% 0 0)` }}>
        <AdvancedImageViewer src={afterSrc} alt="Current compare" state={state} onChange={onChange} heightClass="h-[480px]" />
      </div>
      <div className="absolute inset-y-0 z-20 w-px bg-cyan-400/70" style={{ left: `${divider}%` }} />
      <div className="pointer-events-none absolute left-4 top-4 z-30 rounded-full border border-white/10 bg-black/60 px-3 py-1 text-[10px] uppercase tracking-[0.25em] text-zinc-300 backdrop-blur">
        prior / current compare
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={divider}
        onChange={(event) => setDivider(Number(event.target.value))}
        className="absolute bottom-4 left-1/2 z-30 w-64 -translate-x-1/2 accent-cyan-400"
      />
    </div>
  )
}
