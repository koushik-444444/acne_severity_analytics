import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Camera, Eye, EyeOff, Loader2, Maximize2, Minimize2, Search, SearchX, Shield, Sparkles, Trash2 } from 'lucide-react'
import { motion } from 'framer-motion'

import { api } from '../../services/api'
import {
  clamp,
  describeDelta,
  formatDate,
  formatSignedDelta,
  getConfidenceTextTone,
  getClinicalDeltaLabel,
  getDeltaTone,
  getProfileStorageKey,
  getSeverityTone,
  readStorageItem,
  removeStorageItem,
  writeStorageItem,
} from '../../lib/clinical-utils'
import { useAnalysisWorkflow } from '../../hooks/useAnalysisWorkflow'
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts'
import { useProfileSwitchNotice } from '../../hooks/useProfileSwitchNotice'
import { useStatusPoller } from '../../hooks/useStatusPoller'
import { useWorkspacePrefs } from '../../hooks/useWorkspacePrefs'
import type {
  ComparePayload,
  PrivacyConfig,
  ProfileSummary,
  RegionStats,
  SessionDetail,
  SessionSummary,
} from '../../types/api'
import { AdvancedImageViewer } from './AdvancedImageViewer'
import { MetricCard } from './MetricCard'
import { SplitCompareViewer } from './SplitCompareViewer'
import type { ViewerState } from './types'

type CompareRegionDelta = NonNullable<ComparePayload>['regions'][string]
const BASELINE_SESSION_KEY = 'clearskin-baseline-session'
const ONBOARDING_SEEN_KEY = 'clearskin-ui-prefs-onboarding-seen'
const DEFAULT_PROFILE_ID = 'default-profile'

export function ClinicalWorkspace() {
  const [caseState, dispatch] = useAnalysisWorkflow()
  const {
    isAnalyzing, selectedFile, previewUrl, sessionId,
    active, compare, previousSession, noteDraft,
    activeLesionKey, profileGuard, error,
    isExporting, isSavingNotes, isLoadingSession, isPurging,
  } = caseState

  const [history, setHistory] = useState<SessionSummary[]>([])
  const [profiles, setProfiles] = useState<ProfileSummary[]>([])
  const [activeProfileId, setActiveProfileId] = useState<string>(DEFAULT_PROFILE_ID)
  const [baselineSession, setBaselineSession] = useState<SessionSummary | null>(null)
  const [privacy, setPrivacy] = useState<PrivacyConfig | null>(null)
  const [privacyMode, setPrivacyMode] = useState(false)
  const [retentionHours, setRetentionHours] = useState(72)
  const [compareFullscreen, setCompareFullscreen] = useState(false)
  const [mainViewer, setMainViewer] = useState<ViewerState>({ scale: 1, x: 0, y: 0 })
  const [compareViewer, setCompareViewer] = useState<ViewerState>({ scale: 1, x: 0, y: 0 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [lesionRegionFilter, setLesionRegionFilter] = useState<string>('all')
  const [lesionConfidenceFilter, setLesionConfidenceFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all')
  const [showOnboarding, setShowOnboarding] = useState(false)
  const workspaceRef = useRef<HTMLDivElement | null>(null)

  const {
    leftRailWidth, setLeftRailWidth,
    rightRailWidth, setRightRailWidth,
    compareViewerMode, setCompareViewerMode,
    showDetectionOverlay, setShowDetectionOverlay,
    showCompareOverlay, setShowCompareOverlay,
    exportPreset, setExportPreset,
  } = useWorkspacePrefs(activeProfileId)

  const [status, setStatus] = useStatusPoller(sessionId, isAnalyzing)
  const profileSwitchNotice = useProfileSwitchNotice(activeProfileId)

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
          dispatch({ type: 'SET_ERROR', error: err instanceof Error ? err.message : 'Failed to initialize workspace' })
        }
      })

    return () => {
      activeRequest = false
    }
  }, [])

  useEffect(() => {
    setBaselineSession(null)

    let activeRequest = true
    const baselineStorageKey = getProfileStorageKey(BASELINE_SESSION_KEY, activeProfileId)

    void Promise.all([api.getHistory(30, activeProfileId), api.getProfiles()])
      .then(([historyPage, profileItems]) => {
        if (!activeRequest) {
          return
        }

        setHistory(historyPage.items)
        setProfiles(profileItems)

        const storedBaseline = readStorageItem(baselineStorageKey)
        if (!storedBaseline) {
          return
        }

        const match = historyPage.items.find((item) => item.session_id === storedBaseline)
        if (match) {
          setBaselineSession(match)
          return
        }

        removeStorageItem(baselineStorageKey)
      })
      .catch((err) => {
        if (activeRequest) {
          dispatch({ type: 'SET_ERROR', error: err instanceof Error ? err.message : 'Failed to load profile workspace' })
        }
      })

    return () => {
      activeRequest = false
    }
  }, [activeProfileId])

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
    const [historyPage, profileItems] = await Promise.all([api.getHistory(30, activeProfileId), api.getProfiles()])
    setHistory(historyPage.items)
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

    let nextPrevious: SessionDetail | null = null
    if (nextCompare?.previous_session_id) {
      nextPrevious = await api.getSession(nextCompare.previous_session_id)
    }

    dispatch({ type: 'SET_COMPARE', compare: nextCompare, previousSession: nextPrevious })
    return nextCompare
  }

  const handleFileChange = (event: import('react').ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    dispatch({
      type: 'SELECT_FILE',
      file,
      previewUrl: file ? URL.createObjectURL(file) : null,
    })
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
      dispatch({ type: 'START_ANALYSIS' })
      setStatus({
        stage: 'warming_up',
        detail: 'Waking backend — this may take a moment on first use...',
        progress: 1,
      })

      // Pre-flight: wake the backend if it's sleeping (HF Space cold start)
      try {
        await api.wakeBackend()
      } catch {
        // Health check failed — continue anyway, startAnalysis will surface
        // the real error if the backend is truly unreachable.
      }

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
      dispatch({ type: 'SET_SESSION', sessionId: start.session_id })
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
      dispatch({ type: 'ANALYSIS_COMPLETE', active: result })
      await loadCompareContext(result.session_id, {
        previousSessionId: baselineSession?.session_id,
        fallbackCompare: result.compare,
      })
      setStatus(result.status)
      setShowDetectionOverlay(true)
      setCompareViewerMode('single')
      await refreshHistory()
    } catch (err) {
      const message = err instanceof Error
        ? (err.name === 'ApiTimeoutError'
          ? err.message
          : err.message)
        : 'Analysis failed — please try again'
      dispatch({ type: 'ANALYSIS_FAILED', error: message })
    } finally {
      dispatch({ type: 'ANALYSIS_FINISHED' })
    }
  }

  const handleSelectHistory = async (item: SessionSummary) => {
    try {
      dispatch({ type: 'LOAD_SESSION_START' })
      const session = await api.getSession(item.session_id)
      const sessionProfileId = item.profile_id ?? DEFAULT_PROFILE_ID
      if (sessionProfileId !== activeProfileId) {
        setActiveProfileId(sessionProfileId)
      }
      const comparePayload = await loadCompareContext(item.session_id, {
        previousSessionId: baselineSession?.session_id,
      })
      dispatch({
        type: 'LOAD_SESSION_COMPLETE',
        sessionId: item.session_id,
        active: {
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
        },
        noteDraft: session.note ?? '',
      })
      setStatus(session.status ?? null)
      setShowDetectionOverlay(true)
      setCompareViewerMode('single')
    } catch (err) {
      dispatch({ type: 'LOAD_SESSION_FAILED', error: err instanceof Error ? err.message : 'Failed to load session' })
    }
  }

  const handlePurge = async (item: SessionSummary) => {
    if (!window.confirm('Permanently delete this session? This cannot be undone.')) return
    try {
      dispatch({ type: 'PURGE_START' })
      await api.purgeSession(item.session_id)
      if (active?.session_id === item.session_id) {
        dispatch({ type: 'PURGE_ACTIVE_CLEARED' })
        setStatus(null)
      }
      await refreshHistory()
      dispatch({ type: 'PURGE_COMPLETE' })
    } catch (err) {
      dispatch({ type: 'PURGE_FAILED', error: err instanceof Error ? err.message : 'Failed to purge session' })
    }
  }

  const resetCaseWorkspace = () => {
    dispatch({ type: 'RESET_CASE' })
    setStatus(null)
  }

  const handleExport = async () => {
    if (!active?.session_id) return
    try {
      dispatch({ type: 'EXPORT_START' })
      const bundle = await api.exportBundle(active.session_id, exportPreset, compare?.previous_session_id)
      if (bundle.pdf_data_uri) {
        window.open(bundle.pdf_data_uri, '_blank')
      }
      dispatch({ type: 'EXPORT_COMPLETE' })
    } catch (err) {
      dispatch({ type: 'EXPORT_FAILED', error: err instanceof Error ? err.message : 'Export failed' })
    }
  }

  const pinBaseline = async (item: SessionSummary) => {
    setBaselineSession(item)
    writeStorageItem(getProfileStorageKey(BASELINE_SESSION_KEY, activeProfileId), item.session_id)

    if (active?.session_id) {
      try {
        await loadCompareContext(active.session_id, { previousSessionId: item.session_id })
      } catch (err) {
        dispatch({ type: 'SET_ERROR', error: err instanceof Error ? err.message : 'Failed to load baseline comparison' })
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
        dispatch({ type: 'SET_ERROR', error: err instanceof Error ? err.message : 'Failed to restore default comparison' })
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
      dispatch({ type: 'SAVE_NOTES_START' })
      await api.updateSessionNotes(active.session_id, noteDraft)
      dispatch({ type: 'SAVE_NOTES_COMPLETE' })
    } catch (err) {
      dispatch({ type: 'SAVE_NOTES_FAILED', error: err instanceof Error ? err.message : 'Failed to save notes' })
    }
  }

  const resetMainViewer = () => setMainViewer({ scale: 1, x: 0, y: 0 })
  const resetCompareViewer = () => setCompareViewer({ scale: 1, x: 0, y: 0 })
  const handleLesionHover = (key: string | null) => dispatch({ type: 'SET_ACTIVE_LESION', key })
  const handleNoteDraftChange = (draft: string) => dispatch({ type: 'SET_NOTE_DRAFT', draft })

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

  useKeyboardShortcuts({
    active,
    compare,
    setMainViewer,
    resetMainViewer,
    resetCompareViewer,
    setShowDetectionOverlay,
    setCompareViewerMode,
    toggleFullscreen,
  })

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
            role="status"
            aria-live="polite"
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
            role="alert"
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
                  dispatch({ type: 'SET_PROFILE_GUARD', guard: null })
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
                    dispatch({ type: 'SET_PROFILE_GUARD', guard: {
                      sessionId: active.session_id,
                      sessionProfileId: activeSessionProfileId,
                      requestedProfileId: nextProfileId,
                    } })
                  } else {
                    dispatch({ type: 'SET_PROFILE_GUARD', guard: null })
                  }
                }}
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
              aria-label="Left panel width"
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
                        <button aria-label="Zoom in" onClick={() => setMainViewer((state) => ({ ...state, scale: clamp(state.scale + 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                          <Search className="h-4 w-4" />
                        </button>
                        <button aria-label="Zoom out" onClick={() => setMainViewer((state) => ({ ...state, scale: clamp(state.scale - 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                          <SearchX className="h-4 w-4" />
                        </button>
                        <button onClick={resetMainViewer} className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                          Reset
                        </button>
                        <button aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'} onClick={() => void toggleFullscreen()} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
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
                      onLesionHover={handleLesionHover}
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
                          <button aria-label="Zoom in compare" onClick={() => setCompareViewer((state) => ({ ...state, scale: clamp(state.scale + 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
                            <Search className="h-4 w-4" />
                          </button>
                          <button aria-label="Zoom out compare" onClick={() => setCompareViewer((state) => ({ ...state, scale: clamp(state.scale - 0.25, 1, 4) }))} className="rounded-full border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white">
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
                        onLesionHover={handleLesionHover}
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
                        onLesionHover={handleLesionHover}
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
                        aria-label="Export preset"
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
                        onChange={(e) => setLesionConfidenceFilter(e.target.value as 'all' | 'high' | 'medium' | 'low')}
                        aria-label="Filter by confidence tier"
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
                          onMouseEnter={() => handleLesionHover(lesion.key)}
                          onMouseLeave={() => handleLesionHover(null)}
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
                    onChange={(e) => handleNoteDraftChange(e.target.value)}
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

            {error && <div role="alert" className="mt-6 rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</div>}

            {showOnboarding && (
              <div
                role="dialog"
                aria-modal="true"
                aria-label="Workspace onboarding"
                className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm"
                onKeyDown={(e) => { if (e.key === 'Escape') dismissOnboarding() }}
              >
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
                      autoFocus
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
              aria-label="Right panel width"
              className="mt-4 w-full accent-cyan-400"
              />
            </div>
          </aside>
        </div>
      </div>
    </section>
  )
}
