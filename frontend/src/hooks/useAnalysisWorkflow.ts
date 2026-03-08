import { useReducer } from 'react'

import type {
  AnalyzeResponse,
  ComparePayload,
  SessionDetail,
} from '../types/api'

type ActiveCaseProfileGuard = {
  sessionId: string
  sessionProfileId: string
  requestedProfileId: string
}

export type CaseWorkflowState = {
  isAnalyzing: boolean
  selectedFile: File | null
  previewUrl: string | null
  sessionId: string | null
  active: AnalyzeResponse | null
  compare: ComparePayload | null
  previousSession: SessionDetail | null
  noteDraft: string
  activeLesionKey: string | null
  profileGuard: ActiveCaseProfileGuard | null
  error: string | null
  isExporting: boolean
  isSavingNotes: boolean
  isLoadingSession: boolean
  isPurging: boolean
}

export type CaseWorkflowAction =
  | { type: 'START_ANALYSIS' }
  | { type: 'SET_SESSION'; sessionId: string }
  | { type: 'ANALYSIS_COMPLETE'; active: AnalyzeResponse }
  | { type: 'ANALYSIS_FAILED'; error: string }
  | { type: 'ANALYSIS_FINISHED' }
  | { type: 'SELECT_FILE'; file: File | null; previewUrl: string | null }
  | { type: 'LOAD_SESSION_START' }
  | { type: 'LOAD_SESSION_COMPLETE'; sessionId: string; active: AnalyzeResponse; noteDraft: string }
  | { type: 'LOAD_SESSION_FAILED'; error: string }
  | { type: 'SET_COMPARE'; compare: ComparePayload | null; previousSession: SessionDetail | null }
  | { type: 'PURGE_START' }
  | { type: 'PURGE_ACTIVE_CLEARED' }
  | { type: 'PURGE_COMPLETE' }
  | { type: 'PURGE_FAILED'; error: string }
  | { type: 'EXPORT_START' }
  | { type: 'EXPORT_COMPLETE' }
  | { type: 'EXPORT_FAILED'; error: string }
  | { type: 'SAVE_NOTES_START' }
  | { type: 'SAVE_NOTES_COMPLETE' }
  | { type: 'SAVE_NOTES_FAILED'; error: string }
  | { type: 'RESET_CASE' }
  | { type: 'SET_ERROR'; error: string | null }
  | { type: 'SET_PROFILE_GUARD'; guard: ActiveCaseProfileGuard | null }
  | { type: 'SET_NOTE_DRAFT'; draft: string }
  | { type: 'SET_ACTIVE_LESION'; key: string | null }

const INITIAL_STATE: CaseWorkflowState = {
  isAnalyzing: false,
  selectedFile: null,
  previewUrl: null,
  sessionId: null,
  active: null,
  compare: null,
  previousSession: null,
  noteDraft: '',
  activeLesionKey: null,
  profileGuard: null,
  error: null,
  isExporting: false,
  isSavingNotes: false,
  isLoadingSession: false,
  isPurging: false,
}

function caseWorkflowReducer(state: CaseWorkflowState, action: CaseWorkflowAction): CaseWorkflowState {
  switch (action.type) {
    case 'START_ANALYSIS':
      return { ...state, error: null, isAnalyzing: true }

    case 'SET_SESSION':
      return { ...state, sessionId: action.sessionId }

    case 'ANALYSIS_COMPLETE':
      return {
        ...state,
        active: action.active,
        activeLesionKey: null,
        profileGuard: null,
      }

    case 'ANALYSIS_FAILED':
      return { ...state, error: action.error }

    case 'ANALYSIS_FINISHED':
      return { ...state, isAnalyzing: false }

    case 'SELECT_FILE':
      return {
        ...state,
        selectedFile: action.file,
        previewUrl: action.previewUrl,
        error: null,
      }

    case 'LOAD_SESSION_START':
      return { ...state, isLoadingSession: true }

    case 'LOAD_SESSION_COMPLETE':
      return {
        ...state,
        sessionId: action.sessionId,
        active: action.active,
        noteDraft: action.noteDraft,
        activeLesionKey: null,
        profileGuard: null,
        isLoadingSession: false,
      }

    case 'LOAD_SESSION_FAILED':
      return { ...state, error: action.error, isLoadingSession: false }

    case 'SET_COMPARE':
      return {
        ...state,
        compare: action.compare,
        previousSession: action.previousSession,
      }

    case 'PURGE_START':
      return { ...state, isPurging: true }

    case 'PURGE_ACTIVE_CLEARED':
      return {
        ...state,
        active: null,
        compare: null,
        previousSession: null,
        sessionId: null,
      }

    case 'PURGE_COMPLETE':
      return { ...state, isPurging: false }

    case 'PURGE_FAILED':
      return { ...state, error: action.error, isPurging: false }

    case 'EXPORT_START':
      return { ...state, isExporting: true }

    case 'EXPORT_COMPLETE':
      return { ...state, isExporting: false }

    case 'EXPORT_FAILED':
      return { ...state, error: action.error, isExporting: false }

    case 'SAVE_NOTES_START':
      return { ...state, isSavingNotes: true }

    case 'SAVE_NOTES_COMPLETE':
      return { ...state, isSavingNotes: false }

    case 'SAVE_NOTES_FAILED':
      return { ...state, error: action.error, isSavingNotes: false }

    case 'RESET_CASE':
      return {
        ...state,
        active: null,
        compare: null,
        previousSession: null,
        selectedFile: null,
        previewUrl: null,
        sessionId: null,
        noteDraft: '',
        activeLesionKey: null,
        profileGuard: null,
      }

    case 'SET_ERROR':
      return { ...state, error: action.error }

    case 'SET_PROFILE_GUARD':
      return { ...state, profileGuard: action.guard }

    case 'SET_NOTE_DRAFT':
      return { ...state, noteDraft: action.draft }

    case 'SET_ACTIVE_LESION':
      return { ...state, activeLesionKey: action.key }

    default:
      return state
  }
}

/**
 * Manages the core analysis case workflow via useReducer.
 * Consolidates 15 individual useState hooks into a single
 * state object with typed action dispatch.
 */
export function useAnalysisWorkflow() {
  return useReducer(caseWorkflowReducer, INITIAL_STATE)
}
