import type {
  AnalysisStartResponse,
  AnalyzeResponse,
  ComparePayload,
  ExportPreset,
  ExportResponse,
  HistoryPage,
  MetricsResponse,
  PrivacyConfig,
  ProfileSummary,
  ReportResponse,
  SessionDetail,
  SessionStatus,
} from '../types/api'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

class ApiTimeoutError extends Error {
  constructor(url: string, timeoutMs: number) {
    super(
      `Request to ${url} timed out after ${Math.round(timeoutMs / 1000)}s. ` +
      'The backend may be waking up from a cold start — please try again in a moment.'
    )
    this.name = 'ApiTimeoutError'
  }
}

async function request<T>(path: string, init?: RequestInit, signal?: AbortSignal, timeoutMs = 30_000): Promise<T> {
  const timeoutSignal = AbortSignal.timeout(timeoutMs)
  const combinedSignal = signal
    ? AbortSignal.any([signal, timeoutSignal])
    : timeoutSignal

  const url = `${API_BASE}${path}`
  let response: Response
  try {
    response = await fetch(url, {
      ...init,
      signal: combinedSignal,
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'TimeoutError') {
      throw new ApiTimeoutError(path, timeoutMs)
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw err
    }
    throw new Error(
      `Network error reaching ${path} — is the backend running? (${err instanceof Error ? err.message : String(err)})`
    )
  }
  if (!response.ok) {
    const text = await response.text()
    let detail: string
    try {
      detail = (JSON.parse(text) as { detail?: string }).detail ?? text
    } catch {
      detail = text
    }
    throw new Error(detail || `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  /** Lightweight ping to wake a sleeping HF Space before heavy work. */
  wakeBackend: () => request<{ status: string }>('/health', undefined, undefined, 90_000),

  startAnalysis: (payload: { session_id?: string; profile_id?: string; privacy_mode: boolean; retention_hours: number }) =>
    request<AnalysisStartResponse>('/analysis/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }, undefined, 90_000),

  analyze: (payload: FormData) =>
    request<AnalyzeResponse>('/analyze', {
      method: 'POST',
      body: payload,
    }, undefined, 300_000),

  getLatestStatus: async () => {
    const data = await request<{ status: SessionStatus }>('/status/latest')
    return data.status
  },

  getStatus: (sessionId: string) => request<SessionStatus>(`/status/${sessionId}`),

  getHistory: async (limit = 30, profileId?: string, cursor?: string): Promise<HistoryPage> => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (profileId) params.set('profile_id', profileId)
    if (cursor) params.set('cursor', cursor)
    return request<HistoryPage>(`/history?${params.toString()}`)
  },

  getProfiles: async () => {
    const data = await request<{ items: ProfileSummary[] }>('/profiles')
    return data.items
  },

  getSession: (sessionId: string) => request<SessionDetail>(`/session/${sessionId}`),

  getCompare: async (sessionId: string, previousSessionId?: string) => {
    const suffix = previousSessionId ? `?previous_session_id=${encodeURIComponent(previousSessionId)}` : ''
    const data = await request<{ current_session_id: string; compare: ComparePayload }>(`/compare/${sessionId}${suffix}`)
    return data.compare
  },

  getPrivacy: () => request<PrivacyConfig>('/privacy'),

  purgeSession: (sessionId: string) =>
    request<{ purged: boolean; session_id: string }>(`/privacy/purge/${sessionId}`, {
      method: 'DELETE',
    }),

  getHealth: () => request<{ status: string; version: string }>('/health', undefined, undefined, 10_000),

  getVersion: () => request<{ app: string; version: string }>('/version', undefined, undefined, 10_000),

  getReport: (sessionId: string, previousSessionId?: string) =>
    request<ReportResponse>(`/report/${sessionId}${previousSessionId ? `?previous_session_id=${encodeURIComponent(previousSessionId)}` : ''}`),

  exportBundle: (
    sessionId: string,
    preset: ExportPreset = 'clinical',
    previousSessionId?: string,
  ) =>
    request<ExportResponse>(`/export/${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ include_pdf_data: true, preset, previous_session_id: previousSessionId }),
    }),

  updateSessionNotes: (sessionId: string, note: string) =>
    request<{ session_id: string; note: string }>(`/session/${sessionId}/notes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ note }),
    }),

  getMetrics: () => request<MetricsResponse>('/metrics'),
}
