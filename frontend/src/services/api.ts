import type {
  AnalysisStartResponse,
  AnalyzeResponse,
  ComparePayload,
  PrivacyConfig,
  ProfileSummary,
  SessionDetail,
  SessionStatus,
  SessionSummary,
} from '../types/api'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  startAnalysis: (payload: { session_id?: string; profile_id?: string; privacy_mode: boolean; retention_hours: number }) =>
    request<AnalysisStartResponse>('/analysis/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  analyze: (payload: FormData) =>
    request<AnalyzeResponse>('/analyze', {
      method: 'POST',
      body: payload,
    }),

  getLatestStatus: async () => {
    const data = await request<{ status: SessionStatus }>('/status/latest')
    return data.status
  },

  getStatus: (sessionId: string) => request<SessionStatus>(`/status/${sessionId}`),

  streamStatus: (sessionId: string) => new EventSource(`${API_BASE}/status/stream/${sessionId}`),

  getHistory: async (limit = 30, profileId?: string) => {
    const suffix = profileId ? `&profile_id=${encodeURIComponent(profileId)}` : ''
    const data = await request<{ items: SessionSummary[] }>(`/history?limit=${limit}${suffix}`)
    return data.items
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
      method: 'POST',
    }),

  getHealth: () => request<{ status: string; version: string }>('/health'),

  getVersion: () => request<{ app: string; version: string }>('/version'),

  getReport: (sessionId: string, previousSessionId?: string) =>
    request(`/report/${sessionId}${previousSessionId ? `?previous_session_id=${encodeURIComponent(previousSessionId)}` : ''}`),

  exportBundle: (
    sessionId: string,
    preset: 'clinical' | 'compact' | 'presentation' = 'clinical',
    previousSessionId?: string,
  ) =>
    request<{ session_id: string; pdf_path: string; pdf_data_uri?: string }>(`/export/${sessionId}`, {
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
}
