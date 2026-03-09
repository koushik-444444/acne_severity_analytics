export type RegionStats = {
  count: number
  lpi?: number
  area_px?: number
  gags_score?: number
}

export type ClinicalAnalysis = {
  regions: Record<string, RegionStats>
  total_lesions: number
  gags_total_score: number
  clinical_severity: ClinicalSeverity | (string & {})
  symmetry_delta: number
}

export type ConsensusLesion = {
  region: string
  bbox: [number, number, number, number]
  center?: [number, number]
  confidence: number
  votes: number
  reliability_score: number
  confidence_level: string
  class_name?: string
  severity_grade?: number
}

export type ConsensusSummary = {
  verified_lesions: number
  average_confidence: number
  top_regions: Array<{ region: string; count: number }>
  region_counts: Record<string, number>
  unassigned_count: number
  summary: string
  lesions?: ConsensusLesion[]
  type_counts?: Record<string, number>
}

export type SessionStage =
  | 'idle'
  | 'queued'
  | 'uploading'
  | 'segmenting'
  | 'cloud_inference'
  | 'mapping'
  | 'scoring'
  | 'completed'
  | 'failed'

export type ClinicalSeverity =
  | 'Clear'
  | 'Mild'
  | 'Moderate'
  | 'Severe'
  | 'Very Severe / Cystic'

export type SessionStatus = {
  session_id?: string
  stage: SessionStage | (string & {})
  detail: string
  progress: number
  updated_at?: string
  completed?: boolean
  failed?: boolean
}

export type SessionSummary = {
  session_id: string
  profile_id?: string | null
  timestamp: string
  severity: string | null
  gags_score: number | null
  lesion_count: number | null
  symmetry_delta: number | null
  privacy_mode: boolean
  retention_hours: number
  status?: SessionStatus | null
  note?: string
}

export type SessionDetail = SessionSummary & {
  results: ({
    clinical_analysis?: ClinicalAnalysis
    consensus_summary?: ConsensusSummary
    lesions?: Record<string, ConsensusLesion[]>
    cloud_results?: Record<string, unknown>
    source_stream_provenance?: {
      streams: Record<string, number>
      strongest_stream: string | null
      stream_total: number
      stream_classes?: Record<string, Record<string, number>>
    }
  } & Record<string, unknown>) | null
  diagnostic_image_path?: string | null
  original_image_path?: string | null
  diagnostic_image?: string | null
  original_image?: string | null
  note?: string
}

export type ComparePayload = {
  previous_session_id: string
  current_session_id: string
  previous_timestamp: string
  current_timestamp: string
  severity_change: { from: string; to: string }
  lesion_delta: number
  gags_delta: number
  symmetry_delta_change: number
  regions: Record<
    string,
    {
      previous_count: number
      current_count: number
      count_delta: number
      previous_lpi: number
      current_lpi: number
      lpi_delta: number
    }
  >
} | null

export type AnalyzeResponse = {
  session_id: string
  status: SessionStatus
  profile_id?: string | null
  severity: string | null
  gags_score: number | null
  lesion_count: number | null
  symmetry_delta: number | null
  results: ({
    clinical_analysis?: ClinicalAnalysis
    consensus_summary?: ConsensusSummary
    lesions?: Record<string, ConsensusLesion[]>
    cloud_results?: Record<string, unknown>
    source_stream_provenance?: {
      streams: Record<string, number>
      strongest_stream: string | null
      stream_total: number
      stream_classes?: Record<string, Record<string, number>>
    }
  } & Record<string, unknown>)
  compare: ComparePayload
  diagnostic_image: string | null
  original_image: string | null
}

export type AnalysisStartResponse = {
  session_id: string
  profile_id?: string
  privacy_mode: boolean
  retention_hours: number
  status: SessionStatus
}

export type ProfileSummary = {
  profile_id: string
  sessions: number
  latest_timestamp: string
  latest_severity: string | null
}

export type PrivacyConfig = {
  privacy_mode_supported: boolean
  default_retention_hours: number
  max_retention_hours: number
  purge_endpoint: string
  stored_fields: string[]
}

export type HistoryPage = {
  items: SessionSummary[]
  next_cursor?: string | null
}

export type ExportPreset = 'clinical' | 'compact' | 'presentation'

export type ExportResponse = {
  session_id: string
  pdf_path: string
  preset: string
  pdf_data_uri?: string
}

export type ReportResponse = {
  session_id: string
  report: {
    clinical_analysis: Record<string, unknown>
    consensus_summary: Record<string, unknown>
    compare: Record<string, unknown> | null
    pdf_path: string
    pdf_data_uri?: string
  }
}

export type LatencyStats = {
  count: number
  mean_ms: number
  min_ms: number
  max_ms: number
  p50_ms: number
  p95_ms: number
}

export type MetricsResponse = {
  api_usage: {
    total_calls: number
    calls_by_model: Record<string, number>
    calls_by_status: Record<string, number>
    latency_stats: LatencyStats | null
    error_rate: number
    recent_errors: Array<{ timestamp: string; model: string; error: string | null }>
  }
  session_stats: {
    total_sessions: number
    sessions_with_results: number
    detection_counts: { mean: number | null; min: number | null; max: number | null }
    gags_scores: { mean: number | null; min: number | null; max: number | null }
  }
  timing: {
    local_pipeline: Record<string, number | null>
    cloud_inference: Record<string, number | null>
    sample_count: number
  }
  pipeline_metrics: {
    sample_count: number
    total_raw_detections: number
    total_post_nms: number
    total_post_gating: number
    total_proximity_propagated: number
    nms_reduction_pct: number | null
    gating_reduction_pct: number | null
    type_coverage_aggregate: Record<string, number>
  } | null
}
