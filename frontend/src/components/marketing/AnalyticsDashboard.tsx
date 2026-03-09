import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'

import { api } from '../../services/api'
import type { MetricsResponse } from '../../types/api'
import { MetadataLabel } from './MetadataLabel'

const CHART_FRAME = {
  left: 10,
  right: 96,
  top: 12,
  bottom: 82,
}

type ChartPoint = {
  name: string
  value: number
  x: number
  y: number
}

function buildStepLinePath(points: ChartPoint[]) {
  const first = points[0]
  if (!points.length || !first) {
    return ''
  }

  let path = `M ${first.x} ${first.y}`
  for (let index = 1; index < points.length; index += 1) {
    const pt = points[index]
    if (pt) path += ` H ${pt.x} V ${pt.y}`
  }
  return path
}

function buildStepAreaPath(points: ChartPoint[]) {
  const first = points[0]
  if (!points.length || !first) {
    return ''
  }

  const linePath = buildStepLinePath(points)
  return `${linePath} V ${CHART_FRAME.bottom} H ${first.x} Z`
}

function formatMs(val: number | null | undefined): string {
  if (val == null) return '--'
  return val < 1000 ? `${Math.round(val)}ms` : `${(val / 1000).toFixed(1)}s`
}

function formatPct(val: number | null | undefined): string {
  if (val == null) return '--'
  return `${val.toFixed(1)}%`
}

/** Build chart data from pipeline timing keys. */
function buildTimingChartData(metrics: MetricsResponse): Array<{ name: string; value: number }> {
  const local = metrics.timing.local_pipeline
  const cloud = metrics.timing.cloud_inference
  const entries: Array<{ name: string; value: number }> = []

  const keys: Array<[string, string]> = [
    ['bisenet_mean', 'BISENET'],
    ['landmarks_mean', 'LNDMRK'],
    ['geometry_mean', 'GEOM'],
    ['combine_mean', 'MERGE'],
  ]
  for (const [key, label] of keys) {
    const v = local[key]
    if (v != null) entries.push({ name: label, value: v })
  }

  const cloudKeys: Array<[string, string]> = [
    ['model_a_1280_ms_mean', 'CLD_A'],
    ['model_b_ms_mean', 'CLD_B'],
  ]
  for (const [key, label] of cloudKeys) {
    const v = cloud[key]
    if (v != null) entries.push({ name: label, value: v })
  }

  // Add total if available
  const total = local['total_mean']
  if (total != null) entries.push({ name: 'TOTAL', value: total })

  // Fallback if no data
  if (!entries.length) {
    return [{ name: 'N/A', value: 0 }]
  }
  return entries
}

export function AnalyticsDashboard() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeIndex, setActiveIndex] = useState(-1)

  useEffect(() => {
    let cancelled = false
    void api.getMetrics()
      .then((data) => { if (!cancelled) setMetrics(data) })
      .catch((err) => { if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load metrics') })
    return () => { cancelled = true }
  }, [])

  const chartData = useMemo(() => {
    if (!metrics) return []
    return buildTimingChartData(metrics)
  }, [metrics])

  const resolvedActiveIndex = activeIndex >= 0 ? activeIndex : chartData.length - 1

  const { activePoint, chartPoints, areaPath, linePath, tickValues } = useMemo(() => {
    if (!chartData.length) {
      return {
        activePoint: { name: '--', value: 0, x: 0, y: 0 },
        chartPoints: [] as ChartPoint[],
        areaPath: '',
        linePath: '',
        tickValues: [] as Array<{ value: number; y: number }>,
      }
    }
    const maxValue = Math.max(...chartData.map((item) => item.value))
    const normalizedMax = Math.ceil(maxValue / 100) * 100 || 100
    const xStep = (CHART_FRAME.right - CHART_FRAME.left) / Math.max(chartData.length - 1, 1)
    const yRange = CHART_FRAME.bottom - CHART_FRAME.top
    const points = chartData.map((item, index) => ({
      ...item,
      x: Number((CHART_FRAME.left + xStep * index).toFixed(2)),
      y: Number((CHART_FRAME.bottom - (item.value / normalizedMax) * yRange).toFixed(2)),
    }))

    return {
      activePoint: points[resolvedActiveIndex] ?? points[points.length - 1] ?? { name: '', value: 0, x: 0, y: 0 },
      chartPoints: points,
      areaPath: buildStepAreaPath(points),
      linePath: buildStepLinePath(points),
      tickValues: Array.from({ length: 5 }, (_, index) => {
        const ratio = index / 4
        return {
          value: Math.round(normalizedMax * (1 - ratio)),
          y: Number((CHART_FRAME.top + yRange * ratio).toFixed(2)),
        }
      }),
    }
  }, [chartData, resolvedActiveIndex])

  // Sidebar stat cards — derive from real metrics
  const statCards = useMemo(() => {
    if (!metrics) {
      return [
        { label: 'TOTAL_SESSIONS', value: '--', delta: 'LOADING' },
        { label: 'MEAN_LATENCY', value: '--', delta: 'LOADING' },
        { label: 'API_SUCCESS', value: '--', delta: 'LOADING' },
      ]
    }
    const sess = metrics.session_stats
    const latency = metrics.api_usage.latency_stats
    const errorRate = metrics.api_usage.error_rate
    const successRate = (1 - errorRate) * 100

    return [
      {
        label: 'TOTAL_SESSIONS',
        value: String(sess.total_sessions),
        delta: sess.sessions_with_results > 0
          ? `${sess.sessions_with_results} analyzed`
          : 'NONE',
      },
      {
        label: 'MEAN_LATENCY',
        value: latency ? formatMs(latency.mean_ms) : '--',
        delta: latency ? `p95 ${formatMs(latency.p95_ms)}` : 'NO DATA',
      },
      {
        label: 'API_SUCCESS',
        value: metrics.api_usage.total_calls > 0 ? formatPct(successRate) : '--',
        delta: metrics.api_usage.total_calls > 0
          ? `${metrics.api_usage.total_calls} calls`
          : 'NO CALLS',
      },
    ]
  }, [metrics])

  const streamStatus = metrics ? 'LIVE' : (error ? 'ERROR' : 'LOADING')
  const sampleCount = metrics?.timing.sample_count ?? 0

  return (
    <section className="relative overflow-hidden bg-[#010101] py-60">
      <h2 className="editorial-title -bottom-10 right-0 translate-x-1/4 rotate-6 text-[20vw]">Analytics_Core</h2>

      <div className="relative z-10 mx-auto max-w-7xl px-8">
        <div className="grid grid-cols-1 items-start gap-12 lg:grid-cols-12">
          <div className="group lg:col-span-8">
            <div className="holographic-panel h-[600px] overflow-hidden rounded-[3rem] p-16">
              <div className="absolute left-12 top-12 z-10 flex w-full items-start justify-between pr-24">
                <div>
                  <h3 className="terminal-text mb-4 text-[10px] font-bold tracking-[0.4em] text-cyan-400">
                    PIPELINE_TIMING
                  </h3>
                  <p className="text-3xl font-light tracking-tighter">Stage Latency Breakdown</p>
                </div>
                <div className="text-right">
                  <MetadataLabel className="mb-2">STREAM: {streamStatus}</MetadataLabel>
                  <MetadataLabel>SAMPLES: {sampleCount}</MetadataLabel>
                </div>
              </div>

              <div className="h-full pt-32">
                <div className="relative h-full overflow-hidden rounded-[2rem] border border-white/5 bg-[radial-gradient(circle_at_50%_20%,rgba(0,242,255,0.08),rgba(0,0,0,0)_55%)]">
                  <div className="pointer-events-none absolute left-6 top-6 z-10 rounded-full border border-cyan-400/10 bg-black/45 px-4 py-2 backdrop-blur">
                    <div className="terminal-text text-[8px] tracking-[0.32em] text-cyan-400/75">ACTIVE STAGE</div>
                    <div className="mt-1 flex items-end gap-3">
                      <span className="text-2xl font-semibold text-white">{formatMs(activePoint.value)}</span>
                      <span className="terminal-text mb-1 text-[8px] tracking-[0.28em] text-zinc-500">{activePoint.name}</span>
                    </div>
                  </div>

                  {error ? (
                    <div className="flex h-full items-center justify-center">
                      <p className="terminal-text text-[10px] tracking-[0.2em] text-red-400/60">{error}</p>
                    </div>
                  ) : (
                    <>
                      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-full w-full">
                        <defs>
                          <linearGradient id="analytics-area" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#00f2ff" stopOpacity="0.24" />
                            <stop offset="100%" stopColor="#00f2ff" stopOpacity="0" />
                          </linearGradient>
                          <linearGradient id="analytics-line" x1="0" y1="0" x2="1" y2="0">
                            <stop offset="0%" stopColor="rgba(255,255,255,0.35)" />
                            <stop offset="50%" stopColor="#00f2ff" />
                            <stop offset="100%" stopColor="rgba(255,255,255,0.35)" />
                          </linearGradient>
                        </defs>

                        {tickValues.map((tick) => (
                          <g key={`tick-${tick.value}`}>
                            <line
                              x1={CHART_FRAME.left}
                              x2={CHART_FRAME.right}
                              y1={tick.y}
                              y2={tick.y}
                              stroke="rgba(255,255,255,0.05)"
                              strokeDasharray="1.2 2.4"
                            />
                            <text
                              x="3"
                              y={tick.y + 1.5}
                              fill="rgba(255,255,255,0.28)"
                              fontSize="3"
                              fontFamily="JetBrains Mono"
                            >
                              {tick.value}
                            </text>
                          </g>
                        ))}

                        <path d={areaPath} fill="url(#analytics-area)" />
                        <path d={linePath} fill="none" stroke="url(#analytics-line)" strokeWidth="0.7" strokeLinejoin="round" />

                        {chartPoints.map((point, index) => (
                          <g key={point.name}>
                            {index === resolvedActiveIndex ? (
                              <line
                                x1={point.x}
                                x2={point.x}
                                y1={CHART_FRAME.top}
                                y2={CHART_FRAME.bottom}
                                stroke="rgba(0,242,255,0.22)"
                                strokeDasharray="1.5 2.5"
                              />
                            ) : null}
                            <circle
                              cx={point.x}
                              cy={point.y}
                              r={index === resolvedActiveIndex ? '1.6' : '1'}
                              fill={index === resolvedActiveIndex ? '#00f2ff' : 'rgba(255,255,255,0.75)'}
                              stroke={index === resolvedActiveIndex ? 'rgba(255,255,255,0.9)' : 'transparent'}
                              strokeWidth="0.35"
                            />
                          </g>
                        ))}
                      </svg>

                      <div className="absolute inset-0 grid" style={{ gridTemplateColumns: `repeat(${chartPoints.length || 1}, 1fr)` }}>
                        {chartPoints.map((point, index) => (
                          <button
                            key={`hit-${point.name}`}
                            type="button"
                            aria-label={`Inspect ${point.name}`}
                            onMouseEnter={() => setActiveIndex(index)}
                            onFocus={() => setActiveIndex(index)}
                            className="h-full w-full outline-none"
                          />
                        ))}
                      </div>

                      <div className="pointer-events-none absolute inset-x-10 bottom-5 flex items-center justify-between text-[10px] uppercase tracking-[0.3em] text-zinc-500">
                        {chartPoints.map((point) => (
                          <span key={`label-${point.name}`} className={point.name === activePoint.name ? 'text-cyan-300' : ''}>
                            {point.name}
                          </span>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-8 lg:col-span-4">
            {statCards.map((stat) => (
              <motion.div
                key={stat.label}
                whileHover={{ x: 10 }}
                className="holographic-panel rounded-[2rem] border-cyan-400/5 p-10 transition-all duration-500 hover:border-cyan-400/20"
              >
                <div className="mb-6 flex items-center gap-3">
                  <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_10px_#00f2ff]" />
                  <p className="terminal-text text-[9px] tracking-[0.3em] text-zinc-600">{stat.label}</p>
                </div>
                <div className="flex items-end justify-between">
                  <span className="text-5xl font-bold tracking-tighter">{stat.value}</span>
                  <span className="terminal-text mb-1 text-[9px] text-cyan-400">{stat.delta}</span>
                </div>
              </motion.div>
            ))}

            {metrics?.pipeline_metrics ? (
              <motion.div
                whileHover={{ x: 10 }}
                className="holographic-panel rounded-[2rem] border-cyan-400/5 p-10 transition-all duration-500 hover:border-cyan-400/20"
              >
                <div className="mb-6 flex items-center gap-3">
                  <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_10px_#00f2ff]" />
                  <p className="terminal-text text-[9px] tracking-[0.3em] text-zinc-600">NMS_REDUCTION</p>
                </div>
                <div className="flex items-end justify-between">
                  <span className="text-5xl font-bold tracking-tighter">
                    {formatPct(metrics.pipeline_metrics.nms_reduction_pct)}
                  </span>
                  <span className="terminal-text mb-1 text-[9px] text-cyan-400">
                    {metrics.pipeline_metrics.total_raw_detections} raw
                  </span>
                </div>
              </motion.div>
            ) : null}
          </div>
        </div>
      </div>
    </section>
  )
}
