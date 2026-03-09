import { memo } from 'react'

export const MetricCard = memo(function MetricCard({ label, value, accent = false, tone }: { label: string; value: string; accent?: boolean; tone?: string }) {
  return (
    <div className={`rounded-[1.5rem] border border-white/5 bg-white/3 p-6 ${tone ?? ''}`}>
      <div className="terminal-text mb-2 text-[9px] text-zinc-500">{label}</div>
      <div className={accent ? 'text-3xl font-bold text-cyan-400' : 'text-3xl font-bold text-white'}>{value}</div>
    </div>
  )
})
