import { cn } from '../../lib/utils'

export function SkeletonPulse({ className }: { className?: string }) {
  return (
    <div className={cn('animate-pulse rounded-xl bg-white/5', className)} />
  )
}

export function SessionLoadingSkeleton() {
  return (
    <div className="space-y-8" aria-busy="true" aria-label="Loading session">
      <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="holographic-panel rounded-[1.75rem] p-5">
            <SkeletonPulse className="mb-2 h-3 w-20" />
            <SkeletonPulse className="h-7 w-28" />
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.3fr_0.7fr]">
        <div className="rounded-[1.75rem] border border-white/5 bg-black/30 p-4">
          <SkeletonPulse className="mb-4 h-4 w-48" />
          <SkeletonPulse className="h-[520px] w-full rounded-[1.25rem]" />
        </div>
        <div className="space-y-6">
          <div className="holographic-panel rounded-[1.75rem] p-6">
            <SkeletonPulse className="mb-4 h-3 w-32" />
            <SkeletonPulse className="mb-3 h-14 w-full" />
            <SkeletonPulse className="h-14 w-full" />
          </div>
          <div className="holographic-panel rounded-[1.75rem] p-6">
            <SkeletonPulse className="mb-4 h-3 w-40" />
            {Array.from({ length: 3 }).map((_, i) => (
              <SkeletonPulse key={i} className="mb-2 h-12 w-full" />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export function HistoryCardSkeleton() {
  return (
    <div className="w-full rounded-2xl border border-white/5 bg-white/3 p-4" aria-hidden="true">
      <div className="mb-2 flex items-start justify-between">
        <SkeletonPulse className="h-3 w-16" />
        <SkeletonPulse className="h-3 w-20" />
      </div>
      <SkeletonPulse className="mb-2 h-5 w-24" />
      <SkeletonPulse className="h-3 w-36" />
    </div>
  )
}
