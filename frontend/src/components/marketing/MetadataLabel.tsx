import type { ReactNode } from 'react'

import { cn } from '../../lib/utils'

export function MetadataLabel({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn('metadata-micro flex items-center gap-2 animate-flicker', className)}>
      <div className="h-1 w-1 rounded-full bg-cyan-400/40" />
      {children}
    </div>
  )
}
