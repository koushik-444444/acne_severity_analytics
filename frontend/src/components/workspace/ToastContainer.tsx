import { type ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react'

import { cn } from '../../lib/utils'
import { ToastContext, useToast, useToastState } from '../../hooks/useToast'
import type { ToastItem, ToastType } from '../../hooks/useToast'

const ICON_MAP: Record<ToastType, typeof CheckCircle2> = {
  success: CheckCircle2,
  error: AlertCircle,
  info: Info,
}

const TONE_CLASSES: Record<ToastType, { border: string; icon: string; bg: string; glow: string }> = {
  success: {
    border: 'border-emerald-500/30',
    icon: 'text-emerald-400',
    bg: 'bg-emerald-500/5',
    glow: 'shadow-emerald-500/10',
  },
  error: {
    border: 'border-rose-500/30',
    icon: 'text-rose-400',
    bg: 'bg-rose-500/5',
    glow: 'shadow-rose-500/10',
  },
  info: {
    border: 'border-cyan-500/30',
    icon: 'text-cyan-400',
    bg: 'bg-cyan-500/5',
    glow: 'shadow-cyan-500/10',
  },
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const state = useToastState()

  return (
    <ToastContext.Provider value={state}>
      {children}
      <ToastContainer />
    </ToastContext.Provider>
  )
}

function ToastContainer() {
  const { toasts, dismiss } = useToast()

  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed bottom-6 right-6 z-[9999] flex flex-col gap-3"
    >
      <AnimatePresence mode="popLayout">
        {toasts.map(item => (
          <ToastCard key={item.id} item={item} onDismiss={dismiss} />
        ))}
      </AnimatePresence>
    </div>
  )
}

function ToastCard({ item, onDismiss }: { item: ToastItem; onDismiss: (id: string) => void }) {
  const tone = TONE_CLASSES[item.type]
  const Icon = ICON_MAP[item.type]

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, x: 80, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      className={cn(
        'pointer-events-auto flex w-80 items-start gap-3 rounded-xl border p-4',
        'backdrop-blur-xl shadow-lg',
        'bg-[rgba(5,10,12,0.85)]',
        tone.border,
        tone.bg,
        tone.glow,
      )}
    >
      <Icon className={cn('mt-0.5 size-5 shrink-0', tone.icon)} />

      <div className="min-w-0 flex-1">
        <p className="terminal-text text-[10px] text-zinc-500 mb-1">
          {item.type}
        </p>
        <p className="text-sm leading-snug text-zinc-200">
          {item.message}
        </p>
        {item.action && (
          <button
            type="button"
            onClick={item.action.onClick}
            className={cn(
              'mt-2 text-xs font-medium transition-colors',
              tone.icon,
              'hover:brightness-125',
            )}
          >
            {item.action.label}
          </button>
        )}
      </div>

      <button
        type="button"
        onClick={() => onDismiss(item.id)}
        className="shrink-0 rounded p-0.5 text-zinc-500 transition-colors hover:text-zinc-300"
        aria-label="Dismiss notification"
      >
        <X className="size-4" />
      </button>
    </motion.div>
  )
}
