import { createContext, useCallback, useContext, useRef, useState } from 'react'

export type ToastType = 'success' | 'error' | 'info'

export type ToastAction = {
  label: string
  onClick: () => void
}

export type ToastItem = {
  id: string
  type: ToastType
  message: string
  action?: ToastAction
}

type ToastOptions = {
  type?: ToastType
  duration?: number
  action?: ToastAction
}

export type ToastContextValue = {
  toasts: ToastItem[]
  toast: (message: string, options?: ToastOptions) => void
  dismiss: (id: string) => void
}

const DEFAULT_DURATIONS: Record<ToastType, number> = {
  success: 3000,
  error: 5000,
  info: 3000,
}

export const ToastContext = createContext<ToastContextValue | null>(null)

/**
 * Provides toast notification state and actions.
 * Must be used within a <ToastProvider>.
 */
export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext)
  if (!ctx) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return ctx
}

let toastCounter = 0

/**
 * Core toast state manager — used internally by ToastProvider.
 * Returns the context value to be passed through ToastContext.Provider.
 */
export function useToastState(): ToastContextValue {
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const timersRef = useRef<Map<string, number>>(new Map())

  const dismiss = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
    const timer = timersRef.current.get(id)
    if (timer !== undefined) {
      window.clearTimeout(timer)
      timersRef.current.delete(id)
    }
  }, [])

  const toast = useCallback((message: string, options?: ToastOptions) => {
    const type = options?.type ?? 'info'
    const duration = options?.duration ?? DEFAULT_DURATIONS[type]
    const id = `toast-${++toastCounter}-${Date.now()}`

    const item: ToastItem = {
      id,
      type,
      message,
      action: options?.action,
    }

    setToasts(prev => [...prev, item])

    const timer = window.setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
      timersRef.current.delete(id)
    }, duration)

    timersRef.current.set(id, timer)
  }, [])

  return { toasts, toast, dismiss }
}
