import { useEffect, useState } from 'react'

import { api } from '../services/api'
import type { SessionStatus } from '../types/api'

/**
 * Manages session status state and polls every 1500ms while
 * an analysis is running. Stops when the session reports
 * completed or failed. Uses AbortController to cancel in-flight
 * requests on cleanup to prevent orphaned network activity.
 *
 * Returns `[status, setStatus]` — the caller can push status
 * updates via `setStatus` and the poller will also write to it.
 */
export function useStatusPoller(
  sessionId: string | null,
  isAnalyzing: boolean,
) {
  const [status, setStatus] = useState<SessionStatus | null>(null)

  useEffect(() => {
    if (!sessionId || !isAnalyzing) return
    let cancelled = false
    const abortController = new AbortController()

    const pollStatus = async () => {
      while (!cancelled) {
        try {
          const nextStatus = await api.getStatus(sessionId, abortController.signal)
          if (cancelled) {
            return
          }
          setStatus(nextStatus)
          if (nextStatus.completed || nextStatus.failed) {
            return
          }
        } catch {
          if (cancelled) {
            return
          }
        }

        await new Promise((resolve) => window.setTimeout(resolve, 1500))
      }
    }

    void pollStatus()

    return () => {
      cancelled = true
      abortController.abort()
    }
  }, [sessionId, isAnalyzing])

  return [status, setStatus] as const
}
