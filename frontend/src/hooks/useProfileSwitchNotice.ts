import { useEffect, useRef, useState } from 'react'

/**
 * Tracks active profile changes and surfaces a timed notice banner
 * for 2600ms after each switch.
 */
export function useProfileSwitchNotice(activeProfileId: string) {
  const [profileSwitchNotice, setProfileSwitchNotice] = useState<{ from: string; to: string } | null>(null)
  const hasHydratedInitialProfileRef = useRef(false)
  const previousProfileRef = useRef(activeProfileId)
  const timerRef = useRef<number | null>(null)

  useEffect(() => {
    if (!hasHydratedInitialProfileRef.current) {
      hasHydratedInitialProfileRef.current = true
      previousProfileRef.current = activeProfileId
      return
    }

    const previousProfileId = previousProfileRef.current
    if (previousProfileId === activeProfileId) {
      return
    }

    setProfileSwitchNotice({ from: previousProfileId, to: activeProfileId })
    previousProfileRef.current = activeProfileId

    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current)
    }

    timerRef.current = window.setTimeout(() => {
      setProfileSwitchNotice(null)
      timerRef.current = null
    }, 2600)
  }, [activeProfileId])

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current)
      }
    }
  }, [])

  return profileSwitchNotice
}
