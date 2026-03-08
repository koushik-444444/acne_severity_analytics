/**
 * Pure utility functions extracted from ClinicalWorkspace.tsx.
 *
 * Every export here is a side-effect-free function that can be tested
 * without rendering any React component.
 */

const DEFAULT_PROFILE_ID = 'default-profile'

export function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

export function getSeverityTone(severity: string) {
  const normalized = severity.toLowerCase()
  if (normalized.includes('very severe') || normalized.includes('cystic')) {
    return 'border-red-500/35 bg-red-500/15 text-red-200 shadow-[0_0_30px_rgba(239,68,68,0.12)]'
  }
  if (normalized.includes('severe')) {
    return 'border-orange-500/35 bg-orange-500/15 text-orange-200 shadow-[0_0_30px_rgba(249,115,22,0.12)]'
  }
  if (normalized.includes('moderate')) {
    return 'border-amber-500/35 bg-amber-500/15 text-amber-200 shadow-[0_0_30px_rgba(245,158,11,0.12)]'
  }
  if (normalized.includes('mild')) {
    return 'border-cyan-400/35 bg-cyan-400/10 text-cyan-200 shadow-[0_0_30px_rgba(0,242,255,0.10)]'
  }
  return 'border-white/10 bg-white/5 text-zinc-200'
}

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export function getProfileStorageKey(baseKey: string, profileId: string) {
  return `${baseKey}:${profileId || DEFAULT_PROFILE_ID}`
}

export function formatSignedDelta(value: number, suffix = '') {
  return `${value > 0 ? '+' : ''}${value}${suffix}`
}

export function getClinicalDeltaStatus(value: number, betterWhenLower = true): 'improved' | 'stable' | 'worsened' {
  if (value === 0) {
    return 'stable'
  }

  const improved = betterWhenLower ? value < 0 : value > 0
  return improved ? 'improved' : 'worsened'
}

export function getClinicalDeltaLabel(value: number, betterWhenLower = true) {
  const status = getClinicalDeltaStatus(value, betterWhenLower)
  if (status === 'stable') {
    return 'Stable'
  }
  return status === 'improved' ? 'Improved' : 'Worsened'
}

export function getDeltaTone(value: number, betterWhenLower = true) {
  if (value === 0) {
    return 'border-white/10 bg-white/5 text-zinc-200'
  }

  const improved = betterWhenLower ? value < 0 : value > 0
  return improved
    ? 'border-emerald-400/25 bg-emerald-400/10 text-emerald-100'
    : 'border-rose-400/25 bg-rose-400/10 text-rose-100'
}

export function describeDelta(value: number, label: string, betterWhenLower = true) {
  if (value === 0) {
    return `${label} remained stable`
  }

  const status = getClinicalDeltaStatus(value, betterWhenLower)
  const magnitude = Math.abs(value)
  return status === 'improved'
    ? `${label} improved by ${magnitude}`
    : `${label} worsened by ${magnitude}`
}

export function getConfidenceTextTone(confidence: number) {
  if (confidence >= 0.7) return 'text-cyan-300'
  if (confidence >= 0.4) return 'text-amber-300'
  return 'text-zinc-300'
}

export function readStorageItem(key: string) {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

export function writeStorageItem(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value)
  } catch {
    // ignore storage write failures
  }
}

export function removeStorageItem(key: string) {
  try {
    window.localStorage.removeItem(key)
  } catch {
    // ignore storage removal failures
  }
}
