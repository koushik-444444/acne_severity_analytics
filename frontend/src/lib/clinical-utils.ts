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

// ---------------------------------------------------------------------------
// Acne type label and colour utilities
// ---------------------------------------------------------------------------

/**
 * Canonical display labels for acne types returned by the detection models.
 * Keys are lowercase to allow case-insensitive lookup.
 */
const ACNE_TYPE_LABELS: Record<string, string> = {
  blackhead: 'Blackhead',
  blackheads: 'Blackhead',
  whitehead: 'Whitehead',
  whiteheads: 'Whitehead',
  papule: 'Papule',
  papules: 'Papule',
  pustule: 'Pustule',
  pustules: 'Pustule',
  nodule: 'Nodule',
  nodules: 'Nodule',
  cyst: 'Cyst',
  cystic: 'Cyst',
  comedone: 'Comedone',
  'dark spot': 'Dark Spot',
  'dark_spot': 'Dark Spot',
  acne: 'Acne',
}

/**
 * Tailwind colour classes keyed by canonical label.
 * Each entry provides text, bg, and border classes for a
 * consistent colour-coded badge.
 */
const ACNE_TYPE_COLORS: Record<string, string> = {
  Blackhead: 'text-zinc-300 bg-zinc-500/20 border-zinc-500/30',
  Whitehead: 'text-slate-200 bg-slate-400/20 border-slate-400/30',
  Comedone: 'text-zinc-300 bg-zinc-500/20 border-zinc-500/30',
  Papule: 'text-amber-300 bg-amber-500/20 border-amber-500/30',
  Pustule: 'text-orange-300 bg-orange-500/20 border-orange-500/30',
  Nodule: 'text-red-300 bg-red-500/20 border-red-500/30',
  Cyst: 'text-rose-300 bg-rose-500/20 border-rose-500/30',
  'Dark Spot': 'text-purple-300 bg-purple-500/20 border-purple-500/30',
  Acne: 'text-cyan-300 bg-cyan-500/20 border-cyan-500/30',
}

const DEFAULT_ACNE_COLOR = 'text-cyan-300 bg-cyan-500/20 border-cyan-500/30'

/**
 * Convert a raw model class_name into a human-readable label.
 *
 * Returns `'Acne'` for generic / unrecognised labels.
 */
export function getAcneTypeLabel(className?: string): string {
  if (!className) return 'Acne'
  return ACNE_TYPE_LABELS[className.toLowerCase().trim()] ?? 'Acne'
}

/**
 * Return Tailwind utility classes for an acne-type badge.
 *
 * Accepts either the raw model class_name or a canonical label.
 */
export function getAcneTypeColor(className?: string): string {
  const label = getAcneTypeLabel(className)
  return ACNE_TYPE_COLORS[label] ?? DEFAULT_ACNE_COLOR
}

/**
 * GAGS severity grade as a short label.
 */
const SEVERITY_GRADE_LABELS: Record<number, string> = {
  1: 'Grade 1 — Comedone',
  2: 'Grade 2 — Papule',
  3: 'Grade 3 — Pustule',
  4: 'Grade 4 — Nodule/Cyst',
}

export function getSeverityGradeLabel(grade?: number): string {
  if (grade == null) return 'Grade 2 — Papule'
  return SEVERITY_GRADE_LABELS[grade] ?? `Grade ${grade}`
}

/**
 * Return true when the class label carries real type information
 * (i.e. it is NOT a generic catch-all like "acne").
 */
const GENERIC_LABELS = new Set(['acne', 'acne_detected', 'lesion', ''])

export function isTypedAcneLabel(className?: string): boolean {
  if (!className) return false
  return !GENERIC_LABELS.has(className.toLowerCase().trim())
}
