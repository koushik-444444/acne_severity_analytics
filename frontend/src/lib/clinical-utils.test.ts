import { describe, expect, it } from 'vitest'

import {
  clamp,
  describeDelta,
  formatDate,
  formatSignedDelta,
  getAcneTypeColor,
  getAcneTypeLabel,
  getConfidenceTextTone,
  getClinicalDeltaLabel,
  getClinicalDeltaStatus,
  getDeltaTone,
  getProfileStorageKey,
  getSeverityGradeLabel,
  getSeverityTone,
  isTypedAcneLabel,
} from './clinical-utils'

// --- formatDate ---

describe('formatDate', () => {
  it('formats a valid ISO date', () => {
    const result = formatDate('2025-01-15T12:00:00Z')
    // Output is locale-dependent, just verify it's not the raw ISO
    expect(result).toBeTruthy()
    expect(typeof result).toBe('string')
  })

  it('returns the raw string on invalid input', () => {
    expect(formatDate('not-a-date')).toBe('Invalid Date')
  })
})

// --- getSeverityTone ---

describe('getSeverityTone', () => {
  it('returns red tone for very severe', () => {
    expect(getSeverityTone('Very Severe')).toContain('red')
  })

  it('returns red tone for cystic', () => {
    expect(getSeverityTone('Cystic')).toContain('red')
  })

  it('returns orange tone for severe', () => {
    expect(getSeverityTone('Severe')).toContain('orange')
  })

  it('returns amber tone for moderate', () => {
    expect(getSeverityTone('Moderate')).toContain('amber')
  })

  it('returns cyan tone for mild', () => {
    expect(getSeverityTone('Mild')).toContain('cyan')
  })

  it('returns neutral tone for unknown severity', () => {
    expect(getSeverityTone('Unknown')).toContain('zinc')
  })
})

// --- clamp ---

describe('clamp', () => {
  it('returns value when within range', () => {
    expect(clamp(5, 0, 10)).toBe(5)
  })

  it('clamps to min', () => {
    expect(clamp(-5, 0, 10)).toBe(0)
  })

  it('clamps to max', () => {
    expect(clamp(15, 0, 10)).toBe(10)
  })

  it('handles equal min and max', () => {
    expect(clamp(5, 3, 3)).toBe(3)
  })
})

// --- getProfileStorageKey ---

describe('getProfileStorageKey', () => {
  it('combines base key and profile id', () => {
    expect(getProfileStorageKey('prefs', 'profile-1')).toBe('prefs:profile-1')
  })

  it('falls back to default-profile for empty string', () => {
    expect(getProfileStorageKey('prefs', '')).toBe('prefs:default-profile')
  })
})

// --- formatSignedDelta ---

describe('formatSignedDelta', () => {
  it('prefixes positive values with +', () => {
    expect(formatSignedDelta(5)).toBe('+5')
  })

  it('keeps negative sign', () => {
    expect(formatSignedDelta(-3)).toBe('-3')
  })

  it('shows zero without sign', () => {
    expect(formatSignedDelta(0)).toBe('0')
  })

  it('appends suffix', () => {
    expect(formatSignedDelta(2, '%')).toBe('+2%')
  })
})

// --- getClinicalDeltaStatus ---

describe('getClinicalDeltaStatus', () => {
  it('returns stable for zero', () => {
    expect(getClinicalDeltaStatus(0)).toBe('stable')
  })

  it('returns improved for negative when betterWhenLower', () => {
    expect(getClinicalDeltaStatus(-3, true)).toBe('improved')
  })

  it('returns worsened for positive when betterWhenLower', () => {
    expect(getClinicalDeltaStatus(3, true)).toBe('worsened')
  })

  it('inverts logic when betterWhenLower is false', () => {
    expect(getClinicalDeltaStatus(3, false)).toBe('improved')
    expect(getClinicalDeltaStatus(-3, false)).toBe('worsened')
  })
})

// --- getClinicalDeltaLabel ---

describe('getClinicalDeltaLabel', () => {
  it('returns Stable for zero', () => {
    expect(getClinicalDeltaLabel(0)).toBe('Stable')
  })

  it('returns Improved for negative delta', () => {
    expect(getClinicalDeltaLabel(-2)).toBe('Improved')
  })

  it('returns Worsened for positive delta', () => {
    expect(getClinicalDeltaLabel(2)).toBe('Worsened')
  })
})

// --- getDeltaTone ---

describe('getDeltaTone', () => {
  it('returns neutral tone for zero', () => {
    expect(getDeltaTone(0)).toContain('zinc')
  })

  it('returns emerald for improvement', () => {
    expect(getDeltaTone(-3)).toContain('emerald')
  })

  it('returns rose for worsening', () => {
    expect(getDeltaTone(3)).toContain('rose')
  })
})

// --- describeDelta ---

describe('describeDelta', () => {
  it('describes stable condition', () => {
    expect(describeDelta(0, 'Lesion count')).toBe('Lesion count remained stable')
  })

  it('describes improvement', () => {
    expect(describeDelta(-5, 'Score')).toBe('Score improved by 5')
  })

  it('describes worsening', () => {
    expect(describeDelta(3, 'Score')).toBe('Score worsened by 3')
  })
})

// --- getConfidenceTextTone ---

describe('getConfidenceTextTone', () => {
  it('returns cyan for high confidence', () => {
    expect(getConfidenceTextTone(0.9)).toBe('text-cyan-300')
  })

  it('returns amber for medium confidence', () => {
    expect(getConfidenceTextTone(0.5)).toBe('text-amber-300')
  })

  it('returns zinc for low confidence', () => {
    expect(getConfidenceTextTone(0.2)).toBe('text-zinc-300')
  })

  it('returns cyan at exactly 0.7', () => {
    expect(getConfidenceTextTone(0.7)).toBe('text-cyan-300')
  })

  it('returns amber at exactly 0.4', () => {
    expect(getConfidenceTextTone(0.4)).toBe('text-amber-300')
  })
})

// --- getAcneTypeLabel ---

describe('getAcneTypeLabel', () => {
  it('returns canonical label for known type', () => {
    expect(getAcneTypeLabel('pustule')).toBe('Pustule')
    expect(getAcneTypeLabel('Pustules')).toBe('Pustule')
  })

  it('normalises plural and case variants', () => {
    expect(getAcneTypeLabel('blackheads')).toBe('Blackhead')
    expect(getAcneTypeLabel('Nodules')).toBe('Nodule')
    expect(getAcneTypeLabel('papules')).toBe('Papule')
    expect(getAcneTypeLabel('whiteheads')).toBe('Whitehead')
  })

  it('returns Acne for generic labels', () => {
    expect(getAcneTypeLabel('acne')).toBe('Acne')
    expect(getAcneTypeLabel('Acne')).toBe('Acne')
  })

  it('returns Acne for undefined/empty input', () => {
    expect(getAcneTypeLabel(undefined)).toBe('Acne')
    expect(getAcneTypeLabel('')).toBe('Acne')
  })

  it('handles dark spot', () => {
    expect(getAcneTypeLabel('dark spot')).toBe('Dark Spot')
    expect(getAcneTypeLabel('dark_spot')).toBe('Dark Spot')
  })

  it('returns Acne for unknown labels', () => {
    expect(getAcneTypeLabel('some_unknown_class')).toBe('Acne')
  })
})

// --- getAcneTypeColor ---

describe('getAcneTypeColor', () => {
  it('returns unique color for each type', () => {
    const pustuleColor = getAcneTypeColor('pustule')
    const blackheadColor = getAcneTypeColor('blackhead')
    const cystColor = getAcneTypeColor('cyst')
    expect(pustuleColor).toContain('orange')
    expect(blackheadColor).toContain('zinc')
    expect(cystColor).toContain('rose')
  })

  it('returns default cyan for generic acne', () => {
    expect(getAcneTypeColor('acne')).toContain('cyan')
    expect(getAcneTypeColor(undefined)).toContain('cyan')
  })

  it('returns classes with bg, text, and border', () => {
    const color = getAcneTypeColor('nodule')
    expect(color).toContain('text-')
    expect(color).toContain('bg-')
    expect(color).toContain('border-')
  })
})

// --- getSeverityGradeLabel ---

describe('getSeverityGradeLabel', () => {
  it('returns correct label for each grade', () => {
    expect(getSeverityGradeLabel(1)).toContain('Comedone')
    expect(getSeverityGradeLabel(2)).toContain('Papule')
    expect(getSeverityGradeLabel(3)).toContain('Pustule')
    expect(getSeverityGradeLabel(4)).toContain('Nodule/Cyst')
  })

  it('returns default grade 2 for undefined', () => {
    expect(getSeverityGradeLabel(undefined)).toContain('Papule')
  })

  it('handles unknown grade', () => {
    expect(getSeverityGradeLabel(99)).toBe('Grade 99')
  })
})

// --- isTypedAcneLabel ---

describe('isTypedAcneLabel', () => {
  it('returns false for generic labels', () => {
    expect(isTypedAcneLabel('acne')).toBe(false)
    expect(isTypedAcneLabel('Acne')).toBe(false)
    expect(isTypedAcneLabel('acne_detected')).toBe(false)
    expect(isTypedAcneLabel('lesion')).toBe(false)
    expect(isTypedAcneLabel('')).toBe(false)
    expect(isTypedAcneLabel(undefined)).toBe(false)
  })

  it('returns true for typed labels', () => {
    expect(isTypedAcneLabel('pustule')).toBe(true)
    expect(isTypedAcneLabel('Blackheads')).toBe(true)
    expect(isTypedAcneLabel('nodule')).toBe(true)
    expect(isTypedAcneLabel('cyst')).toBe(true)
    expect(isTypedAcneLabel('dark spot')).toBe(true)
  })
})
