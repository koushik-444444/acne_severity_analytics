import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { createElement } from 'react'

import { ErrorBoundary } from './ErrorBoundary'

function ThrowingComponent({ error }: { error: Error }) {
  throw error
}

describe('ErrorBoundary', () => {
  // Suppress React error boundary console.error noise
  const originalError = console.error
  beforeEach(() => {
    console.error = vi.fn()
  })
  afterEach(() => {
    console.error = originalError
  })

  it('renders children when no error', () => {
    render(
      createElement(ErrorBoundary, null,
        createElement('div', null, 'Safe content')
      )
    )
    expect(screen.getByText('Safe content')).toBeInTheDocument()
  })

  it('renders default fallback on error', () => {
    render(
      createElement(ErrorBoundary, null,
        createElement(ThrowingComponent, { error: new Error('Test crash') })
      )
    )
    expect(screen.getByText('Runtime Error')).toBeInTheDocument()
    expect(screen.getByText('Test crash')).toBeInTheDocument()
    expect(screen.getByText('Reload Page')).toBeInTheDocument()
  })

  it('renders custom fallback when provided', () => {
    const fallback = createElement('div', null, 'Custom fallback')
    render(
      createElement(ErrorBoundary, { fallback },
        createElement(ThrowingComponent, { error: new Error('Boom') })
      )
    )
    expect(screen.getByText('Custom fallback')).toBeInTheDocument()
  })

  it('shows error message in the default fallback', () => {
    render(
      createElement(ErrorBoundary, null,
        createElement(ThrowingComponent, { error: new Error('Specific error msg') })
      )
    )
    expect(screen.getByText('Specific error msg')).toBeInTheDocument()
  })
})
