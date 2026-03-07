import { Component, type ReactNode } from 'react'

type ErrorBoundaryProps = {
  children: ReactNode
  fallback?: ReactNode
}

type ErrorBoundaryState = {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback

      return (
        <div className="mx-auto max-w-xl px-6 py-24 text-center">
          <div className="rounded-2xl border border-red-500/20 bg-red-500/5 px-8 py-12">
            <p className="metadata-micro mb-4 text-red-400">Runtime Error</p>
            <p className="mb-4 text-sm text-zinc-300">
              Something went wrong in this section. Try reloading the page.
            </p>
            {this.state.error && (
              <pre className="mt-4 overflow-x-auto rounded-lg bg-black/50 px-4 py-3 text-left text-xs text-zinc-500">
                {this.state.error.message}
              </pre>
            )}
            <button
              onClick={() => window.location.reload()}
              className="mt-6 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-6 py-2 text-sm text-cyan-400 transition hover:bg-cyan-400/20"
            >
              Reload Page
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
