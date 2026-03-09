import { lazy, Suspense } from 'react'

import { Features } from './components/marketing/Features'
import { Footer } from './components/marketing/Footer'
import { Hero } from './components/marketing/Hero'
import { Navbar } from './components/marketing/Navbar'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ToastProvider } from './components/workspace/ToastContainer'

const AnalyticsDashboard = lazy(() =>
  import('./components/marketing/AnalyticsDashboard').then(m => ({ default: m.AnalyticsDashboard }))
)

const ClinicalWorkspace = lazy(() =>
  import('./components/workspace/ClinicalWorkspace').then(m => ({ default: m.ClinicalWorkspace }))
)

function WorkspaceLoadingFallback() {
  return (
    <div role="status" aria-live="polite" className="flex min-h-[400px] items-center justify-center">
      <div className="text-center">
        <p className="metadata-micro mb-2 text-cyan-400/60">Initializing</p>
        <p className="text-sm text-zinc-500">Loading clinical workspace...</p>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <ToastProvider>
      <div className="bg-[#010101] text-white selection:bg-cyan-400 selection:text-black">
        <a href="#workspace" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[200] focus:rounded-lg focus:bg-cyan-400 focus:px-4 focus:py-2 focus:text-sm focus:font-semibold focus:text-black">
          Skip to main content
        </a>
        <header>
          <Navbar />
        </header>
        <main>
          <Hero />
          <section id="features" aria-label="Features"><Features /></section>
          <section id="analytics" aria-label="Analytics">
            <Suspense fallback={null}>
              <AnalyticsDashboard />
            </Suspense>
          </section>
          <ErrorBoundary>
            <Suspense fallback={<WorkspaceLoadingFallback />}>
              <section id="workspace"><ClinicalWorkspace /></section>
            </Suspense>
          </ErrorBoundary>
        </main>
        <Footer />
      </div>
    </ToastProvider>
  )
}
