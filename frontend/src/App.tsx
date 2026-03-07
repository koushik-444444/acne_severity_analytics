import { lazy, Suspense } from 'react'
import { LazyMotion, domAnimation } from 'framer-motion'

import { AnalyticsDashboard } from './components/marketing/AnalyticsDashboard'
import { Features } from './components/marketing/Features'
import { Footer } from './components/marketing/Footer'
import { Hero } from './components/marketing/Hero'
import { Navbar } from './components/marketing/Navbar'
import { ErrorBoundary } from './components/ErrorBoundary'

const ClinicalWorkspace = lazy(() =>
  import('./components/workspace/ClinicalWorkspace').then(m => ({ default: m.ClinicalWorkspace }))
)

function WorkspaceLoadingFallback() {
  return (
    <div className="flex min-h-[400px] items-center justify-center">
      <div className="text-center">
        <p className="metadata-micro mb-2 text-cyan-400/60">Initializing</p>
        <p className="text-sm text-zinc-500">Loading clinical workspace...</p>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <LazyMotion features={domAnimation}>
      <div className="bg-[#010101] text-white selection:bg-cyan-400 selection:text-black">
        <Navbar />
        <Hero />
        <section id="features"><Features /></section>
        <section id="analytics"><AnalyticsDashboard /></section>
        <ErrorBoundary>
          <Suspense fallback={<WorkspaceLoadingFallback />}>
            <section id="workspace"><ClinicalWorkspace /></section>
          </Suspense>
        </ErrorBoundary>
        <Footer />
      </div>
    </LazyMotion>
  )
}
