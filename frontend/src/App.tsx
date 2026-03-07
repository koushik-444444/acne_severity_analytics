import { LazyMotion, domMax } from 'framer-motion'

import { AnalyticsDashboard } from './components/marketing/AnalyticsDashboard'
import { Features } from './components/marketing/Features'
import { Footer } from './components/marketing/Footer'
import { Hero } from './components/marketing/Hero'
import { Navbar } from './components/marketing/Navbar'
import { ClinicalWorkspace } from './components/workspace/ClinicalWorkspace'

export default function App() {
  return (
    <LazyMotion features={domMax}>
      <div className="bg-[#010101] text-white selection:bg-cyan-400 selection:text-black">
        <Navbar />
        <Hero />
        <Features />
        <AnalyticsDashboard />
        <ClinicalWorkspace />
        <Footer />
      </div>
    </LazyMotion>
  )
}
