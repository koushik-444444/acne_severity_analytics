import { motion } from 'framer-motion'

import { HeroScene } from './HeroScene'
import { MetadataLabel } from './MetadataLabel'

export function Hero() {
  return (
    <section className="medical-grid relative flex min-h-screen flex-col items-center justify-center overflow-hidden pt-20">
      <div aria-hidden="true" className="editorial-title pointer-events-none absolute left-0 top-1/2 -translate-y-1/2 -rotate-12 select-none text-[25vw]">
        Archival_07
      </div>

      <div aria-hidden="true" className="pointer-events-none absolute inset-0 opacity-40">
        <MetadataLabel className="absolute left-12 top-32">LN_LOAD: 0.992ms</MetadataLabel>
        <MetadataLabel className="absolute left-8 top-1/2 origin-left -rotate-90">SEQ_BUFFER_ACTIVE</MetadataLabel>
        <MetadataLabel className="absolute bottom-12 right-12">REF_SYNC: UTC_STABLE</MetadataLabel>
        <div className="absolute right-[20%] top-1/4 h-32 w-px bg-gradient-to-b from-cyan-400/0 via-cyan-400/20 to-cyan-400/0" />
      </div>

      <div className="relative z-10 mx-auto w-full max-w-7xl px-8">
        <div className="grid grid-cols-1 items-center gap-12 lg:grid-cols-12">
          <motion.div
            initial={{ opacity: 0, x: -50, filter: 'blur(10px)' }}
            animate={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
            transition={{ duration: 1.2, ease: [0.19, 1, 0.22, 1] }}
            className="relative z-30 lg:col-span-5"
          >
            <div className="mb-12 inline-flex items-center gap-4 rounded-full border border-cyan-400/10 bg-cyan-400/2 px-4 py-2">
              <span aria-hidden="true" className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-400 shadow-[0_0_10px_#00f2ff]" />
              <span className="font-mono text-[9px] font-bold uppercase tracking-[0.3em] text-cyan-400/60">
                Neural Protocol Established
              </span>
            </div>

            <h1 className="mb-10 text-7xl font-extrabold leading-[0.85] tracking-tighter md:text-9xl">
              LIQUID <br />
              <span className="italic text-cyan-400">GENESIS.</span>
            </h1>

            <p className="mb-16 max-w-md border-l border-cyan-400/20 pl-8 text-xl font-light leading-relaxed text-zinc-400 md:text-2xl">
              Clinical-grade acne severity analytics powered by{' '}
              <span className="font-medium text-white">Liquid Neural Networks</span>.
            </p>

            <div className="flex flex-wrap gap-8">
              <a href="#workspace" aria-label="Go to diagnostic workspace" className="btn-precision terminal-text px-12 py-6 text-[10px] font-bold text-cyan-400">
                INIT_DIAGNOSTIC
              </a>
              <a href="#features" aria-label="View features and research" className="terminal-text border border-white/5 bg-white/2 px-12 py-6 text-[10px] font-bold text-white/40 transition-all hover:border-white/20 hover:text-white">
                WHITE_PAPER
              </a>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.8, x: 100 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            transition={{ duration: 1.8, ease: [0.19, 1, 0.22, 1] }}
            className="relative flex h-full items-center justify-end lg:col-span-7"
          >
            <div className="relative max-w-4xl translate-x-[20%] translate-y-[5%] aspect-square w-full">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative h-[160%] w-[160%] overflow-hidden">
                  <div className="relative h-full w-full scale-[0.6]">
                    <HeroScene />
                    <div className="pointer-events-none absolute bottom-0 right-0 z-50 h-[20%] w-[40%] bg-black" />
                  </div>
                </div>
              </div>

              <div aria-hidden="true" className="absolute right-1/4 top-1/4 h-px w-32 rotate-45 bg-cyan-400/20" />
              <div aria-hidden="true" className="absolute right-[35%] top-[20%]">
                <MetadataLabel>ROBOT_ID: CS_001</MetadataLabel>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}
