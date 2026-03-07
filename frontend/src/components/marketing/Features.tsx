import { Activity, Brain, Sparkles, Target } from 'lucide-react'
import { motion } from 'framer-motion'

const FEATURES = [
  {
    icon: <Brain className="h-6 w-6 text-cyan-400" />,
    title: 'NEURAL_SEGMENTATION',
    description: 'Deep-layer facial region mapping via BiSeNet consensus architecture.',
  },
  {
    icon: <Target className="h-6 w-6 text-cyan-400" />,
    title: 'LESION_PRECISION',
    description: 'Multi-scale WBF fusion for sub-pixel detection of inflammatory triggers.',
  },
  {
    icon: <Activity className="h-6 w-6 text-cyan-400" />,
    title: 'GAGS_ANALYTICS',
    description: 'Automated Global Acne Grading System compliance for clinical validation.',
  },
  {
    icon: <Sparkles className="h-6 w-6 text-cyan-400" />,
    title: 'EDGE_INFERENCE',
    description: 'Privacy-first workflows with retained archive intelligence and case comparison.',
  },
]

export function Features() {
  return (
    <section className="relative bg-black py-40">
      <div className="mx-auto max-w-7xl px-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((feature, idx) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.1 }}
              className="glass-panel group rounded-3xl p-10 transition-all duration-500 hover:border-cyan-400/40"
            >
              <div className="mb-8 w-fit rounded-2xl bg-cyan-400/5 p-4 transition-colors group-hover:bg-cyan-400/10">
                {feature.icon}
              </div>
              <h3 className="terminal-text mb-4 text-sm font-bold tracking-widest">{feature.title}</h3>
              <p className="text-sm font-light leading-relaxed text-zinc-500">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
