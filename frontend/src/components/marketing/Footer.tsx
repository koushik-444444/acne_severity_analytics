import { Activity, Database, X } from 'lucide-react'

export function Footer() {
  return (
    <footer className="border-t border-white/5 bg-black py-32">
      <div className="mx-auto max-w-7xl px-8">
        <div className="mb-32 flex flex-col items-start justify-between gap-20 md:flex-row">
          <div className="max-w-xs">
            <div className="mb-8 flex items-center gap-3">
              <div className="flex h-8 w-8 rotate-45 items-center justify-center bg-cyan-400">
                <Activity className="h-5 w-5 -rotate-45 text-black" />
              </div>
              <span className="terminal-text text-lg font-bold tracking-widest">
                ClearSkin<span className="text-cyan-400">AI</span>
              </span>
            </div>
            <p className="text-sm font-light leading-relaxed text-zinc-600">
              Advancing human health through liquid-neural precision and ethical AI integration.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-24 sm:grid-cols-3">
            <div>
              <h4 className="terminal-text mb-8 text-[10px] font-bold text-white">NODE_NETWORK</h4>
              <ul className="terminal-text space-y-4 text-[11px] text-zinc-600">
                <li><a href="#workspace" className="transition-colors hover:text-cyan-400">DIAGNOSTICS</a></li>
                <li><a href="#analytics" className="transition-colors hover:text-cyan-400">AGI_MAPPING</a></li>
                <li><a href="#workspace" className="transition-colors hover:text-cyan-400">PRESCRIPTION</a></li>
              </ul>
            </div>
            <div>
              <h4 className="terminal-text mb-8 text-[10px] font-bold text-white">PROTOCOL</h4>
              <ul className="terminal-text space-y-4 text-[11px] text-zinc-600">
                <li><a href="#features" className="transition-colors hover:text-cyan-400">RESEARCH</a></li>
                <li><a href="#workspace" className="transition-colors hover:text-cyan-400">HIPAA_SECURE</a></li>
                <li><a href="#analytics" className="transition-colors hover:text-cyan-400">OPEN_LAB</a></li>
              </ul>
            </div>
          </div>
        </div>

        <div className="flex flex-col items-center justify-between gap-8 border-t border-white/5 pt-12 sm:flex-row">
          <p className="terminal-text text-[9px] tracking-[0.2em] text-zinc-700">© {new Date().getFullYear()} CLEARSKIN_AI_CORE. ALL_RIGHTS_RESERVED.</p>
          <div className="flex gap-10">
            <X className="h-4 w-4 cursor-pointer text-zinc-700 transition-colors hover:text-cyan-400" />
            <Activity className="h-4 w-4 cursor-pointer text-zinc-700 transition-colors hover:text-cyan-400" />
            <Database className="h-4 w-4 cursor-pointer text-zinc-700 transition-colors hover:text-cyan-400" />
          </div>
        </div>
      </div>
    </footer>
  )
}
