import { useEffect, useState } from 'react'
import { Activity } from 'lucide-react'

import { cn } from '../../lib/utils'

export function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <nav
      className={cn(
        'fixed left-0 right-0 top-0 z-50 px-8 py-6 transition-all duration-500',
        isScrolled ? 'border-b border-cyan-400/10 bg-black/80 py-4 backdrop-blur-2xl' : 'bg-transparent',
      )}
    >
      <div className="mx-auto flex max-w-screen-2xl items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="flex h-10 w-10 rotate-45 items-center justify-center bg-cyan-400">
              <Activity className="h-6 w-6 -rotate-45 text-black" />
            </div>
            <div className="absolute -inset-1 rotate-45 animate-pulse border border-cyan-400/30" />
          </div>
          <span className="terminal-text text-xl font-bold tracking-widest">
            ClearSkin<span className="text-cyan-400">AI</span>
          </span>
        </div>

        <div className="hidden items-center gap-12 md:flex">
          {['Analysis', 'Clinical', 'Research', 'Archive'].map((item) => (
            <a
              key={item}
              href="#"
              className="terminal-text text-[11px] font-bold text-zinc-500 transition-colors hover:text-cyan-400"
            >
              {item}
            </a>
          ))}
          <button className="terminal-text border border-cyan-400/50 bg-transparent px-6 py-2 text-[10px] font-bold text-cyan-400 transition-all hover:bg-cyan-400 hover:text-black">
            INIT_SESSION
          </button>
        </div>
      </div>
    </nav>
  )
}
