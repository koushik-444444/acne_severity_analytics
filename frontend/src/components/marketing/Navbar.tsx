import { useEffect, useState } from 'react'
import { Activity, Menu, X } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'

import { cn } from '../../lib/utils'

const NAV_LINKS = [
  { label: 'Analysis', href: '#workspace' },
  { label: 'Clinical', href: '#analytics' },
  { label: 'Research', href: '#features' },
  { label: 'Archive', href: '#workspace' },
]

export function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  // Close mobile menu on navigation
  const handleNavClick = () => setMobileOpen(false)

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

        {/* Desktop nav */}
        <div className="hidden items-center gap-12 md:flex">
          {NAV_LINKS.map((item) => (
            <a
              key={item.label}
              href={item.href}
              className="terminal-text text-[11px] font-bold text-zinc-500 transition-colors hover:text-cyan-400"
            >
              {item.label}
            </a>
          ))}
          <a href="#workspace" className="terminal-text border border-cyan-400/50 bg-transparent px-6 py-2 text-[10px] font-bold text-cyan-400 transition-all hover:bg-cyan-400 hover:text-black">
            INIT_SESSION
          </a>
        </div>

        {/* Mobile hamburger */}
        <button
          type="button"
          onClick={() => setMobileOpen((v) => !v)}
          aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
          className="inline-flex items-center justify-center rounded-lg border border-white/10 p-2 text-zinc-400 transition-colors hover:border-cyan-400/20 hover:text-white md:hidden"
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {/* Mobile menu panel */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="overflow-hidden md:hidden"
          >
            <div className="flex flex-col gap-4 pb-6 pt-4">
              {NAV_LINKS.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  onClick={handleNavClick}
                  className="terminal-text text-[11px] font-bold text-zinc-400 transition-colors hover:text-cyan-400"
                >
                  {item.label}
                </a>
              ))}
              <a
                href="#workspace"
                onClick={handleNavClick}
                className="terminal-text inline-block border border-cyan-400/50 bg-transparent px-6 py-2 text-center text-[10px] font-bold text-cyan-400 transition-all hover:bg-cyan-400 hover:text-black"
              >
                INIT_SESSION
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  )
}
