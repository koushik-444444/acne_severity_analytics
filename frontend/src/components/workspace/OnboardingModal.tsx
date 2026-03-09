import { useEffect, useRef } from 'react'

export function OnboardingModal({ onDismiss }: { onDismiss: () => void }) {
  const dialogRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return

    const focusable = dialog.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )
    const first = focusable[0]
    const last = focusable[focusable.length - 1]

    function trapFocus(e: KeyboardEvent) {
      if (e.key !== 'Tab') return
      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault()
          last?.focus()
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault()
          first?.focus()
        }
      }
    }

    dialog.addEventListener('keydown', trapFocus)
    return () => dialog.removeEventListener('keydown', trapFocus)
  }, [])

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Workspace onboarding"
      ref={dialogRef}
      className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm"
      onKeyDown={(e) => { if (e.key === 'Escape') onDismiss() }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onDismiss() }}
    >
      <div className="holographic-panel max-w-2xl rounded-[2rem] p-8">
        <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">WORKSPACE ONBOARDING</div>
        <h3 className="mb-4 text-3xl font-bold tracking-tight">How to use the clinical workspace</h3>
        <div className="space-y-3 text-sm leading-relaxed text-zinc-400">
          <div>1. Upload a case and wait for the first warmup analysis to complete.</div>
          <div>2. Use <kbd className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-xs text-zinc-300">+</kbd>, <kbd className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-xs text-zinc-300">-</kbd>, <kbd className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-xs text-zinc-300">R</kbd>, <kbd className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-xs text-zinc-300">O</kbd>, <kbd className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-xs text-zinc-300">C</kbd>, and <kbd className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 font-mono text-xs text-zinc-300">F</kbd> for quick viewer controls.</div>
          <div>3. Pin a baseline session from the archive to compare future sessions against it.</div>
          <div>4. Hover lesion cards to highlight matching detections in the viewer.</div>
          <div>5. Save session notes and export a clinical, compact, or presentation report.</div>
        </div>
        <div className="mt-6 flex justify-end">
          <button
            type="button"
            autoFocus
            onClick={onDismiss}
            className="rounded-full bg-cyan-400 px-5 py-2 text-sm font-semibold text-black"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  )
}
