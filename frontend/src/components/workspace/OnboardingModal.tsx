export function OnboardingModal({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Workspace onboarding"
      className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 p-6 backdrop-blur-sm"
      onKeyDown={(e) => { if (e.key === 'Escape') onDismiss() }}
    >
      <div className="holographic-panel max-w-2xl rounded-[2rem] p-8">
        <div className="terminal-text mb-3 text-[10px] text-cyan-400/80">WORKSPACE ONBOARDING</div>
        <h3 className="mb-4 text-3xl font-bold tracking-tight">How to use the clinical workspace</h3>
        <div className="space-y-3 text-sm leading-relaxed text-zinc-400">
          <div>1. Upload a case and wait for the first warmup analysis to complete.</div>
          <div>2. Use `+`, `-`, `R`, `O`, `C`, and `F` for quick viewer controls.</div>
          <div>3. Pin a baseline session from the archive to compare future sessions against it.</div>
          <div>4. Hover lesion cards to highlight matching detections in the viewer.</div>
          <div>5. Save session notes and export a clinical, compact, or presentation report.</div>
        </div>
        <div className="mt-6 flex justify-end">
          <button
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
