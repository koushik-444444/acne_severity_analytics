export function SessionNotesPanel({
  noteDraft,
  onDraftChange,
  onSave,
  isSaving,
}: {
  noteDraft: string
  onDraftChange: (draft: string) => void
  onSave: () => void
  isSaving: boolean
}) {
  return (
    <div className="holographic-panel rounded-[1.75rem] p-6">
      <h3 className="terminal-text mb-3 text-[10px] text-cyan-400/80">SESSION NOTES</h3>
      <textarea
        aria-label="Session notes"
        value={noteDraft}
        onChange={(e) => onDraftChange(e.target.value)}
        placeholder="Add dermatologist notes, case observations, or treatment context..."
        className="min-h-[140px] w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white outline-none"
      />
      <div className="mt-4 flex items-center justify-between">
        <div className="text-xs text-zinc-500">Notes persist with this session.</div>
        <button
          type="button"
          onClick={onSave}
          disabled={isSaving}
          className="rounded-full border border-white/10 px-4 py-2 text-xs text-white transition-colors hover:border-cyan-400/20 hover:text-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isSaving ? 'Saving...' : 'Save notes'}
        </button>
      </div>
    </div>
  )
}
