import { Camera, Loader2 } from 'lucide-react'
import { motion } from 'framer-motion'

import type { PrivacyConfig, SessionStatus } from '../../types/api'

export function UploadPanel({
  previewUrl,
  isAnalyzing,
  status,
  privacyMode,
  retentionHours,
  privacy,
  selectedFile,
  onFileChange,
  onStart,
}: {
  previewUrl: string | null
  isAnalyzing: boolean
  status: SessionStatus | null
  privacyMode: boolean
  retentionHours: number
  privacy: PrivacyConfig | null
  selectedFile: File | null
  onFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void
  onStart: () => void
}) {
  return (
    <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1.2fr_0.8fr]">
      <div className="rounded-[2rem] border border-dashed border-white/10 bg-white/3 p-8">
        <label className="group block cursor-pointer overflow-hidden rounded-[1.5rem] border border-white/5 bg-black/30 p-8 text-center transition-all hover:border-cyan-400/25 hover:bg-cyan-400/5">
          <input type="file" className="hidden" accept="image/*" onChange={onFileChange} disabled={isAnalyzing} aria-label="Select clinical image for analysis" />
          {previewUrl ? (
            <div className="relative overflow-hidden rounded-[1.5rem]">
              <img src={previewUrl} alt="Preview" className="h-[420px] w-full object-cover opacity-60" />
              {isAnalyzing && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="absolute inset-0 bg-cyan-400/8 backdrop-blur-[1px]">
                  <div className="scanner-line opacity-60" />
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                    <Loader2 aria-hidden="true" className="h-10 w-10 animate-spin text-cyan-400" />
                    <div className="terminal-text text-[10px] tracking-[0.3em] text-cyan-400">{status?.detail ?? 'PROCESSING'}</div>
                    <div className="h-2 w-64 overflow-hidden rounded-full bg-white/10" role="progressbar" aria-valuenow={status?.progress ?? 0} aria-valuemin={0} aria-valuemax={100} aria-label="Analysis progress">
                      <div className="h-full bg-cyan-400 transition-all" style={{ width: `${status?.progress ?? 0}%` }} />
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          ) : (
            <div className="flex h-[420px] flex-col items-center justify-center gap-4 rounded-[1.5rem] border border-dashed border-white/10 bg-black/20">
              <Camera aria-hidden="true" className="h-10 w-10 text-zinc-600 transition-colors group-hover:text-cyan-400" />
              <div className="terminal-text text-[11px] text-zinc-500">SELECT_CLINICAL_IMAGE</div>
            </div>
          )}
        </label>

        <div className="mt-6 flex items-center gap-4">
          <button
            onClick={onStart}
            disabled={!selectedFile || isAnalyzing}
            className="terminal-text flex-1 bg-cyan-400 px-6 py-4 text-[10px] font-bold text-black transition-all hover:tracking-[0.2em] disabled:cursor-not-allowed disabled:opacity-30"
          >
            {isAnalyzing ? 'PROCESSING...' : 'EXECUTE_DIAGNOSTIC'}
          </button>
        </div>

        <div className="mt-4 rounded-2xl border border-cyan-400/10 bg-cyan-400/5 px-4 py-3 text-sm text-zinc-400">
          First run may take longer while the clinical engine warms up. Live progress will begin as soon as the backend finishes lazy initialization.
        </div>
      </div>

      <div className="space-y-6">
        <div className="holographic-panel rounded-[1.75rem] p-6">
          <h4 className="terminal-text mb-3 text-[10px] text-cyan-400/80">PRIVACY CONSOLE</h4>
          <div className="space-y-3 text-sm text-zinc-400">
            <div>Original uploads: {privacyMode ? 'purged after run' : 'retained for archive compare'}</div>
            <div>Retention window: {retentionHours} hours</div>
            <div>Stored fields: {privacy?.stored_fields?.length ?? 0}</div>
          </div>
        </div>

        <div className="holographic-panel rounded-[1.75rem] p-6">
          <h4 className="terminal-text mb-3 text-[10px] text-cyan-400/80">EXPLAINABILITY PREVIEW</h4>
          <p className="text-sm leading-relaxed text-zinc-500">
            Every completed case unlocks per-region lesion burden, consensus confidence, GAGS contributions, and longitudinal delta analysis.
          </p>
        </div>
      </div>
    </div>
  )
}
