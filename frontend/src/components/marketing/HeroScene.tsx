import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'

const SPLINE_SCENE_URL = 'https://prod.spline.design/DtbMNZsG-pT5th3f/scene.splinecode'
const SPLINE_VIEWER_SCRIPT_URL = 'https://unpkg.com/@splinetool/viewer@1.9.82/build/spline-viewer.js'
const SPLINE_BOOT_DELAY_MS = 900

type HeroSceneStatus = 'booting' | 'loading' | 'ready' | 'disabled' | 'error'
type NavigatorWithHints = Navigator & {
  connection?: {
    saveData?: boolean
  }
  deviceMemory?: number
}

let splineViewerScriptPromise: Promise<void> | null = null

function loadSplineViewerScript() {
  if (typeof window === 'undefined') {
    return Promise.resolve()
  }

  if (window.customElements?.get('spline-viewer')) {
    return Promise.resolve()
  }

  if (splineViewerScriptPromise) {
    return splineViewerScriptPromise
  }

  splineViewerScriptPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>('script[data-spline-viewer-script="true"]')

    const handleLoad = () => {
      if (existing) {
        existing.dataset.loaded = 'true'
      }
      resolve()
    }

    const handleError = () => {
      splineViewerScriptPromise = null
      reject(new Error('Failed to load Spline viewer'))
    }

    if (existing) {
      if (existing.dataset.loaded === 'true') {
        resolve()
        return
      }

      existing.addEventListener('load', handleLoad, { once: true })
      existing.addEventListener('error', handleError, { once: true })
      return
    }

    const script = document.createElement('script')
    script.src = SPLINE_VIEWER_SCRIPT_URL
    script.type = 'module'
    script.async = true
    script.dataset.splineViewerScript = 'true'
    script.addEventListener('load', () => {
      script.dataset.loaded = 'true'
      resolve()
    }, { once: true })
    script.addEventListener('error', handleError, { once: true })
    document.head.appendChild(script)
  })

  return splineViewerScriptPromise
}

function StaticFallback({ mode }: { mode: 'disabled' | 'error' }) {
  const label = mode === 'disabled' ? 'LOW_POWER_RENDER' : 'STATIC_SCENE_FALLBACK'
  const detail =
    mode === 'disabled'
      ? 'Adaptive mode keeps the workstation responsive on mobile, reduced-motion, or data-saver devices.'
      : 'The live 3D scene did not initialize, so the hero falls back to a lighter cinematic render.'

  return (
    <div className="relative h-full w-full overflow-hidden rounded-full border border-cyan-400/10 bg-[radial-gradient(circle_at_50%_45%,rgba(0,242,255,0.14),rgba(0,0,0,0)_34%),linear-gradient(180deg,rgba(6,10,12,0.9),rgba(1,1,1,1))]">
      <div className="absolute inset-[8%] rounded-full border border-cyan-400/8" />
      <div
        className="absolute inset-[16%] rounded-full border border-dashed border-cyan-400/15 animate-spin"
        style={{ animationDuration: '26s' }}
      />
      <div
        className="absolute inset-[24%] rounded-full border border-white/6 animate-spin"
        style={{ animationDuration: '18s', animationDirection: 'reverse' }}
      />
      <div className="absolute left-1/2 top-1/2 h-[34%] w-[34%] -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-400/12 blur-3xl" />
      <div className="absolute left-1/2 top-1/2 h-[18%] w-[18%] -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-400/30 bg-cyan-400/8 shadow-[0_0_60px_rgba(0,242,255,0.15)]" />
      <div className="absolute inset-x-[18%] top-1/2 h-px -translate-y-1/2 bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent" />
      <div className="absolute bottom-14 left-1/2 w-[68%] -translate-x-1/2 text-center">
        <div className="terminal-text mb-3 text-[9px] tracking-[0.42em] text-cyan-400/75">{label}</div>
        <p className="text-sm font-light leading-relaxed text-zinc-500">{detail}</p>
      </div>
    </div>
  )
}

export function HeroScene() {
  const [status, setStatus] = useState<HeroSceneStatus>('booting')
  const [shouldLoad, setShouldLoad] = useState(false)

  useEffect(() => {
    const navigatorHints = navigator as NavigatorWithHints
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const compactViewport = window.matchMedia('(max-width: 1023px)').matches
    const saveData = navigatorHints.connection?.saveData === true
    const lowMemory = typeof navigatorHints.deviceMemory === 'number' && navigatorHints.deviceMemory <= 4

    if (reducedMotion || compactViewport || saveData || lowMemory) {
      setStatus('disabled')
      return
    }

    let idleId: number | null = null
    const timer = window.setTimeout(() => {
      if ('requestIdleCallback' in window) {
        idleId = window.requestIdleCallback(() => {
          setShouldLoad(true)
          setStatus('loading')
        }, { timeout: 1500 })
        return
      }

      setShouldLoad(true)
      setStatus('loading')
    }, SPLINE_BOOT_DELAY_MS)

    return () => {
      window.clearTimeout(timer)
      if (idleId !== null && 'cancelIdleCallback' in window) {
        window.cancelIdleCallback(idleId)
      }
    }
  }, [])

  useEffect(() => {
    if (!shouldLoad) {
      return
    }

    let active = true

    loadSplineViewerScript()
      .then(() => {
        if (active) {
          setStatus('ready')
        }
      })
      .catch(() => {
        if (active) {
          setStatus('error')
        }
      })

    return () => {
      active = false
    }
  }, [shouldLoad])

  if (status === 'disabled' || status === 'error') {
    return <StaticFallback mode={status} />
  }

  if (status !== 'ready') {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-6 rounded-full border border-cyan-400/10 bg-cyan-400/3">
        <Loader2 className="h-12 w-12 animate-spin text-cyan-400" />
        <div className="terminal-text text-[9px] tracking-[0.4em] text-cyan-400/70">
          {status === 'loading' ? 'STREAMING_3D_SCENE' : 'DEFERRED_3D_BOOT'}
        </div>
      </div>
    )
  }

  return (
    <spline-viewer
      url={SPLINE_SCENE_URL}
      loading="lazy"
      events-target="global"
      loading-anim="false"
      className="block h-full w-full"
    />
  )
}
