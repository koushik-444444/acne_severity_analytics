import { useRef, useState, type MouseEvent as ReactMouseEvent } from 'react'

import { clamp } from '../../lib/clinical-utils'
import type { ConsensusLesion } from '../../types/api'
import type { ViewerState } from './types'

export function AdvancedImageViewer({
  src,
  alt,
  state,
  onChange,
  heightClass,
  lesions = [],
  activeLesionKey = null,
  onLesionHover,
}: {
  src: string
  alt: string
  state: ViewerState
  onChange: (next: ViewerState) => void
  heightClass: string
  lesions?: Array<ConsensusLesion & { key: string }>
  activeLesionKey?: string | null
  onLesionHover?: (key: string | null) => void
}) {
  const dragRef = useRef<{ x: number; y: number } | null>(null)
  const minimapDragRef = useRef(false)
  const imgRef = useRef<HTMLImageElement | null>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [naturalDims, setNaturalDims] = useState<{ w: number; h: number } | null>(null)

  const startDrag = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (state.scale <= 1) return
    dragRef.current = { x: event.clientX - state.x, y: event.clientY - state.y }
  }

  const onMove = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!dragRef.current || state.scale <= 1) return
    onChange({
      ...state,
      x: event.clientX - dragRef.current.x,
      y: event.clientY - dragRef.current.y,
    })
  }

  const stopDrag = () => {
    dragRef.current = null
    minimapDragRef.current = false
  }

  const moveFromMinimap = (event: ReactMouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const relX = (event.clientX - rect.left) / rect.width
    const relY = (event.clientY - rect.top) / rect.height
    onChange({
      ...state,
      x: (0.5 - relX) * 120,
      y: (0.5 - relY) * 120,
    })
  }

  return (
    <div
      className={`relative overflow-hidden rounded-[1.25rem] border border-white/5 bg-black ${heightClass}`}
      onMouseMove={onMove}
      onMouseUp={stopDrag}
      onMouseLeave={stopDrag}
    >
      <div className="pointer-events-none absolute left-4 top-4 z-10 rounded-full border border-white/10 bg-black/50 px-3 py-1 text-[10px] uppercase tracking-[0.25em] text-zinc-400 backdrop-blur">
        zoom {state.scale.toFixed(2)}x
      </div>
      <div aria-hidden="true" className="pointer-events-none absolute left-4 bottom-4 z-10 rounded-full border border-white/10 bg-black/50 px-3 py-1 text-[10px] uppercase tracking-[0.25em] text-zinc-400 backdrop-blur">
        {cursor ? `x ${cursor.x.toFixed(0)} · y ${cursor.y.toFixed(0)}` : 'x -- · y --'}
      </div>
      <div
        aria-hidden="true"
        className="absolute bottom-4 right-4 z-10 h-24 w-24 overflow-hidden rounded-lg border border-white/10 bg-black/60 backdrop-blur"
        onMouseDown={() => {
          minimapDragRef.current = true
        }}
        onMouseMove={(event) => {
          if (minimapDragRef.current) moveFromMinimap(event)
        }}
        onMouseUp={stopDrag}
      >
        {src ? (
          <>
            <img src={src} alt="" className="h-full w-full object-cover opacity-60" />
            <div
              className="absolute border border-cyan-300/80 bg-cyan-400/10"
              style={{
                left: `${clamp(50 - state.x / 8, 5, 75)}%`,
                top: `${clamp(50 - state.y / 8, 5, 75)}%`,
                width: `${clamp(100 / state.scale, 18, 100)}%`,
                height: `${clamp(100 / state.scale, 18, 100)}%`,
                transform: 'translate(-50%, -50%)',
              }}
            />
          </>
        ) : null}
      </div>
      {src ? (
        <div
          role="application"
          aria-label="Image viewer — drag to pan, double-click to reset"
          tabIndex={0}
          onMouseDown={startDrag}
          onDoubleClick={() => onChange({ scale: 1, x: 0, y: 0 })}
          onKeyDown={(e) => {
            const step = 20
            if (e.key === 'ArrowLeft') { e.preventDefault(); onChange({ ...state, x: state.x + step }) }
            else if (e.key === 'ArrowRight') { e.preventDefault(); onChange({ ...state, x: state.x - step }) }
            else if (e.key === 'ArrowUp') { e.preventDefault(); onChange({ ...state, y: state.y + step }) }
            else if (e.key === 'ArrowDown') { e.preventDefault(); onChange({ ...state, y: state.y - step }) }
          }}
          className="flex h-full w-full cursor-grab items-center justify-center focus:outline-none focus:ring-2 focus:ring-inset focus:ring-cyan-400/40 active:cursor-grabbing"
          onMouseMove={(event) => {
            onMove(event)
            const rect = event.currentTarget.getBoundingClientRect()
            setCursor({ x: event.clientX - rect.left, y: event.clientY - rect.top })
          }}
        >
          <div className="relative max-h-full max-w-full">
            <img
              src={src}
              alt={alt}
              ref={imgRef}
              className="max-h-full max-w-full object-contain select-none transition-transform duration-200"
              style={{ transform: `translate(${state.x}px, ${state.y}px) scale(${state.scale})` }}
              draggable={false}
              onLoad={(e) => {
                const img = e.currentTarget
                setNaturalDims({ w: img.naturalWidth, h: img.naturalHeight })
              }}
            />
            {lesions.length > 0 && naturalDims && (
              <div className="pointer-events-none absolute inset-0">
                {lesions.map((lesion) => {
                  const [x1, y1, x2, y2] = lesion.bbox
                  const pctLeft = (x1 / naturalDims.w) * 100
                  const pctTop = (y1 / naturalDims.h) * 100
                  const pctWidth = (Math.max(0, x2 - x1) / naturalDims.w) * 100
                  const pctHeight = (Math.max(0, y2 - y1) / naturalDims.h) * 100
                  const isActive = activeLesionKey === lesion.key
                  const tone = lesion.confidence >= 0.7
                    ? 'border-cyan-300/90 shadow-[0_0_20px_rgba(0,242,255,0.25)]'
                    : lesion.confidence >= 0.4
                      ? 'border-amber-300/90 shadow-[0_0_18px_rgba(245,158,11,0.18)]'
                      : 'border-white/70 shadow-[0_0_12px_rgba(255,255,255,0.10)]'

                  return (
                    <button
                      key={lesion.key}
                      type="button"
                      aria-label={`${lesion.region} lesion, confidence ${lesion.confidence.toFixed(2)}`}
                      onMouseEnter={() => onLesionHover?.(lesion.key)}
                      onMouseLeave={() => onLesionHover?.(null)}
                      onFocus={() => onLesionHover?.(lesion.key)}
                      onBlur={() => onLesionHover?.(null)}
                      className={`pointer-events-auto absolute border transition-all ${tone} ${isActive ? 'scale-[1.02] bg-cyan-400/10' : 'bg-transparent'}`}
                      style={{
                        left: `${pctLeft}%`,
                        top: `${pctTop}%`,
                        width: `${pctWidth}%`,
                        height: `${pctHeight}%`,
                        transform: `translate(${state.x}px, ${state.y}px) scale(${state.scale})`,
                        transformOrigin: 'top left',
                      }}
                    >
                      <span className={`absolute -top-5 left-0 rounded bg-black/80 px-1 py-0.5 text-[8px] text-white ${isActive ? 'opacity-100' : 'opacity-0'}`}>
                        {lesion.region} {lesion.confidence.toFixed(2)}
                      </span>
                    </button>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex h-full items-center justify-center text-sm text-zinc-500">No image available</div>
      )}
    </div>
  )
}
