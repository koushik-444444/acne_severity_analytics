import type { DetailedHTMLProps, HTMLAttributes } from 'react'

declare global {
  interface IdleDeadline {
    readonly didTimeout: boolean
    timeRemaining(): DOMHighResTimeStamp
  }

  interface Window {
    requestIdleCallback?(
      callback: (deadline: IdleDeadline) => void,
      options?: { timeout: number },
    ): number
    cancelIdleCallback?(handle: number): void
  }
}

declare module 'react' {
  namespace JSX {
    interface IntrinsicElements {
      'spline-viewer': DetailedHTMLProps<HTMLAttributes<HTMLElement>, HTMLElement> & {
        url: string
        loading?: 'auto' | 'lazy' | 'eager'
        'events-target'?: 'canvas' | 'global'
        'loading-anim'?: 'true' | 'false'
        'loading-anim-type'?: string
      }
    }
  }
}
