export type ViewerState = {
  scale: number
  x: number
  y: number
}

export type ViewerMode = 'single' | 'split'

export type WorkspaceUiPrefs = {
  leftRailWidth?: number
  rightRailWidth?: number
  compareViewerMode?: ViewerMode
  showDetectionOverlay?: boolean
  showCompareOverlay?: boolean
  exportPreset?: 'clinical' | 'compact' | 'presentation'
}
