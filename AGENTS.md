# AGENTS.md — Acne Severity Analytics Workstation

## Architecture Overview

Monorepo with two independent stacks:
- **`frontend/`** — React 19 + TypeScript + Vite + Tailwind v4 single-page clinical workstation
- **`backend/`** — FastAPI (Python 3.11) API bridge with OpenCV, BiSeNet face segmentation, Roboflow cloud inference, and ReportLab PDF exports
- SQLite for session persistence; no ORM — raw SQL via `sqlite3`

## Build / Lint / Test Commands

### Frontend (`frontend/`)

```bash
cd frontend
npm install            # install dependencies
npm run dev            # dev server on http://localhost:3000
npm run build          # production build (Vite)
npm run lint           # type-check only (tsc --noEmit)
npm run clean          # remove dist/
```

No test runner is configured (no jest/vitest). Lint is TypeScript type-checking only — no ESLint config exists.

### Backend (`backend/`)

```bash
pip install -r requirements.txt   # from repo root

# Run API server
cd backend
python -m uvicorn api_bridge:app --host 0.0.0.0 --port 8000

# Run CLI single-image analysis
cd backend
python main.py --image path/to/photo.jpg --output output --visualize --smooth

# Run batch processing
cd backend
python batch_process.py --input_dir images/ --output_dir output/

# Run validation (precision/recall/F1 against ground-truth)
cd backend
python validate_v7.py
```

No formal test suite (no pytest, unittest, or test config). `test_local_model.py` is a manual smoke test — run directly with `python test_local_model.py`.

### Docker (backend only)

```bash
docker build -t acne-severity-backend .
docker run --env-file backend/.env -p 8000:8000 acne-severity-backend
```

### Environment Variables

Backend — copy `backend/.env.example` to `backend/.env`:
- `ROBOFLOW_API_KEY` (required, server refuses to start without it)
- `MODEL_A_ID`, `MODEL_B_ID`, `MAX_API_DIM` (optional, have defaults)
- `MAX_UPLOAD_BYTES`, `DEFAULT_RETENTION_HOURS`, `MAX_RETENTION_HOURS` (optional)

Frontend — copy `frontend/.env.example` to `frontend/.env`:
- `VITE_API_BASE_URL` — backend URL (defaults to `http://localhost:8000`)
- `GEMINI_API_KEY` — injected at runtime in AI Studio deployments

## Code Style Guidelines

### TypeScript / Frontend

**File naming:**
- PascalCase for `.tsx` component files: `Hero.tsx`, `ClinicalWorkspace.tsx`
- camelCase for `.ts` utility/service files: `api.ts`, `utils.ts`
- kebab-case for type declaration files: `spline-viewer.d.ts`

**Import ordering** (3 groups, blank-line separated):
1. Third-party (`react`, `framer-motion`, `lucide-react`)
2. Internal modules (`../../services/api`, `../../types/api`, `../../lib/utils`)
3. No CSS imports in components — only in `main.tsx`

**Type-only imports** — always use `import type` for types:
```ts
import type { AnalyzeResponse, SessionStatus } from '../../types/api'
import { type ClassValue, clsx } from 'clsx'
```

**Types over interfaces** — the codebase uses `type` exclusively, never `interface`. Types are PascalCase and live in `types/api.ts`:
```ts
export type RegionStats = { count: number; lpi: number; area_px: number; gags_score: number }
export type SessionDetail = SessionSummary & { results: Record<string, unknown> | null }
```

**Component patterns:**
- Named function declarations, not arrow functions: `export function Hero() { ... }`
- Only `App.tsx` uses `export default`; all others use named exports
- Props are typed inline at the destructuring site (no separate `Props` types)
- Sub-components are private functions at the bottom of the file
- Helper functions are plain `function` declarations at module scope

**Constants:** `UPPER_SNAKE_CASE` at module scope: `UI_PREFS_KEY`, `DEFAULT_PROFILE_ID`

**String style:** Single quotes. No semicolons in most component files.

**Error handling pattern:**
```ts
catch (err) {
  setError(err instanceof Error ? err.message : 'Descriptive fallback')
}
```

**Promise discard:** Use `void` keyword: `onClick={() => void handleStart()}`

**Async effect cleanup:** Use `let cancelled = false` guard pattern:
```ts
useEffect(() => {
  let cancelled = false
  void doWork().then(() => { if (!cancelled) setState(...) })
  return () => { cancelled = true }
}, [dep])
```

**Styling:** Tailwind v4 utility classes everywhere. No CSS modules or styled-components. Use `cn()` from `lib/utils.ts` (clsx + tailwind-merge) for conditional classes. Custom component classes (`.holographic-panel`, `.glass-panel`, `.metadata-micro`, etc.) are in `index.css`.

**API calls:** All in `services/api.ts` using native `fetch`. Generic `request<T>()` helper. No axios. State management via `useState` only — no Redux/Zustand.

### Python / Backend

**File naming:** `snake_case.py` for all files. No hyphens.

**Import ordering** (3 groups, no strict blank-line separation):
1. Standard library (`os`, `sys`, `json`, `time`, `threading`, `uuid`)
2. Third-party (`cv2`, `numpy`, `torch`, `fastapi`, `pydantic`)
3. Local (`from face_segmentation.pipeline import ...`, `from cloud_inference import ...`)

**Naming conventions:**
- Functions/variables: `snake_case` — `log_api_call()`, `validate_upload()`, `clinical_report`
- Classes: `PascalCase` — `FaceSegmentationPipeline`, `EnsembleLesionMapper`, `BridgeStore`
- Constants: `UPPER_SNAKE_CASE` — `ROBOFLOW_API_KEY`, `BASE_DIR`, `REGION_WEIGHTS`
- Private methods: single underscore prefix — `_load_weights()`, `_clean_mask()`, `_init_db()`

**Type hints:** Used extensively. Prefer `Dict[str, Any]`, `List[Dict]`, `Optional[str]` from `typing`. `api_bridge.py` is the most thorough; ML pipeline files sometimes use bare `Dict` returns.

**Docstrings:** Google-style with `Args:` and `Returns:` sections on public methods. Module-level docstrings on every file. Triple double-quotes.

**String quoting:** `api_bridge.py` uses single quotes. ML pipeline and CLI files use double quotes. Match whichever convention the file already uses.

**Error handling patterns:**
- `raise HTTPException(status_code=..., detail=...)` for API errors
- `raise FileNotFoundError(...)` with download instructions for missing weights
- `try/except ImportError` with fallback flags for optional dependencies
- Silent `except: pass` only for JSON file loading and SQLite schema migrations
- Guard clauses on a single line: `if image is None: return 1`

**Logging:** `print()` with bracketed prefixes — `[Bridge]`, `[Pipeline]`, `[V7-API]`, `[Cloud Engine Error]`. No `logging` module.

**Config:** `python-dotenv` with `load_dotenv()` at module level. Each entry-point reads `.env` independently — no centralized config module. `api_bridge.py` fails fast if `ROBOFLOW_API_KEY` is missing.

**Lazy imports:** Heavy ML dependencies (torch, mediapipe) are loaded lazily. `face_segmentation/__init__.py` uses `__getattr__`. `api_bridge.py` uses global `None` sentinels with `ensure_runtime_imports()`.

**Thread safety:** `threading.Lock()` for model initialization (double-checked locking) and SQLite access in `BridgeStore`.

**Data passing:** Plain dicts everywhere — no dataclasses or NamedTuple. Pydantic `BaseModel` only for FastAPI request validation.

**CLI scripts:** Use `argparse` with `if __name__ == "__main__":` guard. No click/typer.

## Key Directories

```
frontend/src/components/marketing/   # Landing page sections
frontend/src/components/workspace/   # ClinicalWorkspace (main app)
frontend/src/services/api.ts         # All API calls
frontend/src/types/api.ts            # All shared types
frontend/src/lib/utils.ts            # cn() utility
backend/face_segmentation/           # Core ML package (BiSeNet, MediaPipe, mapping)
backend/face_segmentation/models/    # PyTorch model definitions
backend/face_segmentation/utils/     # Visualization helpers
backend/weights/                     # Model weight files (large, consider LFS)
```

## CI/CD

GitHub Actions workflow (`.github/workflows/sync-hf-space.yml`) syncs the backend to a Hugging Face Space on push to `main`. Requires `HF_TOKEN` repository secret.
