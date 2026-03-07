# Acne Severity Analytics

AI-powered acne severity analysis workstation with a React clinical review frontend and FastAPI backend for lesion detection, GAGS scoring, longitudinal comparison, and baseline-aware report exports.

## Repository Structure

```text
.
|- frontend/   # React + Vite clinical review workstation
`- backend/    # FastAPI analysis bridge and reporting pipeline
```

## Core Capabilities

- lesion detection with bounding-box visualization
- GAGS scoring and clinical severity grading
- longitudinal session history and baseline pinning
- profile-aware workspace persistence
- privacy mode and retention controls
- baseline-aware compare summaries and PDF exports
- workstation-style viewer tools for zoom, pan, overlays, and split compare

## Frontend

Location: `frontend/`

Tech stack:
- React
- TypeScript
- Vite
- Framer Motion

Run locally:

```bash
cd frontend
npm install
npm run dev
```

Default app URL:
- `http://localhost:3000`

Useful commands:

```bash
npm run lint
npm run build
```

## Backend

Location: `backend/`

Tech stack:
- FastAPI
- OpenCV
- ReportLab
- Roboflow cloud inference
- face segmentation and lesion-region mapping

Run locally:

```bash
cd backend
pip install -r requirements.txt
python api_bridge.py
```

Default API URL:
- `http://localhost:8000`

Required environment variables:
- `ROBOFLOW_API_KEY`

Use `backend/.env.example` as the starting point.

## Workflow

1. Start the backend on port `8000`
2. Start the frontend on port `3000`
3. Upload a facial acne image in the workspace
4. Review lesion detections, GAGS score, severity, and compare deltas
5. Export clinical, compact, or presentation PDF reports

## Notes

- End users see acne detections with lesion boxes, not face segmentation overlays
- Runtime data such as uploads, outputs, logs, local databases, and `.env` files are git-ignored in the root repo
- The backend currently includes local weights under `backend/weights/`; large model files may be better moved to Git LFS later
