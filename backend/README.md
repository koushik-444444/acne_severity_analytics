---
title: Acne Severity Analytics Backend
emoji: 🩺
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Acne Severity Analytics Backend

FastAPI backend for acne lesion detection, face-region mapping, GAGS scoring, longitudinal comparison, and baseline-aware PDF export workflows.

## What It Does

- runs Roboflow-based lesion detection
- maps lesions to facial clinical regions
- computes GAGS score and severity band
- stores session history in SQLite
- supports notes, privacy controls, retention, compare, and export flows
- renders lesion-box diagnostic overlays for the frontend workstation

## Local Run

```bash
cd ..
pip install -r requirements.txt
cd backend
python -m uvicorn api_bridge:app --host 0.0.0.0 --port 8000
```

Required environment variables:

- `ROBOFLOW_API_KEY`

Optional environment variables:

- `MODEL_A_ID`
- `MODEL_B_ID`
- `MAX_API_DIM`
- `DEFAULT_RETENTION_HOURS`
- `MAX_RETENTION_HOURS`

Start from `.env.example`.

## Docker

```bash
docker build -t acne-severity-backend ..
docker run --env-file .env -p 8000:8000 acne-severity-backend
```

## Hugging Face Spaces

This backend is prepared for a Docker Space.

- SDK: `docker`
- app port: `8000`
- runtime entrypoint: `uvicorn api_bridge:app --host 0.0.0.0 --port 8000`

Make sure these files are present in the Space root:

- `requirements.txt`
- `Dockerfile`
- `.dockerignore`
- `backend/`

Set `ROBOFLOW_API_KEY` in the Space secrets before launching.
