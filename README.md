# SwarSense
## AI-Based Intelligent Music Analysis and Indian Raga Recognition System

**AIML Honors Final Year Project**

---

## Overview

SwarSense accepts an uploaded audio file (`.wav` or `.mp3`) and automatically analyses its musical characteristics. It is being built in phases — see **Development Phases** below for what is actually implemented today versus planned.

The target architecture combines:
- **Deterministic DSP** (BPM, spectrograms, MFCC, pitch) via Librosa — ✅ implemented (Phase 1)
- Optional **external song identification + metadata enrichment** (AcoustID/ACRCloud/AudD + MusicBrainz) — ✅ implemented (Phase 2); this is external reference data, not a SwarSense prediction
- **Pretrained AI models** (source separation, instrument detection) — not yet implemented (Phase 3)
- **Trained ML/DL models** (Indian raga classification — the primary research contribution) — not yet implemented; no labelled dataset or trained model exists yet (Phase 4)

---

## Project Structure

```
swarsense/
├── backend/               FastAPI Python backend
│   ├── app/
│   │   ├── main.py        Application entry point
│   │   ├── api/           Routes + Pydantic schemas
│   │   ├── services/      Job queue + analysis orchestration
│   │   ├── audio/         Audio loading + validation
│   │   ├── inference/     ML model inference (Phase 3–4)
│   │   └── utils/         Logging
│   ├── tests/             Pytest test suite
│   ├── requirements.txt   Production dependencies
│   └── requirements-ml.txt  Training-only dependencies
│
├── frontend/              Next.js TypeScript frontend
│
├── ml/                    Research/training code (separate from production)
│   ├── datasets/          Dataset documentation + download scripts
│   ├── notebooks/         EDA + experiment notebooks
│   ├── preprocessing/     Segmentation + augmentation
│   ├── training/          Model training scripts
│   └── evaluation/        Evaluation scripts + results
│
├── models/                Trained model checkpoints (gitignored)
├── data/
│   ├── ragas.json         Static raga knowledge base
│   └── uploads/           Temporary audio files (gitignored)
│
├── docs/                  Project documentation
├── docker-compose.yml
└── README.md
```

---

## Quick Start

### Prerequisites
- Docker and Docker Compose
- (For development) Python 3.11, Node.js 20

### Run with Docker

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/api/docs

### Run without Docker (development)

**Backend:**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # optional — only needed to enable Song Intelligence
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## Development Phases

| Phase | Status | Description |
|---|---|---|
| 0 — Infrastructure | ✅ Complete | Project skeleton, Docker, FastAPI, Next.js |
| 1 — DSP Pipeline | ✅ Complete | Audio validation/preprocessing, loudness, BPM/beat tracking, PYIN pitch, MFCC, chroma, spectral features, HPSS, heuristic tonic estimation, 5 visualizations, results dashboard |
| 2 — Song Intelligence | ✅ Complete (optional, external) | AcoustID (default)/ACRCloud/AudD identification + MusicBrainz enrichment. Requires a free/paid provider API key — reports `unavailable` otherwise. Never affects DSP results. See below. |
| 3 — Pretrained Models | ⏳ Not started | Demucs source separation + YAMNet instrument detection (schema stubs exist; no model integration yet) |
| 4 — Raga Classifier | ⏳ Not started | No labelled dataset and no trained model exist yet. `ml/` is an empty skeleton. `raga_prediction` is a documented placeholder — never fabricated |
| 5 — Explainability | ⏳ Not started | Grad-CAM + SHAP (depends on Phase 4) |
| 6 — Integration | ⏳ Not started | Async queue + caching + polish |
| 7 — Documentation | 🔄 Ongoing | This README + `docs/` |

Backend test suite: 116 tests passing (`backend/tests/`), covering DSP pipeline, API integration, and Song Intelligence (with mocked providers — no live credentials required for tests).

### Song Intelligence setup (optional)

Song Intelligence is **external metadata only** — it identifies which recording was uploaded (via acoustic fingerprinting) and enriches it with MusicBrainz data. It is never a SwarSense-generated prediction, and a raga name surfaced via MusicBrainz tags is always labeled as an external reference, never as SwarSense's own raga classification.

Without any provider credentials configured, identification correctly reports `unavailable` and has no effect on the DSP analysis. To enable it, copy `backend/.env.example` to `backend/.env` and set at least one of:
- `ACOUSTID_API_KEY` (default provider; also requires `fpcalc` from [Chromaprint](https://acoustid.org/chromaprint) on PATH)
- `ACRCLOUD_HOST` + `ACRCLOUD_ACCESS_KEY` + `ACRCLOUD_ACCESS_SECRET`
- `AUDD_API_TOKEN`

Set `SONG_IDENTIFICATION_PROVIDER` to pick which one is used (`acoustid` | `acrcloud` | `audd` | `auto`), or `SONG_INTELLIGENCE_ENABLED=false` to disable the stage entirely.

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11, Uvicorn |
| Audio DSP | Librosa, SoundFile, NumPy, SciPy |
| Deep Learning | PyTorch |
| ML Baseline | Scikit-learn |
| Explainability | captum (Grad-CAM), shap |
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Deployment | Docker, Docker Compose |

---

## Academic Honesty Notice

This project is an AIML Honors research prototype.
All model predictions include confidence scores and explicit source labeling.
Raga classification is limited to a closed vocabulary of trained ragas.
Uncertain predictions are displayed as uncertain — never presented as definitive facts.

---

## Citation

If you build on this work:

```
@misc{swarsense2026,
  title  = {SwarSense: An AI-Based Intelligent Music Analysis and Indian Raga Recognition System},
  author = {[Your Name]},
  year   = {2026},
  note   = {AIML Honors Final Year Project}
}
```

---

## Datasets and Attribution

Datasets used in this project require attribution:
- CompMusic: Cite Gulati et al. (see docs/dataset_documentation.md)
- Saraga: Cite Srinivasamurthy et al. (2021)

See [docs/dataset_documentation.md](docs/dataset_documentation.md) for full details.
