# SwarSense
## AI-Based Intelligent Music Analysis and Indian Raga Recognition System

**AIML Honors Final Year Project**

---

## Overview

SwarSense is an end-to-end AI system that accepts an uploaded audio file (`.wav` or `.mp3`) and automatically analyses its musical characteristics.

The system combines:
- **Deterministic DSP** (BPM, spectrograms, MFCC, pitch) via Librosa
- **Pretrained AI models** (source separation, instrument detection)
- **Trained ML/DL models** (Indian raga classification — the primary research contribution)

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
| 1 — DSP Pipeline | ⏳ Pending | Librosa feature extraction |
| 2 — Visualizations | ⏳ Pending | Waveform, spectrogram, chromagram, pitch |
| 3 — Pretrained Models | ⏳ Pending | Demucs + YAMNet integration |
| 4 — Raga Classifier | ⏳ Pending | Dataset prep + CNN training |
| 5 — Explainability | ⏳ Pending | Grad-CAM + SHAP |
| 6 — Integration | ⏳ Pending | Async queue + caching + polish |
| 7 — Documentation | ⏳ Pending | Testing + final demo |

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
