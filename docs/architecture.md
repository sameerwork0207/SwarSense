# SwarSense — System Architecture
**AIML Honors Project — Phase 0**

```
┌──────────────────────────────────────────────────────────────┐
│                    NEXT.JS FRONTEND                          │
│  Landing page (/) + Results dashboard (/results/[jobId])     │
│  TypeScript + Tailwind CSS                                   │
│                                                              │
│  Components (Phase 2):                                       │
│    WaveformViewer (WaveSurfer.js)                            │
│    SpectrogramViewer (Plotly heatmap)                        │
│    ChromagramViewer                                          │
│    PitchContourViewer                                        │
│    BeatGridOverlay                                           │
│    RagaCard (prediction + Grad-CAM + reference info)         │
│    InstrumentCard (confidence bars)                          │
└─────────────────────────┬────────────────────────────────────┘
                          │ HTTP REST
                          │ POST /api/analyze → 202 + job_id
                          │ GET  /api/status/{id} (poll)
                          │ GET  /api/result/{id}
┌─────────────────────────▼────────────────────────────────────┐
│                    FASTAPI BACKEND                           │
│  Python 3.11 · Uvicorn ASGI                                  │
│                                                              │
│  app/api/routes.py       — All HTTP endpoints               │
│  app/api/schemas.py      — Pydantic request/response types  │
│  app/services/job_service.py    — Job lifecycle             │
│  app/services/analysis_service.py — Pipeline orchestration  │
│  app/audio/validator.py  — File validation                  │
│  app/audio/loader.py     — Librosa audio loading            │
│  app/inference/raga_model.py      — Raga CNN (Phase 4)      │
│  app/inference/instrument_model.py — YAMNet (Phase 3)       │
│  app/inference/separation_model.py — Demucs (Phase 3)       │
└──────────┬──────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────┐
│              ANALYSIS PIPELINE (Phase 1+)                   │
│                                                             │
│  AudioValidator → AudioLoader → Segmenter                   │
│         │                           │                       │
│         ▼                           ▼                       │
│  ┌────────────────┐     ┌─────────────────────────────────┐ │
│  │  DSP MODULE    │     │      ML/DL MODULES              │ │
│  │  (Librosa)     │     │                                 │ │
│  │                │     │  Demucs (Phase 3)               │ │
│  │  BPM           │     │  YAMNet (Phase 3)               │ │
│  │  Beat tracking │     │  Raga Baseline ML (Phase 4)     │ │
│  │  MFCC          │     │  Raga CNN (Phase 4)             │ │
│  │  Mel spectrogram│    │  Grad-CAM (Phase 5)             │ │
│  │  Chromagram    │     │  SHAP (Phase 5)                 │ │
│  │  Pitch/F0      │     │                                 │ │
│  │  Tonic est.    │     └─────────────────────────────────┘ │
│  │  HPSS          │                                         │
│  │  Loudness      │                                         │
│  └────────────────┘                                         │
│                                                             │
│  Result Aggregation → AnalysisResult schema                 │
│  (Partial results returned if any module fails)             │
└─────────────────────────────────────────────────────────────┘

File Storage:
  data/uploads/  — Temporary audio files (gitignored)
  data/ragas.json — Static raga knowledge base
  models/raga/   — Trained model checkpoints (gitignored)
  models/instrument/ — Instrument model files (gitignored)
```

## Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Audio target format | 22050 Hz, mono | Librosa default; sufficient for music |
| Segment length | 30 seconds | Captures melodic phrases; manageable computation |
| Segment aggregation | Average softmax | Preserves calibration vs majority vote |
| Job queue | FastAPI BackgroundTasks (Phase 0–5) | No extra infrastructure for demo |
| Result schema | source field on every AI output | Forces UI to label DSP vs model outputs |
| Raga info source | Static JSON (not ML) | Clear separation: prediction vs. reference |

## Module Independence Contract

Each module MUST:
1. Catch its own exceptions
2. Return None or a placeholder on failure
3. Record its error in `module_errors`
4. NOT block other modules from running

The result JSON is always returned — even if all ML modules fail,
the DSP features are still present.
