# SwarSense — Model Card
**Status: Phase 0 — No Models Trained Yet**

> This document will be populated once models are trained in Phase 4.
> No accuracy numbers, training data statistics, or performance metrics are included
> in this Phase 0 version because none are available.
> All such information will be filled in from actual experiments.

---

## Model 1: Raga Classification — Traditional ML Baseline

| Field | Value |
|---|---|
| **Model type** | Random Forest / SVM |
| **Framework** | Scikit-learn |
| **Input features** | MFCC statistics, chroma statistics, spectral features, pitch histogram |
| **Output** | Softmax probabilities over N raga classes |
| **Training data** | TBD after dataset acquisition |
| **Raga classes** | TBD |
| **Training status** | Not yet trained |
| **Accuracy** | Not yet measured |
| **Macro F1** | Not yet measured |
| **Confusion matrix** | Not yet available |

### Intended Explainability
- SHAP values for feature importance

---

## Model 2: Raga Classification — CNN on Mel Spectrograms

| Field | Value |
|---|---|
| **Model type** | Lightweight CNN or fine-tuned ResNet-18 / EfficientNet-B0 |
| **Framework** | PyTorch |
| **Input** | Mel spectrogram of 30-second audio segment (128 mel bins × T frames) |
| **Output** | Softmax probabilities over N raga classes |
| **Training data** | TBD after dataset acquisition |
| **Raga classes** | TBD |
| **Training status** | Not yet trained |
| **Accuracy** | Not yet measured |
| **Macro F1** | Not yet measured |
| **Confusion matrix** | Not yet available |
| **Training time** | Not yet measured |

### Intended Explainability
- Grad-CAM via captum on the final convolutional layer
- Heatmap overlaid on Mel spectrogram showing which time-frequency regions influenced the prediction

### Segment Aggregation Strategy
- Input audio is split into 30-second non-overlapping segments
- Model produces per-segment softmax probabilities
- Recording-level prediction: average of all segment softmax vectors
- Predicted class: argmax of averaged softmax
- Rationale: Average softmax preserves confidence calibration better than majority vote

---

## Model 3: Instrument Detection (Phase 3)

| Field | Value |
|---|---|
| **Model type** | Pretrained YAMNet (TensorFlow Hub) |
| **Source** | Google / TensorFlow Hub |
| **Training data** | AudioSet (521 classes; ~2 million 10-second clips) |
| **Scope** | Western instruments. Does NOT include Tabla or Harmonium. |
| **Indian instrument gap** | Documented — see dataset_documentation.md |
| **Status** | Phase 3 — not yet integrated |

---

## Model 4: Source Separation (Phase 3)

| Field | Value |
|---|---|
| **Model** | Demucs htdemucs (Meta AI) |
| **Source** | github.com/facebookresearch/demucs |
| **License** | MIT |
| **Stems** | vocals, drums, bass, other |
| **CPU performance** | ~0.2–0.7× real-time on CPU |
| **Applied duration** | First 30 seconds only (due to CPU constraints) |
| **Status** | Phase 3 — not yet integrated |

---

## Known Limitations (All Models)

1. Raga classifier limited to a closed vocabulary (N ragas). Will not work reliably on out-of-vocabulary ragas.
2. Performance degrades on low-quality or polyphonic recordings.
3. Tonic estimation is a heuristic (chromagram peak), not a model prediction.
4. YAMNet does not cover Indian instruments.
5. All models are CPU-only in this deployment (GPU optional).

---

*This model card will be completed with actual performance metrics after Phase 4 training.*
