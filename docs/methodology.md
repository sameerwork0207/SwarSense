# SwarSense — Research Methodology
**AIML Honors Project — Indian Raga Recognition System**

---

## 1. Problem Statement

Given an audio recording of Indian classical music, automatically identify:
1. General audio features (tempo, loudness, spectral properties)
2. Musical features (pitch, tonic, chroma)
3. Probable Indian raga class (from a closed vocabulary of N ragas)
4. Supporting instruments

This is a **supervised multi-class classification** problem for the raga component,
combined with established **signal processing** for audio features.

---

## 2. Methodology Overview

### 2.1 Feature Extraction Strategy

We distinguish three categories of outputs:

| Category | Method | Examples |
|---|---|---|
| **Deterministic DSP** | Librosa / SciPy | BPM, MFCC, Mel spectrogram, chroma |
| **Pretrained model** | Demucs, YAMNet | Source separation, instrument detection |
| **Trained model** | Scikit-learn, PyTorch | Raga classification |

This distinction is preserved throughout the codebase and UI — every AI output declares its `source` field.

### 2.2 Audio Preprocessing

All audio is normalised to:
- **Sample rate:** 22050 Hz (standard for music MIR research)
- **Channels:** Mono
- **Duration cap:** 600 seconds (10 minutes)
- **Segment length:** 30 seconds for classification

**Rationale for 22050 Hz:** Sufficient to capture audio content up to 11 kHz (Nyquist). Most musically relevant content (voice, strings, tabla) is below this.

**Rationale for mono:** Librosa and most MIR features assume mono input. Converting stereo to mono is standard practice.

### 2.3 Raga Classification Approach

**Baseline — Traditional ML:**
- Extract MFCC statistics (mean, std per coefficient)
- Extract chroma statistics (mean per pitch class)
- Extract spectral features (centroid, rolloff, ZCR)
- Compute pitch histogram from F0 contour
- Train: Random Forest and SVM with RBF kernel
- Evaluate: 5-fold stratified cross-validation

**Primary Model — CNN on Mel Spectrogram:**
- Compute 128-bin Mel spectrogram per 30-second segment
- Architecture: Custom lightweight CNN or fine-tuned ResNet-18
- Training: Adam optimizer, cross-entropy loss, SpecAugment augmentation
- Evaluate: Same test split as baseline

**Segment Aggregation:**
- Each recording is split into 30-second non-overlapping windows
- Model predicts per segment
- Recording-level prediction: **average softmax probabilities** across all segments
- Predicted class: argmax of averaged softmax vector
- Uncertainty threshold: 0.5 — predictions below this are reported as "uncertain"

**Rationale for average softmax (vs majority vote):**
Average softmax preserves the full probability distribution and produces calibrated confidence scores. Majority vote discards probability information and performs worse when segment predictions are spread across similar classes (e.g., Yaman vs Bhupali).

---

## 3. Model Comparison Plan

We will compare at minimum two approaches on the same test set:

| Model | Features | Framework |
|---|---|---|
| Baseline | MFCC + chroma + spectral statistics | Scikit-learn |
| CNN | Mel spectrogram (raw pixels) | PyTorch |

All comparisons will use:
- Same train/validation/test split (80/10/10 or dataset-specific splits)
- Same raga class vocabulary
- Metrics: Accuracy, Macro F1, per-class F1, confusion matrix

---

## 4. Evaluation Protocol

### Metrics

For raga classification:
- **Accuracy** (overall)
- **Precision** (per class + macro)
- **Recall** (per class + macro)
- **F1-score** (per class + macro)
- **Confusion matrix** — most important for understanding inter-class confusion

For regression estimates (BPM):
- **MAE** against Librosa's ground truth

### Cross-validation

- 5-fold stratified cross-validation for the traditional ML baseline
- CNN: Single train/val/test split (stratified); report test-set performance only

### Reporting

- All metrics reported from **actual experiments** only
- No expected or estimated values presented as results
- Confidence intervals where sufficient data permits

---

## 5. Explainability

### Grad-CAM (CNN)
- Applied to the final convolutional layer of the raga CNN
- Output: Heatmap of size (H × W) over the Mel spectrogram
- Interpretation: Time–frequency regions that most influenced the prediction
- Correlation with music theory: We will attempt to link activated regions to known characteristic note movements of each raga

### SHAP (Baseline ML)
- Applied to the Random Forest / SVM baseline
- Output: Feature importance scores per prediction
- Interpretation: Which MFCC coefficients / chroma bins / spectral features drove the prediction

### Static Raga Knowledge
- Aroha, Avroha, Vadi, Samvadi, Thaat, characteristic phrases
- Source: Static JSON knowledge base (musicological reference)
- **Always displayed with label: "Reference Information — Not a model prediction"**

---

## 6. Limitations and Scope

1. **Closed vocabulary:** The model only recognises ragas it was trained on. Out-of-vocabulary ragas will be misclassified.
2. **Dataset size:** Small datasets (< 100 clips/class) limit achievable accuracy.
3. **Polyphonic complexity:** Raga performance with multiple instruments is harder than solo recordings.
4. **Tonic not modelled:** PYIN-based tonic estimation is a heuristic, not a learned model.
5. **Indian instrument gap:** No large-scale labelled dataset for Tabla/Harmonium exists publicly.

These limitations are documented honestly and will be discussed in the project report.

---

*Last updated: Phase 0 — September 2026*
