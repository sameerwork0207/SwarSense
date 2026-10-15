# SwarSense — ML Research Directory

This directory contains all research and training code.
**It is intentionally separate from the production backend (`backend/`).**

## Structure

```
ml/
├── datasets/         Dataset documentation + download scripts
├── notebooks/        Jupyter notebooks for EDA and experiments
├── preprocessing/    Audio segmentation and augmentation utilities
├── training/         Model training scripts
└── evaluation/       Evaluation and model comparison scripts
```

## Usage

Install training dependencies:
```bash
pip install -r backend/requirements-ml.txt
```

Start Jupyter:
```bash
jupyter lab
```

## Phase 4 Checklist

- [ ] Dataset downloaded and verified
- [ ] EDA notebook: clip counts, duration distribution, class balance
- [ ] Feature extraction notebook: MFCCs, chroma, spectral
- [ ] Baseline ML training script
- [ ] CNN training script (PyTorch)
- [ ] Model comparison evaluation script
- [ ] Confusion matrix visualization
