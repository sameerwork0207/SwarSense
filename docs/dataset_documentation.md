# SwarSense — Dataset Documentation
**Phase 0 — Research Status**

> This document will be updated as datasets are acquired and evaluated.
> No statistics are fabricated. All entries reflect the actual current state of research.

---

## Status: PENDING DATASET ACQUISITION

Dataset access has not yet been completed. This document tracks:
1. Each candidate dataset
2. Access steps taken
3. Actual clip counts (to be filled after download)
4. Actual raga class list (to be determined from data)

---

## Candidate Dataset 1: CompMusic Indian Art Music Raga Recognition Dataset

| Field | Value |
|---|---|
| Source | Zenodo (DOI: 10.5281/zenodo.7278505 features / 10.5281/zenodo.7278510 audio) |
| Access | **Restricted — requires formal access request** |
| License | Research use; citation of Gulati et al. required |
| Ragas | 30+ (exact list pending access) |
| Clip count | Not yet verified |
| Format | MP3 / WAV (varies) |
| Label quality | Expert annotated (MTG-UPF CompMusic project) |
| Status | ⏳ Access request to be submitted |
| Action | Submit request at zenodo.org → button: "Request access" |

---

## Candidate Dataset 2: Saraga (Hindustani + Carnatic)

| Field | Value |
|---|---|
| Source | Zenodo + github.com/MTG/saraga |
| Access | **Open — Creative Commons license** |
| License | CC (see specific LICENSE file per subset) |
| Coverage | Hindustani concerts + Carnatic concerts with rich metadata |
| Clip count | Not yet verified |
| Format | MP3 / WAV |
| Label quality | Expert annotated; includes tonic, raga labels, tempo |
| Status | ⏳ Download pending |
| Action | `pip install mirdata` then `mirdata.list_datasets()` |

---

## Candidate Dataset 3: Kaggle — Indian Music Raga

| Field | Value |
|---|---|
| Source | kaggle.com/datasets/kcwaghmarewaghmare/indian-music-raga |
| Access | Public (Kaggle login required) |
| License | Unverified — check before academic use |
| Ragas | 8: Asawari, Bageshree, Bhairavi, Boopali, Darbari Kanada, Malkauns, Vrindavani Sarang, Yaman |
| Instruments | Vocal + violin |
| Clip count | Not yet verified |
| Format | Unknown (pending download) |
| Status | ⏳ Pending inspection |
| Action | Download and verify clip count, duration, label quality |

---

## Candidate Dataset 4: Kaggle — INDIAN RAGAS (aashidutt)

| Field | Value |
|---|---|
| Source | kaggle.com/datasets/aashidutt/indian-ragas |
| Access | Public |
| Ragas | 5: Yaman, Bhairav, Shudh Sarang, Jogkuns, Madhuvanti |
| Format | 30-second WAV clips |
| Clip count | Not yet verified |
| Status | ⏳ Prototype use only (too small for training) |

---

## Candidate Dataset 5: Kaggle — Thaat and Raga Forest (TRF)

| Field | Value |
|---|---|
| Source | kaggle.com/datasets/kuchhbhikh/thaat-and-raga-forest-trf-dataset |
| Access | Public |
| Coverage | 10 Thaats; multiple ragas; MP3 + WAV |
| Clip count | Not yet verified |
| Status | ⏳ Pending inspection |

---

## Final Raga Class List

**Status: NOT YET DETERMINED**

The final class list will be selected based on:
1. Which ragas have ≥ 50 clips after dataset inspection
2. Which ragas appear in multiple datasets (for train/test split validity)
3. Which ragas are musically distinctive enough to be separable

Working hypothesis (subject to change after data inspection):
- Yaman
- Bhairav
- Bhairavi
- Malkauns
- Bhupali
- Darbari Kanada
- Bageshree
- Asavari

---

## Dataset Evaluation Criteria

For each dataset, we will measure:
- Total clips per raga class
- Mean/median/std of clip duration
- Audio quality (SNR estimation)
- Label confidence
- Whether train/validation/test splits can be made without data leakage

---

*Last updated: Phase 0 — September 2026*
