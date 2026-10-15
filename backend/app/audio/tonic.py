"""
SwarSense — Tonic Estimation (Heuristic)
==========================================
Estimates the tonic (Sa / root note) of a piece using chroma energy statistics.

Method:
  1. Compute chromagram over the full signal.
  2. Build a 12-bin histogram of mean chroma energy across all frames.
  3. Return the pitch class with the highest mean energy as the estimated tonic.

IMPORTANT RESEARCH LIMITATIONS:
  - This is a HEURISTIC based on chroma energy dominance, NOT a trained model.
  - In Indian classical music, the tonic is determined by the drone (tanpura/shruti)
    and the vocalist's setting, which may not correlate with raw spectral energy.
  - Chroma-peak tonic estimation can be confused by other high-energy notes
    (e.g. Pancham / Pa, which is the 5th and often highly prominent).
  - The output is labeled 'estimated_tonic' with explicit confidence caveats.
  - Confidence is NOT a calibrated probability — it is a relative dominance ratio
    (tonic chroma energy / mean chroma energy). Do NOT interpret it as a
    probability of correctness.
  - Accuracy is not characterized. Do NOT report a percentage accuracy for this
    method without rigorous evaluation.

Planned improvement:
  Phase 4 will explore Salience-based or deep-learning tonic estimation methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import librosa

from app.utils.logger import get_logger

logger = get_logger(__name__)

# 12 pitch class names starting from C
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class TonicResult:
    """
    Estimated tonic result.

    Attributes:
        estimated_note:     Pitch class name (e.g. 'C', 'C#', 'D').
                            Do NOT assume octave.
        chroma_energy:      Mean chroma energy for all 12 pitch classes.
                            (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
        dominance_ratio:    Tonic energy / mean energy across all pitch classes.
                            This is NOT a probability of correctness.
        method:             'chroma_histogram_peak'
        caveat:             Research caveat string that MUST be displayed in the UI.
        analysis_scope:     'global' (always — tonic uses full signal).
    """
    estimated_note: Optional[str]
    chroma_energy: list[float]
    dominance_ratio: Optional[float]
    method: str
    caveat: str
    analysis_scope: str = "global"


TONIC_CAVEAT = (
    "Tonic estimation uses the dominant chroma energy bin as a heuristic. "
    "This is an approximation that may not reflect the true musical tonic (Sa). "
    "Indian classical tonic is determined by the drone/shruti setting and "
    "cannot be reliably inferred from chroma energy alone. "
    "Do not rely on this estimate for musicological analysis without expert verification."
)


# ─── Tonic Estimation ─────────────────────────────────────────────────────────


def estimate_tonic(samples: np.ndarray, sample_rate: int) -> TonicResult:
    """
    Estimate the tonic note using chroma histogram peak heuristic.

    Args:
        samples:     Mono float32 audio array (FULL recording — do not pass excerpt).
        sample_rate: Sample rate in Hz.

    Returns:
        TonicResult with estimated pitch class and energy distribution.
    """
    duration_s = len(samples) / sample_rate
    logger.info(f"Estimating tonic on full {duration_s:.1f}s signal.")

    # ── Compute chromagram ────────────────────────────────────────────────────
    # Using CQT-based chroma (more frequency-stable than STFT chroma)
    chroma = librosa.feature.chroma_cqt(
        y=samples,
        sr=sample_rate,
        bins_per_octave=36,  # Finer resolution for better pitch class separation
    )
    # chroma shape: (12, n_frames)

    # ── Build energy histogram ────────────────────────────────────────────────
    # Mean chroma energy per pitch class across all frames
    chroma_mean = np.mean(chroma, axis=1)  # shape: (12,)
    chroma_mean_normalized = chroma_mean / (np.sum(chroma_mean) + 1e-10)

    # ── Find dominant pitch class ─────────────────────────────────────────────
    dominant_idx = int(np.argmax(chroma_mean))
    estimated_note = PITCH_CLASSES[dominant_idx]

    # ── Dominance ratio ───────────────────────────────────────────────────────
    mean_energy = float(np.mean(chroma_mean))
    dominant_energy = float(chroma_mean[dominant_idx])
    if mean_energy > 1e-10:
        dominance_ratio = round(dominant_energy / mean_energy, 3)
    else:
        dominance_ratio = None

    # ── Prepare per-pitch-class energy list ───────────────────────────────────
    chroma_energy_list = [round(float(v), 4) for v in chroma_mean_normalized]

    logger.info(
        f"Estimated tonic: {estimated_note} "
        f"(dominance_ratio={dominance_ratio}, method=chroma_histogram_peak)"
    )

    return TonicResult(
        estimated_note=estimated_note,
        chroma_energy=chroma_energy_list,
        dominance_ratio=dominance_ratio,
        method="chroma_histogram_peak",
        caveat=TONIC_CAVEAT,
        analysis_scope="global",
    )
