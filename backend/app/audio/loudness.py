"""
SwarSense — Loudness Measurement
==================================
Calculates integrated loudness using ITU-R BS.1770 (LUFS) and RMS (dBFS).

Methods:
  - LUFS (Loudness Units Full Scale): Industry-standard perceptual loudness
    via pyloudnorm. Requires stereo or mono float32 audio.
  - RMS (Root Mean Square): Classical amplitude-based loudness measure.

LUFS interpretation (rough guide):
  Typical music:           -14 to -9 LUFS
  Classical/quiet pieces:  -23 to -18 LUFS
  Very loud/clipping risk: above -6 LUFS

Important:
  These are signal-level measurements, NOT perceived Indian classical loudness.
  Normalization to loudness standards is not enforced — we measure only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class LoudnessResult:
    """
    Loudness measurements.

    Attributes:
        lufs:   Integrated loudness in LUFS (ITU-R BS.1770). None if pyloudnorm
                failed or audio is too short for measurement.
        rms_db: RMS loudness in dBFS (decibels relative to full scale).
        rms_linear: Raw linear RMS value.
        method: 'pyloudnorm' or 'rms_only' depending on what succeeded.
    """
    lufs: Optional[float]
    rms_db: float
    rms_linear: float
    method: str


# ─── Loudness Calculation ─────────────────────────────────────────────────────


def measure_loudness(samples: np.ndarray, sample_rate: int) -> LoudnessResult:
    """
    Calculate LUFS and RMS loudness from mono audio samples.

    Args:
        samples:     Mono float32 audio array (normalized or raw).
        sample_rate: Sample rate in Hz.

    Returns:
        LoudnessResult with LUFS and RMS measurements.

    Notes:
        pyloudnorm requires at least 400ms of audio for LUFS measurement.
        If LUFS calculation fails, only RMS is returned.
    """
    # ── RMS loudness ──────────────────────────────────────────────────────────
    rms_linear = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms_linear < 1e-10:
        rms_db = -120.0  # Silence / near-silence
    else:
        rms_db = float(20 * np.log10(rms_linear))

    # ── LUFS via pyloudnorm ───────────────────────────────────────────────────
    lufs: Optional[float] = None
    method = "rms_only"

    try:
        import pyloudnorm as pyln

        # pyloudnorm expects (n_samples,) or (n_samples, n_channels)
        # We always have mono → shape it as (n_samples, 1) then reshape
        data = samples.astype(np.float64)

        # Minimum 400ms required by BS.1770 gating
        min_samples = int(0.4 * sample_rate)
        if len(data) < min_samples:
            logger.warning(
                f"Audio too short for LUFS measurement ({len(data)/sample_rate:.2f}s < 0.4s). "
                "Returning RMS only."
            )
        else:
            meter = pyln.Meter(sample_rate)  # BS.1770-4 meter
            lufs_raw = meter.integrated_loudness(data)
            # pyloudnorm returns -inf for silence — clamp to a practical floor
            if np.isfinite(lufs_raw):
                lufs = float(lufs_raw)
                method = "pyloudnorm_bs1770"
            else:
                logger.warning("LUFS measurement returned -inf (silence). Using RMS only.")

    except ImportError:
        logger.warning(
            "pyloudnorm not installed. LUFS measurement unavailable. Using RMS only."
        )
    except Exception as exc:
        logger.warning(f"LUFS measurement failed: {exc}. Using RMS only.")

    logger.debug(f"Loudness: LUFS={lufs}, RMS={rms_db:.2f} dBFS, method={method}")
    return LoudnessResult(
        lufs=lufs,
        rms_db=round(rms_db, 3),
        rms_linear=round(rms_linear, 6),
        method=method,
    )
