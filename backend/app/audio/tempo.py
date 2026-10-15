"""
SwarSense — Tempo & Beat Detection
=====================================
Estimates BPM and detects beat positions using librosa's beat tracker.

Method:
  librosa.beat.beat_track() — onset-strength-based autocorrelation.
  This is a standard DSP algorithm, NOT a trained ML model.

Limitations:
  - Indian classical music often has irregular or non-Western rhythmic patterns
    (e.g. Vilambit laya / slow tempo in Khayal may not give stable BPM estimates).
  - The beat tracker is optimized for Western music rhythmic patterns.
  - BPM is an estimate — label it accordingly in the UI.
  - For very slow or free-rhythm pieces, BPM may be unreliable.
  - Confidence is NOT fabricated — only the raw librosa estimate is returned.

Global analysis:
  BPM and beats are always estimated on the FULL audio signal.
  If the signal is long, librosa internally uses an efficient onset detection
  pipeline and does not need to process sample-by-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import librosa

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class TempoResult:
    """
    Tempo estimation and beat detection results.

    Attributes:
        bpm:              Estimated tempo in beats per minute (global).
        beat_times_s:     Beat positions in seconds (global analysis, full recording).
        n_beats:          Number of detected beats.
        analysis_scope:   Always 'global' for this module.
        method:           Algorithm used.
    """
    bpm: float
    beat_times_s: list[float]
    n_beats: int
    analysis_scope: str = "global"
    method: str = "librosa_beat_track"


# ─── Beat Tracking ────────────────────────────────────────────────────────────


def estimate_tempo(samples: np.ndarray, sample_rate: int) -> TempoResult:
    """
    Estimate tempo (BPM) and detect beat positions from the full audio signal.

    Args:
        samples:     Mono float32 audio array (full recording).
        sample_rate: Sample rate in Hz (should be 22050).

    Returns:
        TempoResult with BPM and beat timestamps.

    Notes:
        Performed on the GLOBAL (full) audio signal.
        Beat times may be sparse or unreliable for free-rhythm compositions.
    """
    logger.info(f"Estimating tempo on full {len(samples)/sample_rate:.1f}s signal.")

    try:
        # Compute onset strength envelope (mel-band based)
        onset_env = librosa.onset.onset_strength(
            y=samples,
            sr=sample_rate,
            aggregate=np.median,  # More robust to noise than mean
        )

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sample_rate,
        )

        # Convert to float (numpy scalar → Python float)
        bpm = float(np.round(float(np.atleast_1d(tempo)[0]), 2))

        # Convert beat frames to seconds
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        beat_times_s = [round(float(t), 3) for t in beat_times]

        logger.info(f"Tempo estimated: {bpm} BPM, {len(beat_times_s)} beats detected.")

        return TempoResult(
            bpm=bpm,
            beat_times_s=beat_times_s,
            n_beats=len(beat_times_s),
            analysis_scope="global",
            method="librosa_beat_track",
        )

    except Exception as exc:
        logger.error(f"Tempo estimation failed: {exc}")
        raise
