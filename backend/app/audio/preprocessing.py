"""
SwarSense — Audio Preprocessing
=================================
Handles amplitude normalization and windowing for long recordings.

Design:
  - Global analysis is always performed on the FULL audio signal.
  - For recordings longer than WINDOW_THRESHOLD_S, segment-level analysis
    is ADDITIONALLY performed using sliding windows.
  - Both global and segment results are returned — the segment analysis
    does NOT replace global analysis.

Window parameters (configurable):
  WINDOW_SIZE_S = 30.0 seconds
  OVERLAP_RATIO = 0.5 (50% overlap → hop = 15 s)
  WINDOW_THRESHOLD_S = 60.0 seconds (only window if longer than this)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

WINDOW_SIZE_S: float = 30.0       # Window duration in seconds
OVERLAP_RATIO: float = 0.5        # 50% overlap
WINDOW_THRESHOLD_S: float = 60.0  # Only window audio longer than this


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class AudioWindow:
    """A single analysis window extracted from the full recording."""
    index: int
    start_seconds: float
    end_seconds: float
    samples: np.ndarray
    sample_rate: int
    is_global: bool = False  # True only for a special "global" window entry

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class PreprocessedAudio:
    """
    Result of preprocessing a single audio file.

    Attributes:
        samples:      Full mono, resampled audio array (float32).
        sample_rate:  Target sample rate (22050 Hz).
        duration_s:   Duration of the full recording in seconds.
        windows:      Sliding windows for segment-level analysis.
                      Empty if duration_s <= WINDOW_THRESHOLD_S.
        normalized:   Whether amplitude normalization was applied.
    """
    samples: np.ndarray
    sample_rate: int
    duration_s: float
    windows: list[AudioWindow] = field(default_factory=list)
    normalized: bool = False
    original_sample_rate: int = 0
    original_channels: int = 1


# ─── Core Functions ───────────────────────────────────────────────────────────


def normalize_amplitude(samples: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Normalize audio amplitude.

    Args:
        samples: Float32 audio array.
        method:  'peak'  — peak (max-abs) normalization to [-1, 1].
                 'rms'   — RMS normalization to target RMS of -23 dBFS.

    Returns:
        Normalized audio array.

    Notes:
        Peak normalization is used by default as it is simple and reversible.
        RMS normalization is more appropriate for loudness matching.
        This normalization is applied BEFORE feature extraction, not to the
        final audio output — it improves DSP feature stability.
    """
    if method == "peak":
        peak = np.max(np.abs(samples))
        if peak < 1e-8:
            logger.warning("Audio appears to be silent (near-zero amplitude). Skipping normalization.")
            return samples
        return samples / peak

    elif method == "rms":
        rms = np.sqrt(np.mean(samples ** 2))
        if rms < 1e-8:
            logger.warning("Audio has near-zero RMS. Skipping normalization.")
            return samples
        # Target RMS: -23 dBFS ≈ 0.0708 linear
        target_rms = 10 ** (-23 / 20)
        return samples * (target_rms / rms)

    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Use 'peak' or 'rms'.")


def create_windows(
    samples: np.ndarray,
    sample_rate: int,
    window_size_s: float = WINDOW_SIZE_S,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list[AudioWindow]:
    """
    Split audio into overlapping windows.

    Args:
        samples:       Full audio array.
        sample_rate:   Sample rate in Hz.
        window_size_s: Window duration in seconds.
        overlap_ratio: Overlap fraction (0.5 = 50% overlap).

    Returns:
        List of AudioWindow objects. The global signal is NOT in this list —
        callers must keep the full `samples` array separately.

    Example:
        For a 120-second recording with 30s windows and 50% overlap:
        Window 0:   0s – 30s
        Window 1:  15s – 45s
        Window 2:  30s – 60s
        ...
    """
    total_samples = len(samples)
    window_samples = int(window_size_s * sample_rate)
    hop_samples = int(window_samples * (1 - overlap_ratio))

    if hop_samples <= 0:
        raise ValueError(f"Invalid window/overlap parameters: hop_samples={hop_samples}")

    windows: list[AudioWindow] = []
    start = 0
    idx = 0

    while start < total_samples:
        end = min(start + window_samples, total_samples)
        window_audio = samples[start:end]

        # Skip windows shorter than 1 second (they happen at the tail)
        if len(window_audio) < sample_rate:
            break

        start_s = start / sample_rate
        end_s = end / sample_rate

        windows.append(AudioWindow(
            index=idx,
            start_seconds=start_s,
            end_seconds=end_s,
            samples=window_audio,
            sample_rate=sample_rate,
        ))
        idx += 1
        start += hop_samples

    logger.debug(f"Created {len(windows)} windows from {len(samples)/sample_rate:.1f}s audio.")
    return windows


def preprocess(
    samples: np.ndarray,
    sample_rate: int,
    original_sample_rate: int,
    original_channels: int,
    normalize: bool = True,
) -> PreprocessedAudio:
    """
    Full preprocessing pipeline.

    Args:
        samples:               Mono, resampled audio array (from loader.py).
        sample_rate:           Current sample rate (should be 22050 Hz).
        original_sample_rate:  Original sample rate before resampling.
        original_channels:     Number of channels in the original file.
        normalize:             Whether to apply peak normalization.

    Returns:
        PreprocessedAudio containing the full signal, metadata, and windows.
    """
    duration_s = len(samples) / sample_rate

    # ── Normalization ─────────────────────────────────────────────────────────
    if normalize:
        samples = normalize_amplitude(samples, method="peak")
        normalized = True
    else:
        normalized = False

    # ── Windowing (only for long recordings) ─────────────────────────────────
    windows: list[AudioWindow] = []
    if duration_s > WINDOW_THRESHOLD_S:
        logger.info(
            f"Audio duration {duration_s:.1f}s > {WINDOW_THRESHOLD_S}s. "
            "Creating sliding windows for segment-level analysis."
        )
        windows = create_windows(samples, sample_rate)

    return PreprocessedAudio(
        samples=samples,
        sample_rate=sample_rate,
        duration_s=duration_s,
        windows=windows,
        normalized=normalized,
        original_sample_rate=original_sample_rate,
        original_channels=original_channels,
    )
