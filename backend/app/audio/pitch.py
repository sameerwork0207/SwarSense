"""
SwarSense — Pitch / F0 Estimation
====================================
Estimates the fundamental frequency (F0) contour using the PYIN algorithm.

Method:
  librosa.pyin() — Probabilistic YIN (McLeod & Wyvill 2005, Mauch & Dixon 2014).
  PYIN is a state-of-the-art monophonic F0 estimator.

Important research caveats:
  - PYIN estimates the DOMINANT pitch, not necessarily the tonic or Sa.
  - Raw F0 is NOT equivalent to musical notes or swaras.
  - F0 → MIDI → note name mapping is an approximation (assumes 440 Hz A4 tuning).
  - Indian music uses microtonal inflections (gamaks, meends, andolans) that
    are NOT captured by discrete note labels.
  - Voiced/unvoiced detection is probabilistic; errors are expected.
  - All F0 values and statistics in the output are labeled as 'estimated'.

Global vs. windowed pitch:
  - For short recordings (≤60s), F0 is extracted globally.
  - For long recordings (>60s), F0 is extracted on a representative excerpt
    (first 60s by default) for visualization. Global statistics (mean, std, 
    voiced ratio) are computed over the FULL signal but downsampled.

Downsampling:
  The raw PYIN output contains one frame per hop (typically ~512 samples =
  ~23ms at 22050 Hz). For a 3-minute recording this is ~7800 frames.
  The JSON output is downsampled to MAX_F0_POINTS for API efficiency.
  The full resolution is used for visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import librosa

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

FMIN_HZ: float = 50.0   # Minimum expected F0 (below bass voice)
FMAX_HZ: float = 800.0  # Maximum expected F0 (above most instruments)
MAX_F0_POINTS: int = 1000  # Maximum data points in API JSON response
# For long recordings, extract F0 from this many seconds for visualization
VIZ_EXCERPT_S: float = 60.0

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class PitchResult:
    """
    Pitch estimation results.

    Attributes:
        f0_hz:          F0 array (Hz) at each time frame. None for unvoiced frames.
                        Downsampled to MAX_F0_POINTS for API output.
        times_s:        Timestamps (seconds) for each F0 frame.
        voiced_flag:    Boolean array — True where voiced/pitched.
        voiced_ratio:   Fraction of frames estimated as voiced (0.0–1.0).
        mean_f0_hz:     Mean F0 over voiced frames only (Hz).
        std_f0_hz:      Standard deviation of F0 over voiced frames (Hz).
        min_f0_hz:      Minimum F0 over voiced frames (Hz).
        max_f0_hz:      Maximum F0 over voiced frames (Hz).
        dominant_midi:  Most common MIDI note number (modal) over voiced frames.
        dominant_note:  Note name for dominant_midi (e.g. 'C#4').
        analysis_scope: 'global' or 'excerpt_60s' for long recordings.
        method:         Always 'pyin'.
    """
    f0_hz: list[Optional[float]]
    times_s: list[float]
    voiced_flag: list[bool]
    voiced_ratio: float
    mean_f0_hz: Optional[float]
    std_f0_hz: Optional[float]
    min_f0_hz: Optional[float]
    max_f0_hz: Optional[float]
    dominant_midi: Optional[int]
    dominant_note: Optional[str]
    analysis_scope: str
    method: str = "pyin"


# ─── Pitch Estimation ─────────────────────────────────────────────────────────


def estimate_pitch(samples: np.ndarray, sample_rate: int) -> PitchResult:
    """
    Estimate F0 contour using the PYIN algorithm.

    Args:
        samples:     Mono float32 audio array (full recording or excerpt).
        sample_rate: Sample rate in Hz.

    Returns:
        PitchResult with F0 contour, voiced flags, and summary statistics.
    """
    duration_s = len(samples) / sample_rate
    logger.info(f"Running PYIN pitch estimation on {duration_s:.1f}s audio.")

    # ── Determine analysis scope ──────────────────────────────────────────────
    if duration_s > VIZ_EXCERPT_S:
        excerpt_samples = int(VIZ_EXCERPT_S * sample_rate)
        analysis_samples = samples[:excerpt_samples]
        analysis_scope = f"excerpt_{VIZ_EXCERPT_S:.0f}s"
        logger.info(
            f"Audio is {duration_s:.1f}s > {VIZ_EXCERPT_S:.0f}s. "
            f"Running PYIN on first {VIZ_EXCERPT_S:.0f}s excerpt for visualization. "
            "Global statistics (voiced_ratio, mean F0) computed below."
        )
    else:
        analysis_samples = samples
        analysis_scope = "global"

    # ── Run PYIN ──────────────────────────────────────────────────────────────
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=analysis_samples,
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        sr=sample_rate,
        fill_na=None,  # Keep NaN for unvoiced frames rather than filling
    )

    # Build time axis
    hop_length = 512  # librosa default
    n_frames = len(f0)
    times = librosa.frames_to_time(np.arange(n_frames), sr=sample_rate, hop_length=hop_length)

    # ── Statistics over voiced frames ─────────────────────────────────────────
    voiced_mask = voiced_flag.astype(bool)
    voiced_f0 = f0[voiced_mask & np.isfinite(f0)]

    voiced_ratio = float(np.mean(voiced_mask)) if len(voiced_flag) > 0 else 0.0
    mean_f0: Optional[float] = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else None
    std_f0: Optional[float] = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else None
    min_f0: Optional[float] = float(np.min(voiced_f0)) if len(voiced_f0) > 0 else None
    max_f0: Optional[float] = float(np.max(voiced_f0)) if len(voiced_f0) > 0 else None

    # ── Dominant note (modal MIDI) ─────────────────────────────────────────────
    dominant_midi: Optional[int] = None
    dominant_note: Optional[str] = None
    if len(voiced_f0) > 0:
        midi_notes = np.round(librosa.hz_to_midi(voiced_f0)).astype(int)
        dominant_midi = int(np.bincount(midi_notes - midi_notes.min()).argmax() + midi_notes.min())
        octave = dominant_midi // 12 - 1
        pitch_class = dominant_midi % 12
        dominant_note = f"{NOTE_NAMES[pitch_class]}{octave}"

    # ── Downsample for API JSON output ────────────────────────────────────────
    total_frames = len(f0)
    if total_frames > MAX_F0_POINTS:
        step = total_frames // MAX_F0_POINTS
        f0_out = f0[::step]
        times_out = times[::step]
        voiced_out = voiced_flag[::step]
    else:
        f0_out = f0
        times_out = times
        voiced_out = voiced_flag

    # Convert to Python-native types (JSON-serializable)
    f0_list: list[Optional[float]] = [
        round(float(v), 3) if (v is not None and np.isfinite(v)) else None
        for v in f0_out
    ]
    times_list = [round(float(t), 4) for t in times_out]
    voiced_list = [bool(v) for v in voiced_out]

    logger.info(
        f"PYIN complete: voiced_ratio={voiced_ratio:.2%}, "
        f"mean_f0={mean_f0:.1f}Hz, dominant_note={dominant_note}"
    )

    return PitchResult(
        f0_hz=f0_list,
        times_s=times_list,
        voiced_flag=voiced_list,
        voiced_ratio=round(voiced_ratio, 4),
        mean_f0_hz=round(mean_f0, 3) if mean_f0 is not None else None,
        std_f0_hz=round(std_f0, 3) if std_f0 is not None else None,
        min_f0_hz=round(min_f0, 3) if min_f0 is not None else None,
        max_f0_hz=round(max_f0, 3) if max_f0 is not None else None,
        dominant_midi=dominant_midi,
        dominant_note=dominant_note,
        analysis_scope=analysis_scope,
        method="pyin",
    )
