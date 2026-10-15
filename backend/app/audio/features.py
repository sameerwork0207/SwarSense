"""
SwarSense — DSP Feature Extraction
=====================================
Extracts MFCC, Chroma, Spectral, and HPSS features.

All feature extraction is performed on the FULL audio signal globally.
For long recordings (>60s), segment-level MFCC/Chroma statistics are
additionally computed per window and included as segment_features.

Features:
  A. MFCC (Mel-Frequency Cepstral Coefficients)
     - 13 coefficients (standard for music analysis)
     - Statistics: mean, std per coefficient (global + per segment)
     
  B. Mel Spectrogram
     - 128 mel bins
     - Power in dB
     - Returned as matrix (used for visualization, NOT in JSON response directly)

  C. Chromagram
     - 12 pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
     - Mean + std per pitch class (global)
     - Used for visualization and tonic estimation

  D. Spectral Features (global statistics)
     - Spectral centroid (Hz): frequency centroid of the spectrum
     - Spectral bandwidth (Hz): spread around centroid
     - Spectral rolloff (Hz): frequency below which 85% of energy lies
     - Zero-crossing rate: rate of sign changes (relates to noisiness/texture)

  E. HPSS (Harmonic-Percussive Source Separation)
     - harmonic_energy_ratio: fraction of energy in harmonic component
     - percussive_energy_ratio: fraction in percussive component
     - Useful for characterizing melody vs. rhythm balance

Note on data volumes:
  Mel spectrogram and STFT raw matrices are NOT returned in the main JSON
  response (they would be megabytes). They are stored in the job store and
  rendered as PNG images by visualization.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import librosa

from app.audio.preprocessing import PreprocessedAudio
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

N_MFCC = 13
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class MFCCFeatures:
    """MFCC summary statistics (global)."""
    n_coefficients: int
    mean: list[float]       # Mean per coefficient (length = n_coefficients)
    std: list[float]        # Std per coefficient


@dataclass
class ChromaFeatures:
    """Chroma summary statistics (global)."""
    pitch_classes: list[str]  # ["C", "C#", ..., "B"]
    mean: list[float]         # Mean energy per pitch class (12 values)
    std: list[float]          # Std per pitch class


@dataclass
class SpectralFeatures:
    """Spectral descriptor statistics (global)."""
    centroid_mean_hz: float
    centroid_std_hz: float
    bandwidth_mean_hz: float
    rolloff_mean_hz: float
    zero_crossing_rate_mean: float
    zero_crossing_rate_std: float


@dataclass
class HPSSFeatures:
    """Harmonic-Percussive separation energy ratios."""
    harmonic_energy_ratio: float    # Fraction of total energy in harmonic component
    percussive_energy_ratio: float  # Fraction in percussive component
    method: str = "librosa_hpss"


@dataclass
class SegmentFeatures:
    """Feature statistics for a single window/segment."""
    window_index: int
    start_s: float
    end_s: float
    mfcc_mean: list[float]
    mfcc_std: list[float]
    chroma_mean: list[float]


@dataclass
class AudioFeatures:
    """
    All extracted DSP features for the analysis result.
    
    The mel_spectrogram_db and stft_db attributes are raw matrices used
    internally by visualization.py — they are NOT serialized into the API JSON.
    """
    mfcc: MFCCFeatures
    chroma: ChromaFeatures
    spectral: SpectralFeatures
    hpss: HPSSFeatures

    # Raw matrices for visualization (not in API response)
    mel_spectrogram_db: Optional[np.ndarray] = None  # shape: (n_mels, n_frames)
    stft_db: Optional[np.ndarray] = None              # shape: (1+n_fft//2, n_frames)
    chroma_matrix: Optional[np.ndarray] = None        # shape: (12, n_frames)
    times: Optional[np.ndarray] = None                # frame time axis

    # Segment-level features (only populated for long recordings)
    segment_features: list[SegmentFeatures] = field(default_factory=list)
    analysis_scope: str = "global"


# ─── Feature Extraction ───────────────────────────────────────────────────────


def extract_features(audio: PreprocessedAudio) -> AudioFeatures:
    """
    Extract all DSP features from preprocessed audio.

    Args:
        audio: PreprocessedAudio containing the full mono signal and windows.

    Returns:
        AudioFeatures with global statistics, raw matrices for viz, and
        optional segment-level features for long recordings.
    """
    samples = audio.samples
    sr = audio.sample_rate
    duration_s = audio.duration_s

    logger.info(f"Extracting DSP features from {duration_s:.1f}s audio at {sr} Hz.")

    # ── STFT (full recording) ─────────────────────────────────────────────────
    stft = librosa.stft(samples, n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft_magnitude = np.abs(stft)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    times = librosa.frames_to_time(
        np.arange(stft_db.shape[1]), sr=sr, hop_length=HOP_LENGTH
    )

    # ── Mel Spectrogram (full recording) ─────────────────────────────────────
    mel_spec = librosa.feature.melspectrogram(
        y=samples,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # ── MFCC (global) ─────────────────────────────────────────────────────────
    mfcc_matrix = librosa.feature.mfcc(
        y=samples,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    mfcc_mean = [round(float(v), 4) for v in np.mean(mfcc_matrix, axis=1)]
    mfcc_std = [round(float(v), 4) for v in np.std(mfcc_matrix, axis=1)]

    mfcc_features = MFCCFeatures(
        n_coefficients=N_MFCC,
        mean=mfcc_mean,
        std=mfcc_std,
    )

    # ── Chroma (global) ───────────────────────────────────────────────────────
    chroma_matrix = librosa.feature.chroma_stft(
        S=stft_magnitude ** 2,
        sr=sr,
        n_chroma=12,
        hop_length=HOP_LENGTH,
    )
    chroma_mean = [round(float(v), 4) for v in np.mean(chroma_matrix, axis=1)]
    chroma_std = [round(float(v), 4) for v in np.std(chroma_matrix, axis=1)]

    chroma_features = ChromaFeatures(
        pitch_classes=PITCH_CLASSES,
        mean=chroma_mean,
        std=chroma_std,
    )

    # ── Spectral Features (global) ────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(S=stft_magnitude, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft_magnitude, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft_magnitude, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(samples, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

    spectral_features = SpectralFeatures(
        centroid_mean_hz=round(float(np.mean(centroid)), 2),
        centroid_std_hz=round(float(np.std(centroid)), 2),
        bandwidth_mean_hz=round(float(np.mean(bandwidth)), 2),
        rolloff_mean_hz=round(float(np.mean(rolloff)), 2),
        zero_crossing_rate_mean=round(float(np.mean(zcr)), 6),
        zero_crossing_rate_std=round(float(np.std(zcr)), 6),
    )

    # ── HPSS (global) ─────────────────────────────────────────────────────────
    harmonic, percussive = librosa.effects.hpss(samples)
    total_energy = float(np.sum(samples ** 2))
    if total_energy > 1e-10:
        harmonic_ratio = round(float(np.sum(harmonic ** 2)) / total_energy, 4)
        percussive_ratio = round(float(np.sum(percussive ** 2)) / total_energy, 4)
    else:
        harmonic_ratio = 0.0
        percussive_ratio = 0.0

    hpss_features = HPSSFeatures(
        harmonic_energy_ratio=harmonic_ratio,
        percussive_energy_ratio=percussive_ratio,
    )

    # ── Segment-level features (for long recordings) ──────────────────────────
    segment_features: list[SegmentFeatures] = []
    if audio.windows:
        logger.info(f"Extracting features for {len(audio.windows)} segments.")
        for win in audio.windows:
            seg_mfcc = librosa.feature.mfcc(
                y=win.samples,
                sr=win.sample_rate,
                n_mfcc=N_MFCC,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
            seg_chroma = librosa.feature.chroma_stft(
                y=win.samples,
                sr=win.sample_rate,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
            segment_features.append(SegmentFeatures(
                window_index=win.index,
                start_s=round(win.start_seconds, 3),
                end_s=round(win.end_seconds, 3),
                mfcc_mean=[round(float(v), 4) for v in np.mean(seg_mfcc, axis=1)],
                mfcc_std=[round(float(v), 4) for v in np.std(seg_mfcc, axis=1)],
                chroma_mean=[round(float(v), 4) for v in np.mean(seg_chroma, axis=1)],
            ))

    analysis_scope = f"global + {len(audio.windows)} segments" if audio.windows else "global"
    logger.info(f"Feature extraction complete. Scope: {analysis_scope}")

    return AudioFeatures(
        mfcc=mfcc_features,
        chroma=chroma_features,
        spectral=spectral_features,
        hpss=hpss_features,
        mel_spectrogram_db=mel_spec_db,
        stft_db=stft_db,
        chroma_matrix=chroma_matrix,
        times=times,
        segment_features=segment_features,
        analysis_scope=analysis_scope,
    )
