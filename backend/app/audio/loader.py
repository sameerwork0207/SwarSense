"""
SwarSense — Audio Loader
=========================
Loads audio files from raw bytes using soundfile + librosa.

Responsibilities:
  - Decode WAV/MP3 bytes to a numpy float32 array.
  - Convert multi-channel to mono (average channels).
  - Resample to the target sample rate (22050 Hz).
  - Read original metadata: channels, sample rate, format.
  - Validate decoded audio: duration, empty signal.

This module is the single entry-point for converting raw file bytes into
a numpy array that the feature-extraction pipeline can work with.

Error handling:
  - sf.SoundFileError → AudioLoadError (corrupted/unsupported codec)
  - librosa.ParameterError → AudioLoadError
  - Empty signal after decoding → AudioLoadError
  - Duration exceeding limit → AudioDurationError
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
import librosa

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

TARGET_SAMPLE_RATE: int = 22050    # Hz — standard for music analysis
MAX_DURATION_SECONDS: float = 600.0  # 10 minutes


# ─── Exceptions ───────────────────────────────────────────────────────────────


class AudioLoadError(Exception):
    """Raised when audio cannot be loaded or decoded."""


class AudioDurationError(AudioLoadError):
    """Raised when audio exceeds the maximum allowed duration."""


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class LoadedAudio:
    """
    Decoded audio ready for preprocessing and feature extraction.

    Attributes:
        samples:         Mono float32 audio array at TARGET_SAMPLE_RATE.
        sample_rate:     Actual sample rate after resampling (22050 Hz).
        original_sr:     Sample rate of the original file before resampling.
        original_channels: Number of channels in the original file (1=mono, 2=stereo).
        duration_seconds: Total duration in seconds.
        detected_format:  Detected audio format ('wav', 'mp3', etc.).
        n_samples:       Total number of samples at target sample rate.
    """
    samples: np.ndarray
    sample_rate: int
    original_sr: int
    original_channels: int
    duration_seconds: float
    detected_format: str
    n_samples: int


# ─── Loader ───────────────────────────────────────────────────────────────────


class AudioLoader:
    """
    Loads audio files from raw bytes.

    Supports WAV (all standard bit depths) and MP3 via soundfile/libsndfile.
    Falls back to librosa.load (which uses audioread) for MP3 on systems where
    soundfile cannot decode MP3 directly.
    """

    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        max_duration_s: float = MAX_DURATION_SECONDS,
    ):
        self.target_sr = target_sr
        self.max_duration_s = max_duration_s

    def load_bytes(self, content: bytes, filename: str = "audio") -> LoadedAudio:
        """
        Decode raw audio bytes into a mono float32 numpy array.

        Args:
            content:   Raw file bytes (WAV or MP3).
            filename:  Original filename (used for format detection and logging).

        Returns:
            LoadedAudio containing mono samples at target_sr and metadata.

        Raises:
            AudioLoadError:     If decoding fails or audio is empty.
            AudioDurationError: If audio exceeds max_duration_s.
        """
        if not content:
            raise AudioLoadError("Cannot load empty file content.")

        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        logger.info(f"Loading audio: '{filename}', size={len(content)/1024:.1f} KB, ext='{ext}'")

        # ── Primary load: soundfile (fast, handles WAV + some MP3 builds) ─────
        samples, original_sr, original_channels = self._load_soundfile(content, filename)

        # ── Detect format ─────────────────────────────────────────────────────
        detected_format = ext if ext in ("wav", "mp3") else "unknown"

        # ── Duration check on original samples ────────────────────────────────
        original_n_samples = len(samples) if samples.ndim == 1 else samples.shape[0]
        duration_s = original_n_samples / original_sr

        if duration_s > self.max_duration_s:
            raise AudioDurationError(
                f"Audio duration {duration_s:.1f}s exceeds the maximum "
                f"allowed duration of {self.max_duration_s:.0f}s (10 minutes)."
            )

        if duration_s < 0.1:
            raise AudioLoadError(
                f"Audio is too short ({duration_s:.3f}s). "
                "Minimum supported duration is 0.1 seconds."
            )

        # ── Convert to mono ───────────────────────────────────────────────────
        if samples.ndim > 1:
            # samples is (n_samples, n_channels) from soundfile
            mono = np.mean(samples, axis=1).astype(np.float32)
            logger.debug(f"Converted {original_channels}-channel audio to mono.")
        else:
            mono = samples.astype(np.float32)

        # ── Resample to target SR ─────────────────────────────────────────────
        if original_sr != self.target_sr:
            logger.debug(f"Resampling {original_sr} Hz → {self.target_sr} Hz.")
            mono = librosa.resample(mono, orig_sr=original_sr, target_sr=self.target_sr)

        n_samples = len(mono)
        actual_duration = n_samples / self.target_sr

        logger.info(
            f"Audio loaded: duration={actual_duration:.2f}s, "
            f"original_sr={original_sr}, target_sr={self.target_sr}, "
            f"channels={original_channels}, format='{detected_format}'"
        )

        return LoadedAudio(
            samples=mono,
            sample_rate=self.target_sr,
            original_sr=original_sr,
            original_channels=original_channels,
            duration_seconds=actual_duration,
            detected_format=detected_format,
            n_samples=n_samples,
        )

    def _load_soundfile(
        self, content: bytes, filename: str
    ) -> tuple[np.ndarray, int, int]:
        """
        Attempt to decode audio using soundfile.
        Falls back to librosa.load (audioread) for MP3.

        Returns:
            (samples, sample_rate, n_channels)
        """
        buf = io.BytesIO(content)
        try:
            samples, sr = sf.read(buf, dtype="float32", always_2d=False)
            # Determine channel count from shape
            n_channels = 1 if samples.ndim == 1 else samples.shape[1]
            return samples, sr, n_channels

        except sf.SoundFileError as e:
            # soundfile may not decode MP3 on all platforms — fall back to librosa
            logger.warning(
                f"soundfile failed to decode '{filename}': {e}. "
                "Falling back to librosa.load (audioread)."
            )
            return self._load_librosa(content, filename)

    def _load_librosa(
        self, content: bytes, filename: str
    ) -> tuple[np.ndarray, int, int]:
        """
        Decode audio using librosa.load (uses audioread backend).
        Returns mono signal at original sample rate.
        """
        buf = io.BytesIO(content)
        try:
            # librosa always returns mono (mono=True) at original sr by default
            samples, sr = librosa.load(buf, sr=None, mono=False)
        except Exception as exc:
            raise AudioLoadError(
                f"Failed to decode audio '{filename}': {exc}"
            ) from exc

        if samples is None or len(samples) == 0:
            raise AudioLoadError(f"Audio '{filename}' decoded to an empty signal.")

        # librosa.load with mono=False returns (channels, samples) if stereo
        if samples.ndim == 2:
            n_channels = samples.shape[0]
            # Convert (channels, samples) → (samples, channels) for consistency
            samples = samples.T
        else:
            n_channels = 1

        return samples.astype(np.float32), sr, n_channels
