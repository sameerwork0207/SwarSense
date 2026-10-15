"""
SwarSense — Test Fixtures (conftest.py)
========================================
Shared pytest fixtures for all tests.

Audio fixtures:
  - synthetic_wav_bytes:      Minimal 3-second 440Hz WAV (22050 Hz, mono).
  - synthetic_wav_bytes_long: 90-second WAV (tests windowing).
  - synthetic_mp3_bytes:      WAV with .mp3 extension spoofed (for rejection test).
  - invalid_file_bytes:       Random bytes (not audio).

All audio is generated programmatically using scipy/numpy.
No large audio files are committed to the repository.
"""

import io
import struct

import numpy as np
import pytest
import scipy.io.wavfile as wav_io


def _generate_sine_wav(
    duration_s: float = 3.0,
    freq_hz: float = 440.0,
    sample_rate: int = 22050,
    amplitude: float = 0.5,
) -> bytes:
    """
    Generate a mono WAV file (PCM 16-bit) as bytes.

    Args:
        duration_s:   Duration in seconds.
        freq_hz:      Sine wave frequency.
        sample_rate:  Sample rate in Hz.
        amplitude:    Amplitude (0–1.0).

    Returns:
        Raw WAV bytes.
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Mix 440Hz + 880Hz for richer content that pyin can track
    signal = amplitude * (np.sin(2 * np.pi * freq_hz * t) + 0.3 * np.sin(2 * np.pi * freq_hz * 2 * t))
    signal = (signal * 32767).astype(np.int16)

    buf = io.BytesIO()
    wav_io.write(buf, sample_rate, signal)
    buf.seek(0)
    return buf.read()


def _generate_stereo_sine_wav(
    duration_s: float = 3.0,
    freq_hz: float = 440.0,
    sample_rate: int = 44100,
) -> bytes:
    """Generate a stereo WAV (tests mono conversion)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    left = (0.5 * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    right = (0.5 * np.sin(2 * np.pi * freq_hz * 1.5 * t) * 32767).astype(np.int16)
    stereo = np.column_stack([left, right])

    buf = io.BytesIO()
    wav_io.write(buf, sample_rate, stereo)
    buf.seek(0)
    return buf.read()


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def synthetic_wav_bytes() -> bytes:
    """3-second mono 440Hz sine wave WAV at 22050 Hz."""
    return _generate_sine_wav(duration_s=3.0, freq_hz=440.0, sample_rate=22050)


@pytest.fixture(scope="session")
def synthetic_wav_bytes_stereo() -> bytes:
    """3-second stereo 440Hz sine wave WAV at 44100 Hz (tests stereo→mono)."""
    return _generate_stereo_sine_wav(duration_s=3.0, freq_hz=440.0, sample_rate=44100)


@pytest.fixture(scope="session")
def synthetic_wav_bytes_long() -> bytes:
    """90-second mono WAV at 22050 Hz (triggers windowing logic)."""
    return _generate_sine_wav(duration_s=90.0, freq_hz=220.0, sample_rate=22050)


@pytest.fixture(scope="session")
def invalid_file_bytes() -> bytes:
    """Random bytes that are not valid audio."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=1024, dtype=np.uint8).tobytes()
