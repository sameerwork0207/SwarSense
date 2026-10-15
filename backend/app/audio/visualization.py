"""
SwarSense — Audio Visualization (Server-Side)
===============================================
Generates Matplotlib chart images (PNG bytes) for the frontend visualization API.

All charts are generated with the non-interactive Agg backend (no display required).
This module is called by the analysis pipeline and stores PNG bytes in the job store,
served by dedicated GET /api/result/{job_id}/{chart_type} endpoints.

Chart types:
  1. Waveform         — Amplitude over time with beat markers overlaid.
  2. Spectrogram      — STFT magnitude in dB (log-frequency scale).
  3. Mel Spectrogram  — Mel-filterbank energies in dB.
  4. Chromagram       — 12-bin chroma energy over time.
  5. Pitch Contour    — F0 (Hz) over time with voiced/unvoiced coloring.

Chart labeling:
  Each chart is labeled to indicate whether it represents:
    - The complete recording (global analysis), or
    - A selected/representative excerpt (e.g., first 60s of a long file).
  This is displayed in the chart subtitle.

Design notes:
  - All figures use a dark theme (facecolor '#1a1a2e') for UI consistency.
  - PNG bytes are returned (not file paths) so they are kept in-memory.
  - Figures are explicitly closed after rendering to prevent memory leaks.
  - DPI=150 produces ~1200×600 px images (good quality without bloat).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import librosa
import librosa.display

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Visual Theme ─────────────────────────────────────────────────────────────

DARK_BG = "#0f0f1a"
PANEL_BG = "#1a1a2e"
ACCENT = "#8b5cf6"
ACCENT2 = "#06b6d4"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#2d2d4e"
BEAT_COLOR = "#f59e0b"
DPI = 150
FIGSIZE = (10, 4)


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class VisualizationBundle:
    """
    Collection of PNG images generated for one analysis job.
    Each field holds raw PNG bytes (or None if generation failed).

    Attributes are stored in the job store and served via individual endpoints.
    They are NOT serialized into the main AnalysisResult JSON.
    """
    waveform: Optional[bytes] = None
    spectrogram: Optional[bytes] = None
    mel_spectrogram: Optional[bytes] = None
    chromagram: Optional[bytes] = None
    pitch: Optional[bytes] = None
    errors: dict[str, str] = field(default_factory=dict)


# ─── Chart Helpers ────────────────────────────────────────────────────────────


def _apply_dark_theme(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)


def _to_png(fig) -> bytes:
    """Render figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _scope_subtitle(analysis_scope: str, duration_s: float) -> str:
    """Build a subtitle string indicating the analysis scope."""
    if analysis_scope == "global":
        return f"Full recording ({duration_s:.1f}s)"
    else:
        return f"Excerpt: {analysis_scope} of full recording ({duration_s:.1f}s)"


# ─── Chart Generators ─────────────────────────────────────────────────────────


def plot_waveform(
    samples: np.ndarray,
    sample_rate: int,
    duration_s: float,
    beat_times_s: Optional[list[float]] = None,
    analysis_scope: str = "global",
) -> bytes:
    """
    Plot audio waveform with optional beat markers.

    Args:
        samples:        Full or excerpt mono audio array.
        sample_rate:    Sample rate in Hz.
        duration_s:     Duration of the FULL recording (for subtitle label).
        beat_times_s:   Beat position timestamps in seconds.
        analysis_scope: Label for the chart subtitle.

    Returns:
        PNG image bytes.
    """
    # Downsample waveform for plotting (max 8000 points for performance)
    max_pts = 8000
    if len(samples) > max_pts:
        step = len(samples) // max_pts
        wave_display = samples[::step]
    else:
        wave_display = samples

    time_axis = np.linspace(0, len(samples) / sample_rate, len(wave_display))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _apply_dark_theme(ax, fig)

    ax.plot(time_axis, wave_display, color=ACCENT, linewidth=0.6, alpha=0.9)
    ax.fill_between(time_axis, wave_display, alpha=0.15, color=ACCENT)

    # Beat markers
    if beat_times_s:
        for bt in beat_times_s:
            if bt <= time_axis[-1]:
                ax.axvline(bt, color=BEAT_COLOR, linewidth=0.8, alpha=0.6, linestyle="--")

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    subtitle = _scope_subtitle(analysis_scope, duration_s)
    ax.set_title(f"Waveform  |  {subtitle}", fontsize=10, pad=8)
    ax.set_xlim(0, time_axis[-1])

    plt.tight_layout(pad=1.5)
    return _to_png(fig)


def plot_spectrogram(
    stft_db: np.ndarray,
    sample_rate: int,
    hop_length: int,
    duration_s: float,
    analysis_scope: str = "global",
) -> bytes:
    """
    Plot log-frequency magnitude spectrogram.

    Args:
        stft_db:        STFT magnitude in dB, shape (1+n_fft//2, n_frames).
        sample_rate:    Sample rate in Hz.
        hop_length:     STFT hop length.
        duration_s:     Duration of the full recording (for subtitle).
        analysis_scope: Label for subtitle.

    Returns:
        PNG image bytes.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _apply_dark_theme(ax, fig)

    img = librosa.display.specshow(
        stft_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        ax=ax,
        cmap="magma",
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=8)

    subtitle = _scope_subtitle(analysis_scope, duration_s)
    ax.set_title(f"Spectrogram (log-freq, dB)  |  {subtitle}", fontsize=10, pad=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)

    plt.tight_layout(pad=1.5)
    return _to_png(fig)


def plot_mel_spectrogram(
    mel_db: np.ndarray,
    sample_rate: int,
    hop_length: int,
    duration_s: float,
    analysis_scope: str = "global",
) -> bytes:
    """
    Plot Mel-filterbank spectrogram in dB.

    Args:
        mel_db:         Mel spectrogram in dB, shape (n_mels, n_frames).
        sample_rate:    Sample rate in Hz.
        hop_length:     Hop length.
        duration_s:     Full recording duration for subtitle.
        analysis_scope: Scope label.

    Returns:
        PNG image bytes.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _apply_dark_theme(ax, fig)

    img = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap="inferno",
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=8)

    subtitle = _scope_subtitle(analysis_scope, duration_s)
    ax.set_title(f"Mel Spectrogram (dB)  |  {subtitle}", fontsize=10, pad=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Mel Frequency", fontsize=9)

    plt.tight_layout(pad=1.5)
    return _to_png(fig)


def plot_chromagram(
    chroma_matrix: np.ndarray,
    sample_rate: int,
    hop_length: int,
    duration_s: float,
    analysis_scope: str = "global",
) -> bytes:
    """
    Plot chroma energy over time.

    Args:
        chroma_matrix:  Chroma matrix, shape (12, n_frames).
        sample_rate:    Sample rate in Hz.
        hop_length:     Hop length.
        duration_s:     Full recording duration for subtitle.
        analysis_scope: Scope label.

    Returns:
        PNG image bytes.
    """
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _apply_dark_theme(ax, fig)

    img = librosa.display.specshow(
        chroma_matrix,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="chroma",
        ax=ax,
        cmap="YlOrRd",
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR, fontsize=8)

    subtitle = _scope_subtitle(analysis_scope, duration_s)
    ax.set_title(f"Chromagram  |  {subtitle}", fontsize=10, pad=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Pitch Class", fontsize=9)
    # NOTE: Do NOT call set_yticklabels manually here.
    # librosa.display.specshow with y_axis='chroma' sets its own FixedLocator;
    # overriding the labels with a different count raises ValueError.
    # The default chroma tick labels (C, C#, ..., B) are already set by librosa.

    plt.tight_layout(pad=1.5)
    return _to_png(fig)


def plot_pitch_contour(
    f0_hz: list[Optional[float]],
    times_s: list[float],
    voiced_flag: list[bool],
    duration_s: float,
    beat_times_s: Optional[list[float]] = None,
    analysis_scope: str = "global",
) -> bytes:
    """
    Plot F0 contour with voiced/unvoiced distinction.

    Args:
        f0_hz:          F0 values (Hz), None for unvoiced frames.
        times_s:        Corresponding timestamps.
        voiced_flag:    Voiced/unvoiced boolean per frame.
        duration_s:     Full recording duration for subtitle.
        beat_times_s:   Beat positions to overlay.
        analysis_scope: Scope label.

    Returns:
        PNG image bytes.
    """
    times_arr = np.array(times_s)
    f0_arr = np.array([v if v is not None else np.nan for v in f0_hz])
    voiced_arr = np.array(voiced_flag, dtype=bool)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _apply_dark_theme(ax, fig)

    # Unvoiced frames as grey scatter
    unvoiced_mask = ~voiced_arr & np.isfinite(f0_arr)
    if np.any(unvoiced_mask):
        ax.scatter(
            times_arr[unvoiced_mask], f0_arr[unvoiced_mask],
            color="#475569", s=2, alpha=0.4, label="Unvoiced (estimated)"
        )

    # Voiced frames as colored scatter
    voiced_mask = voiced_arr & np.isfinite(f0_arr)
    if np.any(voiced_mask):
        ax.scatter(
            times_arr[voiced_mask], f0_arr[voiced_mask],
            color=ACCENT2, s=3, alpha=0.85, label="Voiced (estimated)"
        )

    # Beat markers
    if beat_times_s:
        for bt in beat_times_s:
            ax.axvline(bt, color=BEAT_COLOR, linewidth=0.7, alpha=0.5, linestyle=":")

    # MIDI note reference lines (C3–C6)
    import librosa
    for midi_note in range(48, 84, 12):  # C3, C4, C5, C6
        freq = librosa.midi_to_hz(midi_note)
        note_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][midi_note % 12]
        octave = midi_note // 12 - 1
        ax.axhline(
            freq, color="#334155", linewidth=0.6, linestyle="--", alpha=0.7
        )
        ax.text(
            0.01, freq, f"{note_name}{octave}", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#64748b", va="center"
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, markerscale=3)

    subtitle = _scope_subtitle(analysis_scope, duration_s)
    ax.set_title(f"Pitch Contour (F0, estimated)  |  {subtitle}", fontsize=10, pad=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Frequency (Hz, log scale)", fontsize=9)

    plt.tight_layout(pad=1.5)
    return _to_png(fig)


# ─── Orchestrator ─────────────────────────────────────────────────────────────


def generate_all_visualizations(
    samples: np.ndarray,
    sample_rate: int,
    duration_s: float,
    stft_db: np.ndarray,
    mel_db: np.ndarray,
    chroma_matrix: np.ndarray,
    hop_length: int,
    beat_times_s: list[float],
    f0_hz: list[Optional[float]],
    f0_times_s: list[float],
    voiced_flag: list[bool],
    viz_scope: str = "global",
) -> VisualizationBundle:
    """
    Generate all visualization PNGs for a completed analysis job.

    Each chart is generated in a try/except — if one fails, the others continue.
    Errors are recorded in VisualizationBundle.errors.

    Returns:
        VisualizationBundle with PNG bytes for each chart type.
    """
    bundle = VisualizationBundle()

    # 1. Waveform
    try:
        bundle.waveform = plot_waveform(
            samples=samples,
            sample_rate=sample_rate,
            duration_s=duration_s,
            beat_times_s=beat_times_s,
            analysis_scope=viz_scope,
        )
        logger.debug("Waveform chart generated.")
    except Exception as e:
        logger.error(f"Waveform chart failed: {e}")
        bundle.errors["waveform"] = str(e)

    # 2. Spectrogram
    try:
        bundle.spectrogram = plot_spectrogram(
            stft_db=stft_db,
            sample_rate=sample_rate,
            hop_length=hop_length,
            duration_s=duration_s,
            analysis_scope=viz_scope,
        )
        logger.debug("Spectrogram chart generated.")
    except Exception as e:
        logger.error(f"Spectrogram chart failed: {e}")
        bundle.errors["spectrogram"] = str(e)

    # 3. Mel spectrogram
    try:
        bundle.mel_spectrogram = plot_mel_spectrogram(
            mel_db=mel_db,
            sample_rate=sample_rate,
            hop_length=hop_length,
            duration_s=duration_s,
            analysis_scope=viz_scope,
        )
        logger.debug("Mel spectrogram chart generated.")
    except Exception as e:
        logger.error(f"Mel spectrogram chart failed: {e}")
        bundle.errors["mel_spectrogram"] = str(e)

    # 4. Chromagram
    try:
        bundle.chromagram = plot_chromagram(
            chroma_matrix=chroma_matrix,
            sample_rate=sample_rate,
            hop_length=hop_length,
            duration_s=duration_s,
            analysis_scope=viz_scope,
        )
        logger.debug("Chromagram chart generated.")
    except Exception as e:
        logger.error(f"Chromagram chart failed: {e}")
        bundle.errors["chromagram"] = str(e)

    # 5. Pitch contour
    try:
        bundle.pitch = plot_pitch_contour(
            f0_hz=f0_hz,
            times_s=f0_times_s,
            voiced_flag=voiced_flag,
            duration_s=duration_s,
            beat_times_s=beat_times_s,
            analysis_scope=viz_scope,
        )
        logger.debug("Pitch contour chart generated.")
    except Exception as e:
        logger.error(f"Pitch contour chart failed: {e}")
        bundle.errors["pitch"] = str(e)

    logger.info(
        f"Visualization complete: "
        f"{sum(1 for v in [bundle.waveform, bundle.spectrogram, bundle.mel_spectrogram, bundle.chromagram, bundle.pitch] if v is not None)}/5 charts OK, "
        f"{len(bundle.errors)} errors."
    )
    return bundle
