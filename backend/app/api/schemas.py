"""
SwarSense — Pydantic Request / Response Schemas
================================================
All API contracts are defined here.

Design principles:
  - Every field has a description for Swagger auto-docs.
  - Optional fields use None defaults so partial results are valid.
  - Never add computed/AI fields here without a corresponding "source" field
    that explicitly states whether the value came from DSP, a pretrained model,
    or our own trained model.

Phase 1 additions:
  - Phase1AudioMetadata: file info, loudness
  - Phase1TempoResult: BPM, beat times
  - Phase1LoudnessResult: LUFS, RMS
  - Phase1PitchResult: F0 contour, voiced ratio, statistics
  - Phase1MFCCResult: coefficient statistics
  - Phase1ChromaResult: pitch class energy
  - Phase1SpectralResult: centroid, bandwidth, rolloff, ZCR
  - Phase1HPSSResult: harmonic/percussive energy ratios
  - Phase1EstimatedTonic: chroma heuristic tonic with explicit caveats
  - Phase1SegmentFeatures: per-window features for long recordings
  - VisualizationLinks: URL paths for PNG image endpoints
  - Phase1AnalysisResult: top-level Phase 1 result
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ─── Enums ─────────────────────────────────────────────────────────────────────


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some modules succeeded; others failed


class AnalysisSource(str, Enum):
    """
    Every AI/ML result must declare its source so the UI can render it
    with the correct label and caveat.
    """
    DSP = "dsp"                         # Deterministic signal processing (Librosa)
    PRETRAINED_MODEL = "pretrained"     # Externally pretrained model (YAMNet, Demucs)
    TRAINED_MODEL = "trained_model"     # Our own trained model (Raga CNN)
    HEURISTIC = "heuristic"             # Rule-based / lookup (tonic estimation)
    STATIC_KNOWLEDGE = "static"         # Static musicological reference data


# ─── Error Schema ──────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")
    detail: Optional[str] = Field(None, description="Optional technical detail.")

    model_config = {"json_schema_extra": {"example": {
        "error": "file_too_large",
        "message": "Uploaded file exceeds the 50 MB limit.",
        "detail": "Received 72.3 MB.",
    }}}


# ─── Upload / Job Schemas ──────────────────────────────────────────────────────


class AnalyzeResponse(BaseModel):
    """
    Returned immediately when a file is accepted for analysis.
    The client should poll /api/status/{job_id} for progress.
    """
    job_id: str = Field(..., description="Unique identifier for this analysis job.")
    status: JobStatus = Field(JobStatus.PENDING, description="Initial job status.")
    message: str = Field(
        "File accepted. Analysis queued.",
        description="Human-readable status message.",
    )
    filename: str = Field(..., description="Original filename of the uploaded audio.")
    estimated_duration_seconds: Optional[float] = Field(
        None, description="Estimated processing time in seconds (rough estimate only)."
    )

    model_config = {"json_schema_extra": {"example": {
        "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "status": "pending",
        "message": "File accepted. Analysis queued.",
        "filename": "raga_yaman_sample.wav",
        "estimated_duration_seconds": 30.0,
    }}}


class JobStatusResponse(BaseModel):
    """
    Returned by GET /api/status/{job_id}.
    Includes progress percentage and the latest partial results if available.
    """
    job_id: str
    status: JobStatus
    progress: int = Field(0, ge=0, le=100, description="Progress percentage (0–100).")
    message: str = Field("", description="Current processing stage description.")
    partial_result: Optional[dict[str, Any]] = Field(
        None, description="Partial results as they become available."
    )
    error: Optional[ErrorResponse] = Field(None, description="Error if status is 'failed'.")


# ─── Phase 1 Audio Feature Schemas ─────────────────────────────────────────────


class Phase1AudioMetadata(BaseModel):
    """Basic file and format metadata. Source: DSP (SoundFile/Librosa)."""
    source: AnalysisSource = AnalysisSource.DSP
    duration_seconds: float = Field(..., description="Total duration in seconds (full recording).")
    sample_rate: int = Field(..., description="Analysis sample rate in Hz (22050 after resampling).")
    original_sample_rate: int = Field(..., description="Original sample rate of the uploaded file.")
    channels: int = Field(..., description="Number of audio channels in the original file.")
    n_samples: int = Field(..., description="Total sample count at analysis sample rate.")
    file_format: str = Field(..., description="Detected audio format ('wav' or 'mp3').")
    normalized: bool = Field(..., description="Whether peak normalization was applied before feature extraction.")


class Phase1LoudnessResult(BaseModel):
    """Loudness measurements. Source: DSP (pyloudnorm + numpy)."""
    source: AnalysisSource = AnalysisSource.DSP
    lufs: Optional[float] = Field(
        None,
        description=(
            "Integrated loudness in LUFS (ITU-R BS.1770-4). Computed with pyloudnorm. "
            "Typical music: -14 to -9 LUFS. None if too short or computation failed."
        ),
    )
    rms_db: float = Field(..., description="RMS loudness in dBFS (decibels relative to full scale).")
    rms_linear: float = Field(..., description="Linear RMS amplitude value.")
    method: str = Field(..., description="'pyloudnorm_bs1770' or 'rms_only' depending on availability.")


class Phase1TempoResult(BaseModel):
    """Tempo and beat detection. Source: DSP (librosa beat tracker, global analysis)."""
    source: AnalysisSource = AnalysisSource.DSP
    bpm: float = Field(..., description="Estimated tempo in beats per minute (global analysis).")
    beat_times_seconds: list[float] = Field(
        default_factory=list,
        description="Beat positions in seconds (global analysis, full recording).",
    )
    n_beats: int = Field(..., description="Number of detected beats.")
    analysis_scope: str = Field("global", description="Always 'global' — BPM estimated on full recording.")
    method: str = Field("librosa_beat_track", description="Beat tracking algorithm.")
    caveat: str = Field(
        "BPM estimation uses librosa's onset-strength beat tracker. "
        "This may be unreliable for free-rhythm (alap) or slow-tempo (vilambit) Indian classical music.",
        description="Important limitation note.",
    )


class Phase1PitchResult(BaseModel):
    """Pitch / F0 estimation. Source: DSP (PYIN algorithm, librosa)."""
    source: AnalysisSource = AnalysisSource.DSP
    f0_hz: list[Optional[float]] = Field(
        default_factory=list,
        description=(
            "Estimated F0 per frame (Hz). None for unvoiced frames. "
            "Downsampled to max 1000 points for API efficiency."
        ),
    )
    times_seconds: list[float] = Field(
        default_factory=list,
        description="Timestamps (seconds) for each F0 frame.",
    )
    voiced_flag: list[bool] = Field(
        default_factory=list,
        description="Whether each frame is estimated as voiced (True) or unvoiced (False).",
    )
    voiced_ratio: float = Field(..., description="Fraction of frames estimated as voiced (0.0–1.0).")
    mean_f0_hz: Optional[float] = Field(None, description="Mean F0 over voiced frames (Hz).")
    std_f0_hz: Optional[float] = Field(None, description="Standard deviation of F0 over voiced frames (Hz).")
    min_f0_hz: Optional[float] = Field(None, description="Minimum F0 over voiced frames (Hz).")
    max_f0_hz: Optional[float] = Field(None, description="Maximum F0 over voiced frames (Hz).")
    dominant_midi: Optional[int] = Field(None, description="Modal MIDI note number over voiced frames.")
    dominant_note: Optional[str] = Field(None, description="Note name for dominant MIDI note (e.g. 'C#4').")
    analysis_scope: str = Field(
        ...,
        description=(
            "'global' for short recordings, 'excerpt_60s' for recordings >60s. "
            "For long recordings, voiced_ratio and statistics are over the excerpt."
        ),
    )
    method: str = Field("pyin", description="F0 estimation algorithm (PYIN).")
    caveat: str = Field(
        "F0 estimation uses the probabilistic YIN (PYIN) algorithm. "
        "This provides an estimate of the dominant pitch, not the musical tonic or swara. "
        "Microtonal inflections (gamaks, meends) are not resolved at this quantization.",
        description="Research limitation note.",
    )


class Phase1MFCCResult(BaseModel):
    """MFCC statistics. Source: DSP (librosa, global analysis)."""
    source: AnalysisSource = AnalysisSource.DSP
    n_coefficients: int = Field(13, description="Number of MFCC coefficients computed.")
    mean: list[float] = Field(..., description="Mean of each coefficient across all frames.")
    std: list[float] = Field(..., description="Standard deviation of each coefficient.")
    analysis_scope: str = Field("global", description="'global' or 'global + N segments'.")


class Phase1ChromaResult(BaseModel):
    """Chroma (pitch class energy) statistics. Source: DSP (librosa, global analysis)."""
    source: AnalysisSource = AnalysisSource.DSP
    pitch_classes: list[str] = Field(
        default_factory=lambda: ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        description="12 pitch class names in semitone order starting from C.",
    )
    mean: list[float] = Field(..., description="Mean chroma energy per pitch class (12 values).")
    std: list[float] = Field(..., description="Standard deviation of chroma energy per pitch class.")
    analysis_scope: str = Field("global", description="Analysis scope.")


class Phase1SpectralResult(BaseModel):
    """Spectral descriptor statistics. Source: DSP (librosa, global analysis)."""
    source: AnalysisSource = AnalysisSource.DSP
    centroid_mean_hz: float = Field(..., description="Mean spectral centroid (frequency center of mass) in Hz.")
    centroid_std_hz: float = Field(..., description="Standard deviation of spectral centroid.")
    bandwidth_mean_hz: float = Field(..., description="Mean spectral bandwidth (spread around centroid) in Hz.")
    rolloff_mean_hz: float = Field(..., description="Mean spectral rolloff frequency (85% energy threshold) in Hz.")
    zero_crossing_rate_mean: float = Field(..., description="Mean zero-crossing rate (proxy for noisiness).")
    zero_crossing_rate_std: float = Field(..., description="Standard deviation of zero-crossing rate.")


class Phase1HPSSResult(BaseModel):
    """Harmonic-Percussive Source Separation ratios. Source: DSP (librosa HPSS)."""
    source: AnalysisSource = AnalysisSource.DSP
    harmonic_energy_ratio: float = Field(
        ..., description="Fraction of total signal energy in the harmonic component (0.0–1.0)."
    )
    percussive_energy_ratio: float = Field(
        ..., description="Fraction in the percussive component (0.0–1.0)."
    )
    method: str = Field("librosa_hpss", description="Separation method.")
    note: str = Field(
        "HPSS provides a rough harmonic/percussive split based on median filtering. "
        "Ratios may not sum to 1.0 due to energy in the residual.",
        description="Method limitation note.",
    )


class Phase1SegmentFeature(BaseModel):
    """Feature statistics for a single analysis window (long recordings only)."""
    window_index: int
    start_seconds: float
    end_seconds: float
    mfcc_mean: list[float]
    mfcc_std: list[float]
    chroma_mean: list[float]


class Phase1EstimatedTonic(BaseModel):
    """
    Estimated tonic note. Source: HEURISTIC (chroma histogram peak).
    IMPORTANT: This is NOT a model prediction.
    Always display this with the caveat text.
    """
    source: AnalysisSource = AnalysisSource.HEURISTIC
    estimated_note: Optional[str] = Field(
        None,
        description=(
            "Estimated tonic pitch class (e.g. 'C', 'C#', 'D'). "
            "Octave is not determined. This is a heuristic estimate only."
        ),
    )
    chroma_energy: list[float] = Field(
        default_factory=list,
        description=(
            "Normalized mean chroma energy for all 12 pitch classes (C, C#, ..., B). "
            "Used to determine the dominant pitch class."
        ),
    )
    dominance_ratio: Optional[float] = Field(
        None,
        description=(
            "Ratio of dominant pitch class energy to mean energy across all classes. "
            "This is NOT a probability of correctness — it measures spectral dominance only."
        ),
    )
    method: str = Field("chroma_histogram_peak", description="Estimation method.")
    caveat: str = Field(
        ...,
        description=(
            "MUST be displayed in the UI. "
            "Tonic estimation uses chroma energy dominance and is not reliable for Indian classical music "
            "where the drone/shruti setting determines Sa."
        ),
    )
    analysis_scope: str = Field("global", description="Always global — full recording used for tonic estimation.")


class VisualizationLinks(BaseModel):
    """
    URLs for server-rendered visualization PNG images.
    Clients should request these endpoints independently.
    Images are NOT embedded in the main result JSON.
    """
    waveform: Optional[str] = Field(None, description="URL path for waveform PNG image.")
    spectrogram: Optional[str] = Field(None, description="URL path for spectrogram PNG image.")
    mel_spectrogram: Optional[str] = Field(None, description="URL path for Mel spectrogram PNG image.")
    chromagram: Optional[str] = Field(None, description="URL path for chromagram PNG image.")
    pitch: Optional[str] = Field(None, description="URL path for pitch contour PNG image.")


# ─── Phase 0 Placeholder Schemas (retained for future phases) ─────────────────


class AudioMetadata(BaseModel):
    """Basic file and format metadata. Source: DSP (SoundFile/Librosa). [Phase 0 — kept for backwards compat]"""
    source: AnalysisSource = AnalysisSource.DSP
    duration_seconds: float = Field(..., description="Total duration in seconds.")
    sample_rate: int = Field(..., description="Sample rate in Hz.")
    channels: int = Field(..., description="Number of audio channels in original file.")
    bit_depth: Optional[int] = Field(None, description="Bit depth if available.")
    format: str = Field(..., description="Audio file format (e.g. 'wav', 'mp3').")
    loudness_lufs: Optional[float] = Field(
        None, description="Integrated loudness in LUFS (ITU-R BS.1770). Source: pyloudnorm."
    )
    rms_db: Optional[float] = Field(None, description="RMS loudness in dB.")


class TempoFeatures(BaseModel):
    """Tempo and beat features. Source: DSP (Librosa beat tracker)."""
    source: AnalysisSource = AnalysisSource.DSP
    bpm: float = Field(..., description="Estimated tempo in beats per minute.")
    beat_frames: list[int] = Field(default_factory=list, description="Frame indices of detected beats.")
    beat_times_seconds: list[float] = Field(default_factory=list, description="Beat positions in seconds.")


class PitchFeatures(BaseModel):
    """Pitch / F0 contour. Source: DSP (PYIN algorithm via Librosa)."""
    source: AnalysisSource = AnalysisSource.DSP
    f0_hz: list[Optional[float]] = Field(default_factory=list, description="F0 per frame in Hz. None where unvoiced.")
    f0_times_seconds: list[float] = Field(default_factory=list, description="Time stamps for each F0 frame.")
    voiced_flag: list[bool] = Field(default_factory=list, description="Whether each frame is estimated to be voiced.")


class TonicEstimate(BaseModel):
    """Estimated tonic note. Source: heuristic (chromagram peak)."""
    source: AnalysisSource = AnalysisSource.HEURISTIC
    estimated_tonic: Optional[str] = Field(None, description="Estimated tonic pitch class (e.g. 'C', 'C#', 'D').")
    method: str = Field("chromagram_peak", description="Estimation method used.")
    confidence_note: str = Field(
        "Tonic estimation uses the dominant chroma bin as a heuristic. "
        "This is an approximation and may not reflect the true musical tonic.",
        description="Caveat about estimation reliability.",
    )


class SpectralFeatures(BaseModel):
    """Spectral descriptors. Source: DSP (Librosa)."""
    source: AnalysisSource = AnalysisSource.DSP
    mfcc_mean: list[float] = Field(default_factory=list, description="Mean of each MFCC coefficient.")
    mfcc_std: list[float] = Field(default_factory=list, description="Standard deviation of each MFCC coefficient.")
    chroma_mean: list[float] = Field(default_factory=list, description="Mean chroma energy per pitch class.")
    spectral_centroid_mean: Optional[float] = Field(None, description="Mean spectral centroid in Hz.")
    spectral_rolloff_mean: Optional[float] = Field(None, description="Mean spectral rolloff in Hz.")
    zero_crossing_rate_mean: Optional[float] = Field(None, description="Mean zero-crossing rate.")


class VisualizationData(BaseModel):
    """Pre-computed visualization data. Phase 0 schema — replaced by VisualizationLinks in Phase 1."""
    waveform_samples: list[float] = Field(default_factory=list, description="Downsampled waveform amplitude values.")
    waveform_times: list[float] = Field(default_factory=list, description="Corresponding time values.")
    mel_spectrogram_image_b64: Optional[str] = Field(None, description="Base64-encoded PNG of Mel spectrogram.")
    chromagram_image_b64: Optional[str] = Field(None, description="Base64-encoded PNG of chromagram.")
    pitch_contour_image_b64: Optional[str] = Field(None, description="Base64-encoded PNG of pitch contour.")


# ─── Instrument / Source Separation Schemas ────────────────────────────────────


class InstrumentPrediction(BaseModel):
    instrument: str = Field(..., description="Instrument name.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0–1).")
    source: AnalysisSource = Field(..., description="Which model produced this prediction.")


class InstrumentDetectionResult(BaseModel):
    """
    Instrument detection results.
    Source: Pretrained YAMNet (Phase 3+). Placeholder in Phase 0–1.
    """
    status: str = Field("not_implemented", description="'completed', 'not_implemented', or 'failed'.")
    instruments: list[InstrumentPrediction] = Field(default_factory=list)
    note: str = Field(
        "Instrument detection will be available in Phase 3. "
        "Note: Standard pretrained models (YAMNet) do not include Tabla or Harmonium. "
        "This is a documented research gap.",
        description="Implementation status note.",
    )


class SourceSeparationResult(BaseModel):
    """
    Vocal vs. instrumental energy split.
    Source: Pretrained Demucs (htdemucs). Placeholder in Phase 0–1.
    """
    status: str = Field("not_implemented", description="'completed', 'not_implemented', or 'failed'.")
    vocal_energy_percent: Optional[float] = Field(None, description="Percentage of energy in vocal stem.")
    instrumental_energy_percent: Optional[float] = Field(None, description="Percentage in instrumental stems.")
    analysis_excerpt_seconds: Optional[float] = Field(None, description="Duration of audio used for separation.")
    note: str = Field(
        "Source separation (Demucs) will be available in Phase 3. "
        "Will be applied to a 30-second excerpt due to CPU runtime constraints.",
        description="Implementation status note.",
    )


# ─── Raga Classification Schema ───────────────────────────────────────────────


class RagaPrediction(BaseModel):
    """
    Raga classification result from our trained model.
    IMPORTANT: Always display source and confidence.
    Never present uncertain predictions as facts.
    """
    source: AnalysisSource = AnalysisSource.TRAINED_MODEL
    status: str = Field("not_implemented", description="'completed', 'not_implemented', or 'failed'.")
    predicted_raga: Optional[str] = Field(None, description="Predicted raga name.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence (0–1).")
    uncertainty_threshold: float = Field(0.5, description="Predictions below this are marked 'uncertain'.")
    top_predictions: list[dict[str, Any]] = Field(default_factory=list, description="Top-k predictions.")
    grad_cam_image_b64: Optional[str] = Field(None, description="Grad-CAM heatmap (Phase 5+).")
    model_version: Optional[str] = Field(None, description="Model checkpoint identifier.")

    model_config = {"protected_namespaces": ()}
    note: str = Field(
        "Raga classification model will be trained and integrated in Phase 4. "
        "System currently supports a limited vocabulary of ragas.",
        description="Implementation status note.",
    )


class RagaKnowledgeEntry(BaseModel):
    """
    Static musicological reference information for a raga.
    Source: STATIC_KNOWLEDGE — this is reference data, NOT a model prediction.
    """
    source: AnalysisSource = AnalysisSource.STATIC_KNOWLEDGE
    name: str
    thaat: str
    aroha: str = Field(..., description="Ascending scale pattern.")
    avroha: str = Field(..., description="Descending scale pattern.")
    vadi: str = Field(..., description="Primary (most prominent) note.")
    samvadi: str = Field(..., description="Secondary prominent note.")
    time_of_day: Optional[str] = None
    characteristic_phrases: list[str] = Field(default_factory=list)
    description: str = Field("", description="Brief musicological description.")


# ─── Song Intelligence Schema (Phase 2) ───────────────────────────────────────
# Optional Stage 9. Identifies the recording via an external fingerprinting
# provider (AcoustID by default) and enriches it via MusicBrainz.
# This is EXTERNAL METADATA, never a SwarSense prediction — the raga_prediction
# field above is SwarSense's own (future) model output and must never be
# confused with information retrieved here.


class IdentificationStatusSchema(str, Enum):
    """
    identified:      One candidate clearly and reliably identified.
    ambiguous:       Multiple plausible candidates, none clearly dominant.
    not_identified:  Provider responded but found no reliable candidate.
    unavailable:     Provider not configured, failed, or timed out.
    """
    IDENTIFIED = "identified"
    AMBIGUOUS = "ambiguous"
    NOT_IDENTIFIED = "not_identified"
    UNAVAILABLE = "unavailable"


class IdentificationCandidateSchema(BaseModel):
    """A single candidate match returned by the identification provider."""
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    release_year: Optional[int] = None
    score: Optional[float] = Field(None, description="Provider confidence score, if available.")
    score_available: bool = Field(False, description="True if score is a genuine provider value.")
    acoustid: Optional[str] = None
    mbid: Optional[str] = Field(None, description="MusicBrainz Recording ID, if available.")
    provider_track_id: Optional[str] = None
    isrc: Optional[str] = None


class SongIdentificationResult(BaseModel):
    """
    Resolved song identification result. Source: external provider (AcoustID/
    ACRCloud/AudD), NOT a SwarSense model prediction.
    """
    status: IdentificationStatusSchema
    provider: Optional[str] = Field(None, description="Which provider produced this result.")
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    release_year: Optional[int] = None
    confidence: Optional[float] = Field(None, description="Confidence of the winning candidate.")
    confidence_available: bool = False
    candidates: list[IdentificationCandidateSchema] = Field(
        default_factory=list,
        description="All candidates considered — shown in full when status='ambiguous'.",
    )
    mbid: Optional[str] = None
    attribution: Optional[str] = Field(None, description="Required attribution string, if any.")
    error: Optional[str] = Field(None, description="Error description if status='unavailable'.")
    intelligence_note: Optional[str] = Field(
        None, description="Human-readable explanation of the status."
    )


class EnrichmentFactSchema(BaseModel):
    field: str
    value: str


class EnrichmentSourceSchema(BaseModel):
    """
    A source of enrichment data. The disclaimer MUST always be shown alongside
    these facts in the UI — they are external reference, never a SwarSense
    prediction. In particular, a raga name appearing here is NOT the output
    of SwarSense's raga classifier.
    """
    source_name: str
    url: Optional[str] = None
    disclaimer: str = "External reference — not generated by SwarSense."
    facts: list[EnrichmentFactSchema] = Field(default_factory=list)


class SongIntelligenceResult(BaseModel):
    """Combined Phase 2 result: identification + optional enrichment."""
    identification: SongIdentificationResult
    enrichment_sources: list[EnrichmentSourceSchema] = Field(default_factory=list)
    enrichment_error: Optional[str] = None


# ─── Full Analysis Result ──────────────────────────────────────────────────────


class AnalysisResult(BaseModel):
    """
    Complete analysis result for a job.
    Returned by GET /api/result/{job_id}.

    Phase 1 adds real DSP features. Phase 3/4 will add ML model results.
    Modules not yet implemented return their placeholder schemas with status='not_implemented'.
    """
    job_id: str
    status: JobStatus
    filename: str

    # ── Phase 1: DSP analysis results (populated when analysis completes) ──────
    audio: Optional[Phase1AudioMetadata] = Field(None, description="Audio file metadata.")
    loudness: Optional[Phase1LoudnessResult] = Field(None, description="Loudness measurements.")
    tempo: Optional[Phase1TempoResult] = Field(None, description="Tempo and beat detection.")
    pitch: Optional[Phase1PitchResult] = Field(None, description="Pitch / F0 contour.")
    mfcc: Optional[Phase1MFCCResult] = Field(None, description="MFCC coefficient statistics.")
    chroma: Optional[Phase1ChromaResult] = Field(None, description="Chroma (pitch class) energy.")
    spectral: Optional[Phase1SpectralResult] = Field(None, description="Spectral descriptor statistics.")
    hpss: Optional[Phase1HPSSResult] = Field(None, description="Harmonic/percussive energy ratios.")
    estimated_tonic: Optional[Phase1EstimatedTonic] = Field(
        None,
        description="Estimated tonic — heuristic only. See caveat field.",
    )
    segment_features: list[Phase1SegmentFeature] = Field(
        default_factory=list,
        description=(
            "Per-segment feature statistics for long recordings (>60s). "
            "Empty for short recordings. Does NOT replace global analysis."
        ),
    )
    visualizations: Optional[VisualizationLinks] = Field(
        None, description="URL paths for PNG visualization images."
    )

    # ── Phase 0: deprecated DSP fields (kept for backwards compat) ─────────────
    audio_metadata: Optional[AudioMetadata] = Field(None, description="[Deprecated — use audio]")
    tonic: Optional[TonicEstimate] = Field(None, description="[Deprecated — use estimated_tonic]")
    visualization: Optional[VisualizationData] = Field(None, description="[Deprecated — use visualizations]")

    # ── Phase 3: Pretrained models (not implemented yet) ──────────────────────
    source_separation: SourceSeparationResult = Field(default_factory=SourceSeparationResult)
    instrument_detection: InstrumentDetectionResult = Field(default_factory=InstrumentDetectionResult)

    # ── Phase 4: Trained model (not implemented yet) ──────────────────────────
    raga_prediction: RagaPrediction = Field(default_factory=RagaPrediction)

    # ── Phase 5: Static knowledge (not implemented yet) ───────────────────────
    raga_reference: Optional[RagaKnowledgeEntry] = Field(None, description="Reference raga musicology.")

    # ── Phase 2: Song Intelligence (optional, external metadata) ──────────────
    song_identification: Optional[SongIntelligenceResult] = Field(
        None,
        description=(
            "Optional external song identification + MusicBrainz enrichment. "
            "None if Song Intelligence did not run. Never a SwarSense prediction."
        ),
    )

    # ── Error tracking ────────────────────────────────────────────────────────
    module_errors: dict[str, str] = Field(
        default_factory=dict,
        description="Maps module name to error message if that module failed.",
    )
