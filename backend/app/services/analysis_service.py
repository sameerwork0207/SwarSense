"""
SwarSense — Analysis Service (Phase 1)
=======================================
Orchestrates the full Phase 1 audio DSP analysis pipeline.

Pipeline stages (with progress reporting):
  Stage 1 [0→15%]:  Load & validate audio
  Stage 2 [15→25%]: Preprocess (normalize, windowing)
  Stage 3 [25→40%]: Tempo & beat detection (global)
  Stage 4 [40→60%]: Pitch estimation (PYIN)
  Stage 5 [60→75%]: Spectral features (MFCC, Chroma, Spectral, HPSS, Loudness)
  Stage 6 [75→85%]: Tonic estimation (heuristic)
  Stage 7 [85→97%]: Render visualizations
  Stage 8 [97→100%]: Assemble final result

Design:
  - Each module is wrapped in try/except. A single module failure does NOT
    crash the whole pipeline — the error is recorded in module_errors.
  - Progress is updated at each stage boundary.
  - Visualization PNG bytes are stored in the job store under 'visualizations'.
  - The main AnalysisResult JSON does NOT contain raw matrix data.

Global vs. segment analysis:
  - All feature extractors operate on the FULL audio signal (global analysis).
  - For recordings >60s, segment features are ADDITIONALLY extracted per window.
  - Visualizations are generated from the global signal (full recording).
"""

from __future__ import annotations

import time
from typing import Optional

from app.api.schemas import (
    AnalysisResult,
    EnrichmentFactSchema,
    EnrichmentSourceSchema,
    IdentificationCandidateSchema,
    InstrumentDetectionResult,
    JobStatus,
    Phase1AudioMetadata,
    Phase1ChromaResult,
    Phase1EstimatedTonic,
    Phase1HPSSResult,
    Phase1LoudnessResult,
    Phase1MFCCResult,
    Phase1PitchResult,
    Phase1SegmentFeature,
    Phase1SpectralResult,
    Phase1TempoResult,
    RagaPrediction,
    SongIdentificationResult,
    SongIntelligenceResult,
    SourceSeparationResult,
    VisualizationLinks,
)
from app.services.job_service import JobService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisService:
    """
    Orchestrates all analysis modules for a given audio job.

    Each module corresponds to one Phase 1 DSP component.
    All methods fail gracefully — errors are recorded without crashing the pipeline.
    """

    def __init__(self):
        # Module implementations injected as DSP pipeline is built.
        pass

    def _run_song_intelligence(
        self,
        file_content: bytes,
        job_id: str,
        filename: str,
    ) -> Optional[SongIntelligenceResult]:
        """
        Run the optional Phase 2 Song Intelligence pipeline and translate its
        result into the API schema.

        Failure isolation: ANY exception here is caught and reported as
        status='unavailable' — it is never raised, and never recorded in
        module_errors, so it cannot flip the job's overall status or affect
        DSP results in any way.
        """
        try:
            from app.utils.config import get_settings
            if not get_settings().SONG_INTELLIGENCE_ENABLED:
                return None
        except Exception:
            pass  # Settings unavailable — fall through and attempt with defaults.

        try:
            from app.intelligence.service import SongIntelligenceService

            service = SongIntelligenceService.from_settings()
            identification, enrichment = service.run_sync(
                audio_bytes=file_content, job_id=job_id, filename=filename
            )

            candidates = [
                IdentificationCandidateSchema(
                    title=c.title,
                    artist=c.artist,
                    album=c.album,
                    release_year=c.release_year,
                    score=c.score,
                    score_available=c.score_available,
                    acoustid=c.acoustid,
                    mbid=c.mbid,
                    provider_track_id=c.provider_track_id,
                    isrc=c.isrc,
                )
                for c in identification.candidates
            ]

            identification_schema = SongIdentificationResult(
                status=identification.status.value,
                provider=identification.provider.value if identification.provider else None,
                title=identification.title,
                artist=identification.artist,
                album=identification.album,
                release_year=identification.release_year,
                confidence=identification.confidence,
                confidence_available=identification.confidence_available,
                candidates=candidates,
                mbid=identification.mbid,
                attribution=identification.attribution,
                error=identification.error,
                intelligence_note=identification.intelligence_note,
            )

            enrichment_sources = []
            enrichment_error = None
            if enrichment is not None:
                enrichment_error = enrichment.error
                enrichment_sources = [
                    EnrichmentSourceSchema(
                        source_name=s.source_name,
                        url=s.url,
                        disclaimer=s.disclaimer,
                        facts=[
                            EnrichmentFactSchema(field=f.field, value=f.value)
                            for f in s.facts
                        ],
                    )
                    for s in enrichment.sources
                ]

            logger.info(
                f"[{job_id}] Song Intelligence result: status={identification.status.value}"
            )
            return SongIntelligenceResult(
                identification=identification_schema,
                enrichment_sources=enrichment_sources,
                enrichment_error=enrichment_error,
            )
        except Exception as exc:
            logger.error(f"[{job_id}] Song Intelligence stage crashed (isolated): {exc}", exc_info=True)
            return SongIntelligenceResult(
                identification=SongIdentificationResult(
                    status="unavailable",
                    error=f"Song Intelligence failed unexpectedly: {exc}",
                    intelligence_note="Song Intelligence failed. DSP analysis is complete and unaffected.",
                ),
                enrichment_sources=[],
                enrichment_error=None,
            )

    def run_analysis(
        self,
        job_id: str,
        file_content: bytes,
        filename: str,
    ) -> None:
        """
        Entry point called by the FastAPI background task.

        Runs the full Phase 1 DSP pipeline sequentially.
        Updates job progress at each stage.
        Records module-level errors in module_errors.
        """
        job_svc = JobService()  # Shared via class-level _jobs dict
        module_errors: dict[str, str] = {}

        logger.info(f"[{job_id}] Analysis pipeline starting for '{filename}'.")
        job_svc.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            progress=2,
            message="Starting analysis...",
        )

        # ── Stage 1: Load audio ────────────────────────────────────────────────
        job_svc.update_job(job_id, progress=5, message="Loading and validating audio...")
        loaded_audio = None
        try:
            from app.audio.loader import AudioLoader, AudioLoadError, AudioDurationError
            loader = AudioLoader()
            loaded_audio = loader.load_bytes(file_content, filename)
            logger.info(f"[{job_id}] Audio loaded: {loaded_audio.duration_seconds:.2f}s")
            job_svc.update_job(job_id, progress=15, message="Audio loaded successfully.")
        except Exception as exc:
            module_errors["audio_loader"] = str(exc)
            logger.error(f"[{job_id}] Audio loading failed: {exc}")
            job_svc.update_job(
                job_id,
                status=JobStatus.FAILED,
                progress=15,
                message=f"Audio loading failed: {exc}",
            )
            return  # Cannot proceed without audio

        samples = loaded_audio.samples
        sr = loaded_audio.sample_rate
        duration_s = loaded_audio.duration_seconds

        # ── Stage 2: Preprocess ────────────────────────────────────────────────
        job_svc.update_job(job_id, progress=18, message="Preprocessing audio...")
        preprocessed = None
        try:
            from app.audio.preprocessing import preprocess
            preprocessed = preprocess(
                samples=samples,
                sample_rate=sr,
                original_sample_rate=loaded_audio.original_sr,
                original_channels=loaded_audio.original_channels,
                normalize=True,
            )
            samples = preprocessed.samples  # Use normalized samples for features
            logger.info(
                f"[{job_id}] Preprocessed: normalized={preprocessed.normalized}, "
                f"windows={len(preprocessed.windows)}"
            )
            job_svc.update_job(job_id, progress=25, message="Audio preprocessed.")
        except Exception as exc:
            module_errors["preprocessing"] = str(exc)
            logger.warning(f"[{job_id}] Preprocessing failed (using raw audio): {exc}")
            # Create a minimal PreprocessedAudio with unnormalized samples
            from app.audio.preprocessing import PreprocessedAudio
            preprocessed = PreprocessedAudio(
                samples=samples,
                sample_rate=sr,
                duration_s=duration_s,
                windows=[],
                normalized=False,
                original_sample_rate=loaded_audio.original_sr,
                original_channels=loaded_audio.original_channels,
            )

        # ── Stage 3: Loudness ──────────────────────────────────────────────────
        job_svc.update_job(job_id, progress=27, message="Measuring loudness...")
        loudness_result = None
        try:
            from app.audio.loudness import measure_loudness
            loudness_raw = measure_loudness(preprocessed.samples, sr)
            loudness_result = Phase1LoudnessResult(
                lufs=loudness_raw.lufs,
                rms_db=loudness_raw.rms_db,
                rms_linear=loudness_raw.rms_linear,
                method=loudness_raw.method,
            )
            logger.info(f"[{job_id}] Loudness: LUFS={loudness_raw.lufs}, RMS={loudness_raw.rms_db:.1f}dB")
        except Exception as exc:
            module_errors["loudness"] = str(exc)
            logger.error(f"[{job_id}] Loudness measurement failed: {exc}")

        # ── Stage 4: Tempo & Beat Detection ───────────────────────────────────
        job_svc.update_job(job_id, progress=30, message="Detecting tempo and beats (global analysis)...")
        tempo_result = None
        try:
            from app.audio.tempo import estimate_tempo
            tempo_raw = estimate_tempo(preprocessed.samples, sr)
            tempo_result = Phase1TempoResult(
                bpm=tempo_raw.bpm,
                beat_times_seconds=tempo_raw.beat_times_s,
                n_beats=tempo_raw.n_beats,
                analysis_scope=tempo_raw.analysis_scope,
                method=tempo_raw.method,
                caveat=(
                    "BPM estimation uses librosa's onset-strength beat tracker. "
                    "May be unreliable for free-rhythm (alap) or slow-tempo (vilambit) Indian classical music."
                ),
            )
            logger.info(f"[{job_id}] Tempo: {tempo_raw.bpm} BPM, {tempo_raw.n_beats} beats.")
            job_svc.update_job(job_id, progress=40, message=f"Detected {tempo_raw.bpm:.1f} BPM.")
        except Exception as exc:
            module_errors["tempo"] = str(exc)
            logger.error(f"[{job_id}] Tempo estimation failed: {exc}")
            job_svc.update_job(job_id, progress=40, message="Tempo estimation failed — continuing.")

        # ── Stage 5: Pitch Estimation ──────────────────────────────────────────
        job_svc.update_job(job_id, progress=42, message="Estimating pitch contour (PYIN)...")
        pitch_result = None
        try:
            from app.audio.pitch import estimate_pitch
            pitch_raw = estimate_pitch(preprocessed.samples, sr)
            pitch_result = Phase1PitchResult(
                f0_hz=pitch_raw.f0_hz,
                times_seconds=pitch_raw.times_s,
                voiced_flag=pitch_raw.voiced_flag,
                voiced_ratio=pitch_raw.voiced_ratio,
                mean_f0_hz=pitch_raw.mean_f0_hz,
                std_f0_hz=pitch_raw.std_f0_hz,
                min_f0_hz=pitch_raw.min_f0_hz,
                max_f0_hz=pitch_raw.max_f0_hz,
                dominant_midi=pitch_raw.dominant_midi,
                dominant_note=pitch_raw.dominant_note,
                analysis_scope=pitch_raw.analysis_scope,
                method=pitch_raw.method,
                caveat=(
                    "F0 estimation uses PYIN (probabilistic YIN). "
                    "This estimates the dominant pitch, not the musical tonic or swara. "
                    "Microtonal inflections (gamaks, meends) are not resolved."
                ),
            )
            logger.info(
                f"[{job_id}] Pitch: voiced_ratio={pitch_raw.voiced_ratio:.2%}, "
                f"dominant_note={pitch_raw.dominant_note}"
            )
            job_svc.update_job(
                job_id, progress=60,
                message=f"Pitch estimated. Voiced ratio: {pitch_raw.voiced_ratio:.0%}."
            )
        except Exception as exc:
            module_errors["pitch"] = str(exc)
            logger.error(f"[{job_id}] Pitch estimation failed: {exc}")
            job_svc.update_job(job_id, progress=60, message="Pitch estimation failed — continuing.")

        # ── Stage 6: DSP Features (MFCC, Chroma, Spectral, HPSS) ─────────────
        job_svc.update_job(job_id, progress=62, message="Extracting spectral features (MFCC, Chroma, Spectral, HPSS)...")
        mfcc_result = None
        chroma_result = None
        spectral_result = None
        hpss_result = None
        features_raw = None
        try:
            from app.audio.features import extract_features
            features_raw = extract_features(preprocessed)

            mfcc_result = Phase1MFCCResult(
                n_coefficients=features_raw.mfcc.n_coefficients,
                mean=features_raw.mfcc.mean,
                std=features_raw.mfcc.std,
                analysis_scope=features_raw.analysis_scope,
            )
            chroma_result = Phase1ChromaResult(
                pitch_classes=features_raw.chroma.pitch_classes,
                mean=features_raw.chroma.mean,
                std=features_raw.chroma.std,
                analysis_scope=features_raw.analysis_scope,
            )
            spectral_result = Phase1SpectralResult(
                centroid_mean_hz=features_raw.spectral.centroid_mean_hz,
                centroid_std_hz=features_raw.spectral.centroid_std_hz,
                bandwidth_mean_hz=features_raw.spectral.bandwidth_mean_hz,
                rolloff_mean_hz=features_raw.spectral.rolloff_mean_hz,
                zero_crossing_rate_mean=features_raw.spectral.zero_crossing_rate_mean,
                zero_crossing_rate_std=features_raw.spectral.zero_crossing_rate_std,
            )
            hpss_result = Phase1HPSSResult(
                harmonic_energy_ratio=features_raw.hpss.harmonic_energy_ratio,
                percussive_energy_ratio=features_raw.hpss.percussive_energy_ratio,
                method=features_raw.hpss.method,
                note=(
                    "HPSS provides a rough harmonic/percussive split based on median filtering. "
                    "Ratios may not sum to 1.0 due to residual energy."
                ),
            )
            logger.info(f"[{job_id}] DSP features extracted. Scope: {features_raw.analysis_scope}")
            job_svc.update_job(job_id, progress=75, message="Spectral features extracted.")
        except Exception as exc:
            module_errors["features"] = str(exc)
            logger.error(f"[{job_id}] Feature extraction failed: {exc}")
            job_svc.update_job(job_id, progress=75, message="Feature extraction partially failed — continuing.")

        # ── Stage 7: Tonic Estimation ──────────────────────────────────────────
        job_svc.update_job(job_id, progress=77, message="Estimating tonic (heuristic)...")
        tonic_result = None
        try:
            from app.audio.tonic import estimate_tonic
            tonic_raw = estimate_tonic(preprocessed.samples, sr)
            tonic_result = Phase1EstimatedTonic(
                estimated_note=tonic_raw.estimated_note,
                chroma_energy=tonic_raw.chroma_energy,
                dominance_ratio=tonic_raw.dominance_ratio,
                method=tonic_raw.method,
                caveat=tonic_raw.caveat,
                analysis_scope=tonic_raw.analysis_scope,
            )
            logger.info(f"[{job_id}] Estimated tonic: {tonic_raw.estimated_note}")
        except Exception as exc:
            module_errors["tonic"] = str(exc)
            logger.error(f"[{job_id}] Tonic estimation failed: {exc}")

        # ── Stage 8: Visualizations ────────────────────────────────────────────
        job_svc.update_job(job_id, progress=82, message="Rendering visualizations...")
        viz_links = None
        viz_bundle = None
        try:
            from app.audio.visualization import generate_all_visualizations

            # Determine visualization scope for chart subtitles
            viz_scope = "global"
            if duration_s > 60:
                viz_scope = f"global ({duration_s:.0f}s full recording)"

            # Beat times for overlays (empty list if tempo failed)
            beat_times = tempo_result.beat_times_seconds if tempo_result else []

            # F0 for pitch chart (empty if pitch failed)
            f0_hz_list = pitch_result.f0_hz if pitch_result else []
            f0_times_list = pitch_result.times_seconds if pitch_result else []
            voiced_list = pitch_result.voiced_flag if pitch_result else []

            # Need raw matrices from features for spectrogram/mel/chroma charts
            stft_db = features_raw.stft_db if features_raw is not None else None
            mel_db = features_raw.mel_spectrogram_db if features_raw is not None else None
            chroma_matrix = features_raw.chroma_matrix if features_raw is not None else None

            if stft_db is not None and mel_db is not None and chroma_matrix is not None:
                from app.audio.features import HOP_LENGTH
                viz_bundle = generate_all_visualizations(
                    samples=preprocessed.samples,
                    sample_rate=sr,
                    duration_s=duration_s,
                    stft_db=stft_db,
                    mel_db=mel_db,
                    chroma_matrix=chroma_matrix,
                    hop_length=HOP_LENGTH,
                    beat_times_s=beat_times,
                    f0_hz=f0_hz_list,
                    f0_times_s=f0_times_list,
                    voiced_flag=voiced_list,
                    viz_scope=viz_scope,
                )
                # Record visualization errors
                module_errors.update({
                    f"viz_{k}": v for k, v in viz_bundle.errors.items()
                })

                # Build URL links
                base_url = f"/api/result/{job_id}"
                viz_links = VisualizationLinks(
                    waveform=f"{base_url}/waveform" if viz_bundle.waveform else None,
                    spectrogram=f"{base_url}/spectrogram" if viz_bundle.spectrogram else None,
                    mel_spectrogram=f"{base_url}/mel-spectrogram" if viz_bundle.mel_spectrogram else None,
                    chromagram=f"{base_url}/chromagram" if viz_bundle.chromagram else None,
                    pitch=f"{base_url}/pitch" if viz_bundle.pitch else None,
                )
                logger.info(f"[{job_id}] Visualizations generated.")
            else:
                logger.warning(f"[{job_id}] Cannot generate spectral charts — feature extraction failed.")

            job_svc.update_job(job_id, progress=95, message="Visualizations rendered.")
        except Exception as exc:
            module_errors["visualization"] = str(exc)
            logger.error(f"[{job_id}] Visualization failed: {exc}")

        # ── Stage 9: Song Intelligence (Phase 2, optional) ─────────────────────
        # Runs only if enabled. Any failure here is isolated — it is reported
        # inside song_identification as status='unavailable' and never affects
        # DSP results or the overall job status (never added to module_errors).
        job_svc.update_job(job_id, progress=96, message="Identifying recording (optional)...")
        song_identification_result = self._run_song_intelligence(file_content, job_id, filename)

        # ── Stage 10: Assemble Result ──────────────────────────────────────────
        job_svc.update_job(job_id, progress=97, message="Assembling final result...")

        # Build audio metadata
        audio_meta = Phase1AudioMetadata(
            duration_seconds=round(duration_s, 3),
            sample_rate=sr,
            original_sample_rate=loaded_audio.original_sr,
            channels=loaded_audio.original_channels,
            n_samples=loaded_audio.n_samples,
            file_format=loaded_audio.detected_format,
            normalized=preprocessed.normalized,
        )

        # Build segment features
        segment_feats: list[Phase1SegmentFeature] = []
        if features_raw and features_raw.segment_features:
            for sf in features_raw.segment_features:
                segment_feats.append(Phase1SegmentFeature(
                    window_index=sf.window_index,
                    start_seconds=sf.start_s,
                    end_seconds=sf.end_s,
                    mfcc_mean=sf.mfcc_mean,
                    mfcc_std=sf.mfcc_std,
                    chroma_mean=sf.chroma_mean,
                ))

        final_status = JobStatus.COMPLETED if not module_errors else JobStatus.PARTIAL

        result = AnalysisResult(
            job_id=job_id,
            status=final_status,
            filename=filename,
            audio=audio_meta,
            loudness=loudness_result,
            tempo=tempo_result,
            pitch=pitch_result,
            mfcc=mfcc_result,
            chroma=chroma_result,
            spectral=spectral_result,
            hpss=hpss_result,
            estimated_tonic=tonic_result,
            segment_features=segment_feats,
            visualizations=viz_links,
            # Phase 3/4/5 — stubs
            source_separation=SourceSeparationResult(),
            instrument_detection=InstrumentDetectionResult(),
            raga_prediction=RagaPrediction(),
            raga_reference=None,
            song_identification=song_identification_result,
            module_errors=module_errors,
        )

        # Store visualization PNG bytes separately (not in the main JSON)
        job_svc.update_job(
            job_id,
            status=final_status,
            progress=100,
            message="Analysis complete." if not module_errors else "Analysis complete with some module errors.",
            result=result,
            visualizations=viz_bundle,  # stored as extra field
        )

        logger.info(
            f"[{job_id}] Pipeline finished. "
            f"Status: {final_status}. "
            f"Errors: {list(module_errors.keys()) if module_errors else 'none'}."
        )
