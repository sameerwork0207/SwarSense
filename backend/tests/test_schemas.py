"""
SwarSense — Tests: Pydantic Schema Validation
==============================================
Verifies that schemas correctly validate and reject data.
"""

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AnalysisResult,
    AnalysisSource,
    AudioMetadata,
    ErrorResponse,
    JobStatus,
    RagaPrediction,
    SpectralFeatures,
    TonicEstimate,
)


class TestJobStatus:
    def test_valid_statuses(self):
        for status in ["pending", "processing", "completed", "failed", "partial"]:
            assert JobStatus(status) is not None

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            JobStatus("unknown_status")


class TestAnalysisSource:
    def test_all_sources_defined(self):
        """All required source types must be defined."""
        required = ["dsp", "pretrained", "trained_model", "heuristic", "static"]
        for src in required:
            assert AnalysisSource(src) is not None


class TestAudioMetadata:
    def test_valid_audio_metadata(self):
        meta = AudioMetadata(
            duration_seconds=120.5,
            sample_rate=22050,
            channels=2,
            format="wav",
        )
        assert meta.duration_seconds == 120.5
        assert meta.sample_rate == 22050
        assert meta.source == AnalysisSource.DSP

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            AudioMetadata(sample_rate=22050)  # missing duration_seconds, format


class TestTonicEstimate:
    def test_tonic_has_correct_source(self):
        tonic = TonicEstimate(estimated_tonic="C#")
        assert tonic.source == AnalysisSource.HEURISTIC

    def test_tonic_can_be_none(self):
        tonic = TonicEstimate()
        assert tonic.estimated_tonic is None


class TestRagaPrediction:
    def test_default_status_is_not_implemented(self):
        raga = RagaPrediction()
        assert raga.status == "not_implemented"

    def test_raga_source_is_trained_model(self):
        raga = RagaPrediction()
        assert raga.source == AnalysisSource.TRAINED_MODEL

    def test_confidence_bounds(self):
        """Confidence must be between 0 and 1."""
        with pytest.raises(ValidationError):
            RagaPrediction(confidence=1.5)
        with pytest.raises(ValidationError):
            RagaPrediction(confidence=-0.1)

    def test_valid_confidence(self):
        raga = RagaPrediction(confidence=0.82, predicted_raga="Yaman")
        assert raga.confidence == 0.82


class TestErrorResponse:
    def test_valid_error(self):
        err = ErrorResponse(error="file_too_large", message="File exceeds 50 MB.")
        assert err.error == "file_too_large"
        assert err.detail is None

    def test_error_with_detail(self):
        err = ErrorResponse(
            error="decode_failed",
            message="Could not decode audio.",
            detail="libsndfile returned FLAC error.",
        )
        assert err.detail is not None


class TestAnalysisResult:
    def test_minimal_result(self):
        """AnalysisResult with only required fields should be valid."""
        result = AnalysisResult(
            job_id="test-123",
            status=JobStatus.COMPLETED,
            filename="test.wav",
        )
        assert result.audio_metadata is None
        assert result.raga_prediction.status == "not_implemented"

    def test_partial_result_allowed(self):
        """PARTIAL status is valid when some modules failed."""
        result = AnalysisResult(
            job_id="test-456",
            status=JobStatus.PARTIAL,
            filename="test.wav",
            module_errors={"raga_model": "Model not loaded."},
        )
        assert result.status == JobStatus.PARTIAL
        assert "raga_model" in result.module_errors
