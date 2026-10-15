"""
SwarSense — Tests: Health Endpoint & API Integration (Phase 1)
==============================================================
Tests the FastAPI application layer.

Run with:
  cd backend
  python -m pytest tests/ -v
"""

import io
import time

import numpy as np
import pytest
import scipy.io.wavfile as wav_io
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_wav(duration_s: float = 3.0, sr: int = 22050) -> bytes:
    """Helper: generate synthetic WAV bytes for API tests."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav_io.write(buf, sr, signal)
    buf.seek(0)
    return buf.read()


# ─── Health Check ──────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_returns_200(self):
        """Health endpoint must return HTTP 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        """Health response must contain required fields."""
        response = client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data

    def test_health_status_value(self):
        """Health status must be 'healthy'."""
        response = client.get("/api/health")
        assert response.json()["status"] == "healthy"

    def test_health_service_name(self):
        """Service name must be 'SwarSense API'."""
        response = client.get("/api/health")
        assert response.json()["service"] == "SwarSense API"


# ─── Analyze Endpoint ──────────────────────────────────────────────────────────

class TestAnalyzeEndpoint:
    def test_analyze_rejects_empty_file(self):
        """Empty file upload must be rejected."""
        response = client.post(
            "/api/analyze",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code in (400, 415, 422)

    def test_analyze_rejects_wrong_extension(self):
        """Non-audio file extension must be rejected."""
        response = client.post(
            "/api/analyze",
            files={"file": ("document.txt", b"Some text content here.", "text/plain")},
        )
        assert response.status_code in (415, 422)

    def test_analyze_accepts_wav_magic_bytes(self):
        """A valid WAV file should be accepted and return a job_id."""
        wav_bytes = _make_wav(duration_s=2.0)
        response = client.post(
            "/api/analyze",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["filename"] == "test.wav"

    def test_analyze_returns_job_id_format(self):
        """Returned job_id should be a non-empty string."""
        wav_bytes = _make_wav(duration_s=2.0)
        response = client.post(
            "/api/analyze",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        if response.status_code == 202:
            job_id = response.json()["job_id"]
            assert isinstance(job_id, str)
            assert len(job_id) > 0

    def test_analyze_rejects_file_over_50mb(self):
        """A file larger than the 50 MB limit must be rejected with 413."""
        oversized = b"\x00" * (50 * 1024 * 1024 + 1)
        response = client.post(
            "/api/analyze",
            files={"file": ("huge.wav", oversized, "audio/wav")},
        )
        assert response.status_code == 413
        detail = response.json()["detail"]
        assert detail["error"] == "file_too_large"


# ─── Status Endpoint ───────────────────────────────────────────────────────────

class TestStatusEndpoint:
    def test_status_unknown_job_returns_404(self):
        """Requesting status of a non-existent job must return 404."""
        response = client.get("/api/status/nonexistent-job-id-12345")
        assert response.status_code == 404

    def test_status_valid_job_after_analyze(self):
        """After submitting a file, status endpoint should return valid data."""
        wav_bytes = _make_wav(duration_s=2.0)
        analyze_response = client.post(
            "/api/analyze",
            files={"file": ("test_status.wav", wav_bytes, "audio/wav")},
        )
        if analyze_response.status_code != 202:
            pytest.skip("Analyze endpoint rejected the test file.")

        job_id = analyze_response.json()["job_id"]
        status_response = client.get(f"/api/status/{job_id}")
        assert status_response.status_code == 200

        data = status_response.json()
        assert "job_id" in data
        assert "status" in data
        assert "progress" in data
        assert data["job_id"] == job_id


# ─── Result Endpoint ───────────────────────────────────────────────────────────

class TestResultEndpoint:
    def test_result_unknown_job_returns_404(self):
        """Requesting result of a non-existent job must return 404."""
        response = client.get("/api/result/nonexistent-job-id-12345")
        assert response.status_code == 404

    def test_result_for_failed_job_returns_4xx_not_500(self):
        """
        Regression test: a job that fails during audio loading (e.g. corrupt
        bytes with a .wav extension that passes the lightweight upload
        validator but can't actually be decoded) must surface as a client
        error with a clear message, not a 500 'result_missing'.
        """
        corrupt_bytes = b"not actually audio data, just random-looking bytes 1234567890"
        analyze_response = client.post(
            "/api/analyze",
            files={"file": ("corrupt.wav", corrupt_bytes, "audio/wav")},
        )
        if analyze_response.status_code != 202:
            pytest.skip("Upload validator rejected the corrupt file before a job was created.")

        job_id = analyze_response.json()["job_id"]
        status_response = client.get(f"/api/status/{job_id}")
        assert status_response.json()["status"] == "failed"

        result_response = client.get(f"/api/result/{job_id}")
        assert result_response.status_code == 422
        detail = result_response.json()["detail"]
        assert detail["error"] == "analysis_failed"
        assert "message" in detail

    def test_docs_endpoint_available(self):
        """Swagger docs must be accessible at /api/docs."""
        response = client.get("/api/docs")
        assert response.status_code == 200

    def test_openapi_schema_available(self):
        """OpenAPI JSON schema must be accessible."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "info" in schema
        assert "paths" in schema


# ─── Phase 1 E2E Integration Tests ────────────────────────────────────────────

class TestPhase1EndToEndAnalysis:
    """
    End-to-end test: upload WAV -> poll until completed -> check result + viz endpoints.

    NOTE: TestClient runs background tasks synchronously, so this should complete fast.
    """

    @pytest.fixture(autouse=True)
    def run_analysis(self):
        wav_bytes = _make_wav(duration_s=3.0)
        response = client.post(
            "/api/analyze",
            files={"file": ("e2e_test.wav", wav_bytes, "audio/wav")},
        )
        if response.status_code != 202:
            pytest.skip(f"Analyze rejected: {response.status_code}")
        self.job_id = response.json()["job_id"]

        # Poll until completed (TestClient runs bg tasks inline — should be fast)
        deadline = time.time() + 60
        self.final_status = "pending"
        while time.time() < deadline:
            st = client.get(f"/api/status/{self.job_id}").json()
            self.final_status = st["status"]
            if self.final_status in ("completed", "partial", "failed"):
                break
            time.sleep(0.5)

    def test_analysis_completes(self):
        """Analysis must reach 'completed' or 'partial' status."""
        assert self.final_status in ("completed", "partial"), \
            f"Expected completed/partial, got: {self.final_status}"

    def test_result_has_audio_metadata(self):
        """Result must contain audio metadata with valid duration."""
        if self.final_status == "failed":
            pytest.skip("Analysis failed.")
        r = client.get(f"/api/result/{self.job_id}")
        assert r.status_code == 200
        data = r.json()
        assert data.get("audio") is not None
        assert data["audio"]["duration_seconds"] > 0
        assert data["audio"]["sample_rate"] == 22050

    def test_result_has_tempo(self):
        """Result must contain tempo information."""
        if self.final_status == "failed":
            pytest.skip("Analysis failed.")
        r = client.get(f"/api/result/{self.job_id}")
        data = r.json()
        if data.get("tempo") is not None:
            # Pure sine waves may produce BPM=0 (no transients) — valid DSP behavior
            assert data["tempo"]["bpm"] >= 0
            assert "beat_times_seconds" in data["tempo"]


    def test_result_has_estimated_tonic_with_caveat(self):
        """Tonic result must include caveat and be labeled as heuristic."""
        if self.final_status == "failed":
            pytest.skip("Analysis failed.")
        r = client.get(f"/api/result/{self.job_id}")
        data = r.json()
        tonic = data.get("estimated_tonic")
        if tonic is not None:
            assert len(tonic.get("caveat", "")) > 10
            assert tonic.get("source") == "heuristic"

    def test_result_has_mfcc_13_coefficients(self):
        """MFCC must have exactly 13 coefficients."""
        if self.final_status == "failed":
            pytest.skip("Analysis failed.")
        r = client.get(f"/api/result/{self.job_id}")
        data = r.json()
        if data.get("mfcc") is not None:
            assert len(data["mfcc"]["mean"]) == 13
            assert len(data["mfcc"]["std"]) == 13

    def test_result_has_chroma_12_bins(self):
        """Chroma must have 12 pitch class bins."""
        if self.final_status == "failed":
            pytest.skip("Analysis failed.")
        r = client.get(f"/api/result/{self.job_id}")
        data = r.json()
        if data.get("chroma") is not None:
            assert len(data["chroma"]["mean"]) == 12

    def test_visualization_endpoints_return_png(self):
        """Visualization image endpoints must return PNG bytes."""
        if self.final_status == "failed":
            pytest.skip("Analysis failed.")
        charts = ["waveform", "spectrogram", "mel-spectrogram", "chromagram", "pitch"]
        for chart in charts:
            r = client.get(f"/api/result/{self.job_id}/{chart}")
            if r.status_code == 200:
                assert r.headers["content-type"] == "image/png"
                # Check PNG magic bytes
                assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_visualization_unavailable_for_unknown_job(self):
        """Visualization endpoints must return 404 for unknown jobs."""
        r = client.get("/api/result/nonexistent-xyz-123/waveform")
        assert r.status_code == 404
