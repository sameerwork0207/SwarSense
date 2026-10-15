"""
SwarSense — Song Intelligence Tests (Phase 2)
===============================================
Covers:
  - CandidateResolver: all resolution branches (0/1/multi candidates,
    threshold, dominance margin, provider error/unavailable).
  - ProviderFactory: missing credentials -> NullProvider -> unavailable.
  - SongIntelligenceService: timeout handling, exception isolation
    (a provider that raises or hangs never crashes the service).
  - AnalysisService integration: Song Intelligence failures never affect
    DSP results or flip the job status to anything but completed.

All provider calls are mocked — no real network access or API keys needed.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.intelligence.base import (
    IdentificationCandidate,
    IdentificationStatus,
    ProviderName,
    RawIdentificationResult,
    SongIdentifier,
)
from app.intelligence.resolver import CandidateResolver
from app.intelligence.service import ProviderFactory, SongIntelligenceService


# ─── CandidateResolver ──────────────────────────────────────────────────────


class TestCandidateResolver:
    def setup_method(self):
        self.resolver = CandidateResolver(threshold=0.75, dominance_margin=0.15)

    def test_zero_candidates_is_not_identified(self):
        raw = RawIdentificationResult(provider=ProviderName.ACOUSTID, status="no_match", candidates=[])
        result = self.resolver.resolve(raw)
        assert result.status == IdentificationStatus.NOT_IDENTIFIED
        assert result.candidates == []

    def test_single_high_confidence_candidate_is_identified(self):
        candidate = IdentificationCandidate(title="Raga Yaman Alap", artist="Some Artist", score=0.9, score_available=True)
        raw = RawIdentificationResult(provider=ProviderName.ACOUSTID, status="ok", candidates=[candidate])
        result = self.resolver.resolve(raw)
        assert result.status == IdentificationStatus.IDENTIFIED
        assert result.title == "Raga Yaman Alap"
        assert result.confidence == 0.9

    def test_single_low_confidence_candidate_is_not_identified(self):
        candidate = IdentificationCandidate(title="Maybe This Song", score=0.4, score_available=True)
        raw = RawIdentificationResult(provider=ProviderName.ACOUSTID, status="ok", candidates=[candidate])
        result = self.resolver.resolve(raw)
        assert result.status == IdentificationStatus.NOT_IDENTIFIED
        # The candidate must still be surfaced for reference, but not auto-selected.
        assert result.title is None

    def test_single_candidate_without_score_is_not_identified(self):
        """A single candidate with no score must never be auto-identified."""
        candidate = IdentificationCandidate(title="AudD Result", artist="Someone", score=None, score_available=False)
        raw = RawIdentificationResult(provider=ProviderName.AUDD, status="ok", candidates=[candidate])
        result = self.resolver.resolve(raw)
        assert result.status == IdentificationStatus.NOT_IDENTIFIED

    def test_multiple_candidates_clearly_dominant_is_identified(self):
        top = IdentificationCandidate(title="Winner", score=0.9, score_available=True)
        second = IdentificationCandidate(title="Loser", score=0.5, score_available=True)
        raw = RawIdentificationResult(provider=ProviderName.ACOUSTID, status="ok", candidates=[second, top])
        result = self.resolver.resolve(raw)
        assert result.status == IdentificationStatus.IDENTIFIED
        assert result.title == "Winner"

    def test_multiple_candidates_close_scores_is_ambiguous(self):
        """Two candidates both above threshold but within the dominance margin."""
        c1 = IdentificationCandidate(title="A", score=0.82, score_available=True)
        c2 = IdentificationCandidate(title="B", score=0.79, score_available=True)
        raw = RawIdentificationResult(provider=ProviderName.ACOUSTID, status="ok", candidates=[c1, c2])
        result = self.resolver.resolve(raw)
        assert result.status == IdentificationStatus.AMBIGUOUS
        assert len(result.candidates) == 2
        assert result.title is None  # Nothing auto-selected.

    def test_multiple_candidates_below_threshold_is_ambiguous(self):
        c1 = IdentificationCandidate(title="A", score=0.5, score_available=True)
        c2 = IdentificationCandidate(title="B", score=0.3, score_available=True)
        raw = RawIdentificationResult(provider=ProviderName.ACOUSTID, status="ok", candidates=[c1, c2])
        result = self.resolver.resolve(raw)
        assert result.status == IdentificationStatus.AMBIGUOUS

    def test_provider_error_is_unavailable(self):
        raw = RawIdentificationResult(
            provider=ProviderName.ACOUSTID, status="error", candidates=[], error="network broke"
        )
        result = self.resolver.resolve(raw)
        assert result.status == IdentificationStatus.UNAVAILABLE
        assert result.error == "network broke"

    def test_provider_unavailable_status_is_unavailable(self):
        raw = RawIdentificationResult(
            provider=ProviderName.ACOUSTID, status="unavailable", candidates=[], error="fpcalc missing"
        )
        result = self.resolver.resolve(raw)
        assert result.status == IdentificationStatus.UNAVAILABLE

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            CandidateResolver(threshold=0.0)
        with pytest.raises(ValueError):
            CandidateResolver(threshold=1.5)

    def test_invalid_dominance_margin_raises(self):
        with pytest.raises(ValueError):
            CandidateResolver(dominance_margin=-0.1)
        with pytest.raises(ValueError):
            CandidateResolver(dominance_margin=1.0)


# ─── ProviderFactory ────────────────────────────────────────────────────────


class TestProviderFactory:
    def test_acoustid_without_api_key_returns_null_provider(self):
        provider = ProviderFactory.get_provider(provider_name="acoustid", acoustid_api_key="")
        assert provider.provider_name == ProviderName.NULL
        assert provider.is_available is False

    def test_acrcloud_with_incomplete_credentials_returns_null_provider(self):
        provider = ProviderFactory.get_provider(
            provider_name="acrcloud", acrcloud_host="host.only.set"
        )
        assert provider.provider_name == ProviderName.NULL

    def test_audd_without_token_returns_null_provider(self):
        provider = ProviderFactory.get_provider(provider_name="audd", audd_api_token="")
        assert provider.provider_name == ProviderName.NULL

    def test_unknown_provider_name_returns_null_provider(self):
        provider = ProviderFactory.get_provider(provider_name="not_a_real_provider")
        assert provider.provider_name == ProviderName.NULL

    def test_auto_with_no_credentials_returns_null_provider(self):
        provider = ProviderFactory.get_provider(provider_name="auto")
        assert provider.provider_name == ProviderName.NULL

    def test_acoustid_null_provider_identify_returns_unavailable(self):
        provider = ProviderFactory.get_provider(provider_name="acoustid", acoustid_api_key="")
        result = provider.identify("irrelevant/path.wav")
        assert result.status == "unavailable"
        assert result.candidates == []


# ─── Fake providers for service-level tests ────────────────────────────────


class _FakeProvider(SongIdentifier):
    """Returns a fixed RawIdentificationResult, synchronously."""

    def __init__(self, raw_result: RawIdentificationResult):
        self._raw_result = raw_result

    @property
    def provider_name(self) -> ProviderName:
        return self._raw_result.provider

    @property
    def is_available(self) -> bool:
        return True

    def identify(self, audio_path: str) -> RawIdentificationResult:
        return self._raw_result


class _HangingProvider(SongIdentifier):
    """Simulates a provider that never returns within the timeout window."""

    def __init__(self, hang_seconds: float):
        self._hang_seconds = hang_seconds

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.ACOUSTID

    @property
    def is_available(self) -> bool:
        return True

    def identify(self, audio_path: str) -> RawIdentificationResult:
        time.sleep(self._hang_seconds)
        return RawIdentificationResult(provider=ProviderName.ACOUSTID, status="ok", candidates=[])


class _RaisingProvider(SongIdentifier):
    """Simulates a provider whose identify() raises instead of catching internally."""

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.ACOUSTID

    @property
    def is_available(self) -> bool:
        return True

    def identify(self, audio_path: str) -> RawIdentificationResult:
        raise RuntimeError("simulated provider crash")


# ─── SongIntelligenceService ───────────────────────────────────────────────


class TestSongIntelligenceService:
    def _make_service(self, provider) -> SongIntelligenceService:
        service = SongIntelligenceService(provider_name="acoustid", acoustid_api_key="irrelevant")
        service._provider = provider  # Inject fake/hanging/raising provider directly.
        return service

    def test_ambiguous_result_isolated_from_enrichment(self):
        c1 = IdentificationCandidate(title="A", score=0.5, score_available=True, mbid="mbid-a")
        c2 = IdentificationCandidate(title="B", score=0.48, score_available=True, mbid="mbid-b")
        raw = RawIdentificationResult(provider=ProviderName.ACOUSTID, status="ok", candidates=[c1, c2])
        service = self._make_service(_FakeProvider(raw))

        identification, enrichment = service.run_sync(b"fake-audio-bytes", job_id="job-1")

        assert identification.status == IdentificationStatus.AMBIGUOUS
        # Enrichment must never run for a non-identified result.
        assert enrichment is None

    def test_not_identified_result_skips_enrichment(self):
        raw = RawIdentificationResult(provider=ProviderName.ACOUSTID, status="no_match", candidates=[])
        service = self._make_service(_FakeProvider(raw))

        identification, enrichment = service.run_sync(b"fake-audio-bytes", job_id="job-2")

        assert identification.status == IdentificationStatus.NOT_IDENTIFIED
        assert enrichment is None

    def test_identified_without_mbid_skips_enrichment(self):
        candidate = IdentificationCandidate(title="Winner", score=0.95, score_available=True, mbid=None)
        raw = RawIdentificationResult(provider=ProviderName.ACOUSTID, status="ok", candidates=[candidate])
        service = self._make_service(_FakeProvider(raw))

        identification, enrichment = service.run_sync(b"fake-audio-bytes", job_id="job-3")

        assert identification.status == IdentificationStatus.IDENTIFIED
        assert enrichment is None  # No MBID -> no MusicBrainz call.

    def test_timeout_returns_unavailable_without_raising(self):
        """A provider that hangs past the configured timeout must not block or crash."""
        service = SongIntelligenceService(
            provider_name="acoustid", acoustid_api_key="irrelevant", timeout_seconds=1
        )
        service._provider = _HangingProvider(hang_seconds=5)

        started = time.monotonic()
        identification, enrichment = service.run_sync(b"fake-audio-bytes", job_id="job-timeout")
        elapsed = time.monotonic() - started

        assert identification.status == IdentificationStatus.UNAVAILABLE
        assert "timed out" in (identification.error or "").lower()
        assert enrichment is None
        assert elapsed < 4  # Must return promptly at the configured timeout, not wait for the hang.

    def test_provider_exception_is_isolated_as_unavailable(self):
        """If a provider implementation raises instead of catching internally,
        the service must still never raise."""
        service = self._make_service(_RaisingProvider())

        identification, enrichment = service.run_sync(b"fake-audio-bytes", job_id="job-crash")

        assert identification.status == IdentificationStatus.UNAVAILABLE
        assert enrichment is None

    def test_from_settings_falls_back_to_null_provider_without_config(self):
        """With no app settings module reachable, from_settings() must not raise."""
        with patch("app.intelligence.service._get_settings", return_value=None):
            service = SongIntelligenceService.from_settings()
        identification, enrichment = service.run_sync(b"fake-audio-bytes", job_id="job-no-settings")
        assert identification.status == IdentificationStatus.UNAVAILABLE
        assert enrichment is None

    def test_from_settings_propagates_provider_and_thresholds(self):
        """
        from_settings() must read the configured provider name, confidence
        threshold, dominance margin, and timeout verbatim — not silently
        substitute defaults or a different provider.
        """
        from app.utils.config import Settings

        settings = Settings(
            SONG_IDENTIFICATION_PROVIDER="audd",
            AUDD_API_TOKEN="fake-token-for-test",
            IDENTIFICATION_CONFIDENCE_THRESHOLD=0.9,
            IDENTIFICATION_DOMINANCE_MARGIN=0.2,
            SONG_INTELLIGENCE_TIMEOUT_SECONDS=3,
        )
        with patch("app.intelligence.service._get_settings", return_value=settings):
            service = SongIntelligenceService.from_settings()

        assert service._provider.provider_name == ProviderName.AUDD
        assert service._resolver.threshold == 0.9
        assert service._resolver.dominance_margin == 0.2
        assert service._timeout == 3

    def test_from_settings_respects_explicit_provider_over_other_credentials(self):
        """
        Provider selection must be explicit: configuring SONG_IDENTIFICATION_PROVIDER=
        acoustid must select AcoustID even when ACRCloud/AudD credentials also
        happen to be present (no silent 'best available' substitution unless
        provider='auto' was explicitly requested).
        """
        from app.utils.config import Settings

        settings = Settings(
            SONG_IDENTIFICATION_PROVIDER="acoustid",
            ACOUSTID_API_KEY="fake-acoustid-key",
            ACRCLOUD_HOST="host", ACRCLOUD_ACCESS_KEY="k", ACRCLOUD_ACCESS_SECRET="s",
            AUDD_API_TOKEN="fake-audd-token",
        )
        with patch("app.intelligence.service._get_settings", return_value=settings):
            service = SongIntelligenceService.from_settings()

        assert service._provider.provider_name == ProviderName.ACOUSTID


# ─── Pipeline failure isolation (AnalysisService) ──────────────────────────


class TestAnalysisServiceIsolation:
    def test_song_intelligence_exception_does_not_propagate(self, synthetic_wav_bytes):
        """
        Simulate SongIntelligenceService.from_settings() raising an unexpected
        error. AnalysisService._run_song_intelligence must swallow it and
        return a status='unavailable' schema object — never raise, and never
        write to module_errors (DSP-unrelated failures must not flip job
        status away from 'completed').
        """
        from app.services.analysis_service import AnalysisService

        service = AnalysisService()
        with patch(
            "app.intelligence.service.SongIntelligenceService.from_settings",
            side_effect=RuntimeError("boom"),
        ):
            result = service._run_song_intelligence(
                file_content=synthetic_wav_bytes, job_id="job-isolated", filename="test.wav"
            )

        assert result is not None
        assert result.identification.status == "unavailable"
        assert "boom" in (result.identification.error or "")

    def test_song_intelligence_disabled_returns_none(self, synthetic_wav_bytes):
        from app.services.analysis_service import AnalysisService
        from app.utils.config import Settings

        service = AnalysisService()
        disabled_settings = Settings(SONG_INTELLIGENCE_ENABLED=False)
        with patch("app.utils.config.get_settings", return_value=disabled_settings):
            result = service._run_song_intelligence(
                file_content=synthetic_wav_bytes, job_id="job-disabled", filename="test.wav"
            )
        assert result is None

    def test_song_intelligence_disabled_via_real_env_var(self, synthetic_wav_bytes, monkeypatch):
        """
        End-to-end: setting SONG_INTELLIGENCE_ENABLED=false in the actual
        environment (as a deployer would via .env) must disable the stage,
        through the real Settings() env parsing — not a mocked return value.
        """
        from app.services.analysis_service import AnalysisService
        from app.utils import config as config_module

        monkeypatch.setenv("SONG_INTELLIGENCE_ENABLED", "false")
        config_module.get_settings.cache_clear()
        try:
            service = AnalysisService()
            result = service._run_song_intelligence(
                file_content=synthetic_wav_bytes, job_id="job-disabled-env", filename="test.wav"
            )
            assert result is None
        finally:
            config_module.get_settings.cache_clear()

    def test_full_pipeline_completes_when_song_intelligence_unavailable(self, synthetic_wav_bytes):
        """
        End-to-end: with no provider credentials configured (the default in
        this test environment), the full DSP pipeline must still complete
        and song_identification must report 'unavailable' rather than being
        absent or crashing the job.
        """
        from app.services.analysis_service import AnalysisService
        from app.services.job_service import JobService

        job_id = "job-e2e-intel"
        job_svc = JobService()
        job_svc.create_job(job_id=job_id, filename="test.wav", file_content=synthetic_wav_bytes)

        service = AnalysisService()
        service.run_analysis(job_id=job_id, file_content=synthetic_wav_bytes, filename="test.wav")

        job = job_svc.get_job(job_id)
        assert job["status"] in ("completed", "partial")
        result = job["result"]
        assert result.song_identification is not None
        assert result.song_identification.identification.status in ("unavailable", "not_identified")
        # DSP results must be present and unaffected regardless of Song Intelligence outcome.
        assert result.audio is not None
        assert result.tempo is not None
