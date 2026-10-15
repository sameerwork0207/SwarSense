"""
SwarSense — Song Intelligence Service (Phase 2)
================================================
Orchestrates song identification and MusicBrainz enrichment as an optional
Stage 9 that runs after the DSP pipeline.

Provider selection (explicit — configuration-driven):
  SONG_IDENTIFICATION_PROVIDER=acoustid   → use AcoustID only
  SONG_IDENTIFICATION_PROVIDER=acrcloud   → use ACRCloud only
  SONG_IDENTIFICATION_PROVIDER=audd       → use AudD only
  SONG_IDENTIFICATION_PROVIDER=auto       → use best available (ACRCloud→AudD→AcoustID)
                                            "best" = most comprehensive with available creds

  DEFAULT = acoustid

  A provider is NEVER selected silently because credentials happen to exist.
  The SONG_IDENTIFICATION_PROVIDER setting is authoritative.

Failure isolation:
  Any exception during Stage 9 is caught and returns:
    identification.status = unavailable
  The DSP analysis result is never affected.

Timeout:
  Bounded by SONG_INTELLIGENCE_TIMEOUT_SECONDS (default 15).
  Uses Python's signal-based timeout on Unix or a threading approach on Windows.
"""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Optional, Tuple

from app.intelligence.base import (
    EnrichmentResult,
    IdentificationStatus,
    ProviderName,
    RawIdentificationResult,
    ResolvedIdentification,
)
from app.intelligence.resolver import CandidateResolver
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Settings access (lazy import to avoid circular deps) ─────────────────────

def _get_settings():
    """Get application settings lazily."""
    try:
        from app.utils.config import get_settings
        return get_settings()
    except Exception:
        return None


# ─── Provider Factory ─────────────────────────────────────────────────────────


class ProviderFactory:
    """
    Creates identification provider instances based on configuration.

    Raises no exceptions — always returns a provider (possibly NullProvider).
    Logs clearly which provider was selected and why.

    Provider selection is EXPLICIT — the configured provider name is used.
    No silent substitution based on credential availability.
    """

    @staticmethod
    def get_provider(
        provider_name: str,
        acoustid_api_key: str = "",
        acrcloud_host: str = "",
        acrcloud_access_key: str = "",
        acrcloud_access_secret: str = "",
        audd_api_token: str = "",
    ):
        """
        Return the configured provider or NullProvider if unavailable.

        Args:
            provider_name:        'acoustid', 'acrcloud', 'audd', or 'auto'
            acoustid_api_key:     AcoustID API key
            acrcloud_host:        ACRCloud host
            acrcloud_access_key:  ACRCloud access key
            acrcloud_access_secret: ACRCloud secret
            audd_api_token:       AudD API token

        Returns:
            A SongIdentifier implementation (never raises).
        """
        from app.intelligence.providers.null_provider import NullProvider
        from app.intelligence.providers.acoustid_provider import AcoustIDProviderWithTempFile
        from app.intelligence.providers.acrcloud_provider import ACRCloudProvider
        from app.intelligence.providers.audd_provider import AudDProvider

        name = (provider_name or "acoustid").strip().lower()
        logger.info(f"[ProviderFactory] Requested provider: '{name}'")

        if name == "acoustid":
            if not acoustid_api_key:
                return NullProvider(
                    "ACOUSTID_API_KEY is not configured. "
                    "Register free at https://acoustid.org/api-key"
                )
            try:
                p = AcoustIDProviderWithTempFile(api_key=acoustid_api_key)
                logger.info("[ProviderFactory] Selected: AcoustID")
                return p
            except Exception as exc:
                return NullProvider(f"AcoustID provider init failed: {exc}")

        elif name == "acrcloud":
            if not (acrcloud_host and acrcloud_access_key and acrcloud_access_secret):
                return NullProvider(
                    "ACRCloud credentials incomplete. "
                    "Set ACRCLOUD_HOST, ACRCLOUD_ACCESS_KEY, and ACRCLOUD_ACCESS_SECRET."
                )
            try:
                p = ACRCloudProvider(acrcloud_host, acrcloud_access_key, acrcloud_access_secret)
                logger.info("[ProviderFactory] Selected: ACRCloud")
                return p
            except Exception as exc:
                return NullProvider(f"ACRCloud provider init failed: {exc}")

        elif name == "audd":
            if not audd_api_token:
                return NullProvider(
                    "AUDD_API_TOKEN is not configured. "
                    "Register at https://audd.io/"
                )
            try:
                p = AudDProvider(api_token=audd_api_token)
                logger.info("[ProviderFactory] Selected: AudD")
                return p
            except Exception as exc:
                return NullProvider(f"AudD provider init failed: {exc}")

        elif name == "auto":
            # Deterministic priority: ACRCloud → AudD → AcoustID
            # Only uses a provider if its credentials are ALL present.
            # Provider selection is still explicit — the 'auto' setting means
            # "use the first fully-configured provider in this order."
            logger.info("[ProviderFactory] Mode=auto — trying ACRCloud → AudD → AcoustID")

            if acrcloud_host and acrcloud_access_key and acrcloud_access_secret:
                try:
                    p = ACRCloudProvider(acrcloud_host, acrcloud_access_key, acrcloud_access_secret)
                    logger.info("[ProviderFactory] auto → Selected: ACRCloud")
                    return p
                except Exception as exc:
                    logger.warning(f"[ProviderFactory] auto: ACRCloud init failed: {exc}")

            if audd_api_token:
                try:
                    p = AudDProvider(api_token=audd_api_token)
                    logger.info("[ProviderFactory] auto → Selected: AudD")
                    return p
                except Exception as exc:
                    logger.warning(f"[ProviderFactory] auto: AudD init failed: {exc}")

            if acoustid_api_key:
                try:
                    p = AcoustIDProviderWithTempFile(api_key=acoustid_api_key)
                    logger.info("[ProviderFactory] auto → Selected: AcoustID")
                    return p
                except Exception as exc:
                    logger.warning(f"[ProviderFactory] auto: AcoustID init failed: {exc}")

            return NullProvider(
                "Provider=auto but no provider has complete credentials. "
                "Set at least one of: ACOUSTID_API_KEY, or ACRCLOUD_HOST + "
                "ACRCLOUD_ACCESS_KEY + ACRCLOUD_ACCESS_SECRET, or AUDD_API_TOKEN."
            )

        else:
            return NullProvider(
                f"Unknown SONG_IDENTIFICATION_PROVIDER='{name}'. "
                f"Allowed values: acoustid, acrcloud, audd, auto."
            )


# ─── Intelligence Service ──────────────────────────────────────────────────────


class SongIntelligenceService:
    """
    Orchestrates the complete Song Intelligence pipeline:
      1. Provider selection (via ProviderFactory, config-driven)
      2. Audio identification (provider.identify())
      3. Candidate resolution (CandidateResolver.resolve())
      4. MusicBrainz enrichment (only if status=IDENTIFIED and MBID available)
      5. Return (ResolvedIdentification, EnrichmentResult | None)

    Any exception is caught and returned as UNAVAILABLE status.
    The DSP analysis pipeline is never affected.
    """

    def __init__(
        self,
        provider_name: str = "acoustid",
        acoustid_api_key: str = "",
        acrcloud_host: str = "",
        acrcloud_access_key: str = "",
        acrcloud_access_secret: str = "",
        audd_api_token: str = "",
        confidence_threshold: float = 0.75,
        dominance_margin: float = 0.15,
        timeout_seconds: int = 15,
    ) -> None:
        self._provider = ProviderFactory.get_provider(
            provider_name=provider_name,
            acoustid_api_key=acoustid_api_key,
            acrcloud_host=acrcloud_host,
            acrcloud_access_key=acrcloud_access_key,
            acrcloud_access_secret=acrcloud_access_secret,
            audd_api_token=audd_api_token,
        )
        self._resolver = CandidateResolver(
            threshold=confidence_threshold,
            dominance_margin=dominance_margin,
        )
        self._timeout = timeout_seconds

    @classmethod
    def from_settings(cls) -> "SongIntelligenceService":
        """
        Construct service from application settings/environment variables.
        Falls back to safe defaults if settings are unavailable.
        """
        settings = _get_settings()
        if settings is None:
            return cls()  # All defaults — will produce NullProvider

        return cls(
            provider_name=getattr(settings, "SONG_IDENTIFICATION_PROVIDER", "acoustid"),
            acoustid_api_key=getattr(settings, "ACOUSTID_API_KEY", "") or "",
            acrcloud_host=getattr(settings, "ACRCLOUD_HOST", "") or "",
            acrcloud_access_key=getattr(settings, "ACRCLOUD_ACCESS_KEY", "") or "",
            acrcloud_access_secret=getattr(settings, "ACRCLOUD_ACCESS_SECRET", "") or "",
            audd_api_token=getattr(settings, "AUDD_API_TOKEN", "") or "",
            confidence_threshold=getattr(settings, "IDENTIFICATION_CONFIDENCE_THRESHOLD", 0.75),
            dominance_margin=getattr(settings, "IDENTIFICATION_DOMINANCE_MARGIN", 0.15),
            timeout_seconds=getattr(settings, "SONG_INTELLIGENCE_TIMEOUT_SECONDS", 15),
        )

    def run_sync(
        self,
        audio_bytes: bytes,
        job_id: str,
        filename: str = "audio.wav",
    ) -> Tuple[ResolvedIdentification, Optional[EnrichmentResult]]:
        """
        Run the complete Song Intelligence pipeline synchronously.

        Designed to be called from the existing sync analysis pipeline.
        Bounded by self._timeout seconds via a background thread timeout.

        Args:
            audio_bytes: Raw audio file bytes.
            job_id:      Job ID for logging context.
            filename:    Original filename (used to determine temp file suffix).

        Returns:
            (ResolvedIdentification, EnrichmentResult | None)
            Always returns — never raises.
        """
        result_container: list = [None]
        exception_container: list = [None]

        def _run():
            try:
                result_container[0] = self._run_internal(audio_bytes, job_id, filename)
            except Exception as exc:
                exception_container[0] = exc

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self._timeout)

        if t.is_alive():
            # Thread still running after timeout
            logger.warning(
                f"[{job_id}] Song Intelligence timed out after {self._timeout}s."
            )
            unavailable = ResolvedIdentification(
                status=IdentificationStatus.UNAVAILABLE,
                provider=self._provider.provider_name,
                error=f"Song Intelligence timed out after {self._timeout}s.",
                intelligence_note="Song Intelligence timed out. DSP analysis is complete and unaffected.",
            )
            return unavailable, None

        if exception_container[0] is not None:
            exc = exception_container[0]
            logger.error(f"[{job_id}] Song Intelligence exception: {exc}", exc_info=exc)
            unavailable = ResolvedIdentification(
                status=IdentificationStatus.UNAVAILABLE,
                provider=self._provider.provider_name,
                error=str(exc),
                intelligence_note="Song Intelligence failed. DSP analysis is complete and unaffected.",
            )
            return unavailable, None

        return result_container[0]

    def _run_internal(
        self,
        audio_bytes: bytes,
        job_id: str,
        filename: str,
    ) -> Tuple[ResolvedIdentification, Optional[EnrichmentResult]]:
        """
        Internal pipeline: write temp file → identify → resolve → enrich.
        May raise; caller wraps in thread with timeout.
        """
        logger.info(
            f"[{job_id}] Song Intelligence: provider={self._provider.provider_name.value}"
        )

        # Determine file suffix for temp file
        suffix = ".wav"
        if filename:
            ext = os.path.splitext(filename)[-1].lower()
            if ext in (".mp3", ".wav", ".flac", ".ogg"):
                suffix = ext

        # Write audio to temp file and identify
        # AcoustID provider handles its own temp file.
        # Other providers (ACRCloud, AudD) also need a path.
        from app.intelligence.providers.acoustid_provider import AcoustIDProviderWithTempFile
        from app.intelligence.providers.null_provider import NullProvider

        if isinstance(self._provider, NullProvider):
            raw = self._provider.identify("")
        elif isinstance(self._provider, AcoustIDProviderWithTempFile):
            raw = self._provider.identify_bytes(audio_bytes, suffix=suffix)
        else:
            # ACRCloud and AudD: write temp file and pass path
            tmp_path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False, prefix="swarsense_intel_"
                ) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                raw = self._provider.identify(tmp_path)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        # Resolve candidates
        identification = self._resolver.resolve(raw)
        logger.info(
            f"[{job_id}] Identification status: {identification.status.value}"
        )

        # Enrich if identified with a usable MBID
        enrichment: Optional[EnrichmentResult] = None
        if identification.status == IdentificationStatus.IDENTIFIED and identification.mbid:
            logger.info(f"[{job_id}] Running MusicBrainz enrichment (MBID={identification.mbid})")
            from app.intelligence.enrichers.musicbrainz_enricher import MusicBrainzEnricher
            enricher = MusicBrainzEnricher()
            enrichment = enricher.enrich(identification)
            if enrichment.error:
                logger.warning(f"[{job_id}] MusicBrainz enrichment error: {enrichment.error}")
            else:
                logger.info(
                    f"[{job_id}] MusicBrainz enrichment: "
                    f"{sum(len(s.facts) for s in enrichment.sources)} fact(s)"
                )
        else:
            logger.info(f"[{job_id}] MusicBrainz enrichment skipped (status={identification.status.value})")

        return identification, enrichment
