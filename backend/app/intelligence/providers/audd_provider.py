"""
SwarSense — AudD Provider (Phase 2)
======================================
Identifies audio using the AudD Music Recognition API.

IMPORTANT:
  AudD transmits audio data to its servers for identification.
  Only use this provider when this is acceptable for your use case.

Configuration:
  AUDD_API_TOKEN=<your token>   (register at audd.io)

Confidence:
  AudD does NOT expose a confidence/score field in its standard API response.
  Therefore:
    confidence = None
    confidence_available = False
  A confidence value is NEVER invented or inferred.

Documentation: https://docs.audd.io/
"""

from __future__ import annotations

import os
from typing import Optional

from app.intelligence.base import (
    IdentificationCandidate,
    ProviderName,
    RawIdentificationResult,
    SongIdentifier,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_AUDD_API_URL = "https://api.audd.io/"
_HTTP_TIMEOUT = 12


class AudDProvider(SongIdentifier):
    """
    AudD audio identification provider.

    Note: AudD does not expose a confidence/score field.
    Results have confidence_available=False. The resolver will treat
    a single AudD candidate as not_identified (no reliable score to compare
    against the threshold).

    This means AudD results can only be 'identified' if CandidateResolver
    is configured to allow score-unavailable matches — which this implementation
    does NOT do, in accordance with the specification (single candidate without
    score → not_identified).

    AudD is therefore most useful as a secondary enrichment source or to
    confirm title/artist when combined with other logic.
    """

    def __init__(self, api_token: str, http_timeout: int = _HTTP_TIMEOUT) -> None:
        self._api_token = api_token.strip()
        self._http_timeout = http_timeout

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.AUDD

    @property
    def is_available(self) -> bool:
        return bool(self._api_token)

    def identify(self, audio_path: str) -> RawIdentificationResult:
        """Identify audio at audio_path via AudD."""
        try:
            return self._identify_internal(audio_path)
        except Exception as exc:
            logger.error(f"[AudD] Unexpected error: {exc}", exc_info=True)
            return RawIdentificationResult(
                provider=ProviderName.AUDD,
                status="error",
                error=f"AudD identification failed: {exc}",
            )

    def _identify_internal(self, audio_path: str) -> RawIdentificationResult:
        try:
            import httpx
        except ImportError:
            return RawIdentificationResult(
                provider=ProviderName.AUDD,
                status="unavailable",
                error="httpx is not installed.",
            )

        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
        except OSError as exc:
            return RawIdentificationResult(
                provider=ProviderName.AUDD,
                status="error",
                error=f"Could not read audio file: {exc}",
            )

        logger.info("[AudD] Submitting audio to AudD API.")
        try:
            response = httpx.post(
                _AUDD_API_URL,
                data={"api_token": self._api_token, "return": "spotify,apple_music"},
                files={"file": ("sample.wav", audio_data, "audio/wav")},
                timeout=self._http_timeout,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException:
            return RawIdentificationResult(
                provider=ProviderName.AUDD,
                status="error",
                error=f"AudD request timed out after {self._http_timeout}s.",
            )
        except httpx.HTTPStatusError as exc:
            return RawIdentificationResult(
                provider=ProviderName.AUDD,
                status="error",
                error=f"AudD HTTP error {exc.response.status_code}.",
            )
        except Exception as exc:
            return RawIdentificationResult(
                provider=ProviderName.AUDD,
                status="error",
                error=f"AudD network error: {exc}",
            )

        return self._parse_response(result)

    def _parse_response(self, result: dict) -> RawIdentificationResult:
        """Parse AudD JSON response."""
        status = result.get("status", "")

        if status == "error":
            error_info = result.get("error", {})
            error_msg = error_info.get("error_message", str(error_info))
            # Quota exceeded
            if "limit" in error_msg.lower() or "403" in str(error_info.get("http_code", "")):
                return RawIdentificationResult(
                    provider=ProviderName.AUDD,
                    status="error",
                    error=f"AudD quota exceeded or API error: {error_msg}",
                )
            return RawIdentificationResult(
                provider=ProviderName.AUDD,
                status="error",
                error=f"AudD API error: {error_msg}",
            )

        result_data = result.get("result")
        if not result_data:
            logger.info("[AudD] No match found.")
            return RawIdentificationResult(
                provider=ProviderName.AUDD,
                status="no_match",
                candidates=[],
            )

        # AudD returns a single result object (not a list)
        title = result_data.get("title")
        artist = result_data.get("artist")
        album = result_data.get("album")

        release_year: Optional[int] = None
        release_date = result_data.get("release_date", "")
        if release_date and len(str(release_date)) >= 4:
            try:
                release_year = int(str(release_date)[:4])
            except ValueError:
                pass

        # AudD does NOT expose a confidence/score. We do NOT invent one.
        candidate = IdentificationCandidate(
            title=title,
            artist=artist,
            album=album,
            release_year=release_year,
            score=None,         # AudD provides no score
            score_available=False,  # Explicitly false — not guessed
            provider_track_id=result_data.get("song_link"),
            isrc=None,
        )

        logger.info(f"[AudD] Match found: '{title}' by '{artist}' (no confidence score).")
        return RawIdentificationResult(
            provider=ProviderName.AUDD,
            status="ok",
            candidates=[candidate],
        )
