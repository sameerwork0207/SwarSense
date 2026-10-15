"""
SwarSense — AcoustID Provider (Phase 2)
=========================================
Identifies audio using Chromaprint + AcoustID.

Technical approach:
  1. Write audio bytes to a secure NamedTemporaryFile.
  2. Run fpcalc (Chromaprint CLI) locally to compute the acoustic fingerprint.
     → Only the compact fingerprint hash is sent to AcoustID.
     → Raw audio is NEVER uploaded to AcoustID.
  3. Query the AcoustID API with fingerprint + duration + API key.
  4. Parse the response into IdentificationCandidate objects.

Configuration:
  ACOUSTID_API_KEY=<your key>   (register free at acoustid.org)

Requirements:
  - pyacoustid (pip)
  - fpcalc on PATH  (Windows: download from acoustid.org/chromaprint;
                     Docker: apt-get install libchromaprint-tools)

Privacy note:
  AcoustID receives only the compact fingerprint hash and duration — not the
  audio content itself. This is the safest available identification method.

Errors handled:
  - fpcalc not found
  - Audio file unreadable / too short
  - Network timeout
  - HTTP errors
  - API key missing or invalid
  - Empty/malformed response
  - Multiple candidates (all returned; CandidateResolver decides)
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

from app.intelligence.base import (
    IdentificationCandidate,
    ProviderName,
    RawIdentificationResult,
    SongIdentifier,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# AcoustID API endpoint
_ACOUSTID_URL = "https://api.acoustid.org/v2/lookup"

# Metadata includes to request (recordings + releases + MBIDs)
_META = "recordings releasegroups compress"

# HTTP timeout for the AcoustID API call (seconds)
_HTTP_TIMEOUT = 10


class AcoustIDProvider(SongIdentifier):
    """
    AcoustID provider using local Chromaprint fingerprinting.

    The provider uses pyacoustid which calls fpcalc internally.
    All fingerprint computation is local; only the hash is sent to AcoustID.
    """

    def __init__(self, api_key: str, http_timeout: int = _HTTP_TIMEOUT) -> None:
        """
        Args:
            api_key:      AcoustID API key (register free at acoustid.org).
            http_timeout: HTTP request timeout in seconds.
        """
        if not api_key or not api_key.strip():
            raise ValueError("AcoustID API key must not be empty.")
        self._api_key = api_key.strip()
        self._http_timeout = http_timeout

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.ACOUSTID

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def identify(self, audio_path: str) -> RawIdentificationResult:
        """
        Identify audio at the given local path using AcoustID.

        This method never raises — all exceptions are caught and returned
        as an error result so they don't crash the analysis pipeline.

        Args:
            audio_path: Absolute path to a local WAV or MP3 file.

        Returns:
            RawIdentificationResult with status and candidates.
        """
        try:
            return self._identify_internal(audio_path)
        except Exception as exc:
            logger.error(f"[AcoustID] Unexpected error: {exc}", exc_info=True)
            return RawIdentificationResult(
                provider=ProviderName.ACOUSTID,
                status="error",
                candidates=[],
                error=f"AcoustID identification failed: {exc}",
            )

    def _identify_internal(self, audio_path: str) -> RawIdentificationResult:
        """Internal implementation — may raise; caller wraps in try/except."""
        # Import acoustid lazily so missing library doesn't break startup
        try:
            import acoustid
        except ImportError:
            return RawIdentificationResult(
                provider=ProviderName.ACOUSTID,
                status="unavailable",
                error="pyacoustid is not installed. Run: pip install pyacoustid",
            )

        logger.info(f"[AcoustID] Fingerprinting '{os.path.basename(audio_path)}'")

        # Fingerprint the audio file using fpcalc
        try:
            duration, fingerprint = acoustid.fingerprint_file(audio_path)
        except acoustid.FingerprintGenerationError as exc:
            # fpcalc missing, audio too short, or unreadable
            err = str(exc)
            logger.warning(f"[AcoustID] Fingerprint generation failed: {err}")
            if "fpcalc" in err.lower() or "chromaprint" in err.lower():
                return RawIdentificationResult(
                    provider=ProviderName.ACOUSTID,
                    status="unavailable",
                    error=(
                        "fpcalc (Chromaprint) is not available on PATH. "
                        "Windows: download from https://acoustid.org/chromaprint. "
                        "Docker: apt-get install libchromaprint-tools."
                    ),
                )
            return RawIdentificationResult(
                provider=ProviderName.ACOUSTID,
                status="error",
                error=f"Fingerprint generation failed: {err}",
            )
        except Exception as exc:
            return RawIdentificationResult(
                provider=ProviderName.ACOUSTID,
                status="error",
                error=f"Fingerprint generation error: {exc}",
            )

        logger.info(f"[AcoustID] Fingerprint obtained. Duration={duration:.1f}s. Querying API...")

        # Query the AcoustID API
        try:
            results = list(acoustid.lookup(
                self._api_key,
                fingerprint,
                duration,
                meta=_META,
            ))
        except acoustid.WebServiceError as exc:
            err = str(exc)
            logger.warning(f"[AcoustID] API error: {err}")
            if "apikey" in err.lower() or "401" in err or "403" in err:
                return RawIdentificationResult(
                    provider=ProviderName.ACOUSTID,
                    status="error",
                    error=f"AcoustID API key is invalid or unauthorized: {err}",
                )
            return RawIdentificationResult(
                provider=ProviderName.ACOUSTID,
                status="error",
                error=f"AcoustID web service error: {err}",
            )
        except Exception as exc:
            return RawIdentificationResult(
                provider=ProviderName.ACOUSTID,
                status="error",
                error=f"AcoustID lookup failed: {exc}",
            )

        # Parse results
        if not results:
            logger.info("[AcoustID] No results returned.")
            return RawIdentificationResult(
                provider=ProviderName.ACOUSTID,
                status="no_match",
                candidates=[],
            )

        candidates = []
        for score, acoustid_id, title, artist in results:
            if title is None and artist is None:
                # Result has no recording metadata — skip
                continue
            candidate = IdentificationCandidate(
                title=title,
                artist=artist,
                score=float(score) if score is not None else None,
                score_available=(score is not None),
                acoustid=acoustid_id,
            )
            candidates.append(candidate)

        # If all results lacked metadata, try extracting from raw response
        if not candidates and results:
            # Return the raw acoustid_id as a minimal candidate without title/artist
            # so the resolver can still log the fingerprint match
            score, acoustid_id, _, _ = results[0]
            candidates = [IdentificationCandidate(
                score=float(score) if score is not None else None,
                score_available=(score is not None),
                acoustid=acoustid_id,
            )]

        logger.info(f"[AcoustID] {len(candidates)} candidate(s) found.")
        return RawIdentificationResult(
            provider=ProviderName.ACOUSTID,
            status="ok" if candidates else "no_match",
            candidates=candidates,
        )


class AcoustIDProviderWithTempFile(AcoustIDProvider):
    """
    AcoustID provider that accepts raw audio bytes and manages a temp file.

    Usage within the analysis pipeline:
        provider = AcoustIDProviderWithTempFile(api_key=settings.ACOUSTID_API_KEY)
        result = provider.identify_bytes(file_content, suffix=".wav")
    """

    def identify_bytes(
        self, audio_bytes: bytes, suffix: str = ".wav"
    ) -> RawIdentificationResult:
        """
        Write audio_bytes to a temp file, identify it, then clean up.

        The temp file is created with delete=False (required on Windows when
        fpcalc needs to open it by name after Python has it open).
        Cleanup is guaranteed via finally block.

        Args:
            audio_bytes: Raw audio file bytes (WAV or MP3).
            suffix:      File extension for fpcalc to detect format ('.wav' or '.mp3').
        """
        tmp_path: Optional[str] = None
        try:
            # delete=False required on Windows: fpcalc must open the file by name
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, prefix="swarsense_acoustid_"
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            logger.debug(f"[AcoustID] Temp file: {tmp_path} ({len(audio_bytes)/1024:.1f}KB)")
            return self.identify(tmp_path)

        finally:
            # Always clean up — even if identify() raises (it shouldn't)
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    logger.debug(f"[AcoustID] Temp file cleaned up: {tmp_path}")
                except OSError as e:
                    logger.warning(f"[AcoustID] Could not delete temp file {tmp_path}: {e}")
