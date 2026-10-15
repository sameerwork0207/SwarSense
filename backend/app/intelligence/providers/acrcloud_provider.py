"""
SwarSense — ACRCloud Provider (Phase 2)
=========================================
Identifies audio using the ACRCloud Humming/Music Recognition API.

IMPORTANT PRIVACY NOTE:
  Unlike AcoustID, ACRCloud receives a raw audio segment (typically the
  first 10–20 seconds) transmitted to ACRCloud's servers. Only use this
  provider when this data transmission is acceptable for your use case.

Configuration:
  ACRCLOUD_HOST=identify-eu-west-1.acrcloud.com   (or your region)
  ACRCLOUD_ACCESS_KEY=<your key>
  ACRCLOUD_ACCESS_SECRET=<your secret>

Authentication:
  ACRCloud uses HMAC-SHA1 request signing.
  The signature is built from: method + url + access_key + data_type + signature_version + timestamp.
  See: https://docs.acrcloud.com/reference/identification-api

Confidence:
  ACRCloud returns a 'score' field (0–100) in the music metadata.
  This is normalized to 0.0–1.0 to be consistent with AcoustID.

Attribution:
  When results are used, attribution to ACRCloud may be required by their
  terms of service. The attribution field is set accordingly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
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

_ACRCLOUD_DATA_TYPE = "audio"
_ACRCLOUD_SIGNATURE_VERSION = "1"
_ACRCLOUD_HTTP_METHOD = "POST"
_ACRCLOUD_URL = "/v1/identify"
_SAMPLE_BYTES = 10 * 1024  # 10KB audio sample
_HTTP_TIMEOUT = 12


class ACRCloudProvider(SongIdentifier):
    """
    ACRCloud audio identification provider.

    Transmits a raw audio excerpt to ACRCloud's API.
    Only active when ACRCLOUD_HOST, ACRCLOUD_ACCESS_KEY, and
    ACRCLOUD_ACCESS_SECRET are all configured.
    """

    def __init__(
        self,
        host: str,
        access_key: str,
        access_secret: str,
        http_timeout: int = _HTTP_TIMEOUT,
    ) -> None:
        self._host = host.strip().rstrip("/")
        self._access_key = access_key.strip()
        self._access_secret = access_secret.strip()
        self._http_timeout = http_timeout

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.ACRCLOUD

    @property
    def is_available(self) -> bool:
        return bool(self._host and self._access_key and self._access_secret)

    def identify(self, audio_path: str) -> RawIdentificationResult:
        """Identify audio file at audio_path via ACRCloud."""
        try:
            return self._identify_internal(audio_path)
        except Exception as exc:
            logger.error(f"[ACRCloud] Unexpected error: {exc}", exc_info=True)
            return RawIdentificationResult(
                provider=ProviderName.ACRCLOUD,
                status="error",
                error=f"ACRCloud identification failed: {exc}",
            )

    def _build_signature(self, timestamp: str) -> str:
        """Build the HMAC-SHA1 request signature per ACRCloud spec."""
        string_to_sign = "\n".join([
            _ACRCLOUD_HTTP_METHOD,
            _ACRCLOUD_URL,
            self._access_key,
            _ACRCLOUD_DATA_TYPE,
            _ACRCLOUD_SIGNATURE_VERSION,
            timestamp,
        ])
        secret_bytes = self._access_secret.encode("utf-8")
        msg_bytes = string_to_sign.encode("utf-8")
        digest = hmac.new(secret_bytes, msg_bytes, hashlib.sha1).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _identify_internal(self, audio_path: str) -> RawIdentificationResult:
        try:
            import httpx
        except ImportError:
            return RawIdentificationResult(
                provider=ProviderName.ACRCLOUD,
                status="unavailable",
                error="httpx is not installed.",
            )

        timestamp = str(int(time.time()))
        signature = self._build_signature(timestamp)

        # Read audio bytes (ACRCloud needs the actual audio sample)
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
        except OSError as exc:
            return RawIdentificationResult(
                provider=ProviderName.ACRCLOUD,
                status="error",
                error=f"Could not read audio file: {exc}",
            )

        url = f"https://{self._host}{_ACRCLOUD_URL}"
        data = {
            "access_key": self._access_key,
            "data_type": _ACRCLOUD_DATA_TYPE,
            "signature_version": _ACRCLOUD_SIGNATURE_VERSION,
            "signature": signature,
            "timestamp": timestamp,
        }
        files = {"sample": ("sample.wav", audio_data, "audio/wav")}

        logger.info(f"[ACRCloud] Submitting audio sample to {self._host}")
        try:
            response = httpx.post(
                url, data=data, files=files, timeout=self._http_timeout
            )
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException:
            return RawIdentificationResult(
                provider=ProviderName.ACRCLOUD,
                status="error",
                error=f"ACRCloud request timed out after {self._http_timeout}s.",
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                return RawIdentificationResult(
                    provider=ProviderName.ACRCLOUD,
                    status="error",
                    error="ACRCloud: Invalid credentials.",
                )
            return RawIdentificationResult(
                provider=ProviderName.ACRCLOUD,
                status="error",
                error=f"ACRCloud HTTP error {exc.response.status_code}.",
            )
        except Exception as exc:
            return RawIdentificationResult(
                provider=ProviderName.ACRCLOUD,
                status="error",
                error=f"ACRCloud network error: {exc}",
            )

        return self._parse_response(result)

    def _parse_response(self, result: dict) -> RawIdentificationResult:
        """Parse ACRCloud JSON response into RawIdentificationResult."""
        status_code = result.get("status", {}).get("code", -1)
        status_msg = result.get("status", {}).get("msg", "")

        # 0 = success, 1001 = no result
        if status_code == 1001:
            logger.info("[ACRCloud] No match found.")
            return RawIdentificationResult(
                provider=ProviderName.ACRCLOUD,
                status="no_match",
                candidates=[],
            )
        if status_code != 0:
            logger.warning(f"[ACRCloud] API error code {status_code}: {status_msg}")
            return RawIdentificationResult(
                provider=ProviderName.ACRCLOUD,
                status="error",
                error=f"ACRCloud status {status_code}: {status_msg}",
            )

        music_list = result.get("metadata", {}).get("music", [])
        if not music_list:
            return RawIdentificationResult(
                provider=ProviderName.ACRCLOUD,
                status="no_match",
                candidates=[],
            )

        candidates = []
        for item in music_list:
            # Extract title, artist, album
            title = item.get("title")
            artists = item.get("artists", [])
            artist = artists[0].get("name") if artists else None
            album = item.get("album", {}).get("name")

            # Release year from release_date (YYYY-MM-DD or YYYY)
            release_year: Optional[int] = None
            release_date = item.get("release_date", "")
            if release_date and len(release_date) >= 4:
                try:
                    release_year = int(release_date[:4])
                except ValueError:
                    pass

            # Score: ACRCloud provides 0-100, normalize to 0.0-1.0
            raw_score = item.get("score")
            if raw_score is not None:
                try:
                    score = float(raw_score) / 100.0
                    score_available = True
                except (TypeError, ValueError):
                    score = None
                    score_available = False
            else:
                score = None
                score_available = False

            # ISRC
            isrc = item.get("external_ids", {}).get("isrc")
            provider_track_id = str(item.get("acrid", "")) or None

            candidates.append(IdentificationCandidate(
                title=title,
                artist=artist,
                album=album,
                release_year=release_year,
                score=score,
                score_available=score_available,
                provider_track_id=provider_track_id,
                isrc=isrc,
            ))

        logger.info(f"[ACRCloud] {len(candidates)} candidate(s) found.")
        return RawIdentificationResult(
            provider=ProviderName.ACRCLOUD,
            status="ok",
            candidates=candidates,
            attribution=(
                "Music identification powered by ACRCloud. "
                "Results provided by ACRCloud music recognition service."
            ),
        )
