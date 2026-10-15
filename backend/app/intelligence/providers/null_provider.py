"""
SwarSense — Null Provider (Phase 2)
=====================================
Used when no provider is configured or the requested provider's credentials
are absent. Returns status=unavailable immediately without making any network call.
"""

from __future__ import annotations

from app.intelligence.base import (
    IdentificationStatus,
    ProviderName,
    RawIdentificationResult,
    SongIdentifier,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class NullProvider(SongIdentifier):
    """
    A no-op provider that returns 'unavailable' immediately.

    Used when:
    - SONG_INTELLIGENCE_ENABLED=false
    - The configured provider's credentials are missing
    - An unknown provider name is specified
    - An error occurred during provider selection
    """

    def __init__(self, reason: str = "No identification provider configured.") -> None:
        self._reason = reason

    @property
    def provider_name(self) -> ProviderName:
        return ProviderName.NULL

    @property
    def is_available(self) -> bool:
        return False  # Always unavailable by design

    def identify(self, audio_path: str) -> RawIdentificationResult:
        logger.info(f"[NullProvider] Identification skipped: {self._reason}")
        return RawIdentificationResult(
            provider=ProviderName.NULL,
            status="unavailable",
            candidates=[],
            error=self._reason,
        )
