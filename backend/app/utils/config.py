"""
SwarSense — Application Settings
=================================
Centralized configuration loaded from environment variables / .env.

Song Intelligence (Phase 2) settings:
  SONG_INTELLIGENCE_ENABLED            — master switch (default True)
  SONG_IDENTIFICATION_PROVIDER         — 'acoustid' | 'acrcloud' | 'audd' | 'auto' (default 'acoustid')
  ACOUSTID_API_KEY                     — AcoustID API key
  ACRCLOUD_HOST, ACRCLOUD_ACCESS_KEY, ACRCLOUD_ACCESS_SECRET — ACRCloud credentials
  AUDD_API_TOKEN                       — AudD API token
  IDENTIFICATION_CONFIDENCE_THRESHOLD  — minimum score to consider identified (default 0.75)
  IDENTIFICATION_DOMINANCE_MARGIN      — min gap between top-2 scores (default 0.15)
  SONG_INTELLIGENCE_TIMEOUT_SECONDS    — bound on the identification pipeline (default 15)

All Song Intelligence settings are optional. With no credentials configured,
the system falls back to a NullProvider and reports identification as
'unavailable' — the DSP pipeline is never affected.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Song Intelligence (Phase 2) ────────────────────────────────────────────
    SONG_INTELLIGENCE_ENABLED: bool = True
    SONG_IDENTIFICATION_PROVIDER: str = "acoustid"
    ACOUSTID_API_KEY: str = ""
    ACRCLOUD_HOST: str = ""
    ACRCLOUD_ACCESS_KEY: str = ""
    ACRCLOUD_ACCESS_SECRET: str = ""
    AUDD_API_TOKEN: str = ""
    IDENTIFICATION_CONFIDENCE_THRESHOLD: float = 0.75
    IDENTIFICATION_DOMINANCE_MARGIN: float = 0.15
    SONG_INTELLIGENCE_TIMEOUT_SECONDS: int = 15


@lru_cache
def get_settings() -> Settings:
    """Cached application settings singleton."""
    return Settings()
