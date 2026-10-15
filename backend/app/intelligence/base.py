"""
SwarSense — Song Intelligence Base Interfaces & Data Models
===========================================================
Defines the abstract provider interface and all shared data structures
used across providers, resolvers, and enrichers.

These classes are NOT Pydantic models — they are internal dataclasses
used within the intelligence pipeline. Pydantic schemas (for the HTTP API)
live in app/api/schemas.py.

Data flow:
  Provider.identify() → RawIdentificationResult
  CandidateResolver.resolve() → ResolvedIdentification
  MusicBrainzEnricher.enrich() → EnrichmentResult
  SongIntelligenceService.run_sync() → (ResolvedIdentification, EnrichmentResult|None)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─── Enums ─────────────────────────────────────────────────────────────────────


class IdentificationStatus(str, Enum):
    """
    Final identification status returned to the API and UI.

    IDENTIFIED:     One candidate clearly and reliably identified.
    AMBIGUOUS:      Multiple plausible candidates with no clearly dominant match.
    NOT_IDENTIFIED: Provider responded but found no reliable candidate.
    UNAVAILABLE:    Provider not configured, failed, or timed out.
    """
    IDENTIFIED = "identified"
    AMBIGUOUS = "ambiguous"
    NOT_IDENTIFIED = "not_identified"
    UNAVAILABLE = "unavailable"


class ProviderName(str, Enum):
    """Registered provider names used in config and result attribution."""
    ACOUSTID = "acoustid"
    ACRCLOUD = "acrcloud"
    AUDD = "audd"
    NULL = "null"


# ─── Provider Result Models ────────────────────────────────────────────────────


@dataclass
class IdentificationCandidate:
    """
    A single candidate match returned by a provider.

    Fields:
        title:             Track title (if returned by provider).
        artist:            Primary artist name (if returned by provider).
        album:             Album or release title (if returned by provider).
        release_year:      Release year as integer (if returned by provider).
        score:             Provider-specific confidence score.
                           For AcoustID: float 0.0–1.0.
                           For ACRCloud: int 0–100 (normalized to 0.0–1.0 internally).
                           None if the provider does not expose a score.
        score_available:   True if score is a genuine provider value; False if None/inferred.
        acoustid:          AcoustID identifier string (AcoustID provider only).
        mbid:              MusicBrainz Recording ID (UUID string), if provided.
        provider_track_id: Provider-specific internal track ID (non-MBID).
        isrc:              ISRC code if returned.
    """
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    release_year: Optional[int] = None
    score: Optional[float] = None
    score_available: bool = False
    acoustid: Optional[str] = None
    mbid: Optional[str] = None
    provider_track_id: Optional[str] = None
    isrc: Optional[str] = None


@dataclass
class RawIdentificationResult:
    """
    Raw result returned by a provider before resolution.

    Fields:
        provider:          Which provider produced this result.
        status:            Preliminary status ('ok', 'no_match', 'error', 'unavailable').
                           This is NOT the final IdentificationStatus.
        candidates:        Ordered list of candidates (best-first if scored).
        error:             Error message if status == 'error' or 'unavailable'.
        attribution:       Attribution string required by some providers (e.g. ACRCloud).
        raw_response:      Optional dict of the raw provider JSON (for debugging).
    """
    provider: ProviderName
    status: str  # 'ok' | 'no_match' | 'error' | 'unavailable'
    candidates: list[IdentificationCandidate] = field(default_factory=list)
    error: Optional[str] = None
    attribution: Optional[str] = None
    raw_response: Optional[dict] = None


# ─── Resolved Result Models ────────────────────────────────────────────────────


@dataclass
class ResolvedIdentification:
    """
    Final resolved identification after applying the resolution algorithm.

    Fields:
        status:             Final IdentificationStatus.
        provider:           Provider name that produced the result.
        title:              Resolved title (from top candidate, only if identified).
        artist:             Resolved artist name.
        album:              Resolved album name.
        release_year:       Resolved release year.
        confidence:         Score of the top/winning candidate (or None).
        confidence_available: Whether confidence is a real provider value.
        candidates:         All raw candidates (for ambiguous display).
        acoustid:           AcoustID string from winning candidate.
        mbid:               MusicBrainz Recording ID from winning candidate.
        provider_track_id:  Provider-specific track ID from winning candidate.
        attribution:        Provider attribution string if required.
        error:              Error description if status == UNAVAILABLE.
        intelligence_note:  Human-readable note explaining the status.
    """
    status: IdentificationStatus
    provider: Optional[ProviderName] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    release_year: Optional[int] = None
    confidence: Optional[float] = None
    confidence_available: bool = False
    candidates: list[IdentificationCandidate] = field(default_factory=list)
    acoustid: Optional[str] = None
    mbid: Optional[str] = None
    provider_track_id: Optional[str] = None
    attribution: Optional[str] = None
    error: Optional[str] = None
    intelligence_note: Optional[str] = None


# ─── Enrichment Models ─────────────────────────────────────────────────────────


@dataclass
class EnrichmentFact:
    """
    A single externally retrieved fact with field name and value.
    Every fact must be traceable to a source — never displayed without attribution.
    """
    field: str
    value: str


@dataclass
class EnrichmentSource:
    """
    A source of enrichment data with its associated facts and disclaimer.

    The disclaimer MUST always be shown alongside these facts in the UI.
    Facts from external sources must never be presented as SwarSense model output.

    CRITICAL RAGA RULE:
        If a source returns a raga name (e.g., "Raga: Yaman"), the fact must be
        labeled as an external reference. It must NEVER be presented as:
        "SwarSense detected Yaman."
        The UI must show: "External reference: associated with Raga Yaman."
    """
    source_name: str
    url: Optional[str] = None
    disclaimer: str = "External reference — not generated by SwarSense."
    facts: list[EnrichmentFact] = field(default_factory=list)


@dataclass
class EnrichmentResult:
    """
    Combined enrichment result from all enrichment sources.
    """
    sources: list[EnrichmentSource] = field(default_factory=list)
    error: Optional[str] = None


# ─── Abstract Provider Interface ───────────────────────────────────────────────


class SongIdentifier(abc.ABC):
    """
    Abstract interface that all identification providers must implement.

    Providers must:
    - Accept a local audio file path (str).
    - Return a RawIdentificationResult with status + candidates.
    - NEVER crash the calling code — catch all exceptions internally and
      return status='error' with an error message.
    - NEVER fabricate or invent candidate scores, titles, artists, or metadata.
    - NEVER log API keys, tokens, or secrets.
    """

    @abc.abstractmethod
    def identify(self, audio_path: str) -> RawIdentificationResult:
        """
        Identify the audio recording at the given local file path.

        Args:
            audio_path: Absolute path to a local audio file (WAV or MP3).

        Returns:
            RawIdentificationResult — always; never raises.
        """
        ...

    @property
    @abc.abstractmethod
    def provider_name(self) -> ProviderName:
        """Return the provider's ProviderName enum value."""
        ...

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        """
        Return True if this provider is properly configured and can be used.
        Providers should check for required credentials/config here.
        """
        ...
