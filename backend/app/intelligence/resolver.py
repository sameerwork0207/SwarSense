"""
SwarSense — Candidate Resolver (Phase 2)
=========================================
Applies deterministic resolution logic to convert a RawIdentificationResult
(a list of candidates with optional scores) into a final ResolvedIdentification.

Resolution algorithm (approved):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  0 candidates              → NOT_IDENTIFIED                             │
  │  1 candidate               → IDENTIFIED   if score ≥ threshold          │
  │                            → NOT_IDENTIFIED if score < threshold         │
  │                              (or score unavailable)                      │
  │  Multiple candidates       → IDENTIFIED   if top candidate is "clearly   │
  │                              dominant" (defined below)                   │
  │                            → AMBIGUOUS    otherwise                      │
  │  Provider unavailable/err  → UNAVAILABLE                                │
  └─────────────────────────────────────────────────────────────────────────┘

"Clearly dominant" definition:
  A top candidate is considered clearly dominant when ALL of the following hold:
    1. score_available is True  (the score is a real provider value, not inferred)
    2. top_score >= configured threshold (default 0.75)
    3. top_score - second_score >= DOMINANCE_MARGIN (default 0.15)

  Rationale for the dominance margin:
    If two candidates are both above the threshold and their scores are close
    (e.g. 0.82 vs 0.79), it is not safe to auto-select the top candidate.
    The margin of 0.15 ensures the winner is meaningfully more reliable.
    This is especially important for Indian classical content where database
    coverage is sparse and multiple recordings may be superficially similar.

Configuration keys (from app/api/config.py or env):
  IDENTIFICATION_CONFIDENCE_THRESHOLD  — minimum score to consider identified (default 0.75)
  IDENTIFICATION_DOMINANCE_MARGIN      — min gap between top-2 scores (default 0.15)
"""

from __future__ import annotations

from app.intelligence.base import (
    IdentificationCandidate,
    IdentificationStatus,
    ProviderName,
    RawIdentificationResult,
    ResolvedIdentification,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Default constants (also used as fallback if config unavailable) ────────────
DEFAULT_THRESHOLD: float = 0.75
DEFAULT_DOMINANCE_MARGIN: float = 0.15


class CandidateResolver:
    """
    Stateless resolver: converts a RawIdentificationResult into a
    ResolvedIdentification using the approved deterministic algorithm.

    Instantiate with threshold and dominance_margin from application settings.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
    ) -> None:
        """
        Args:
            threshold:         Minimum score for a candidate to be considered identified.
                               Provider-specific; for AcoustID this is a 0.0–1.0 float.
            dominance_margin:  Minimum difference between the top and second candidate's
                               scores required to call the top candidate "clearly dominant".
        """
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if not (0.0 <= dominance_margin < 1.0):
            raise ValueError(f"dominance_margin must be in [0, 1), got {dominance_margin}")
        self.threshold = threshold
        self.dominance_margin = dominance_margin

    # ─── Public API ───────────────────────────────────────────────────────────

    def resolve(self, raw: RawIdentificationResult) -> ResolvedIdentification:
        """
        Apply the approved resolution algorithm to a raw provider result.

        Args:
            raw: The RawIdentificationResult from a provider.

        Returns:
            ResolvedIdentification with a definitive IdentificationStatus.
        """
        # Case: provider not available or returned an error before we even got candidates
        if raw.status in ("error", "unavailable"):
            return self._make_unavailable(
                provider=raw.provider,
                error=raw.error or f"Provider returned status={raw.status}",
                candidates=raw.candidates,
            )

        candidates = raw.candidates

        # Case: no candidates at all
        if not candidates:
            logger.info(
                f"[resolver] Provider={raw.provider.value} returned 0 candidates → not_identified"
            )
            return ResolvedIdentification(
                status=IdentificationStatus.NOT_IDENTIFIED,
                provider=raw.provider,
                candidates=[],
                attribution=raw.attribution,
                intelligence_note=(
                    "No matching recording was found in the provider's database. "
                    "This is a valid result — Indian classical, live, or archival "
                    "recordings are frequently absent from commercial databases."
                ),
            )

        # Case: exactly one candidate
        if len(candidates) == 1:
            return self._resolve_single(candidates[0], raw)

        # Case: multiple candidates
        return self._resolve_multiple(candidates, raw)

    # ─── Private resolution helpers ───────────────────────────────────────────

    def _resolve_single(
        self,
        candidate: IdentificationCandidate,
        raw: RawIdentificationResult,
    ) -> ResolvedIdentification:
        """
        Resolution logic for exactly one candidate.

        Rules:
          - score_available=True AND score >= threshold  → IDENTIFIED
          - score_available=True AND score <  threshold  → NOT_IDENTIFIED
          - score_available=False (no score at all)      → NOT_IDENTIFIED
            (We refuse to identify without a reliable score for a single candidate.)
        """
        score = candidate.score
        score_available = candidate.score_available

        if score_available and score is not None and score >= self.threshold:
            logger.info(
                f"[resolver] Single candidate identified: score={score:.3f} "
                f">= threshold={self.threshold}"
            )
            return self._make_identified(candidate, raw)

        # Score below threshold or unavailable → not_identified
        reason = (
            f"score={score:.3f} < threshold={self.threshold}"
            if (score_available and score is not None)
            else "provider did not supply a confidence score"
        )
        logger.info(
            f"[resolver] Single candidate not_identified: {reason}"
        )
        return ResolvedIdentification(
            status=IdentificationStatus.NOT_IDENTIFIED,
            provider=raw.provider,
            candidates=raw.candidates,
            attribution=raw.attribution,
            intelligence_note=(
                f"One candidate was found but could not be confirmed ({reason}). "
                "This does not affect your DSP analysis."
            ),
        )

    def _resolve_multiple(
        self,
        candidates: list[IdentificationCandidate],
        raw: RawIdentificationResult,
    ) -> ResolvedIdentification:
        """
        Resolution logic for multiple candidates.

        A top candidate is "clearly dominant" when ALL hold:
          1. score_available is True (genuine provider score)
          2. top_score >= threshold
          3. top_score - second_score >= dominance_margin

        If clearly dominant → IDENTIFIED with the top candidate.
        Otherwise → AMBIGUOUS (all candidates shown to user, none auto-selected).
        """
        # Sort by score descending (put score_unavailable candidates last)
        def sort_key(c: IdentificationCandidate) -> float:
            if c.score_available and c.score is not None:
                return c.score
            return -1.0  # No score → treated as worst

        sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
        top = sorted_candidates[0]
        second = sorted_candidates[1] if len(sorted_candidates) > 1 else None

        top_score = top.score if (top.score_available and top.score is not None) else None
        second_score = second.score if (second and second.score_available and second.score is not None) else 0.0

        # Check all three dominance conditions
        if (
            top.score_available
            and top_score is not None
            and top_score >= self.threshold
            and (top_score - second_score) >= self.dominance_margin
        ):
            logger.info(
                f"[resolver] Clearly dominant candidate identified: "
                f"top_score={top_score:.3f} >= threshold={self.threshold}, "
                f"gap={top_score - second_score:.3f} >= margin={self.dominance_margin}"
            )
            return self._make_identified(top, raw)

        # Not clearly dominant → ambiguous
        gap = top_score - second_score if top_score is not None else None
        gap_str = f"{gap:.3f}" if gap is not None else "n/a"
        logger.info(
            f"[resolver] Ambiguous: {len(candidates)} candidates, "
            f"top_score={top_score}, gap={gap_str} (margin={self.dominance_margin})"
        )
        return ResolvedIdentification(
            status=IdentificationStatus.AMBIGUOUS,
            provider=raw.provider,
            candidates=raw.candidates,
            attribution=raw.attribution,
            intelligence_note=(
                f"{len(candidates)} possible matches were found but none is "
                "sufficiently dominant to confirm. The candidates are listed below "
                "for reference. None has been automatically selected."
            ),
        )

    # ─── Result constructors ──────────────────────────────────────────────────

    @staticmethod
    def _make_identified(
        candidate: IdentificationCandidate,
        raw: RawIdentificationResult,
    ) -> ResolvedIdentification:
        """Build a ResolvedIdentification for the IDENTIFIED status."""
        return ResolvedIdentification(
            status=IdentificationStatus.IDENTIFIED,
            provider=raw.provider,
            title=candidate.title,
            artist=candidate.artist,
            album=candidate.album,
            release_year=candidate.release_year,
            confidence=candidate.score,
            confidence_available=candidate.score_available,
            candidates=raw.candidates,
            acoustid=candidate.acoustid,
            mbid=candidate.mbid,
            provider_track_id=candidate.provider_track_id,
            attribution=raw.attribution,
            intelligence_note=None,
        )

    @staticmethod
    def _make_unavailable(
        provider: ProviderName,
        error: str,
        candidates: list[IdentificationCandidate] | None = None,
    ) -> ResolvedIdentification:
        """Build a ResolvedIdentification for the UNAVAILABLE status."""
        return ResolvedIdentification(
            status=IdentificationStatus.UNAVAILABLE,
            provider=provider,
            candidates=candidates or [],
            error=error,
            intelligence_note=(
                "Song identification is currently unavailable. "
                "This does not affect your DSP analysis results."
            ),
        )
