"""
SwarSense — Source Separation (Phase 3 Stub)
=============================================
Phase 3: Integrate Demucs (htdemucs) for vocal/instrumental separation.

PERFORMANCE NOTE:
Demucs on CPU is 1.5–5× slower than real-time.
A 3-minute file may take 5–15 minutes on CPU without GPU acceleration.

System decision:
  Apply source separation only to the first 30 seconds of each audio file.
  Document this limitation clearly in the UI.

Stems produced by htdemucs:
  - vocals
  - drums
  - bass
  - other (covers harmonium, strings, etc.)

We report:
  - vocal_energy_percent: energy of 'vocals' stem / total energy
  - instrumental_energy_percent: energy of remaining stems / total energy
"""

from app.utils.logger import get_logger

logger = get_logger(__name__)

ANALYSIS_EXCERPT_SECONDS = 30.0


class SeparationModel:
    """
    Placeholder for Demucs source separation.
    Phase 3.
    """

    def __init__(self):
        self.model = None
        self.is_loaded = False
        logger.info("SeparationModel: Phase 3 stub — not yet implemented.")

    def load(self) -> None:
        """Load htdemucs model. Phase 3."""
        raise NotImplementedError(
            "Demucs will be integrated in Phase 3. "
            "Install: pip install demucs"
        )

    def separate(self, audio_excerpt) -> dict:
        """
        Separate a 30s audio excerpt into stems.
        Returns energy percentages.
        Phase 3.
        """
        raise NotImplementedError("Source separation: Phase 3.")
