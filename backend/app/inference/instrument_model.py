"""
SwarSense — Instrument Detection (Phase 3 Stub)
================================================
Phase 3: Integrate YAMNet (TensorFlow Hub) for Western instrument detection.

IMPORTANT NOTE:
YAMNet is trained on AudioSet (521 classes) and does NOT reliably include
Tabla or Harmonium as distinct instrument classes. Using it for Indian
instrument detection would produce unreliable results.

System scope:
- YAMNet: Western instruments (guitar, piano, violin, drums, flute, etc.)
- Indian instruments (Tabla, Harmonium): documented research gap.
  Will attempt fine-tuned detection only if labeled data is available.
"""

from app.utils.logger import get_logger

logger = get_logger(__name__)


class InstrumentModel:
    """
    Placeholder for YAMNet-based instrument detection.
    Phase 3.
    """

    def __init__(self):
        self.model = None
        self.is_loaded = False
        logger.info("InstrumentModel: Phase 3 stub — not yet implemented.")

    def load(self) -> None:
        """Load YAMNet from TensorFlow Hub. Phase 3."""
        raise NotImplementedError(
            "Instrument model will be implemented in Phase 3. "
            "Requires: pip install tensorflow tensorflow-hub"
        )

    def predict(self, audio_segment) -> list[dict]:
        """
        Detect instruments in a short audio segment.
        Returns list of {instrument: str, confidence: float}.
        Phase 3.
        """
        raise NotImplementedError("Instrument detection: Phase 3.")
