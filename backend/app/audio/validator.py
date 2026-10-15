"""
SwarSense — Audio File Validator
==================================
Validates uploaded audio files before processing.

Checks:
  1. File extension is .wav or .mp3
  2. MIME type matches the extension (guards against renamed files)
  3. File is not empty
  4. File is not obviously corrupted (basic header check)

Phase 0: Extension-based validation only (python-magic optional).
Phase 1: Full validation integrated into the audio loading pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".wav", ".mp3"}
ALLOWED_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"}

# Magic bytes for format identification (fallback when python-magic is unavailable)
MAGIC_BYTES = {
    "wav": b"RIFF",          # RIFF....WAVEfmt
    "mp3_id3": b"\x49\x44\x33",  # ID3 tag
    "mp3_sync": b"\xff\xfb",      # MPEG sync word
    "mp3_sync2": b"\xff\xfa",
    "mp3_sync3": b"\xff\xf3",
}


@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None
    detected_format: Optional[str] = None


class AudioValidator:
    """
    Validates audio file uploads.

    Uses a two-layer approach:
    1. Extension check (fast)
    2. Magic byte check (reliable — prevents extension spoofing)

    Falls back gracefully if python-magic binary is not available.
    """

    def validate(self, filename: str, content: bytes) -> ValidationResult:
        """
        Validate an uploaded audio file.

        Args:
            filename: Original filename (used for extension check).
            content:  Raw file bytes.

        Returns:
            ValidationResult with is_valid flag and error message if invalid.
        """
        if not content:
            return ValidationResult(is_valid=False, error="Uploaded file is empty.")

        # ── Extension check ────────────────────────────────────────────────────
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                error=(
                    f"Unsupported file extension '{ext}'. "
                    f"Accepted formats: {', '.join(ALLOWED_EXTENSIONS)}."
                ),
            )

        # ── Magic byte check ───────────────────────────────────────────────────
        detected = self._detect_format(content)
        if detected is None:
            logger.warning(
                f"File '{filename}' has extension '{ext}' but unrecognised magic bytes. "
                "Proceeding — may fail during audio loading."
            )
            # Allow through with a warning; Librosa will give a clearer error at load time.
            return ValidationResult(
                is_valid=True,
                detected_format="unknown",
            )

        # Cross-check extension vs magic bytes
        if ext == ".wav" and detected != "wav":
            return ValidationResult(
                is_valid=False,
                error=f"File has .wav extension but appears to be {detected} format.",
            )
        if ext == ".mp3" and detected not in ("mp3", "id3"):
            return ValidationResult(
                is_valid=False,
                error=f"File has .mp3 extension but appears to be {detected} format.",
            )

        logger.debug(f"File '{filename}' validated. Detected format: {detected}.")
        return ValidationResult(is_valid=True, detected_format=detected)

    def _detect_format(self, content: bytes) -> Optional[str]:
        """
        Detect audio format from magic bytes.
        Returns 'wav', 'mp3', 'id3', or None if unrecognised.
        """
        if len(content) < 4:
            return None
        header = content[:4]

        if header[:4] == MAGIC_BYTES["wav"]:
            return "wav"
        if header[:3] == MAGIC_BYTES["mp3_id3"]:
            return "id3"   # MP3 with ID3 metadata tag
        if header[:2] in (
            MAGIC_BYTES["mp3_sync"],
            MAGIC_BYTES["mp3_sync2"],
            MAGIC_BYTES["mp3_sync3"],
        ):
            return "mp3"

        return None
