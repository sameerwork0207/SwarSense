"""
SwarSense — Tests: Audio Validator
====================================
Unit tests for AudioValidator magic-byte and extension checks.
"""

import pytest

from app.audio.validator import AudioValidator, ValidationResult

validator = AudioValidator()


def make_wav_bytes(minimal: bool = True) -> bytes:
    """Return minimal valid WAV header bytes."""
    return (
        b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")    # PCM
        + (1).to_bytes(2, "little")    # mono
        + (22050).to_bytes(4, "little")
        + (44100).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data" + (0).to_bytes(4, "little")
    )


def make_mp3_id3_bytes() -> bytes:
    """Return minimal ID3-tagged MP3 header."""
    return b"\x49\x44\x33\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 100


def make_mp3_sync_bytes() -> bytes:
    """Return minimal MPEG sync-word MP3 header."""
    return b"\xff\xfb\x90\x00" + b"\x00" * 100


class TestAudioValidatorExtensions:
    def test_valid_wav_extension(self):
        result = validator.validate("song.wav", make_wav_bytes())
        assert result.is_valid

    def test_valid_mp3_id3(self):
        result = validator.validate("song.mp3", make_mp3_id3_bytes())
        assert result.is_valid

    def test_valid_mp3_sync(self):
        result = validator.validate("song.mp3", make_mp3_sync_bytes())
        assert result.is_valid

    def test_rejects_txt_extension(self):
        result = validator.validate("notes.txt", b"Some text")
        assert not result.is_valid
        assert "txt" in result.error

    def test_rejects_pdf_extension(self):
        result = validator.validate("report.pdf", b"%PDF-1.4")
        assert not result.is_valid

    def test_rejects_empty_file(self):
        result = validator.validate("song.wav", b"")
        assert not result.is_valid
        assert "empty" in result.error.lower()

    def test_rejects_wav_extension_with_mp3_content(self):
        """File named .wav but containing MP3 data should be rejected."""
        result = validator.validate("fake.wav", make_mp3_id3_bytes())
        assert not result.is_valid

    def test_rejects_mp3_extension_with_wav_content(self):
        """File named .mp3 but containing WAV data should be rejected."""
        result = validator.validate("fake.mp3", make_wav_bytes())
        assert not result.is_valid

    def test_uppercase_extension_normalised(self):
        """Extensions should be case-insensitive."""
        result = validator.validate("SONG.WAV", make_wav_bytes())
        assert result.is_valid

    def test_detected_format_returned(self):
        result = validator.validate("song.wav", make_wav_bytes())
        assert result.detected_format == "wav"
