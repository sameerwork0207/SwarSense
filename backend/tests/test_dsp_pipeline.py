"""
SwarSense — Phase 1 DSP Pipeline Tests
=========================================
Unit and integration tests for all Phase 1 DSP modules.

Tests cover:
  - Audio loading (WAV, stereo→mono conversion, resampling)
  - Preprocessing (normalization, windowing)
  - Loudness measurement (LUFS, RMS)
  - Tempo estimation (BPM, beat detection)
  - Pitch estimation (F0, voiced ratio)
  - Feature extraction (MFCC, Chroma, Spectral, HPSS)
  - Tonic estimation (heuristic)
  - Visualization generation (all 5 charts)
  - Full pipeline integration (load→preprocess→extract→visualize)

Test philosophy:
  - All tests use synthetic WAV files generated in conftest.py.
  - No large audio files are committed to the repository.
  - Tests check data types, value ranges, and structural correctness.
  - Tests do NOT assert exact DSP values (those vary by algorithm version).
"""

import numpy as np
import pytest


# ─── Audio Loader Tests ────────────────────────────────────────────────────────


class TestAudioLoader:
    def test_load_valid_wav(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        loader = AudioLoader()
        result = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        assert result.sample_rate == 22050
        assert result.duration_seconds > 0
        assert result.samples.dtype == np.float32
        assert result.samples.ndim == 1  # Mono

    def test_mono_conversion_from_stereo(self, synthetic_wav_bytes_stereo):
        from app.audio.loader import AudioLoader
        loader = AudioLoader()
        result = loader.load_bytes(synthetic_wav_bytes_stereo, "stereo.wav")
        assert result.samples.ndim == 1  # Should be mono
        assert result.original_channels == 2

    def test_resampling_to_target_sr(self, synthetic_wav_bytes_stereo):
        """Stereo WAV is at 44100 Hz — loader should resample to 22050."""
        from app.audio.loader import AudioLoader
        loader = AudioLoader(target_sr=22050)
        result = loader.load_bytes(synthetic_wav_bytes_stereo, "stereo.wav")
        assert result.sample_rate == 22050
        assert result.original_sr == 44100

    def test_duration_approximately_correct(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        loader = AudioLoader()
        result = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        assert abs(result.duration_seconds - 3.0) < 0.1  # ~3 seconds

    def test_rejects_empty_content(self):
        from app.audio.loader import AudioLoader, AudioLoadError
        loader = AudioLoader()
        with pytest.raises(AudioLoadError):
            loader.load_bytes(b"", "empty.wav")

    def test_rejects_invalid_bytes(self, invalid_file_bytes):
        from app.audio.loader import AudioLoader, AudioLoadError
        loader = AudioLoader()
        with pytest.raises(AudioLoadError):
            loader.load_bytes(invalid_file_bytes, "noise.wav")

    def test_rejects_too_long_audio(self):
        """Generate a 601-second WAV (just over 10-minute limit)."""
        import io
        import scipy.io.wavfile as wav_io
        duration = 601
        sr = 8000  # Low SR to keep memory manageable in tests
        t = np.linspace(0, duration, sr * duration, endpoint=False)
        samples = (0.1 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        buf = io.BytesIO()
        wav_io.write(buf, sr, samples)
        buf.seek(0)
        content = buf.read()

        from app.audio.loader import AudioLoader, AudioDurationError
        loader = AudioLoader(max_duration_s=600.0)
        with pytest.raises(AudioDurationError):
            loader.load_bytes(content, "too_long.wav")


# ─── Preprocessing Tests ──────────────────────────────────────────────────────


class TestPreprocessing:
    def test_normalize_amplitude_peak(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import normalize_amplitude
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        normalized = normalize_amplitude(loaded.samples, method="peak")
        assert np.max(np.abs(normalized)) <= 1.0 + 1e-6
        assert abs(np.max(np.abs(normalized)) - 1.0) < 0.01

    def test_windowing_short_audio_no_windows(self, synthetic_wav_bytes):
        """Short audio (3s) should produce no windows."""
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        assert len(preprocessed.windows) == 0

    def test_windowing_long_audio_has_windows(self, synthetic_wav_bytes_long):
        """Long audio (90s) should produce sliding windows."""
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes_long, "long.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        assert len(preprocessed.windows) > 0
        # Windows should not replace the global signal
        assert preprocessed.duration_s > 60

    def test_global_audio_preserved_in_windowed_result(self, synthetic_wav_bytes_long):
        """Full audio signal should be preserved regardless of windowing."""
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes_long, "long.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        # Global signal duration should match original
        assert abs(preprocessed.duration_s - loaded.duration_seconds) < 0.1

    def test_window_parameters_correct(self, synthetic_wav_bytes_long):
        """Windows should be ~30s each, except possibly the last (tail) window."""
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess, WINDOW_SIZE_S
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes_long, "long.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        # All windows except the last (tail) should be exactly WINDOW_SIZE_S long.
        # The last window may be shorter if audio length is not a multiple of hop size.
        non_tail_windows = preprocessed.windows[:-1]
        for win in non_tail_windows:
            assert abs((win.end_seconds - win.start_seconds) - WINDOW_SIZE_S) < 1.0


# ─── Loudness Tests ───────────────────────────────────────────────────────────


class TestLoudness:
    def test_rms_is_negative_db(self, synthetic_wav_bytes):
        """RMS dB should be negative for any signal with amplitude < 1."""
        from app.audio.loader import AudioLoader
        from app.audio.loudness import measure_loudness
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        result = measure_loudness(loaded.samples, loaded.sample_rate)
        assert result.rms_db < 0

    def test_rms_linear_positive(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.loudness import measure_loudness
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        result = measure_loudness(loaded.samples, loaded.sample_rate)
        assert result.rms_linear > 0

    def test_lufs_is_reasonable_range(self, synthetic_wav_bytes):
        """LUFS for a sine wave should be in a typical range if pyloudnorm works."""
        from app.audio.loader import AudioLoader
        from app.audio.loudness import measure_loudness
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        result = measure_loudness(loaded.samples, loaded.sample_rate)
        if result.lufs is not None:
            # Valid LUFS range: roughly -60 to 0
            assert -60 < result.lufs < 0


# ─── Tempo Tests ──────────────────────────────────────────────────────────────


class TestTempo:
    def test_bpm_is_positive(self, synthetic_wav_bytes):
        """BPM should be >= 0. Pure sine waves may return 0.0 — valid DSP behavior."""
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.tempo import estimate_tempo
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_tempo(preprocessed.samples, preprocessed.sample_rate)
        # librosa may return 0.0 for pure tones with no transients — that is correct
        assert result.bpm >= 0

    def test_beat_times_are_valid(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.tempo import estimate_tempo
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_tempo(preprocessed.samples, preprocessed.sample_rate)
        for bt in result.beat_times_s:
            assert 0 <= bt <= loaded.duration_seconds + 1.0

    def test_analysis_scope_is_global(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.tempo import estimate_tempo
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_tempo(preprocessed.samples, preprocessed.sample_rate)
        assert result.analysis_scope == "global"


# ─── Pitch Tests ──────────────────────────────────────────────────────────────


class TestPitch:
    def test_f0_list_has_correct_length(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.pitch import estimate_pitch
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_pitch(preprocessed.samples, preprocessed.sample_rate)
        assert len(result.f0_hz) == len(result.times_s)
        assert len(result.f0_hz) == len(result.voiced_flag)

    def test_voiced_ratio_in_range(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.pitch import estimate_pitch
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_pitch(preprocessed.samples, preprocessed.sample_rate)
        assert 0.0 <= result.voiced_ratio <= 1.0

    def test_dominant_note_is_string_or_none(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.pitch import estimate_pitch
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_pitch(preprocessed.samples, preprocessed.sample_rate)
        if result.dominant_note is not None:
            assert isinstance(result.dominant_note, str)
            assert len(result.dominant_note) >= 2  # e.g. "A4"

    def test_silent_audio_does_not_raise(self):
        """
        Regression test: fully silent audio produces zero voiced frames, so
        mean_f0/std_f0/min_f0/max_f0 are all None. A completion-logging
        f-string previously used an unguarded `:.1f` format spec on
        mean_f0, raising 'unsupported format string passed to
        NoneType.__format__' and causing the whole pitch module to fail
        (and the job to be marked 'partial') even though the computation
        itself was otherwise valid.
        """
        import numpy as np
        from app.audio.pitch import estimate_pitch

        sr = 22050
        silent_samples = np.zeros(sr * 3, dtype=np.float32)  # 3s of digital silence

        result = estimate_pitch(silent_samples, sr)

        assert result.voiced_ratio == 0.0
        assert result.mean_f0_hz is None
        assert result.dominant_note is None


# ─── Feature Extraction Tests ─────────────────────────────────────────────────


class TestFeatureExtraction:
    def test_mfcc_shape(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.features import extract_features, N_MFCC
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = extract_features(preprocessed)
        assert len(result.mfcc.mean) == N_MFCC
        assert len(result.mfcc.std) == N_MFCC

    def test_chroma_shape(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.features import extract_features
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = extract_features(preprocessed)
        assert len(result.chroma.mean) == 12
        assert len(result.chroma.std) == 12
        assert result.chroma.pitch_classes[0] == "C"
        assert result.chroma.pitch_classes[-1] == "B"

    def test_spectral_values_positive(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.features import extract_features
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = extract_features(preprocessed)
        assert result.spectral.centroid_mean_hz > 0
        assert result.spectral.bandwidth_mean_hz > 0
        assert result.spectral.rolloff_mean_hz > 0
        assert result.spectral.zero_crossing_rate_mean >= 0

    def test_hpss_ratios_in_range(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.features import extract_features
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = extract_features(preprocessed)
        assert 0.0 <= result.hpss.harmonic_energy_ratio <= 1.0
        assert 0.0 <= result.hpss.percussive_energy_ratio <= 1.0

    def test_mel_spectrogram_shape(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.features import extract_features, N_MELS
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = extract_features(preprocessed)
        assert result.mel_spectrogram_db is not None
        assert result.mel_spectrogram_db.shape[0] == N_MELS

    def test_long_audio_produces_segment_features(self, synthetic_wav_bytes_long):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.features import extract_features
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes_long, "long.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = extract_features(preprocessed)
        assert len(result.segment_features) > 0

    def test_short_audio_no_segment_features(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.features import extract_features
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = extract_features(preprocessed)
        assert len(result.segment_features) == 0


# ─── Tonic Estimation Tests ───────────────────────────────────────────────────


class TestTonicEstimation:
    def test_estimated_note_is_valid_pitch_class(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.tonic import estimate_tonic, PITCH_CLASSES
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_tonic(preprocessed.samples, preprocessed.sample_rate)
        assert result.estimated_note in PITCH_CLASSES

    def test_chroma_energy_has_12_bins(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.tonic import estimate_tonic
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_tonic(preprocessed.samples, preprocessed.sample_rate)
        assert len(result.chroma_energy) == 12

    def test_caveat_string_present(self, synthetic_wav_bytes):
        """The tonic result must include a caveat about estimation reliability."""
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.tonic import estimate_tonic
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_tonic(preprocessed.samples, preprocessed.sample_rate)
        assert len(result.caveat) > 20  # Must not be empty or trivial

    def test_analysis_scope_is_global(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.tonic import estimate_tonic
        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        result = estimate_tonic(preprocessed.samples, preprocessed.sample_rate)
        assert result.analysis_scope == "global"


# ─── Visualization Tests ──────────────────────────────────────────────────────


class TestVisualization:
    @pytest.fixture(autouse=True)
    def load_audio(self, synthetic_wav_bytes):
        from app.audio.loader import AudioLoader
        from app.audio.preprocessing import preprocess
        from app.audio.features import extract_features
        from app.audio.tempo import estimate_tempo
        from app.audio.pitch import estimate_pitch

        loader = AudioLoader()
        loaded = loader.load_bytes(synthetic_wav_bytes, "test.wav")
        preprocessed = preprocess(loaded.samples, loaded.sample_rate,
                                  loaded.original_sr, loaded.original_channels)
        self.samples = preprocessed.samples
        self.sr = preprocessed.sample_rate
        self.duration_s = preprocessed.duration_s
        self.features = extract_features(preprocessed)
        self.tempo = estimate_tempo(preprocessed.samples, preprocessed.sample_rate)
        self.pitch = estimate_pitch(preprocessed.samples, preprocessed.sample_rate)

    def test_waveform_returns_png_bytes(self):
        from app.audio.visualization import plot_waveform
        png = plot_waveform(self.samples, self.sr, self.duration_s, self.tempo.beat_times_s)
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_spectrogram_returns_png_bytes(self):
        from app.audio.visualization import plot_spectrogram
        from app.audio.features import HOP_LENGTH
        png = plot_spectrogram(self.features.stft_db, self.sr, HOP_LENGTH, self.duration_s)
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_mel_spectrogram_returns_png_bytes(self):
        from app.audio.visualization import plot_mel_spectrogram
        from app.audio.features import HOP_LENGTH
        png = plot_mel_spectrogram(self.features.mel_spectrogram_db, self.sr, HOP_LENGTH, self.duration_s)
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_chromagram_returns_png_bytes(self):
        from app.audio.visualization import plot_chromagram
        from app.audio.features import HOP_LENGTH
        png = plot_chromagram(self.features.chroma_matrix, self.sr, HOP_LENGTH, self.duration_s)
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_pitch_contour_returns_png_bytes(self):
        from app.audio.visualization import plot_pitch_contour
        png = plot_pitch_contour(
            self.pitch.f0_hz, self.pitch.times_s, self.pitch.voiced_flag, self.duration_s
        )
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_generate_all_returns_bundle(self):
        from app.audio.visualization import generate_all_visualizations
        from app.audio.features import HOP_LENGTH
        bundle = generate_all_visualizations(
            samples=self.samples,
            sample_rate=self.sr,
            duration_s=self.duration_s,
            stft_db=self.features.stft_db,
            mel_db=self.features.mel_spectrogram_db,
            chroma_matrix=self.features.chroma_matrix,
            hop_length=HOP_LENGTH,
            beat_times_s=self.tempo.beat_times_s,
            f0_hz=self.pitch.f0_hz,
            f0_times_s=self.pitch.times_s,
            voiced_flag=self.pitch.voiced_flag,
        )
        assert bundle.waveform is not None
        assert bundle.spectrogram is not None
        assert bundle.mel_spectrogram is not None
        assert bundle.chromagram is not None
        assert bundle.pitch is not None
