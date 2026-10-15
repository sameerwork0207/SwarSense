/**
 * SwarSense API Client (Phase 1)
 * ================================
 * Typed wrapper around the SwarSense FastAPI backend.
 * All API calls go through this module.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─── Enums & Types ────────────────────────────────────────────────────────────

export type JobStatus = "pending" | "processing" | "completed" | "failed" | "partial";
export type AnalysisSource = "dsp" | "heuristic" | "pretrained" | "trained_model" | "static";

// ─── Response Types ───────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  phase: string;
  models_loaded: boolean;
}

export interface AnalyzeResponse {
  job_id: string;
  status: JobStatus;
  message: string;
  filename: string;
  estimated_duration_seconds: number | null;
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  progress: number;
  message: string;
  partial_result: Record<string, unknown> | null;
  error: { error: string; message: string; detail?: string } | null;
}

// ─── Phase 1 Feature Result Types ─────────────────────────────────────────────

export interface AudioMetadata {
  source: AnalysisSource;
  duration_seconds: number;
  sample_rate: number;
  original_sample_rate: number;
  channels: number;
  n_samples: number;
  file_format: string;
  normalized: boolean;
}

export interface LoudnessResult {
  source: AnalysisSource;
  lufs: number | null;
  rms_db: number;
  rms_linear: number;
  method: string;
}

export interface TempoResult {
  source: AnalysisSource;
  bpm: number;
  beat_times_seconds: number[];
  n_beats: number;
  analysis_scope: string;
  method: string;
  caveat: string;
}

export interface PitchResult {
  source: AnalysisSource;
  f0_hz: (number | null)[];
  times_seconds: number[];
  voiced_flag: boolean[];
  voiced_ratio: number;
  mean_f0_hz: number | null;
  std_f0_hz: number | null;
  min_f0_hz: number | null;
  max_f0_hz: number | null;
  dominant_midi: number | null;
  dominant_note: string | null;
  analysis_scope: string;
  method: string;
  caveat: string;
}

export interface MFCCResult {
  source: AnalysisSource;
  n_coefficients: number;
  mean: number[];
  std: number[];
  analysis_scope: string;
}

export interface ChromaResult {
  source: AnalysisSource;
  pitch_classes: string[];
  mean: number[];
  std: number[];
  analysis_scope: string;
}

export interface SpectralResult {
  source: AnalysisSource;
  centroid_mean_hz: number;
  centroid_std_hz: number;
  bandwidth_mean_hz: number;
  rolloff_mean_hz: number;
  zero_crossing_rate_mean: number;
  zero_crossing_rate_std: number;
}

export interface HPSSResult {
  source: AnalysisSource;
  harmonic_energy_ratio: number;
  percussive_energy_ratio: number;
  method: string;
  note: string;
}

export interface EstimatedTonic {
  source: AnalysisSource;
  estimated_note: string | null;
  chroma_energy: number[];
  dominance_ratio: number | null;
  method: string;
  caveat: string;
  analysis_scope: string;
}

// ─── Phase 2: Song Intelligence Types ─────────────────────────────────────────
// External metadata only — never a SwarSense prediction. Distinct from
// raga_prediction, which is SwarSense's own (future) model output.

export type IdentificationStatus = "identified" | "ambiguous" | "not_identified" | "unavailable";

export interface IdentificationCandidate {
  title: string | null;
  artist: string | null;
  album: string | null;
  release_year: number | null;
  score: number | null;
  score_available: boolean;
  acoustid: string | null;
  mbid: string | null;
  provider_track_id: string | null;
  isrc: string | null;
}

export interface SongIdentificationResult {
  status: IdentificationStatus;
  provider: string | null;
  title: string | null;
  artist: string | null;
  album: string | null;
  release_year: number | null;
  confidence: number | null;
  confidence_available: boolean;
  candidates: IdentificationCandidate[];
  mbid: string | null;
  attribution: string | null;
  error: string | null;
  intelligence_note: string | null;
}

export interface EnrichmentFact {
  field: string;
  value: string;
}

export interface EnrichmentSource {
  source_name: string;
  url: string | null;
  disclaimer: string;
  facts: EnrichmentFact[];
}

export interface SongIntelligenceResult {
  identification: SongIdentificationResult;
  enrichment_sources: EnrichmentSource[];
  enrichment_error: string | null;
}

export interface VisualizationLinks {
  waveform: string | null;
  spectrogram: string | null;
  mel_spectrogram: string | null;
  chromagram: string | null;
  pitch: string | null;
}

export interface AnalysisResult {
  job_id: string;
  status: JobStatus;
  filename: string;

  // Phase 1 DSP results
  audio: AudioMetadata | null;
  loudness: LoudnessResult | null;
  tempo: TempoResult | null;
  pitch: PitchResult | null;
  mfcc: MFCCResult | null;
  chroma: ChromaResult | null;
  spectral: SpectralResult | null;
  hpss: HPSSResult | null;
  estimated_tonic: EstimatedTonic | null;
  segment_features: unknown[];
  visualizations: VisualizationLinks | null;

  // Phase 3/4/5 stubs
  raga_prediction: {
    source: string;
    status: string;
    predicted_raga: string | null;
    confidence: number | null;
    note: string;
  };
  source_separation: {
    status: string;
    vocal_energy_percent: number | null;
    instrumental_energy_percent: number | null;
    note: string;
  };
  instrument_detection: {
    status: string;
    instruments: { instrument: string; confidence: number; source: string }[];
    note: string;
  };

  module_errors: Record<string, string>;

  // Phase 2 — optional, external metadata only
  song_identification: SongIntelligenceResult | null;
}

// ─── API Functions ────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function submitAnalysis(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const detail = err?.detail;
    const message = typeof detail === "object" ? detail?.message : typeof detail === "string" ? detail : `Upload failed: ${res.status}`;
    throw new Error(message);
  }

  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/api/status/${jobId}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
  return res.json();
}

export async function getResult(jobId: string): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/api/result/${jobId}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Result fetch failed: ${res.status}`);
  return res.json();
}

/**
 * Returns the full URL for a visualization chart image.
 * The browser can use this directly in <img src="..." />.
 */
export function getVisualizationUrl(jobId: string, chart: "waveform" | "spectrogram" | "mel-spectrogram" | "chromagram" | "pitch"): string {
  return `${API_BASE}/api/result/${jobId}/${chart}`;
}
