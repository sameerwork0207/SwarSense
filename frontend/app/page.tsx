"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AnalysisResult,
  checkHealth,
  getJobStatus,
  getResult,
  getVisualizationUrl,
  submitAnalysis,
  JobStatusResponse,
  SongIntelligenceResult,
} from "../lib/api";

// ─── Types ───────────────────────────────────────────────────────────────────

type AppState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "analyzing"; jobId: string; progress: number; message: string }
  | { phase: "completed"; jobId: string; result: AnalysisResult }
  | { phase: "error"; message: string };

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s.toFixed(1)}s`;
}

function MetricCard({
  label,
  value,
  sub,
  highlight,
  warning,
}: {
  label: string;
  value: string;
  sub?: string;
  highlight?: boolean;
  warning?: boolean;
}) {
  return (
    <div
      className={`rounded-xl p-4 border ${highlight ? "border-violet-500/60 bg-violet-500/10" : warning ? "border-amber-500/40 bg-amber-500/5" : "border-white/10 bg-white/5"}`}
    >
      <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-xl font-bold ${highlight ? "text-violet-300" : warning ? "text-amber-300" : "text-slate-100"}`}>
        {value}
      </p>
      {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
    </div>
  );
}

function SectionHeader({ title, badge }: { title: string; badge?: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
      {badge && (
        <span className="text-xs px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30">
          {badge}
        </span>
      )}
    </div>
  );
}

function VizImage({
  jobId,
  chart,
  label,
}: {
  jobId: string;
  chart: "waveform" | "spectrogram" | "mel-spectrogram" | "chromagram" | "pitch";
  label: string;
}) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);
  const url = getVisualizationUrl(jobId, chart);

  return (
    <div className="rounded-xl overflow-hidden border border-white/10 bg-[#0f0f1a]">
      <div className="px-4 py-2.5 border-b border-white/10 flex items-center justify-between">
        <span className="text-sm font-medium text-slate-300">{label}</span>
        {!error && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-violet-400 hover:text-violet-300 transition-colors"
          >
            Open ↗
          </a>
        )}
      </div>
      {error ? (
        <div className="flex items-center justify-center h-32 text-slate-500 text-sm">
          Chart unavailable
        </div>
      ) : (
        <div className="relative">
          {!loaded && (
            <div className="absolute inset-0 flex items-center justify-center h-40 text-slate-600 text-sm">
              Loading...
            </div>
          )}
          <img
            src={url}
            alt={label}
            className={`w-full transition-opacity duration-300 ${loaded ? "opacity-100" : "opacity-0"}`}
            onLoad={() => setLoaded(true)}
            onError={() => setError(true)}
          />
        </div>
      )}
    </div>
  );
}

function ChromaBar({ values, labels }: { values: number[]; labels: string[] }) {
  const max = Math.max(...values);
  return (
    <div className="flex gap-1 items-end h-20">
      {values.map((v, i) => (
        <div key={labels[i]} className="flex flex-col items-center gap-1 flex-1">
          <div
            className="w-full rounded-t bg-gradient-to-t from-violet-600 to-violet-400 transition-all duration-500"
            style={{ height: `${(v / max) * 64}px`, minHeight: 2 }}
          />
          <span className="text-xs text-slate-500">{labels[i]}</span>
        </div>
      ))}
    </div>
  );
}

function ProgressBar({ progress, message }: { progress: number; message: string }) {
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-slate-400 mb-2">
        <span>{message}</span>
        <span>{progress}%</span>
      </div>
      <div className="w-full bg-white/10 rounded-full h-2">
        <div
          className="h-2 rounded-full bg-gradient-to-r from-violet-500 to-cyan-500 transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

function SongIdentificationCard({ intel }: { intel: SongIntelligenceResult | null }) {
  if (!intel) return null;
  const { identification, enrichment_sources, enrichment_error } = intel;
  const { status } = identification;

  const statusMeta: Record<
    typeof status,
    { label: string; color: string; icon: string }
  > = {
    identified: { label: "Identified", color: "border-emerald-500/40 bg-emerald-500/5 text-emerald-300", icon: "✓" },
    ambiguous: { label: "Ambiguous", color: "border-amber-500/40 bg-amber-500/5 text-amber-300", icon: "?" },
    not_identified: { label: "Not Identified", color: "border-white/10 bg-white/5 text-slate-300", icon: "–" },
    unavailable: { label: "Unavailable", color: "border-slate-500/30 bg-slate-500/5 text-slate-400", icon: "⦸" },
  };
  const meta = statusMeta[status];

  return (
    <div>
      <SectionHeader title="Song Identification" badge="External metadata · Phase 2" />
      <div className={`rounded-xl border p-5 ${meta.color}`}>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-lg">{meta.icon}</span>
          <span className="text-xs font-semibold uppercase tracking-wider">{meta.label}</span>
          {identification.provider && identification.provider !== "null" && (
            <span className="text-xs text-slate-500">via {identification.provider}</span>
          )}
        </div>

        {status === "identified" && (
          <div className="mb-3">
            <p className="text-2xl font-bold">{identification.title ?? "Unknown title"}</p>
            {identification.artist && <p className="text-sm text-slate-300 mt-0.5">{identification.artist}</p>}
            {identification.album && <p className="text-xs text-slate-500">{identification.album}{identification.release_year ? ` · ${identification.release_year}` : ""}</p>}
            {identification.confidence_available && identification.confidence != null && (
              <p className="text-xs text-slate-500 mt-1">Confidence: {(identification.confidence * 100).toFixed(0)}%</p>
            )}
          </div>
        )}

        {identification.intelligence_note && (
          <p className="text-xs text-slate-400">{identification.intelligence_note}</p>
        )}
        {identification.error && status === "unavailable" && (
          <p className="text-xs text-slate-500 mt-1">{identification.error}</p>
        )}
        {identification.attribution && (
          <p className="text-xs text-slate-600 mt-2 italic">{identification.attribution}</p>
        )}

        {(status === "ambiguous" || (status === "not_identified" && identification.candidates.length > 0)) && (
          <div className="mt-3 space-y-2">
            {identification.candidates.map((c, i) => (
              <div key={i} className="rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-xs">
                <span className="text-slate-200 font-medium">{c.title ?? "Untitled"}</span>
                {c.artist && <span className="text-slate-400"> — {c.artist}</span>}
                {c.score_available && c.score != null && (
                  <span className="text-slate-500"> · {(c.score * 100).toFixed(0)}%</span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* MusicBrainz enrichment — always shown with its disclaimer */}
        {enrichment_sources.length > 0 && (
          <div className="mt-4 pt-3 border-t border-white/10 space-y-3">
            {enrichment_sources.map((src, i) => (
              <div key={i}>
                <p className="text-xs text-slate-500 italic mb-1.5">{src.disclaimer}</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                  {src.facts.map((f, j) => (
                    <div key={j} className="text-xs">
                      <span className="text-slate-500">{f.field}: </span>
                      <span className="text-slate-300">{f.value}</span>
                    </div>
                  ))}
                </div>
                {src.url && (
                  <a href={src.url} target="_blank" rel="noopener noreferrer" className="text-xs text-violet-400 hover:text-violet-300 mt-1 inline-block">
                    Source: {src.source_name} ↗
                  </a>
                )}
              </div>
            ))}
          </div>
        )}
        {enrichment_error && (
          <p className="text-xs text-slate-600 mt-2">Enrichment unavailable: {enrichment_error}</p>
        )}
      </div>
    </div>
  );
}

// ─── Main Dashboard Component ─────────────────────────────────────────────────

function ResultsDashboard({ jobId, result }: { jobId: string; result: AnalysisResult }) {
  const { audio, tempo, loudness, pitch, mfcc, chroma, spectral, hpss, estimated_tonic, module_errors, song_identification } = result;
  const errorCount = Object.keys(module_errors).length;

  return (
    <div className="space-y-8">
      {/* Error banner */}
      {errorCount > 0 && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/5 p-4 text-sm text-amber-200">
          <strong>⚠ Partial result:</strong> {errorCount} module(s) encountered errors (
          {Object.keys(module_errors).join(", ")}). The remaining results are valid.
        </div>
      )}

      {/* Summary metrics */}
      <div>
        <SectionHeader title="Audio Overview" badge="Global analysis" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          <MetricCard
            label="Duration"
            value={audio ? formatDuration(audio.duration_seconds) : "—"}
            sub={audio ? `${audio.n_samples.toLocaleString()} samples` : undefined}
          />
          <MetricCard
            label="Format"
            value={audio ? audio.file_format.toUpperCase() : "—"}
            sub={audio ? `${audio.original_sample_rate / 1000}kHz original` : undefined}
          />
          <MetricCard
            label="Channels"
            value={audio ? (audio.channels === 1 ? "Mono" : `${audio.channels}ch → Mono`) : "—"}
          />
          <MetricCard
            label="Est. BPM"
            value={tempo ? `${tempo.bpm.toFixed(1)}` : "—"}
            sub={tempo ? `${tempo.n_beats} beats detected` : undefined}
            highlight
          />
          <MetricCard
            label="LUFS"
            value={loudness?.lufs != null ? `${loudness.lufs.toFixed(1)}` : "N/A"}
            sub={loudness ? `RMS: ${loudness.rms_db.toFixed(1)} dBFS` : undefined}
          />
          <MetricCard
            label="Voiced Ratio"
            value={pitch ? `${(pitch.voiced_ratio * 100).toFixed(0)}%` : "—"}
            sub={pitch?.dominant_note ? `Modal note: ${pitch.dominant_note}` : undefined}
          />
          <MetricCard
            label="Harmonic Ratio"
            value={hpss ? `${(hpss.harmonic_energy_ratio * 100).toFixed(0)}%` : "—"}
            sub={hpss ? `Percussive: ${(hpss.percussive_energy_ratio * 100).toFixed(0)}%` : undefined}
          />
          <MetricCard
            label="Spectral Centroid"
            value={spectral ? `${spectral.centroid_mean_hz.toFixed(0)} Hz` : "—"}
            sub="Mean frequency center"
          />
        </div>
      </div>

      {/* Estimated Tonic — with PROMINENT disclaimer */}
      {estimated_tonic && (
        <div className="rounded-xl border border-amber-500/40 bg-amber-500/5 p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold uppercase tracking-wider text-amber-400">
                  ⚠ Estimated Tonic (Heuristic)
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full border border-amber-500/40 text-amber-400">
                  Not ground truth
                </span>
              </div>
              <p className="text-3xl font-bold text-amber-300 mt-2">
                {estimated_tonic.estimated_note ?? "Unknown"}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Method: {estimated_tonic.method} · Scope: {estimated_tonic.analysis_scope}
                {estimated_tonic.dominance_ratio != null
                  ? ` · Dominance ratio: ${estimated_tonic.dominance_ratio.toFixed(2)}×`
                  : ""}
              </p>
            </div>
            {/* Mini chroma energy bar */}
            {estimated_tonic.chroma_energy.length === 12 && (
              <div className="flex-1 max-w-xs">
                <p className="text-xs text-slate-500 mb-2">Chroma energy distribution</p>
                <ChromaBar
                  values={estimated_tonic.chroma_energy}
                  labels={["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]}
                />
              </div>
            )}
          </div>
          <p className="text-xs text-amber-200/70 mt-3 border-t border-amber-500/20 pt-3">
            {estimated_tonic.caveat}
          </p>
        </div>
      )}

      {/* Song Identification (Phase 2 — external metadata, optional) */}
      <SongIdentificationCard intel={song_identification} />

      {/* Visualizations */}
      <div>
        <SectionHeader title="Visualizations" badge="All: full recording (global)" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <VizImage jobId={jobId} chart="waveform" label="Waveform (with beat markers)" />
          <VizImage jobId={jobId} chart="spectrogram" label="Spectrogram (STFT, log-freq, dB)" />
          <VizImage jobId={jobId} chart="mel-spectrogram" label="Mel Spectrogram (dB)" />
          <VizImage jobId={jobId} chart="chromagram" label="Chromagram (12 pitch classes)" />
        </div>
        <div className="mt-4">
          <VizImage jobId={jobId} chart="pitch" label={`Pitch Contour / F0 (PYIN, estimated)${pitch && pitch.analysis_scope !== 'global' ? ' — Excerpt only' : ''}`} />
        </div>
      </div>

      {/* MFCC */}
      {mfcc && (
        <div>
          <SectionHeader title="MFCC Coefficients" badge="13 coefficients · Global" />
          <div className="rounded-xl border border-white/10 bg-white/5 p-4 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-slate-500">
                  <th className="text-left py-1 pr-4">Coeff</th>
                  {mfcc.mean.map((_, i) => (
                    <th key={i} className="text-center px-1">C{i + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr className="text-slate-300">
                  <td className="py-1 pr-4 text-slate-500">Mean</td>
                  {mfcc.mean.map((v, i) => (
                    <td key={i} className="text-center px-1">{v.toFixed(1)}</td>
                  ))}
                </tr>
                <tr className="text-slate-400">
                  <td className="py-1 pr-4 text-slate-500">Std</td>
                  {mfcc.std.map((v, i) => (
                    <td key={i} className="text-center px-1">{v.toFixed(1)}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Spectral features table */}
      {spectral && (
        <div>
          <SectionHeader title="Spectral Features" badge="Global" />
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <MetricCard
              label="Centroid (mean)"
              value={`${spectral.centroid_mean_hz.toFixed(0)} Hz`}
              sub={`σ = ${spectral.centroid_std_hz.toFixed(0)} Hz`}
            />
            <MetricCard
              label="Bandwidth (mean)"
              value={`${spectral.bandwidth_mean_hz.toFixed(0)} Hz`}
            />
            <MetricCard
              label="Rolloff (mean)"
              value={`${spectral.rolloff_mean_hz.toFixed(0)} Hz`}
            />
            <MetricCard
              label="ZCR (mean)"
              value={spectral.zero_crossing_rate_mean.toFixed(4)}
              sub={`σ = ${spectral.zero_crossing_rate_std.toFixed(4)}`}
            />
          </div>
        </div>
      )}

      {/* Pitch stats */}
      {pitch && (
        <div>
          <SectionHeader title="Pitch Statistics" badge={`PYIN · ${pitch.analysis_scope}`} />
          <p className="text-xs text-amber-400/80 mb-3">
            ⚠ {pitch.caveat}
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <MetricCard label="Voiced Ratio" value={`${(pitch.voiced_ratio * 100).toFixed(1)}%`} />
            <MetricCard label="Mean F0" value={pitch.mean_f0_hz != null ? `${pitch.mean_f0_hz.toFixed(1)} Hz` : "—"} />
            <MetricCard label="Std F0" value={pitch.std_f0_hz != null ? `${pitch.std_f0_hz.toFixed(1)} Hz` : "—"} />
            <MetricCard label="Min F0" value={pitch.min_f0_hz != null ? `${pitch.min_f0_hz.toFixed(1)} Hz` : "—"} />
            <MetricCard label="Max F0" value={pitch.max_f0_hz != null ? `${pitch.max_f0_hz.toFixed(1)} Hz` : "—"} />
            <MetricCard label="Modal Note" value={pitch.dominant_note ?? "—"} sub="Most common over voiced frames" />
          </div>
        </div>
      )}

      {/* Beat list preview */}
      {tempo && tempo.beat_times_seconds.length > 0 && (
        <div>
          <SectionHeader title="Beat Timestamps" badge={`${tempo.n_beats} beats · Global`} />
          <p className="text-xs text-slate-500 mb-2">⚠ {tempo.caveat}</p>
          <div className="rounded-xl border border-white/10 bg-white/5 p-4 max-h-32 overflow-y-auto">
            <div className="flex flex-wrap gap-2">
              {tempo.beat_times_seconds.slice(0, 50).map((t, i) => (
                <span key={i} className="text-xs px-2 py-0.5 bg-white/10 rounded text-slate-300">
                  {t.toFixed(2)}s
                </span>
              ))}
              {tempo.beat_times_seconds.length > 50 && (
                <span className="text-xs text-slate-500">
                  +{tempo.beat_times_seconds.length - 50} more
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Phase 3/4 placeholders */}
      <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
        <p className="text-sm text-slate-500">
          🔬 <strong className="text-slate-400">Phase 3/4/5 modules</strong> — Raga Classification, YAMNet Instrument Detection, and Demucs Source Separation will be available in future phases.
        </p>
      </div>
    </div>
  );
}

// ─── Upload Zone ──────────────────────────────────────────────────────────────

function UploadZone({ onUpload }: { onUpload: (file: File) => void }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) onUpload(file);
    },
    [onUpload]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`
        rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-300 p-12 text-center
        ${dragging
          ? "border-violet-400 bg-violet-500/10 scale-[1.01]"
          : "border-white/20 bg-white/[0.03] hover:border-violet-500/50 hover:bg-violet-500/5"
        }
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,audio/mpeg,audio/wav"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
        }}
      />
      <div className="flex flex-col items-center gap-4">
        <div className={`w-16 h-16 rounded-full flex items-center justify-center text-3xl transition-all ${dragging ? "bg-violet-500/30" : "bg-white/5"}`}>
          🎵
        </div>
        <div>
          <p className="text-lg font-semibold text-slate-200">
            Drop an audio file or click to upload
          </p>
          <p className="text-sm text-slate-400 mt-1">
            MP3 or WAV · Max 50 MB · Max 10 minutes
          </p>
        </div>
        <div className="flex gap-3">
          {[".WAV", ".MP3"].map((ext) => (
            <span
              key={ext}
              className="text-xs px-3 py-1 rounded-full border border-violet-500/40 text-violet-300 bg-violet-500/10"
            >
              {ext}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function SwarSensePage() {
  const [state, setState] = useState<AppState>({ phase: "idle" });
  const [health, setHealth] = useState<{ ok: boolean; version: string } | null>(null);
  const pollerRef = useRef<NodeJS.Timeout | null>(null);

  // Health check on mount
  useEffect(() => {
    checkHealth()
      .then((h) => setHealth({ ok: h.status === "healthy", version: h.version }))
      .catch(() => setHealth({ ok: false, version: "" }));
  }, []);

  // Poll job status
  const startPolling = useCallback((jobId: string) => {
    if (pollerRef.current) clearInterval(pollerRef.current);

    pollerRef.current = setInterval(async () => {
      try {
        const status: JobStatusResponse = await getJobStatus(jobId);

        if (status.status === "completed" || status.status === "partial") {
          clearInterval(pollerRef.current!);
          const result = await getResult(jobId);
          setState({ phase: "completed", jobId, result });
        } else if (status.status === "failed") {
          clearInterval(pollerRef.current!);
          setState({ phase: "error", message: status.message || "Analysis failed." });
        } else {
          setState({
            phase: "analyzing",
            jobId,
            progress: status.progress,
            message: status.message,
          });
        }
      } catch {
        clearInterval(pollerRef.current!);
        setState({ phase: "error", message: "Lost connection to backend." });
      }
    }, 1500);
  }, []);

  const handleUpload = useCallback(
    async (file: File) => {
      setState({ phase: "uploading" });
      try {
        const response = await submitAnalysis(file);
        setState({
          phase: "analyzing",
          jobId: response.job_id,
          progress: 2,
          message: "Analysis queued...",
        });
        startPolling(response.job_id);
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : "Upload failed.";
        setState({ phase: "error", message });
      }
    },
    [startPolling]
  );

  const handleReset = () => {
    if (pollerRef.current) clearInterval(pollerRef.current);
    setState({ phase: "idle" });
  };

  return (
    <div className="min-h-screen bg-[#0a0a14] text-slate-100 font-sans">
      {/* Header */}
      <header className="border-b border-white/10 bg-[#0f0f1a]/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold text-violet-400">SwarSense</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30">
              Phase 1
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className={`flex items-center gap-1.5 ${health?.ok ? "text-emerald-400" : "text-red-400"}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${health?.ok ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`} />
              {health?.ok ? `API v${health.version}` : "API unreachable"}
            </span>
            <a
              href="http://localhost:8000/api/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-500 hover:text-violet-400 transition-colors"
            >
              Swagger ↗
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 space-y-8">
        {/* Title */}
        <div className="text-center">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-violet-400 to-cyan-400 bg-clip-text text-transparent">
            Indian Raga Recognition System
          </h1>
          <p className="text-slate-400 mt-2 text-sm">
            Upload an audio recording for AI-powered musical analysis · Phase 1: DSP Feature Extraction
          </p>
        </div>

        {/* Upload / Status / Results */}
        {state.phase === "idle" && (
          <UploadZone onUpload={handleUpload} />
        )}

        {state.phase === "uploading" && (
          <div className="rounded-2xl border border-white/10 bg-white/5 p-8 text-center">
            <div className="animate-spin text-4xl mb-4">⟳</div>
            <p className="text-slate-300">Uploading file...</p>
          </div>
        )}

        {state.phase === "analyzing" && (
          <div className="rounded-2xl border border-violet-500/30 bg-violet-500/5 p-8 space-y-6">
            <div className="text-center">
              <div className="text-4xl mb-3 animate-pulse">🎶</div>
              <h2 className="text-lg font-semibold text-slate-200">Analysing your audio...</h2>
              <p className="text-xs text-slate-500 mt-1">Running real DSP pipeline with librosa</p>
            </div>
            <ProgressBar progress={state.progress} message={state.message} />
            {/* Stage hints */}
            <div className="flex justify-between text-xs text-slate-600">
              {["Loading", "Preprocessing", "Tempo", "Pitch", "Features", "Tonic", "Visualize", "Done"].map(
                (s, i) => (
                  <span
                    key={s}
                    className={state.progress >= i * 14 ? "text-violet-400" : ""}
                  >
                    {s}
                  </span>
                )
              )}
            </div>
          </div>
        )}

        {state.phase === "error" && (
          <div className="rounded-2xl border border-red-500/30 bg-red-500/5 p-8 text-center space-y-4">
            <p className="text-2xl">❌</p>
            <p className="text-red-300 font-semibold">{state.message}</p>
            <button
              onClick={handleReset}
              className="px-6 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 text-slate-200 text-sm transition-colors"
            >
              Try again
            </button>
          </div>
        )}

        {state.phase === "completed" && (
          <div className="space-y-6">
            {/* Result header */}
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <h2 className="text-lg font-semibold text-slate-100">
                  Analysis Complete
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  {state.result.filename} · Job: <code className="text-slate-500">{state.jobId}</code>
                </p>
              </div>
              <div className="flex gap-2">
                <a
                  href={`http://localhost:8000/api/result/${state.jobId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs px-3 py-1.5 rounded-lg border border-white/20 text-slate-300 hover:bg-white/10 transition-colors"
                >
                  Raw JSON ↗
                </a>
                <button
                  onClick={handleReset}
                  className="text-xs px-3 py-1.5 rounded-lg bg-violet-500/20 text-violet-300 hover:bg-violet-500/30 border border-violet-500/30 transition-colors"
                >
                  Analyse another
                </button>
              </div>
            </div>
            <ResultsDashboard jobId={state.jobId} result={state.result} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 mt-16 py-6">
        <div className="max-w-6xl mx-auto px-4 flex justify-between text-xs text-slate-600">
          <span>SwarSense · Phase 1 — DSP Feature Extraction · No ML models yet</span>
          <span>AIML Honours Project</span>
        </div>
      </footer>
    </div>
  );
}
