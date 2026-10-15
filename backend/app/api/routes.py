"""
SwarSense — API Routes (Phase 1)
=================================
All HTTP endpoints are defined here.

Endpoints (Phase 1):
  GET  /api/health                         — Health check
  POST /api/analyze                        — Submit audio file for analysis
  GET  /api/status/{job_id}               — Poll job progress
  GET  /api/result/{job_id}               — Retrieve completed analysis result (JSON)
  GET  /api/result/{job_id}/waveform      — Waveform PNG image
  GET  /api/result/{job_id}/spectrogram   — Spectrogram PNG image
  GET  /api/result/{job_id}/mel-spectrogram — Mel spectrogram PNG image
  GET  /api/result/{job_id}/chromagram    — Chromagram PNG image
  GET  /api/result/{job_id}/pitch         — Pitch contour PNG image
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from app.api.schemas import (
    AnalysisResult,
    AnalyzeResponse,
    ErrorResponse,
    JobStatus,
    JobStatusResponse,
)
from app.services.job_service import JobService
from app.services.analysis_service import AnalysisService
from app.audio.validator import AudioValidator
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ─── Shared service instances ──────────────────────────────────────────────────
# In Phase 6 these will be injected via FastAPI dependency injection.
job_service = JobService()
analysis_service = AnalysisService()
audio_validator = AudioValidator()


# ─── Health Check ──────────────────────────────────────────────────────────────

@router.get(
    "/health",
    summary="Health Check",
    description="Returns the operational status of the SwarSense backend.",
    tags=["System"],
    responses={200: {"description": "Backend is healthy."}},
)
async def health_check():
    """
    Confirms the API is reachable.

    In later phases this will also report:
    - Whether ML models are loaded
    - Available disk space
    - GPU/CPU availability
    """
    return {
        "status": "healthy",
        "service": "SwarSense API",
        "version": "0.1.0",
        "phase": "1 — DSP Feature Extraction",
        "models_loaded": False,  # Updated in Phase 3/4 when models are integrated
    }


# ─── Submit Audio for Analysis ─────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=202,
    summary="Submit Audio File for Analysis",
    description=(
        "Upload a `.wav` or `.mp3` file. "
        "The system validates the file, creates a background analysis job, "
        "and returns a `job_id` for polling.\n\n"
        "**File constraints:**\n"
        "- Accepted formats: `.wav`, `.mp3`\n"
        "- Maximum size: 50 MB\n"
        "- Maximum duration: 10 minutes\n\n"
        "**Phase 1:** Real DSP analysis pipeline (librosa). "
        "Returns actual BPM, pitch, spectral features, chroma, MFCC, and visualizations."
    ),
    tags=["Analysis"],
    responses={
        202: {"description": "File accepted, analysis job created."},
        400: {"model": ErrorResponse, "description": "Invalid file."},
        413: {"model": ErrorResponse, "description": "File too large."},
        415: {"model": ErrorResponse, "description": "Unsupported file type."},
        422: {"model": ErrorResponse, "description": "Validation error."},
    },
)
async def submit_analysis(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Audio file (.wav or .mp3)")],
):
    """
    Accept an audio file, validate it, and queue it for background DSP analysis.
    """
    logger.info(f"Received upload request: filename='{file.filename}', "
                f"content_type='{file.content_type}'")

    # ── Read file bytes (apply size limit) ─────────────────────────────────────
    MAX_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
    contents = await file.read()
    if len(contents) > MAX_SIZE_BYTES:
        size_mb = len(contents) / (1024 * 1024)
        logger.warning(f"Rejected upload: file too large ({size_mb:.1f} MB).")
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File size {size_mb:.1f} MB exceeds the 50 MB limit.",
            },
        )

    # ── Validate file format ───────────────────────────────────────────────────
    validation_result = audio_validator.validate(
        filename=file.filename or "",
        content=contents,
    )
    if not validation_result.is_valid:
        logger.warning(f"Rejected upload: {validation_result.error}")
        raise HTTPException(
            status_code=415,
            detail={
                "error": "unsupported_format",
                "message": validation_result.error,
            },
        )

    # ── Create job ─────────────────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    job_service.create_job(
        job_id=job_id,
        filename=file.filename or "unknown",
        file_content=contents,
    )
    logger.info(f"Created job {job_id} for file '{file.filename}'.")

    # ── Queue background analysis ──────────────────────────────────────────────
    background_tasks.add_task(
        analysis_service.run_analysis,
        job_id=job_id,
        file_content=contents,
        filename=file.filename or "unknown",
    )

    return AnalyzeResponse(
        job_id=job_id,
        filename=file.filename or "unknown",
        status=JobStatus.PENDING,
        message="File accepted. DSP analysis has been queued.",
        estimated_duration_seconds=None,
    )


# ─── Poll Job Status ───────────────────────────────────────────────────────────

@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Get Job Status",
    description=(
        "Poll the current processing status of an analysis job.\n\n"
        "Poll every 2–3 seconds. When `status` is `completed` or `partial`, "
        "fetch the full result from `GET /api/result/{job_id}`."
    ),
    tags=["Analysis"],
    responses={
        200: {"description": "Job status returned."},
        404: {"model": ErrorResponse, "description": "Job not found."},
    },
)
async def get_job_status(job_id: str):
    """
    Return current status and progress for a queued analysis job.
    """
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "job_not_found",
                "message": f"No job with ID '{job_id}' was found.",
            },
        )

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        message=job.get("message", ""),
        partial_result=job.get("partial_result"),
        error=job.get("error"),
    )


# ─── Retrieve Full Result ──────────────────────────────────────────────────────

@router.get(
    "/result/{job_id}",
    response_model=AnalysisResult,
    summary="Get Full Analysis Result",
    description=(
        "Retrieve the complete DSP analysis result for a finished job.\n\n"
        "Only call this when `GET /api/status/{job_id}` returns "
        "`status = 'completed'` or `status = 'partial'`.\n\n"
        "If some analysis modules failed, their results will be omitted "
        "and the failure will be recorded in `module_errors`.\n\n"
        "Visualization images are NOT embedded here — use the dedicated "
        "image endpoints (`/waveform`, `/spectrogram`, etc.)."
    ),
    tags=["Analysis"],
    responses={
        200: {"description": "Full analysis result returned."},
        202: {"description": "Analysis still in progress."},
        404: {"model": ErrorResponse, "description": "Job not found."},
    },
)
async def get_analysis_result(job_id: str):
    """
    Return the full structured DSP analysis result for a completed job.
    """
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "job_not_found",
                "message": f"No job with ID '{job_id}' was found.",
            },
        )

    status = job["status"]
    if status in (JobStatus.PENDING, JobStatus.PROCESSING):
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": status,
                "message": "Analysis is still in progress. Please poll /api/status/{job_id}.",
            },
        )

    if status == JobStatus.FAILED:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "analysis_failed",
                "message": job.get("message") or "Analysis failed before producing a result.",
            },
        )

    result = job.get("result")
    if result is None:
        # Should not happen for completed/partial jobs — the pipeline always
        # assembles a result before marking a job completed or partial.
        raise HTTPException(
            status_code=500,
            detail={
                "error": "result_missing",
                "message": "Job completed but result is unavailable.",
            },
        )

    return result


# ─── Visualization Image Endpoints ────────────────────────────────────────────

def _serve_visualization_image(job_id: str, chart_name: str) -> Response:
    """
    Helper: retrieve a specific PNG from the job's VisualizationBundle and return it.

    Args:
        job_id:     Analysis job ID.
        chart_name: One of 'waveform', 'spectrogram', 'mel_spectrogram', 'chromagram', 'pitch'.

    Returns:
        FastAPI Response with image/png content type.
    """
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "message": f"No job with ID '{job_id}'."},
        )

    status = job["status"]
    if status in (JobStatus.PENDING, JobStatus.PROCESSING):
        raise HTTPException(
            status_code=202,
            detail={
                "error": "analysis_in_progress",
                "message": f"Analysis is still running ({status}). Check /api/status/{job_id}.",
            },
        )

    viz_bundle = job_service.get_visualizations(job_id)
    if viz_bundle is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "visualizations_unavailable",
                "message": (
                    "Visualizations were not generated for this job. "
                    "This may indicate the feature extraction stage failed. "
                    f"Check module_errors in GET /api/result/{job_id}."
                ),
            },
        )

    png_bytes: bytes | None = getattr(viz_bundle, chart_name, None)
    if png_bytes is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "chart_unavailable",
                "message": (
                    f"Chart '{chart_name}' was not generated. "
                    f"Check module_errors in GET /api/result/{job_id} for details."
                ),
            },
        )

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",
            "Content-Disposition": f'inline; filename="{job_id}_{chart_name}.png"',
        },
    )


@router.get(
    "/result/{job_id}/waveform",
    summary="Get Waveform Image",
    description=(
        "Returns a PNG image of the audio waveform with beat markers overlaid.\n\n"
        "Represents the COMPLETE recording (global analysis).\n"
        "Only available after analysis is completed."
    ),
    tags=["Visualizations"],
    responses={
        200: {"content": {"image/png": {}}, "description": "Waveform PNG image."},
        202: {"description": "Analysis still in progress."},
        404: {"model": ErrorResponse, "description": "Job or chart not found."},
    },
)
async def get_waveform_image(job_id: str):
    return _serve_visualization_image(job_id, "waveform")


@router.get(
    "/result/{job_id}/spectrogram",
    summary="Get Spectrogram Image",
    description=(
        "Returns a PNG image of the log-frequency STFT magnitude spectrogram in dB.\n\n"
        "Represents the COMPLETE recording (global analysis).\n"
        "Only available after analysis is completed."
    ),
    tags=["Visualizations"],
    responses={
        200: {"content": {"image/png": {}}, "description": "Spectrogram PNG image."},
        202: {"description": "Analysis still in progress."},
        404: {"model": ErrorResponse, "description": "Job or chart not found."},
    },
)
async def get_spectrogram_image(job_id: str):
    return _serve_visualization_image(job_id, "spectrogram")


@router.get(
    "/result/{job_id}/mel-spectrogram",
    summary="Get Mel Spectrogram Image",
    description=(
        "Returns a PNG image of the Mel-filterbank spectrogram in dB.\n\n"
        "Represents the COMPLETE recording (global analysis).\n"
        "Only available after analysis is completed."
    ),
    tags=["Visualizations"],
    responses={
        200: {"content": {"image/png": {}}, "description": "Mel spectrogram PNG image."},
        202: {"description": "Analysis still in progress."},
        404: {"model": ErrorResponse, "description": "Job or chart not found."},
    },
)
async def get_mel_spectrogram_image(job_id: str):
    return _serve_visualization_image(job_id, "mel_spectrogram")


@router.get(
    "/result/{job_id}/chromagram",
    summary="Get Chromagram Image",
    description=(
        "Returns a PNG image of the chromagram (12 pitch class energies over time).\n\n"
        "Represents the COMPLETE recording (global analysis).\n"
        "Only available after analysis is completed."
    ),
    tags=["Visualizations"],
    responses={
        200: {"content": {"image/png": {}}, "description": "Chromagram PNG image."},
        202: {"description": "Analysis still in progress."},
        404: {"model": ErrorResponse, "description": "Job or chart not found."},
    },
)
async def get_chromagram_image(job_id: str):
    return _serve_visualization_image(job_id, "chromagram")


@router.get(
    "/result/{job_id}/pitch",
    summary="Get Pitch Contour Image",
    description=(
        "Returns a PNG image of the F0 (pitch) contour estimated by PYIN.\n\n"
        "Voiced frames are shown in cyan; unvoiced in grey.\n"
        "For recordings longer than 60 seconds, this represents the first 60 seconds (excerpt).\n"
        "Only available after analysis is completed."
    ),
    tags=["Visualizations"],
    responses={
        200: {"content": {"image/png": {}}, "description": "Pitch contour PNG image."},
        202: {"description": "Analysis still in progress."},
        404: {"model": ErrorResponse, "description": "Job or chart not found."},
    },
)
async def get_pitch_image(job_id: str):
    return _serve_visualization_image(job_id, "pitch")
