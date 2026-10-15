"""
SwarSense Backend — FastAPI Application Entry Point
====================================================
Project: SwarSense — AI-Based Intelligent Music Analysis and Indian Raga Recognition
Phase: 0 (Infrastructure skeleton)

This module initialises the FastAPI application, configures CORS,
mounts API routes, and registers startup/shutdown lifecycle events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown events.

    Startup:
      - Log application boot
      - (Phase 3+) Pre-load ML models into memory

    Shutdown:
      - Flush logs
      - (Phase 3+) Release GPU/CPU resources
    """
    logger.info("SwarSense backend starting up...")
    logger.info("Phase 0: Infrastructure skeleton — no ML models loaded.")
    yield
    logger.info("SwarSense backend shutting down.")


# ─── Application Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.
    """
    app = FastAPI(
        title="SwarSense API",
        description=(
            "AI-Based Intelligent Music Analysis and Indian Raga Recognition System.\n\n"
            "**SwarSense** accepts an uploaded audio file and returns:\n"
            "- General audio features (BPM, loudness, duration, sample rate)\n"
            "- DSP analysis (MFCC, Mel spectrogram, chromagram, pitch contour)\n"
            "- Source separation results (vocal / instrumental)\n"
            "- Instrument detection\n"
            "- Indian raga classification with confidence and Grad-CAM explanation\n\n"
            "> **Phase 0 Note:** This is the infrastructure skeleton. "
            "ML inference endpoints return placeholder responses until Phase 3–4."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # ─── CORS ──────────────────────────────────────────────────────────────────
    # Allow Next.js dev server and production origin.
    # Tighten in production by replacing "*" with explicit origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://frontend:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Routers ───────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api")

    # ─── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again.",
                "detail": str(exc),
            },
        )

    return app


app = create_app()
