"""
SwarSense — Job Service
=======================
Manages the lifecycle of analysis jobs.

Phase 0: In-memory dictionary store (sufficient for single-worker demo).
Phase 6: Replace with Redis-backed Celery task store for production scalability.

Each job record:
  {
    "job_id":        str,
    "filename":      str,
    "status":        JobStatus,
    "progress":      int (0-100),
    "message":       str,
    "result":        AnalysisResult | None,
    "error":         ErrorResponse | None,
    "partial_result": dict | None,
    "visualizations": VisualizationBundle | None,   # Phase 1: PNG bytes
  }
"""

import threading
from typing import Optional

from app.api.schemas import JobStatus
from app.utils.logger import get_logger

logger = get_logger(__name__)


class JobService:
    """
    Thread-safe in-memory job store.

    NOTE: Jobs are lost on server restart. This is acceptable for Phase 0–5.
    For production, replace with a persistent backend (Redis, PostgreSQL).
    """

    _jobs: dict[str, dict] = {}
    _lock = threading.Lock()

    def __init__(self):
        pass

    def create_job(
        self,
        job_id: str,
        filename: str,
        file_content: bytes,
    ) -> dict:
        """
        Register a new analysis job.

        Args:
            job_id:       Unique identifier.
            filename:     Original filename.
            file_content: Raw bytes of the uploaded audio (held in memory for now;
                          Phase 1 will save to disk and store a path instead).

        Returns:
            The initial job record.
        """
        job = {
            "job_id": job_id,
            "filename": filename,
            "file_content": file_content,  # Phase 1: replace with file path
            "status": JobStatus.PENDING,
            "progress": 0,
            "message": "Queued.",
            "result": None,
            "error": None,
            "partial_result": None,
            "visualizations": None,  # Phase 1: VisualizationBundle (PNG bytes)
        }
        with self._lock:
            self._jobs[job_id] = job
        logger.debug(f"Job created: {job_id}")
        return job

    def get_job(self, job_id: str) -> Optional[dict]:
        """Return the job record or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result=None,
        error=None,
        partial_result=None,
        visualizations=None,  # Phase 1: VisualizationBundle with PNG bytes
    ) -> None:
        """
        Update one or more fields of an existing job record.
        Only non-None arguments are applied.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.error(f"Attempted to update non-existent job: {job_id}")
                return
            if status is not None:
                job["status"] = status
            if progress is not None:
                job["progress"] = progress
            if message is not None:
                job["message"] = message
            if result is not None:
                job["result"] = result
            if error is not None:
                job["error"] = error
            if partial_result is not None:
                job["partial_result"] = partial_result
            if visualizations is not None:
                job["visualizations"] = visualizations
        logger.debug(f"Job updated: {job_id} → status={status}, progress={progress}")

    def get_visualizations(self, job_id: str):
        """Return the VisualizationBundle for a job, or None."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.get("visualizations") if job else None

    def list_jobs(self) -> list[dict]:
        """Return all job records (for debugging/admin use)."""
        with self._lock:
            return list(self._jobs.values())

    def delete_job(self, job_id: str) -> bool:
        """Remove a job from the store. Returns True if it existed."""
        with self._lock:
            existed = job_id in self._jobs
            self._jobs.pop(job_id, None)
        return existed
