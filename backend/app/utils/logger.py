"""
SwarSense — Structured Logging
================================
Provides a consistent logger factory for all modules.

Uses loguru for structured, leveled, colorized output.
Falls back to Python's standard logging if loguru is unavailable.

Usage:
  from app.utils.logger import get_logger
  logger = get_logger(__name__)
  logger.info("Processing job", job_id=job_id)
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Log format: [LEVEL] [module] message
    Log level: Configurable via LOG_LEVEL environment variable (default: INFO).
    """
    import os
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        logger.propagate = False

    return logger
