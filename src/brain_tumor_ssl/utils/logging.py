"""Structured logging via loguru, used everywhere instead of ``print``."""

from __future__ import annotations

import sys

from loguru import logger
from loguru._logger import Logger

_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the global loguru sink (idempotent).

    Args:
        level: Minimum log level to emit (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    global _configured
    logger.remove()
    logger.add(sys.stderr, level=level, format=_FORMAT, colorize=True)
    _configured = True


def get_logger() -> Logger:
    """Return the configured project logger, configuring it on first use.

    Returns:
        The shared loguru logger instance.
    """
    if not _configured:
        configure_logging()
    return logger
