"""Logging configuration utilities."""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: str = "INFO", *, name: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger suitable for CLI execution.

    Args:
        level: Logging level name (e.g., "INFO", "DEBUG").
        name: Optional logger name. If omitted, the root logger is configured.

    Returns:
        A configured `logging.Logger` instance.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
