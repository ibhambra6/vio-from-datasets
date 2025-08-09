"""Logging utilities for the VIO project.

Provides a helper to configure a consistent, colored logger across modules.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Optional logger name. When None, the root logger is used.
        level: Logging level for the stream handler.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured
        return logger

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)

    use_colors = os.environ.get("VIO_LOG_COLOR", "1") != "0"
    level_to_color = {
        logging.DEBUG: "\x1b[36m",  # cyan
        logging.INFO: "\x1b[32m",  # green
        logging.WARNING: "\x1b[33m",  # yellow
        logging.ERROR: "\x1b[31m",  # red
        logging.CRITICAL: "\x1b[35m",  # magenta
    }

    class _ColorFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # noqa: D401
            base = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
            if use_colors:
                color = level_to_color.get(record.levelno, "")
                reset = "\x1b[0m"
                base = f"{color}{base}{reset}"
            self._style._fmt = base  # type: ignore[attr-defined]
            return super().format(record)

    handler.setFormatter(_ColorFormatter(datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
