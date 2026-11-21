"""Single logger setup used across the package."""

from __future__ import annotations

import logging
from pathlib import Path

_ROOT_LOGGER = "pearlroute"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the package logger once: console output, plus a file if given."""
    logger = logging.getLogger(_ROOT_LOGGER)
    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt="%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the package namespace."""
    if not name.startswith(_ROOT_LOGGER):
        name = f"{_ROOT_LOGGER}.{name}"
    return logging.getLogger(name)
