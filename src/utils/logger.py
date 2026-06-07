# src/utils/logger.py

import logging
import os
from src.utils.constants import LOG_LEVEL, LOG_DIR, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with both file and console handlers.
    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Console Handler ──
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler ──
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
