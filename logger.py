"""
AI Classroom Attendance System - Structured Logging Module
===========================================================
Configures standardized logging with console and rotating file handlers.
Guarantees no biometric embeddings, student photographs, or sensitive PII
are written to disk or console.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from config import config


def get_logger(
    name: str = "attendance",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
) -> logging.Logger:
    """
    Retrieve or initialize a standardized structured logger.
    Avoids duplicate handlers on repeated invocations.
    """
    logger = logging.getLogger(name)

    # If handlers already configured, return logger directly
    if logger.handlers:
        return logger

    # Resolve settings
    target_dir = log_dir or config.LOG_DIR
    target_file = log_file or config.LOG_FILE
    level_str = (log_level or config.LOG_LEVEL).upper()
    level = getattr(logging, level_str, logging.INFO)

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Rotating file handler (if log_dir is provided)
    if target_dir:
        try:
            os.makedirs(target_dir, exist_ok=True)
            log_path = os.path.join(target_dir, target_file)
            file_handler = RotatingFileHandler(
                filename=log_path,
                maxBytes=config.LOG_MAX_BYTES,
                backupCount=config.LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console-only if filesystem permissions fail
            console_handler.handle(
                logging.LogRecord(
                    name=name,
                    level=logging.WARNING,
                    pathname=__file__,
                    lineno=0,
                    msg=f"Failed to create rotating file handler in '{target_dir}': {e}",
                    args=(),
                    exc_info=None,
                )
            )

    return logger

