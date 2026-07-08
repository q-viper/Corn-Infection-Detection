"""Loguru configuration helpers."""

from pathlib import Path
import sys

from loguru import logger


LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logger(
    log_dir: Path | None = None,
    log_file: str = "",
    level: str = "INFO",
):
    """Configure Loguru for console and optional file logging."""

    logger.remove()
    logger.add(sys.stderr, level=level.upper(), format=LOG_FORMAT)
    if log_dir is not None and log_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / log_file,
            level=level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="14 days",
            enqueue=True,
        )
    return logger
