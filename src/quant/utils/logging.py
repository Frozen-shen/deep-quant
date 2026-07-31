"""
Unified logging for quant-starter.

Replaces all print() usage with structured logging via loguru.
Provides both console and file output with context information
(module name, timestamp, level).

Usage:
    from quant.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting backtest for {}", config.data.universe)
    logger.warning("Factor IC below threshold: {:.4f}", ic_value)
    logger.error("Data fetch failed for {}", symbol)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger

# Remove default loguru handler to avoid duplicate output
_loguru_logger.remove()

# Track whether global setup has been called
_initialized = False

# Default format for console output
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Format for file output (no color tags)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} - "
    "{message}"
)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str | Path] = None,
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    console: bool = True,
    serialize: bool = False,
) -> None:
    """
    Configure the global logging system.

    Call this once at application startup. Subsequent calls to get_logger()
    will use this configuration.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. If None, no file logging.
        log_file: Log file name. Defaults to "quant_{time}.log" with rotation.
        rotation: When to rotate the log file (size or time-based).
        retention: How long to keep rotated log files.
        console: Whether to output to stderr console.
        serialize: If True, output JSON-structured logs (for log aggregation).
    """
    global _initialized

    # Remove all existing handlers
    _loguru_logger.remove()

    # Console handler
    if console:
        _loguru_logger.add(
            sys.stderr,
            format=_CONSOLE_FORMAT,
            level=level.upper(),
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            log_file = "quant_{time:YYYY-MM-DD}.log"

        file_path = log_path / log_file

        _loguru_logger.add(
            str(file_path),
            format=_FILE_FORMAT,
            level=level.upper(),
            rotation=rotation,
            retention=retention,
            compression="zip",
            serialize=serialize,
            backtrace=True,
            diagnose=False,  # Don't leak variable values to log files
            encoding="utf-8",
        )

    _initialized = True


def get_logger(name: Optional[str] = None) -> "_BoundLoggerProxy":
    """
    Get a logger bound to a specific module/context.

    Args:
        name: Logger name, typically __name__ of the calling module.
              Used as the 'name' context in log output.

    Returns:
        A logger proxy that adds context binding.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing {} stocks", n_stocks)
    """
    if not _initialized:
        # Auto-initialize with sensible defaults if not explicitly set up
        setup_logging()

    if name is None:
        return _BoundLoggerProxy(_loguru_logger)

    return _BoundLoggerProxy(_loguru_logger.bind(name=name))


class _BoundLoggerProxy:
    """
    Thin proxy around loguru logger that provides a consistent interface.

    Supports all standard log levels and loguru's format-string syntax.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger):
        object.__setattr__(self, "_logger", logger)

    def trace(self, message: str, *args, **kwargs) -> None:
        """Log at TRACE level (very detailed debugging)."""
        self._logger.opt(depth=1).trace(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log at DEBUG level (detailed diagnostic info)."""
        self._logger.opt(depth=1).debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log at INFO level (general operational info)."""
        self._logger.opt(depth=1).info(message, *args, **kwargs)

    def success(self, message: str, *args, **kwargs) -> None:
        """Log at SUCCESS level (operation completed successfully)."""
        self._logger.opt(depth=1).success(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log at WARNING level (something unexpected but recoverable)."""
        self._logger.opt(depth=1).warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log at ERROR level (operation failed)."""
        self._logger.opt(depth=1).error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Log at CRITICAL level (system cannot continue)."""
        self._logger.opt(depth=1).critical(message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """Log at ERROR level with full exception traceback."""
        self._logger.opt(depth=1).exception(message, *args, **kwargs)

    def bind(self, **kwargs) -> "_BoundLoggerProxy":
        """Create a child logger with additional context."""
        return _BoundLoggerProxy(self._logger.bind(**kwargs))

    def catch(
        self,
        exception=Exception,
        *,
        level: str = "ERROR",
        reraise: bool = False,
        onerror=None,
        exclude=None,
        message: str = "An error has been caught in function '{record[function]}'",
    ):
        """Decorator/context manager to catch and log exceptions."""
        return self._logger.catch(
            exception=exception,
            level=level,
            reraise=reraise,
            onerror=onerror,
            exclude=exclude,
            message=message,
        )
