"""
Centralized logging factory for the Inventory Optimization Platform.

Design rationale:
- Single call to get_logger() ensures all modules share the same handler
  configuration, preventing duplicate log lines in long-running pipelines.
- File handler writes a persistent audit trail in logs/; the file rotates
  once it hits 10 MB so disk usage stays bounded.
- Console handler can be disabled in config for batch/cron runs.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

import yaml

# Module-level registry so we don't add duplicate handlers across imports
_INITIALIZED_LOGGERS: set[str] = set()

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _load_logging_config(config_path: Optional[str] = None) -> dict:
    """Load logging section from config.yaml, falling back to safe defaults."""
    defaults = {
        "level": "INFO",
        "log_dir": "logs",
        "log_file": "pipeline.log",
        "console": True,
    }
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f) or {}
        return {**defaults, **full_cfg.get("logging", {})}
    return defaults


def get_logger(
    name: str,
    config_path: Optional[str] = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Args:
        name:        Typically ``__name__`` of the calling module.
        config_path: Path to config.yaml. Falls back to defaults if not found.
        level:       Override log level (e.g. 'DEBUG'). Overrides config value.

    Returns:
        logging.Logger: Configured logger instance.

    Example::

        from src.utils.logger import get_logger
        log = get_logger(__name__, config_path="config/config.yaml")
        log.info("Pipeline started")
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers more than once (idempotent)
    if name in _INITIALIZED_LOGGERS:
        return logger

    cfg = _load_logging_config(config_path)

    # Resolve effective log level
    effective_level_str = level or cfg.get("level", "INFO")
    effective_level = getattr(logging, effective_level_str.upper(), logging.INFO)
    logger.setLevel(effective_level)

    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ────────────────────────────────────────────────────
    if cfg.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(effective_level)
        logger.addHandler(console_handler)

    # ── Rotating file handler ──────────────────────────────────────────────
    log_dir = Path(cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / cfg.get("log_file", "pipeline.log")

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(effective_level)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate lines)
    logger.propagate = False

    _INITIALIZED_LOGGERS.add(name)
    return logger
