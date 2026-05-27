"""Structured JSON logging used across all modules."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(log_level: str | None = None) -> None:
    """Configure structlog for JSON output in prod, pretty output in dev."""
    level: str = log_level or os.getenv("LOG_LEVEL", "INFO") or "INFO"
    app_env = os.getenv("APP_ENV", "development")

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if app_env == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)
