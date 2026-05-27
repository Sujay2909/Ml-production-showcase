"""
Pytest configuration: patch structlog before any src module is imported.
Prevents logging side-effects and infrastructure dependencies during tests.
"""

import logging

import structlog


def pytest_configure(config):
    """Configure structlog once before any tests run."""
    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer(colors=False)],
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
