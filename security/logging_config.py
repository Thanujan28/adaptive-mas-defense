"""
Reusable logging setup for the security package.

The security code emits structured ``security_state ...`` records to
the ``security.observer`` logger. Nothing configures that logger by
default, so those records are silently dropped unless you call this.

Usage:

    from security.logging_config import enable_security_logging
    enable_security_logging()          # DEBUG level, readable format
    enable_security_logging(level=logging.INFO)

Or, without touching Python code:

    set SECURITY_LOG=1            (Windows / PowerShell)
    export SECURITY_LOG=1         (bash)

and call ``enable_security_logging_from_env()`` at startup.
"""

from __future__ import annotations

import logging
import os
import sys


# Loggers used across the security package.
SECURITY_LOGGERS = (
    "security",
    "security.observer",
    "security.detector",
    "security.semantic_assessor",
)


_FORMAT = (
    "%(asctime)s %(levelname)-8s %(name)s %(message)s"
)


def enable_security_logging(
    level: int = logging.DEBUG,
    stream=None,
) -> logging.Logger:
    """
    Turn on the ``security`` logger tree and return the root logger.

    By default the observer logs handling of ordinary responses at
    DEBUG and responses that require investigation at WARNING, so
    level DEBUG shows everything.

    Parameters
    ----------
    level:
        Minimum level to emit (default DEBUG).
    stream:
        Optional stream; defaults to stderr.
    """

    if stream is None:
        stream = sys.stdout
    logger = logging.getLogger("security")
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls.
    if not any(
        getattr(handler, "_security_handler", False)
        for handler in logger.handlers
    ):
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(_FORMAT)
        )
        handler._security_handler = True
        logger.addHandler(handler)

    return logger


def enable_security_logging_from_env(
    default_level: int = logging.DEBUG,
) -> bool:
    """
    Enable security logging when the ``SECURITY_LOG`` env var is set.

    Returns True when logging was enabled.
    """

    value = os.getenv("SECURITY_LOG", "").strip().lower()

    if value in {"", "0", "false", "no", "off"}:
        return False

    level_name = os.getenv("SECURITY_LOG_LEVEL", "").strip().upper()
    level = (
        getattr(logging, level_name)
        if level_name and hasattr(logging, level_name)
        else default_level
    )

    enable_security_logging(level=level)
    return True
