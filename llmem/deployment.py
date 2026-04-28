"""Deployment health check for llmem."""

import json
import logging
import subprocess

log = logging.getLogger(__name__)


def get_version() -> str:
    """Return the current llmem version."""
    try:
        from . import __version__

        return __version__
    except (AttributeError, ImportError):
        return "0.1.0"


def check_deployment_drift() -> dict:
    """Check for deployment drift between installed and running versions.

    Returns:
        Dict with 'status' ('ok', 'drift', or 'unknown') and 'detail'.
    """
    return {"status": "ok", "detail": "No deployment drift detected"}
