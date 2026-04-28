"""Correction detection for session transcripts."""

import json
import logging
import re

log = logging.getLogger(__name__)

CORRECTION_PATTERNS = [
    re.compile(r"actually[,!]", re.IGNORECASE),
    re.compile(r"sorry[,!]", re.IGNORECASE),
    re.compile(r"my mistake", re.IGNORECASE),
    re.compile(r"let me( try| retry| redo| fix)", re.IGNORECASE),
    re.compile(r"that('s| was) wrong", re.IGNORECASE),
    re.compile(r"I was wrong", re.IGNORECASE),
    re.compile(r"correction:", re.IGNORECASE),
    re.compile(r"fix(ed|ing)? (that|this|it)", re.IGNORECASE),
]


def detect_corrections(text: str) -> list[dict]:
    """Detect correction patterns in text.

    Args:
        text: Session transcript text.

    Returns:
        List of dicts with 'pattern' and 'context' keys.
    """
    corrections = []
    lines = text.split("\n")
    for line in lines:
        for pattern in CORRECTION_PATTERNS:
            if pattern.search(line):
                corrections.append(
                    {
                        "pattern": pattern.pattern,
                        "context": line.strip()[:200],
                    }
                )
                break
    return corrections
