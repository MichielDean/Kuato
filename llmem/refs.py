"""Code reference resolution for memory-to-code-chunk edges.

Provides resolve_code_ref() — a standalone utility that parses code reference
strings in the format 'path:start_line:end_line' and returns the resolved
file content. Used by Retriever.search() when traverse_refs is enabled.
"""

import logging
import re
from pathlib import Path

from .paths import _is_blocked_path

log = logging.getLogger(__name__)

# Pattern: path:start_line:end_line (1-based, inclusive)
_CODE_REF_PATTERN = re.compile(r"^([^:]+):(\d+):(\d+)$")


def resolve_code_ref(target_id: str) -> dict | None:
    """Resolve a code reference string to actual file content.

    Parses target_id in the format 'path:start_line:end_line' (1-based,
    inclusive). Reads the file at the given path, extracts lines start_line
    through end_line. Uses _is_blocked_path to prevent reading from
    protected system directories.

    Args:
        target_id: Code reference string in 'path:start_line:end_line' format.

    Returns:
        Dict with file_path, start_line, end_line, content, and target_type='code'
        on success, or None if the file doesn't exist, the format is invalid,
        the line range is out of bounds, or the path is blocked. Never raises
        IOError — missing or inaccessible files return None with a log.debug.
    """
    match = _CODE_REF_PATTERN.match(target_id)
    if not match:
        log.debug("llmem: refs: invalid code ref format: %r", target_id)
        return None

    file_path_str = match.group(1)
    start_line = int(match.group(2))
    end_line = int(match.group(3))

    if start_line < 1 or end_line < start_line:
        log.debug(
            "llmem: refs: invalid line range in code ref %r: %d-%d",
            target_id,
            start_line,
            end_line,
        )
        return None

    file_path = Path(file_path_str)

    # Block path traversal using shared security helper
    try:
        resolved = file_path.resolve()
    except OSError:
        log.debug("llmem: refs: cannot resolve path in code ref %r", target_id)
        return None

    if _is_blocked_path(resolved):
        log.debug("llmem: refs: blocked path in code ref %r", target_id)
        return None

    if not resolved.is_file():
        log.debug("llmem: refs: file not found for code ref %r", target_id)
        return None

    try:
        lines = resolved.read_text().splitlines()
    except OSError as exc:
        log.debug("llmem: refs: cannot read file for code ref %r: %s", target_id, exc)
        return None

    if start_line > len(lines):
        log.debug(
            "llmem: refs: start_line %d exceeds file length %d in code ref %r",
            start_line,
            len(lines),
            target_id,
        )
        return None

    # Clamp end_line to file length (best-effort: return what's available)
    clamped_end = min(end_line, len(lines))
    content = "\n".join(lines[start_line - 1 : clamped_end])

    return {
        "file_path": str(file_path),
        "start_line": start_line,
        "end_line": clamped_end,
        "content": content,
        "target_type": "code",
    }
