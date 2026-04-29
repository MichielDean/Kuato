"""Code reference resolution for memory-to-code-chunk edges.

Provides resolve_code_ref() — a standalone utility that parses code reference
strings in the format 'path:start_line:end_line' and returns the resolved
file content. Used by Retriever.search() when traverse_refs is enabled.

Security: Code refs must use relative paths (no leading '/' or '..' traversal).
resolve_code_ref() enforces an allowed-paths allowlist that defaults to
the current working directory, preventing arbitrary file reads.
"""

import logging
import re
from pathlib import Path

from .paths import _is_blocked_path

log = logging.getLogger(__name__)

# Pattern: path:start_line:end_line (1-based, inclusive)
_CODE_REF_PATTERN = re.compile(r"^([^:]+):(\d+):(\d+)$")


def validate_code_ref_path(file_path_str: str) -> str | None:
    """Validate that a code reference path is a safe relative path.

    Code refs must use relative paths that do not escape their containing
    directory. This prevents arbitrary file reads via absolute paths
    (e.g., /home/user/.ssh/id_rsa) or traversal sequences (e.g., ../../etc/passwd).

    Args:
        file_path_str: The path component of a code ref target_id.

    Returns:
        The validated path string if safe, or None if the path is unsafe.
    """
    # Reject absolute paths — code refs must be relative
    if file_path_str.startswith("/"):
        log.debug("llmem: refs: absolute path rejected in code ref: %r", file_path_str)
        return None

    # Reject paths with '..' traversal — prevents directory escape
    if ".." in file_path_str:
        log.debug("llmem: refs: path traversal rejected in code ref: %r", file_path_str)
        return None

    # Reject empty path
    if not file_path_str.strip():
        log.debug("llmem: refs: empty path in code ref")
        return None

    return file_path_str


def resolve_code_ref(
    target_id: str, allowed_paths: list[Path] | None = None
) -> dict | None:
    """Resolve a code reference string to actual file content.

    Parses target_id in the format 'path:start_line:end_line' (1-based,
    inclusive). Reads the file at the given path, extracts lines start_line
    through end_line. Enforces security constraints:

    - Code ref paths must be relative (no leading '/') and contain no '..'
      traversal. Absolute paths and traversals are rejected at parse time.
    - The resolved file path must fall under one of the allowed_paths
      directories (defaults to [Path.cwd()]).
    - The resolved path must not target a blocked system directory.

    When allowed_paths is provided, relative paths are resolved against
    each allowed directory in order. The first match under an allowed
    directory is used. This allows code refs like 'src/lib.rs:1:10' to
    resolve correctly when allowed_paths includes the project root.

    Args:
        target_id: Code reference string in 'path:start_line:end_line' format.
        allowed_paths: List of directories under which code refs are allowed
            to resolve. Defaults to [Path.cwd()] — only files under the
            current working directory can be read. Pass an explicit list to
            allow code refs from additional directories.

    Returns:
        Dict with file_path, start_line, end_line, content, and target_type='code'
        on success, or None if the file doesn't exist, the format is invalid,
        the path is blocked, or the path is outside allowed_paths.
        Never raises IOError — missing or inaccessible files return None
        with a log.debug.
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

    # Validate path is relative and contains no traversal
    validated_path = validate_code_ref_path(file_path_str)
    if validated_path is None:
        return None

    file_path = Path(validated_path)

    # Resolve allowed paths — defaults to current working directory
    if allowed_paths is None:
        allowed_paths = [Path.cwd()]

    # Resolve allowed paths for comparison
    resolved_allowed: list[Path] = []
    for ap in allowed_paths:
        try:
            resolved_allowed.append(ap.resolve())
        except OSError:
            log.debug("llmem: refs: cannot resolve allowed path: %s", ap)

    # For relative paths: resolve against each allowed directory.
    # Try each allowed dir in order; pick the first where the file exists
    # and the resolved path is under that allowed directory.
    resolved: Path | None = None
    for allowed_dir in resolved_allowed:
        candidate = (allowed_dir / file_path).resolve()
        try:
            candidate.relative_to(allowed_dir)
        except ValueError:
            # Resolved path escaped the allowed directory (shouldn't happen
            # for validated relative paths, but check anyway)
            continue
        if _is_blocked_path(candidate):
            continue
        if candidate.is_file():
            resolved = candidate
            break

    if resolved is None:
        log.debug(
            "llmem: refs: path not found under allowed directories for code ref %r",
            target_id,
        )
        return None

    # Final check: resolved path must be under one of the allowed directories
    under_allowed = False
    for allowed_dir in resolved_allowed:
        try:
            resolved.relative_to(allowed_dir)
            under_allowed = True
            break
        except ValueError:
            continue
    if not under_allowed:
        log.debug(
            "llmem: refs: path outside allowed directories in code ref %r",
            target_id,
        )
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
