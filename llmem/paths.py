"""Path resolution for llmem data directories.

All hardcoded ~/.lobsterdog/ paths are replaced with ~/.config/llmem/
(with LMEM_HOME env var override). Backward compatibility:
~/.lobsterdog/ is checked first for migration.
"""

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# Maximum allowed components in LMEM_HOME path (prevents /../../../etc tricks)
_MAX_PATH_DEPTH = 10

# System directories that should never be used as llmem data locations.
# Shared between _validate_home_path, _validate_write_path, and
# OpenCodeAdapter.__init__ to prevent DRY violations.
BLOCKED_SYSTEM_PREFIXES = (
    "/etc",
    "/var",
    "/sys",
    "/proc",
    "/dev",
    "/boot",
    "/root",
    "/sbin",
    "/bin",
    "/usr/sbin",
    "/usr/bin",
)


def _validate_home_path(path: Path, source: str) -> Path:
    """Validate that a home path is safe to use.

    Checks (in order):
    - Must not contain '..' traversal components (checked before resolve)
    - Must not target sensitive system directories (checked before symlink
      check since is_symlink() requires stat access which may fail)
    - Must not be a symlink itself (prevents symlink escalation)
    - Must not exceed a reasonable path depth

    Args:
        path: The candidate home path.
        source: Description of where this path came from (for error messages).

    Returns:
        The resolved, validated path.

    Raises:
        ValueError: If the path is unsafe.
    """
    # Check traversal BEFORE resolving (resolve eliminates ..)
    if ".." in str(path):
        raise ValueError(f"llmem: paths: {source} contains '..' traversal: {path}")

    resolved = path.resolve()

    # Block obvious system directories — checked before symlink check
    # because is_symlink() requires stat access which may fail for
    # inaccessible paths like /root
    for prefix in BLOCKED_SYSTEM_PREFIXES:
        if str(resolved).startswith(prefix):
            raise ValueError(
                f"llmem: paths: {source} targets a system directory: {resolved}"
            )

    # Must not be a symlink itself (prevents symlink escalation)
    # If we can't stat the path (permission denied), treat it as unsafe
    try:
        if path.is_symlink():
            raise ValueError(
                f"llmem: paths: {source} is a symlink (not allowed): {path}"
            )
    except OSError:
        raise ValueError(
            f"llmem: paths: {source} cannot be accessed (permission denied): {path}"
        )

    # Must not exceed a reasonable path depth
    parts = resolved.parts
    if len(parts) > _MAX_PATH_DEPTH:
        raise ValueError(
            f"llmem: paths: {source} path too deep ({len(parts)} components): {resolved}"
        )

    return resolved


def get_llmem_home() -> Path:
    """Return the base directory for all llmem data.

    Resolution order:
    1. LMEM_HOME env var (if set and non-empty after stripping)
    2. ~/.config/llmem/ (if it exists)
    3. ~/.lobsterdog/ (backward compat — if it exists and ~/.config/llmem/ doesn't)
    4. ~/.config/llmem/ (default, even if it doesn't exist yet)

    Returns:
        Path — never None. The path may not exist on disk yet.

    Raises:
        ValueError: If LMEM_HOME is set but points to an unsafe location.
    """
    env_val = os.environ.get("LMEM_HOME", "").strip()
    if env_val:
        candidate = Path(env_val)
        return _validate_home_path(candidate, "LMEM_HOME")

    new_path = Path.home() / ".config" / "llmem"
    old_path = Path.home() / ".lobsterdog"

    if new_path.exists():
        return new_path
    if old_path.exists():
        log.info(
            "paths: using legacy ~/.lobsterdog/ directory; "
            "run migrate_from_lobsterdog() to move to ~/.config/llmem/"
        )
        return old_path

    return new_path


def _validate_write_path(path: Path, label: str) -> Path:
    """Validate that a write target path is safe.

    Checks:
    - Must not contain '..' traversal components (checked before resolve)
    - Must not target protected system directories
    - Must not be a symlink itself

    Does NOT require the path to be within the llmem home directory —
    users may configure custom output paths. This function prevents
    clearly dangerous writes, not all writes outside the default home.

    Args:
        path: The candidate write path.
        label: Description of the file being written (for error messages).

    Returns:
        The resolved, validated path.

    Raises:
        ValueError: If the path is unsafe.
    """
    # Check traversal BEFORE resolving (resolve eliminates ..)
    if ".." in str(path):
        raise ValueError(f"llmem: paths: {label} path contains '..' traversal: {path}")

    resolved = path.resolve()

    # Block system directories
    for prefix in BLOCKED_SYSTEM_PREFIXES:
        if str(resolved).startswith(prefix):
            raise ValueError(
                f"llmem: paths: {label} path targets a protected directory: {resolved}"
            )

    # Must not be a symlink itself
    # If we can't stat the path (permission denied), treat it as unsafe
    try:
        if path.is_symlink():
            raise ValueError(
                f"llmem: paths: {label} path is a symlink (not allowed for write targets): {path}"
            )
    except OSError:
        raise ValueError(
            f"llmem: paths: {label} path cannot be accessed (permission denied): {path}"
        )

    return resolved


def validate_session_id(session_id: str) -> str:
    """Validate that a session ID is safe to use in filesystem paths.

    Rejects session IDs that contain path separators or traversal sequences,
    which could allow writing files outside the intended context directory.

    Args:
        session_id: The session ID to validate.

    Returns:
        The validated session ID string (unchanged if valid).

    Raises:
        ValueError: If the session ID contains '/', '\\', or '..' sequences.
    """
    if not session_id:
        raise ValueError("llmem: paths: session_id must not be empty")
    if "/" in session_id:
        raise ValueError(
            f"llmem: paths: session_id contains '/' (path traversal risk): {session_id!r}"
        )
    if "\\" in session_id:
        raise ValueError(
            f"llmem: paths: session_id contains '\\' (path traversal risk): {session_id!r}"
        )
    if ".." in session_id:
        raise ValueError(
            f"llmem: paths: session_id contains '..' (path traversal risk): {session_id!r}"
        )
    return session_id


def get_config_path() -> Path:
    """Return the path to config.yaml.

    Returns:
        get_llmem_home() / "config.yaml". No preconditions.
    """
    return get_llmem_home() / "config.yaml"


def get_db_path() -> Path:
    """Return the path to the memory database.

    Returns:
        get_llmem_home() / "memory.db". No preconditions.
    """
    return get_llmem_home() / "memory.db"


def get_dream_diary_path() -> Path:
    """Return the path to the dream diary.

    Returns:
        get_llmem_home() / "dream-diary.md". No preconditions.
    """
    return get_llmem_home() / "dream-diary.md"


def get_proposed_changes_path() -> Path:
    """Return the path to the proposed changes file.

    Returns:
        get_llmem_home() / "proposed-changes.md". No preconditions.
    """
    return get_llmem_home() / "proposed-changes.md"


def get_context_dir() -> Path:
    """Return the path to the context directory.

    Returns:
        get_llmem_home() / "context". No preconditions.
    """
    return get_llmem_home() / "context"


def migrate_from_lobsterdog() -> bool:
    """Migrate data from ~/.lobsterdog/ to ~/.config/llmem/.

    Copies config.yaml, memory.db, dream-diary.md, proposed-changes.md,
    and context/ directory from ~/.lobsterdog/ to ~/.config/llmem/
    only if the source exists and the destination doesn't.
    Never deletes the source directory.

    Uses shutil.copy2 with follow_symlinks=False to avoid following
    symlinks — source symlinks are skipped, not followed.

    Returns:
        True if any migration was performed, False if no migration needed.
        Idempotent — calling twice does nothing on the second call.
    """
    old_home = Path.home() / ".lobsterdog"
    new_home = Path.home() / ".config" / "llmem"

    # Resolve to avoid symlink attacks on source/destination
    old_home = old_home.resolve()
    new_home = new_home.resolve()

    if not old_home.exists():
        return False
    if new_home.exists():
        return False

    migrated = False
    new_home.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "config.yaml",
        "memory.db",
        "dream-diary.md",
        "proposed-changes.md",
    ]

    for filename in files_to_copy:
        src = old_home / filename
        dst = new_home / filename
        # Skip symlinks — only copy regular files
        if src.is_symlink():
            log.warning("paths: skipping symlink during migration: %s", src)
            continue
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(str(src), str(dst), follow_symlinks=False)
                log.info(
                    "paths: migrated %s from ~/.lobsterdog/ to ~/.config/llmem/",
                    filename,
                )
                migrated = True
            except OSError as e:
                log.warning("paths: failed to migrate %s: %s", filename, e)

    # Copy context/ directory — skip symlinks
    src_context = old_home / "context"
    dst_context = new_home / "context"
    if src_context.is_symlink():
        log.warning(
            "paths: skipping symlink directory during migration: %s", src_context
        )
    elif src_context.is_dir() and not dst_context.exists():
        try:
            shutil.copytree(
                str(src_context),
                str(dst_context),
                symlinks=False,  # Don't preserve symlinks — copy referenced content
            )
            log.info(
                "paths: migrated context/ directory from ~/.lobsterdog/ to ~/.config/llmem/"
            )
            migrated = True
        except OSError as e:
            log.warning("paths: failed to migrate context/ directory: %s", e)

    return migrated
