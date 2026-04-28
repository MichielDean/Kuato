"""Path resolution for llmem data directories.

All hardcoded ~/.lobsterdog/ paths are replaced with ~/.config/llmem/
(with LMEM_HOME env var override). Backward compatibility:
~/.lobsterdog/ is checked first for migration.
"""

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def get_llmem_home() -> Path:
    """Return the base directory for all llmem data.

    Resolution order:
    1. LMEM_HOME env var (if set and non-empty after stripping)
    2. ~/.config/llmem/ (if it exists)
    3. ~/.lobsterdog/ (backward compat — if it exists and ~/.config/llmem/ doesn't)
    4. ~/.config/llmem/ (default, even if it doesn't exist yet)

    Returns:
        Path — never None. The path may not exist on disk yet.
    """
    import os

    env_val = os.environ.get("LMEM_HOME", "").strip()
    if env_val:
        return Path(env_val)

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

    Returns:
        True if any migration was performed, False if no migration needed.
        Idempotent — calling twice does nothing on the second call.
    """
    old_home = Path.home() / ".lobsterdog"
    new_home = Path.home() / ".config" / "llmem"

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
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(str(src), str(dst))
                log.info(
                    "paths: migrated %s from ~/.lobsterdog/ to ~/.config/llmem/",
                    filename,
                )
                migrated = True
            except OSError as e:
                log.warning("paths: failed to migrate %s: %s", filename, e)

    # Copy context/ directory
    src_context = old_home / "context"
    dst_context = new_home / "context"
    if src_context.is_dir() and not dst_context.exists():
        try:
            shutil.copytree(str(src_context), str(dst_context))
            log.info(
                "paths: migrated context/ directory from ~/.lobsterdog/ to ~/.config/llmem/"
            )
            migrated = True
        except OSError as e:
            log.warning("paths: failed to migrate context/ directory: %s", e)

    return migrated
