"""Configuration loading and writing for llmem."""

import copy
import logging
import os
from pathlib import Path

import yaml

from .paths import (
    get_config_path as _get_config_path,
    _validate_home_path,
)
from .url_validate import is_safe_url, _strip_credentials

log = logging.getLogger(__name__)


def _resolve_config(
    config_path: Path | None = None, config: dict | None = None
) -> dict:
    """Load config lazily if not provided. Internal helper for DRY.

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict (skips file load if provided).

    Returns:
        A config dict (may be empty if file is missing or invalid).
    """
    if config is not None:
        return config
    return load_config(config_path)


DEFAULTS = {
    "memory": {
        "db": None,  # Resolved from get_db_path() at access time
        "ollama_url": "http://localhost:11434",
        "embed_model": "nomic-embed-text",
        "extract_model": "glm-5.1:cloud",
        "context_budget": 4000,
        "auto_extract": True,
        "max_file_size": 10 * 1024 * 1024,
    },
    "dream": {
        "enabled": True,
        "schedule": "*-*-* 03:00:00",
        "similarity_threshold": 0.92,
        "decay_rate": 0.05,
        "decay_interval_days": 30,
        "decay_floor": 0.3,
        "confidence_floor": 0.3,
        "boost_threshold": 5,
        "boost_amount": 0.05,
        "min_score": 0.5,
        "min_recall_count": 3,
        "min_unique_queries": 1,
        "boost_on_promote": 0.1,
        "merge_model": "qwen2.5:1.5b",
        "diary_path": None,  # Resolved from get_dream_diary_path()
        "report_path": None,  # Resolved from get_dream_report_path()
        "behavioral_threshold": 3,
        "behavioral_lookback_days": 30,
        "calibration_enabled": True,
        "stale_procedure_days": 30,
        "calibration_lookback_days": 90,
        "auto_link_threshold": 0.85,
    },
    "opencode": {
        "context_dir": None,  # Resolved from get_context_dir()
        "db_path": str(Path("~/.local/share/opencode/opencode.db").expanduser()),
    },
    "session": {
        "adapter": "opencode",
    },
}


_resolved_defaults_cache: dict | None = None


def _resolve_defaults() -> dict:
    """Resolve path-based defaults that depend on get_llmem_home().

    Returns a dict with path-based defaults resolved. The result is cached
    after the first call — DEFAULTS is immutable after module load, so
    there is no need for repeated deep copies. Each nested value is an
    independent copy of the module-level DEFAULTS constant.

    Returns:
        A dict with resolved path defaults. Callers should NOT mutate
        the returned dict — it is shared across calls for performance.
    """
    global _resolved_defaults_cache
    if _resolved_defaults_cache is not None:
        return _resolved_defaults_cache

    from .paths import (
        get_db_path as _get_db_path,
        get_dream_diary_path,
        get_dream_report_path as _paths_dream_report_path,
        get_context_dir,
    )

    # Deep copy to avoid shared mutable references with module-level DEFAULTS.
    # dict(v) only copies one level — nested dict/list values would
    # still alias the original. copy.deepcopy handles all nesting levels.
    # This is done once and cached — callers only read, never mutate.
    defaults = copy.deepcopy(DEFAULTS)
    defaults["memory"]["db"] = str(_get_db_path())
    defaults["dream"]["diary_path"] = str(get_dream_diary_path())
    defaults["dream"]["report_path"] = str(_paths_dream_report_path())
    defaults["opencode"]["context_dir"] = str(get_context_dir())
    _resolved_defaults_cache = defaults
    return defaults


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from config.yaml.

    Args:
        config_path: Optional path to config.yaml. Defaults to
            get_llmem_home() / "config.yaml".

    Returns:
        A config dict. Returns empty dict if file doesn't exist.
        Logs a warning if file exists but can't be parsed.
    """

    path = Path(config_path) if config_path else _get_config_path()
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            log.warning("llmem: config: file %s is not a YAML mapping, ignoring", path)
            return {}
        return data
    except Exception as e:
        log.warning("llmem: config: failed to load %s: %s", path, e)
        return {}


def get_config_value(key_path: str, config_path: Path | None = None) -> object:
    """Look up a config value by dot-separated key path, merging with defaults.

    The key_path (e.g. 'memory.db') is looked up in the loaded config first.
    If not found, falls back to the corresponding default from _resolve_defaults().

    Args:
        key_path: Dot-separated path (e.g., 'memory.ollama_url').
        config_path: Optional path to config.yaml.

    Returns:
        The config value if found in user config or defaults, otherwise None.
    """
    config = _resolve_config(config_path)
    defaults = _resolve_defaults()
    keys = key_path.split(".")

    # Try user config first
    current = config
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            # Not in user config — fall back to defaults
            current = defaults
            for dk in keys:
                if isinstance(current, dict) and dk in current:
                    current = current[dk]
                else:
                    return None
            return current
    return current


def get_db_path(config_path: Path | None = None, config: dict | None = None) -> Path:
    config = _resolve_config(config_path, config)
    db = config.get("memory", {}).get("db")
    defaults = _resolve_defaults()
    return Path(db).expanduser() if db else Path(defaults["memory"]["db"])


def get_ollama_url(config_path: Path | None = None, config: dict | None = None) -> str:
    """Return the Ollama base URL from config, falling back to the default.

    Validates the URL with is_safe_url() to prevent SSRF attacks
    (e.g. metadata endpoint access). Strips credentials from error
    messages to avoid leaking secrets.

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict.

    Returns:
        The validated Ollama URL string.

    Raises:
        ValueError: If the URL fails scheme or is_safe_url() validation.
    """
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    url = config.get("memory", {}).get("ollama_url") or defaults["memory"]["ollama_url"]
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"llmem: config: unsafe Ollama URL (must be http/https): {_strip_credentials(url)!r}"
        )
    if not is_safe_url(url, allow_remote=True):
        raise ValueError(
            f"llmem: config: Ollama URL blocked (unsafe address): {_strip_credentials(url)!r}"
        )
    return url


def _as_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() not in ("false", "0", "no", "off", "")
    return bool(val)


def is_auto_extract(
    config_path: Path | None = None, config: dict | None = None
) -> bool:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    val = config.get("memory", {}).get("auto_extract")
    if val is None:
        return defaults["memory"]["auto_extract"]
    return _as_bool(val)


def get_max_file_size(
    config_path: Path | None = None, config: dict | None = None
) -> int:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    size = config.get("memory", {}).get("max_file_size")
    if size is None:
        return defaults["memory"]["max_file_size"]
    return int(size)


def get_dream_config(
    config_path: Path | None = None, config: dict | None = None
) -> dict:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    dream = config.get("dream", {})
    dream_defaults = defaults["dream"]
    result = {}
    for key, default_val in dream_defaults.items():
        result[key] = dream.get(key, default_val)
    return result


def is_dream_enabled(
    config_path: Path | None = None, config: dict | None = None
) -> bool:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    val = config.get("dream", {}).get("enabled")
    if val is None:
        return defaults["dream"]["enabled"]
    return _as_bool(val)


def get_dream_schedule(
    config_path: Path | None = None, config: dict | None = None
) -> str:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    return config.get("dream", {}).get("schedule") or defaults["dream"]["schedule"]


def get_dream_report_path(
    config_path: Path | None = None, config: dict | None = None
) -> Path:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    path = (
        config.get("dream", {}).get("report_path") or defaults["dream"]["report_path"]
    )
    return Path(path).expanduser()


def get_server_auth_token(
    config_path: Path | None = None, config: dict | None = None
) -> str | None:
    """Return the server auth token from config, or None if not set.

    Validates minimum token strength when a token is present:
    must be at least 16 characters long. Weak tokens are rejected
    to prevent trivial bypass.

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict.

    Returns:
        The auth token string if set and valid, None otherwise.

    Raises:
        ValueError: If the token is set but too short (< 16 chars).
    """
    config = _resolve_config(config_path, config)
    token = config.get("server", {}).get("auth_token")
    if not token:
        return None
    if not isinstance(token, str):
        raise ValueError(
            "llmem: config: server.auth_token must be a string, "
            f"got {type(token).__name__}"
        )
    min_length = 16
    if len(token) < min_length:
        raise ValueError(
            f"llmem: config: server.auth_token is too short "
            f"({len(token)} chars, minimum {min_length}) — "
            'generate a strong token with: python3 -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    return token


def get_server_port(config_path: Path | None = None, config: dict | None = None) -> int:
    config = _resolve_config(config_path, config)
    port = config.get("server", {}).get("port")
    if port is None:
        return 8322
    try:
        return int(port)
    except (ValueError, TypeError):
        return 8322


def get_opencode_db_path(
    config_path: Path | None = None, config: dict | None = None
) -> Path:
    """Return the path to the opencode SQLite database.

    Validates that the path does not target system directories or contain
    path traversal. The path is resolved and checked against blocked prefixes.

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict.

    Returns:
        Resolved and validated Path to the opencode database file.

    Raises:
        ValueError: If the configured path targets a system directory or
            contains '..' traversal.
    """
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    path = config.get("opencode", {}).get("db_path") or defaults["opencode"]["db_path"]
    candidate = Path(path).expanduser()
    # Validate the path before resolving — _validate_home_path checks for
    # '..' traversal, symlinks, and system directory targeting.
    return _validate_home_path(candidate, "opencode.db_path")


def write_config_yaml(path: Path, config: dict, force: bool = False) -> bool:
    """Write *config* as YAML to *path*, creating parent directories if needed.

    Creates the parent directory with ``0o700`` permissions (matching
    ``llmem/store.py`` and ``llmem/paths.py`` conventions).  Uses
    ``yaml.dump`` with ``default_flow_style=False`` for human-readable
    output.

    Args:
        path: Destination file path (e.g. ``get_llmem_home() / "config.yaml"``).
        config: The configuration dict to write.
        force: If ``False`` (default) and *path* already exists, return
            ``False`` without writing. If ``True``, overwrite the
            existing file.

    Returns:
        ``True`` if the file was written successfully, ``False`` if the
        file already existed and *force* was ``False``.

    Raises:
        OSError: On I/O failures (permission denied, disk full, etc.).
    """
    if path.exists() and not force:
        log.info(
            "llmem: config: %s already exists, skipping (use force=True to overwrite)",
            path,
        )
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(str(path.parent), 0o700)

    content = yaml.dump(config, default_flow_style=False, sort_keys=False)
    # Write with 0o600 permissions to protect API keys and secrets from
    # other users on shared systems. Config files may contain OPENAI_API_KEY,
    # ANTHROPIC_API_KEY, and other credentials.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, content.encode())
    finally:
        os.close(fd)
    return True
