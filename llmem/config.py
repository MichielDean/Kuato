"""Configuration loading for llmem."""

import logging
from pathlib import Path

import yaml

from .paths import get_llmem_home, get_config_path as _get_config_path
from .url_validate import is_safe_url

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
        "extract_model": "qwen2.5:1.5b",
        "prospective_model": "qwen2.5:1.5b",
        "context_budget": 4000,
        "auto_extract": True,
        "max_file_size": 10 * 1024 * 1024,
        "session_dirs": [str(Path("~/.local/share/opencode/sessions").expanduser())],
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
        "report_path": str(Path("~/.agent/diagrams/dream-report.html").expanduser()),
        "behavioral_threshold": 3,
        "behavioral_lookback_days": 30,
        "skill_patch_threshold": 3,
        "proposed_changes_path": None,  # Resolved from get_proposed_changes_path()
        "calibration_enabled": True,
        "stale_procedure_days": 30,
        "calibration_lookback_days": 90,
    },
    "opencode": {
        "context_dir": None,  # Resolved from get_context_dir()
        "db_path": str(Path("~/.local/share/opencode/opencode.db").expanduser()),
    },
    "correction_detection": {
        "enabled": True,
    },
    # NOTE: "resume" and "hook.source_filter" sections are NOT included.
    # These are extension concerns, not core llmem config.
}


def _resolve_defaults() -> dict:
    """Resolve path-based defaults that depend on get_llmem_home().

    Returns a deep copy of DEFAULTS with path-based defaults resolved.
    Each nested dict is independently copied to avoid shared references
    with the module-level DEFAULTS constant.
    """
    from .paths import (
        get_db_path as _get_db_path,
        get_dream_diary_path,
        get_proposed_changes_path,
        get_context_dir,
    )

    # Deep copy every nested dict to avoid shared mutable references
    defaults = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
    defaults["memory"]["db"] = str(_get_db_path())
    defaults["dream"]["diary_path"] = str(get_dream_diary_path())
    defaults["dream"]["proposed_changes_path"] = str(get_proposed_changes_path())
    defaults["opencode"]["context_dir"] = str(get_context_dir())
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
    from .paths import get_config_path as _get_config_path

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
    config = load_config(config_path)
    keys = key_path.split(".")
    current = config
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


def get_db_path(config_path: Path | None = None, config: dict | None = None) -> Path:
    config = _resolve_config(config_path, config)
    db = config.get("memory", {}).get("db")
    defaults = _resolve_defaults()
    return Path(db).expanduser() if db else Path(defaults["memory"]["db"])


def get_ollama_url(config_path: Path | None = None, config: dict | None = None) -> str:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    url = config.get("memory", {}).get("ollama_url") or defaults["memory"]["ollama_url"]
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"llmem: config: unsafe Ollama URL (must be http/https): {url!r}"
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


def get_session_dirs(
    config_path: Path | None = None, config: dict | None = None
) -> list[Path]:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    dirs = config.get("memory", {}).get("session_dirs")
    if dirs is None:
        return [Path(d) for d in defaults["memory"]["session_dirs"]]
    if isinstance(dirs, str):
        dirs = [dirs]
    return [Path(d).expanduser() for d in dirs]


def get_max_file_size(
    config_path: Path | None = None, config: dict | None = None
) -> int:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    size = config.get("memory", {}).get("max_file_size")
    if size is None:
        return defaults["memory"]["max_file_size"]
    return int(size)


def get_prospective_model(
    config_path: Path | None = None, config: dict | None = None
) -> str:
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    return (
        config.get("memory", {}).get("prospective_model")
        or defaults["memory"]["prospective_model"]
    )


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
    config = _resolve_config(config_path, config)
    token = config.get("server", {}).get("auth_token")
    if not token:
        return None
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

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict.

    Returns:
        Resolved Path to the opencode database file.
    """
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    path = config.get("opencode", {}).get("db_path") or defaults["opencode"]["db_path"]
    return Path(path).expanduser().resolve()


def is_correction_detection_enabled(
    config_path: Path | None = None, config: dict | None = None
) -> bool:
    """Return whether correction detection is enabled.

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict.

    Returns:
        True when the key is absent or set to a truthy value.
        False only when explicitly set to false, 0, no, or off.
    """
    config = _resolve_config(config_path, config)
    defaults = _resolve_defaults()
    val = config.get("correction_detection", {}).get("enabled")
    if val is None:
        return defaults["correction_detection"]["enabled"]
    return _as_bool(val)
