"""Configuration loading for llmem (legacy compat shim — prefer llmem.config)."""

import logging
from pathlib import Path

import yaml

from .url_validate import is_safe_url, _strip_credentials

log = logging.getLogger(__name__)

CONFIG_PATH = Path("~/.config/llmem/config.yaml").expanduser()

DEFAULTS = {
    "memory": {
        "db": str(Path("~/.config/llmem/memory.db").expanduser()),
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
        "diary_path": str(Path("~/.config/llmem/dream-diary.md").expanduser()),
        "report_path": str(Path("~/.agent/diagrams/dream-report.html").expanduser()),
        "behavioral_threshold": 3,
        "behavioral_lookback_days": 30,
        "skill_patch_threshold": 3,
        "proposed_changes_path": str(
            Path("~/.config/llmem/proposed-changes.md").expanduser()
        ),
        "calibration_enabled": True,
        "stale_procedure_days": 30,
        "calibration_lookback_days": 90,
    },
    "opencode": {
        "context_dir": str(Path("~/.config/llmem/context").expanduser()),
        "db_path": str(Path("~/.local/share/opencode/opencode.db").expanduser()),
    },
    "resume": {
        "backend": "ollama",
        "model": "qwen2.5:1.5b",
        "output_dir": str(Path("~/.agent/resumes").expanduser()),
        "ollama_url": "http://localhost:11434",
    },
    "hook": {
        "source_filter": "direct",
    },
    "correction_detection": {
        "enabled": True,
    },
    "provider": {
        "default": "ollama",
        "embed": {},
        "generate": {},
        "local": {"model": "all-MiniLM-L6-v2"},
    },
}


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else CONFIG_PATH
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        log.warning("config: failed to load config from %s", path, exc_info=True)
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
    if config is None:
        config = load_config(config_path)
    db = (config.get("memory") or {}).get("db") or DEFAULTS["memory"]["db"]
    return Path(db).expanduser()


def get_ollama_url(config_path: Path | None = None, config: dict | None = None) -> str:
    if config is None:
        config = load_config(config_path)
    url = (config.get("memory") or {}).get("ollama_url") or DEFAULTS["memory"][
        "ollama_url"
    ]
    if not url.startswith(("http://", "https://")):
        raise ValueError("config: Ollama URL must be http/https")
    if not is_safe_url(url, allow_remote=True):
        raise ValueError(
            f"config: Ollama URL blocked (unsafe address): {_strip_credentials(url)!r}"
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
    if config is None:
        config = load_config(config_path)
    val = (config.get("memory") or {}).get("auto_extract")
    if val is None:
        return DEFAULTS["memory"]["auto_extract"]
    return _as_bool(val)


def get_session_dirs(
    config_path: Path | None = None, config: dict | None = None
) -> list[Path]:
    if config is None:
        config = load_config(config_path)
    dirs = (config.get("memory") or {}).get("session_dirs")
    if dirs is None:
        return [Path(d) for d in DEFAULTS["memory"]["session_dirs"]]
    if isinstance(dirs, str):
        dirs = [dirs]
    return [Path(d).expanduser() for d in dirs]


def get_max_file_size(
    config_path: Path | None = None, config: dict | None = None
) -> int:
    if config is None:
        config = load_config(config_path)
    size = (config.get("memory") or {}).get("max_file_size")
    if size is None:
        return DEFAULTS["memory"]["max_file_size"]
    return int(size)


def get_prospective_model(
    config_path: Path | None = None, config: dict | None = None
) -> str:
    if config is None:
        config = load_config(config_path)
    return (config.get("memory") or {}).get("prospective_model") or DEFAULTS["memory"][
        "prospective_model"
    ]


def get_dream_config(
    config_path: Path | None = None, config: dict | None = None
) -> dict:
    if config is None:
        config = load_config(config_path)
    dream = config.get("dream") or {}
    defaults = DEFAULTS["dream"]
    result = {}
    for key, default_val in defaults.items():
        val = dream.get(key)
        result[key] = default_val if val is None else val
    return result


def is_dream_enabled(
    config_path: Path | None = None, config: dict | None = None
) -> bool:
    if config is None:
        config = load_config(config_path)
    val = (config.get("dream") or {}).get("enabled")
    if val is None:
        return DEFAULTS["dream"]["enabled"]
    return _as_bool(val)


def get_dream_schedule(
    config_path: Path | None = None, config: dict | None = None
) -> str:
    if config is None:
        config = load_config(config_path)
    return (config.get("dream") or {}).get("schedule") or DEFAULTS["dream"]["schedule"]


def get_dream_report_path(
    config_path: Path | None = None, config: dict | None = None
) -> Path:
    if config is None:
        config = load_config(config_path)
    path = (config.get("dream") or {}).get("report_path") or DEFAULTS["dream"][
        "report_path"
    ]
    return Path(path).expanduser()


def get_server_auth_token(
    config_path: Path | None = None, config: dict | None = None
) -> str | None:
    if config is None:
        config = load_config(config_path)
    token = (config.get("server") or {}).get("auth_token")
    if not token:
        return None
    return token


def get_server_port(config_path: Path | None = None, config: dict | None = None) -> int:
    if config is None:
        config = load_config(config_path)
    port = (config.get("server") or {}).get("port")
    if port is None:
        return 8322
    try:
        return int(port)
    except (ValueError, TypeError):
        return 8322


def get_resume_config(
    config_path: Path | None = None, config: dict | None = None
) -> dict:
    if config is None:
        config = load_config(config_path)
    resume = config.get("resume") or {}
    defaults = DEFAULTS["resume"]
    result = {}
    for key, default_val in defaults.items():
        val = resume.get(key)
        result[key] = default_val if val is None else val
    return result


def get_resume_model(
    config_path: Path | None = None, config: dict | None = None
) -> str:
    if config is None:
        config = load_config(config_path)
    return (config.get("resume") or {}).get("model") or DEFAULTS["resume"]["model"]


def get_resume_output_dir(
    config_path: Path | None = None, config: dict | None = None
) -> Path:
    if config is None:
        config = load_config(config_path)
    path = (config.get("resume") or {}).get("output_dir") or DEFAULTS["resume"][
        "output_dir"
    ]
    return Path(path).expanduser()


def get_opencode_db_path(
    config_path: Path | None = None, config: dict | None = None
) -> Path:
    """Return the path to the opencode SQLite database.

    Follows the same pattern as get_db_path(): reads from config
    (opencode.db_path key), falling back to the DEFAULTS dict.
    String paths are coerced to Path, ~ is expanded, and the result
    is resolved to an absolute path.

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict.

    Returns:
        Resolved Path to the opencode database file.
    """
    if config is None:
        config = load_config(config_path)
    path = (config.get("opencode") or {}).get("db_path") or DEFAULTS["opencode"][
        "db_path"
    ]
    return Path(path).expanduser().resolve()


_VALID_SOURCE_FILTERS = {"direct", "pipeline", "all"}


def get_source_filter(
    config_path: Path | None = None, config: dict | None = None
) -> str:
    """Return the configured source filter for the hook command.

    Reads the ``hook.source_filter`` key from config. Valid values are
    ``"direct"``, ``"pipeline"``, or ``"all"``. Falls back to
    ``"direct"`` (from DEFAULTS) when the key is absent. Logs a warning
    and falls back to ``"direct"`` for invalid values.

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict.

    Returns:
        One of ``"direct"``, ``"pipeline"``, or ``"all"``.
    """
    if config is None:
        config = load_config(config_path)
    val = (config.get("hook") or {}).get("source_filter")
    if val is None:
        return DEFAULTS["hook"]["source_filter"]
    if val not in _VALID_SOURCE_FILTERS:
        log.warning(
            "config: invalid hook.source_filter value %r, falling back to 'direct'",
            val,
        )
        return "direct"
    return val


def is_correction_detection_enabled(
    config_path: Path | None = None, config: dict | None = None
) -> bool:
    """Return whether correction detection is enabled.

    Reads ``correction_detection.enabled`` from config, defaulting to True.
    Uses ``_as_bool`` for string coercion so that ``"true"``, ``"1"``,
    ``"yes"``, and ``"on"`` all resolve to True, and ``"false"``, ``"0"``,
    ``"no"``, and ``"off"`` resolve to False.

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict.

    Returns:
        True when the key is absent or set to a truthy value.
        False only when explicitly set to ``false``, ``0``, ``no``, or ``off``.
    """
    if config is None:
        config = load_config(config_path)
    val = (config.get("correction_detection") or {}).get("enabled")
    if val is None:
        return DEFAULTS["correction_detection"]["enabled"]
    return _as_bool(val)


def get_provider_config(
    config_path: Path | None = None, config: dict | None = None
) -> dict:
    """Return the provider section from config with defaults applied.

    Reads the ``provider`` key from config, falling back to
    ``DEFAULTS["provider"]``. Returns dict with keys ``default``,
    ``embed``, ``generate``, and ``local``.

    Args:
        config_path: Optional path to config.yaml.
        config: Optional pre-loaded config dict.

    Returns:
        A dict containing ``default``, ``embed``, ``generate``, and
        ``local`` keys with defaults filled in.
    """
    if config is None:
        config = load_config(config_path)
    provider = config.get("provider") or {}
    defaults = DEFAULTS["provider"]
    result = {}
    for key, default_val in defaults.items():
        val = provider.get(key)
        result[key] = default_val if val is None else val
    return result
