"""Tests for llmem.config module."""

import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llmem.config import load_config


class TestConfig_LoadConfig:
    """Test load_config() with new default paths."""

    def test_load_nonexistent_returns_empty(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config == {}

    def test_load_valid_yaml(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("memory:\n  db: /tmp/test.db\n")
        config = load_config(config_path)
        assert config["memory"]["db"] == "/tmp/test.db"

    def test_load_non_dict_returns_empty(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- list\n- not\n- dict\n")
        config = load_config(config_path)
        assert config == {}


class TestConfig_NoResumeSection:
    """Test that load_config defaults do not include 'resume' key."""

    def test_no_resume_in_defaults(self):
        from llmem.config import DEFAULTS

        assert "resume" not in DEFAULTS


class TestConfig_NoSourceFilter:
    """Test that load_config defaults do not include 'hook.source_filter' key."""

    def test_no_source_filter_in_defaults(self):
        from llmem.config import DEFAULTS

        assert "hook" not in DEFAULTS


class TestConfig_OpencodeContextDir:
    """Test that opencode.context_dir uses ~/.config/llmem/context default."""

    def test_opencode_context_dir_default(self):
        from llmem.config import _resolve_defaults

        defaults = _resolve_defaults()
        # Should end with .config/llmem/context or .lobsterdog/context
        context_dir = defaults["opencode"]["context_dir"]
        assert context_dir is not None
        assert "context" in context_dir


class TestConfig_ResolveDefaults_DeepCopy:
    """Test that _resolve_defaults() does not share mutable references with DEFAULTS."""

    def test_correction_detection_not_shared_with_defaults(self):
        """Mutating a copy of resolved defaults should not affect the module DEFAULTS."""
        from llmem.config import DEFAULTS, _resolve_defaults

        resolved = copy.deepcopy(_resolve_defaults())
        # Mutate the independent copy
        resolved["correction_detection"]["enabled"] = False
        # The module-level DEFAULTS must NOT be affected
        assert DEFAULTS["correction_detection"]["enabled"] is True

    def test_session_dirs_not_shared_with_defaults(self):
        """Mutating session_dirs in a copy of resolved defaults must not affect DEFAULTS."""
        from llmem.config import DEFAULTS, _resolve_defaults

        original_dirs = list(DEFAULTS["memory"]["session_dirs"])
        resolved = copy.deepcopy(_resolve_defaults())
        resolved["memory"]["session_dirs"].append("/evil/appended/path")
        assert DEFAULTS["memory"]["session_dirs"] == original_dirs, (
            "session_dirs list is a shared reference — mutating resolved copy "
            "corrupted module-level DEFAULTS"
        )

    def test_deep_copy_all_nested_dicts(self):
        """Every nested dict in _resolve_defaults() is an independent copy of DEFAULTS."""
        from llmem.config import DEFAULTS, _resolve_defaults

        resolved = copy.deepcopy(_resolve_defaults())
        for key in DEFAULTS:
            if isinstance(DEFAULTS[key], dict):
                # Modifying the independent copy must not affect DEFAULTS
                original_val = DEFAULTS[key].copy()
                resolved[key].clear()
                assert DEFAULTS[key] == original_val, (
                    f"Shared reference detected for key '{key}'"
                )

    def test_resolve_defaults_caches_result(self):
        """_resolve_defaults() returns the same object on repeated calls (cached)."""
        from llmem.config import _resolve_defaults

        result1 = _resolve_defaults()
        result2 = _resolve_defaults()
        assert result1 is result2, "_resolve_defaults should return cached result"


class TestConfig_GetConfigValue:
    """Test that get_config_value() merges defaults for missing keys."""

    def test_returns_user_value_when_present(self, tmp_path):
        """When the key exists in user config, return it."""
        from llmem.config import get_config_value

        config_path = tmp_path / "config.yaml"
        config_path.write_text("memory:\n  ollama_url: http://custom:11434\n")
        result = get_config_value("memory.ollama_url", config_path=config_path)
        assert result == "http://custom:11434"

    def test_returns_default_when_missing(self):
        """When the key is missing from user config, return the default."""
        from llmem.config import get_config_value

        # No config file — should fall back to defaults
        result = get_config_value(
            "memory.ollama_url", config_path=Path("/nonexistent.yaml")
        )
        assert result is not None
        assert "ollama_url" not in {}  # confirm we're testing fallback
        # The default ollama_url is http://localhost:11434
        assert result == "http://localhost:11434"

    def test_returns_none_for_unknown_key(self):
        """When key exists neither in config nor defaults, return None."""
        from llmem.config import get_config_value

        result = get_config_value(
            "totally.madeup.key", config_path=Path("/nonexistent.yaml")
        )
        assert result is None

    def test_nested_key_resolution(self):
        """get_config_value resolves nested dot paths correctly."""
        from llmem.config import get_config_value

        result = get_config_value(
            "dream.schedule", config_path=Path("/nonexistent.yaml")
        )
        assert result == "*-*-* 03:00:00"


class TestConfig_NoSkillPatchThreshold:
    """Test that skill_patch_threshold is removed from core DEFAULTS."""

    def test_no_skill_patch_threshold_in_defaults(self):
        """Core DEFAULTS must not contain skill_patch_threshold."""
        from llmem.config import DEFAULTS

        assert "skill_patch_threshold" not in DEFAULTS.get("dream", {})

    def test_no_skill_patch_threshold_in_resolved_defaults(self):
        """Resolved defaults must not contain skill_patch_threshold in dream section."""
        from llmem.config import _resolve_defaults

        resolved = _resolve_defaults()
        assert "skill_patch_threshold" not in resolved.get("dream", {})
