"""Tests for llmem.config module."""

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
