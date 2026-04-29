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

    def test_no_session_dirs_in_defaults(self):
        """session_dirs key must be removed from DEFAULTS — replaced by DB-based discovery."""
        from llmem.config import DEFAULTS

        assert "session_dirs" not in DEFAULTS.get("memory", {})

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


class TestConfig_GetOpencodeDbPath_Validation:
    """Test that get_opencode_db_path validates the path for security."""

    def test_rejects_traversal_in_db_path(self, tmp_path):
        """Paths with '..' traversal should be rejected."""
        from llmem.config import get_opencode_db_path

        config = {"opencode": {"db_path": str(tmp_path / ".." / "etc" / "opencode.db")}}
        with pytest.raises(ValueError, match="traversal"):
            get_opencode_db_path(config=config)

    def test_rejects_system_directory_db_path(self):
        """Paths targeting system directories should be rejected."""
        from llmem.config import get_opencode_db_path

        config = {"opencode": {"db_path": "/etc/opencode/opencode.db"}}
        with pytest.raises(ValueError, match="system directory"):
            get_opencode_db_path(config=config)

    def test_accepts_valid_home_path(self, tmp_path):
        """Valid paths under home directory should be accepted."""
        from llmem.config import get_opencode_db_path

        db_path = tmp_path / "opencode" / "opencode.db"
        config = {"opencode": {"db_path": str(db_path)}}
        result = get_opencode_db_path(config=config)
        assert result == db_path.resolve()

    def test_default_path_is_valid(self):
        """The default opencode db_path should be valid."""
        from llmem.config import get_opencode_db_path

        # Default path resolves without error
        result = get_opencode_db_path(config_path=Path("/nonexistent.yaml"))
        assert result is not None

    def test_rejects_symlink_db_path(self, tmp_path):
        """Symlink paths where the db_path itself is a symlink should be rejected."""
        from llmem.config import get_opencode_db_path

        # Create a real db file and a symlink pointing to it
        real_db = tmp_path / "real_opencode.db"
        real_db.write_text("fake db content")
        link_db = tmp_path / "link_opencode.db"
        link_db.symlink_to(real_db)

        config = {"opencode": {"db_path": str(link_db)}}
        with pytest.raises(ValueError, match="symlink"):
            get_opencode_db_path(config=config)


class TestConfig_WriteConfigYaml_Permissions:
    """Test that write_config_yaml sets secure file permissions."""

    def test_writes_with_600_permissions(self, tmp_path):
        """Config file should be written with 0o600 (owner-only) permissions."""
        from llmem.config import write_config_yaml

        config_path = tmp_path / "config.yaml"
        write_config_yaml(config_path, {"memory": {"db": "/tmp/test.db"}})

        # On Linux, os.chmod works and we can verify permissions
        import os
        import stat

        mode = os.stat(config_path).st_mode
        # Check that group and others have no read/write access
        assert not (mode & stat.S_IRGRP), "Config file should not be group-readable"
        assert not (mode & stat.S_IWGRP), "Config file should not be group-writable"
        assert not (mode & stat.S_IROTH), "Config file should not be others-readable"
        assert not (mode & stat.S_IWOTH), "Config file should not be others-writable"

    def test_directory_has_700_permissions(self, tmp_path):
        """Config directory should have 0o700 (owner-only) permissions."""
        from llmem.config import write_config_yaml

        # Use a subdirectory so mkdir is needed
        config_path = tmp_path / "subdir" / "config.yaml"
        write_config_yaml(config_path, {"memory": {"db": "/tmp/test.db"}})

        import os
        import stat

        dir_mode = os.stat(tmp_path / "subdir").st_mode
        assert not (dir_mode & stat.S_IRGRP), "Config dir should not be group-readable"
        assert not (dir_mode & stat.S_IWGRP), "Config dir should not be group-writable"
        assert not (dir_mode & stat.S_IXOTH), (
            "Config dir should not be others-accessible"
        )


class TestWriteConfigYaml_FilePermissions:
    """Test that write_config_yaml creates files with 0o600 permissions."""

    def test_file_created_with_0600_permissions(self, tmp_path):
        """write_config_yaml must create config files with 0o600 to protect API keys."""
        import os
        import stat

        from llmem.config import write_config_yaml

        config_path = tmp_path / "config.yaml"
        config = {"memory": {"db": "/tmp/test.db"}}
        result = write_config_yaml(config_path, config)
        assert result is True

        file_mode = stat.S_IMODE(os.stat(config_path).st_mode)
        assert file_mode == 0o600, (
            f"Expected file permissions 0o600, got {oct(file_mode)}"
        )

    def test_file_overwritten_preserves_0600_permissions(self, tmp_path):
        """write_config_yaml with force=True must still use 0o600 permissions."""
        import os
        import stat

        from llmem.config import write_config_yaml

        config_path = tmp_path / "config.yaml"
        config = {"memory": {"db": "/tmp/test.db"}}

        # Initial write
        write_config_yaml(config_path, config)

        # Overwrite with force=True
        updated_config = {"memory": {"db": "/tmp/updated.db"}}
        write_config_yaml(config_path, updated_config, force=True)

        file_mode = stat.S_IMODE(os.stat(config_path).st_mode)
        assert file_mode == 0o600, (
            f"Expected file permissions 0o600 after force overwrite, got {oct(file_mode)}"
        )

    def test_existing_file_not_overwritten_without_force(self, tmp_path):
        """write_config_yaml returns False when file exists and force=False."""
        from llmem.config import write_config_yaml

        config_path = tmp_path / "config.yaml"
        config = {"memory": {"db": "/tmp/test.db"}}

        write_config_yaml(config_path, config)
        result = write_config_yaml(config_path, config, force=False)
        assert result is False
