"""Tests for llmem.paths module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llmem.paths import (
    get_llmem_home,
    get_config_path,
    get_db_path,
    get_dream_diary_path,
    get_proposed_changes_path,
    get_context_dir,
    migrate_from_lobsterdog,
)


class TestPaths_GetLlmemHome:
    """Test get_llmem_home() default behavior."""

    def test_returns_path(self):
        result = get_llmem_home()
        assert isinstance(result, Path)

    def test_returns_never_none(self):
        result = get_llmem_home()
        assert result is not None

    def test_default_is_config_llmem(self):
        """When neither ~/.lobsterdog/ nor ~/.config/llmem/ exists, return ~/.config/llmem/."""
        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = False
            # Clear env var
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("LMEM_HOME", None)
                result = get_llmem_home()
                assert result == Path.home() / ".config" / "llmem"


class TestPaths_GetLlmemHome_EnvOverride:
    """Test LMEM_HOME env var override."""

    def test_env_var_overrides_default(self):
        with patch.dict(os.environ, {"LMEM_HOME": "/tmp/llmem-test-home"}):
            result = get_llmem_home()
            assert result == Path("/tmp/llmem-test-home")

    def test_env_var_stripped(self):
        with patch.dict(os.environ, {"LMEM_HOME": "  /tmp/llmem-stripped  "}):
            result = get_llmem_home()
            assert result == Path("/tmp/llmem-stripped")

    def test_empty_env_var_ignored(self):
        with patch.dict(os.environ, {"LMEM_HOME": "  "}):
            # Empty string after stripping should fall through to default
            result = get_llmem_home()
            # Should not be the empty path
            assert str(result) != "." and str(result) != ""


class TestPaths_BackwardCompat:
    """Test backward compatibility with ~/.lobsterdog/."""

    def test_lobsterdog_used_when_new_path_missing(self):
        """When ~/.config/llmem/ doesn't exist but ~/.lobsterdog/ does, use ~/.lobsterdog/."""

        def mock_exists(self_path):
            if str(self_path).endswith(".config/llmem"):
                return False
            if str(self_path).endswith(".lobsterdog"):
                return True
            return False

        with patch.object(Path, "exists", mock_exists):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("LMEM_HOME", None)
                result = get_llmem_home()
                assert result == Path.home() / ".lobsterdog"


class TestPaths_DataMigration:
    """Integration test for migrating from ~/.lobsterdog/ to ~/.config/llmem/."""

    def test_migrate_copies_files(self, tmp_path):
        """migrate_from_lobsterdog copies files from old to new location."""
        old_home = tmp_path / ".lobsterdog"
        new_home = tmp_path / ".config" / "llmem"
        old_home.mkdir()

        # Create files in old location
        (old_home / "config.yaml").write_text("memory: {}")
        (old_home / "memory.db").write_text("fake db")
        (old_home / "dream-diary.md").write_text("# Dream diary")
        (old_home / "proposed-changes.md").write_text("# Changes")
        (old_home / "context").mkdir()
        (old_home / "context" / "file1.txt").write_text("context data")

        with patch("llmem.paths.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            # Make Path(...) still work normally for non-home calls
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw) if a else Path()
            result = migrate_from_lobsterdog()

        assert result is True
        assert (new_home / "config.yaml").exists()
        assert (new_home / "memory.db").exists()
        assert (new_home / "dream-diary.md").exists()
        assert (new_home / "proposed-changes.md").exists()
        assert (new_home / "context").is_dir()
        assert (new_home / "context" / "file1.txt").exists()
        # Old directory is NOT deleted
        assert old_home.exists()

    def test_migrate_idempotent(self, tmp_path):
        """Calling migrate twice does nothing on second call."""
        old_home = tmp_path / ".lobsterdog"
        new_home = tmp_path / ".config" / "llmem"
        old_home.mkdir()
        (old_home / "config.yaml").write_text("test")

        with patch("llmem.paths.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw) if a else Path()
            result1 = migrate_from_lobsterdog()

        # Second call: new_home now exists, so no migration
        with patch("llmem.paths.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw) if a else Path()
            result2 = migrate_from_lobsterdog()

        assert result1 is True
        assert result2 is False

    def test_migrate_no_old_dir(self, tmp_path):
        """No migration when old dir doesn't exist."""
        with patch("llmem.paths.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw) if a else Path()
            result = migrate_from_lobsterdog()
        assert result is False


class TestPaths_ConfigPath:
    def test_returns_config_yaml(self):
        with patch("llmem.paths.get_llmem_home", return_value=Path("/tmp/llmem")):
            result = get_config_path()
            assert result == Path("/tmp/llmem/config.yaml")


class TestPaths_DbPath:
    def test_returns_memory_db(self):
        with patch("llmem.paths.get_llmem_home", return_value=Path("/tmp/llmem")):
            result = get_db_path()
            assert result == Path("/tmp/llmem/memory.db")


class TestPaths_DreamDiaryPath:
    def test_returns_dream_diary(self):
        with patch("llmem.paths.get_llmem_home", return_value=Path("/tmp/llmem")):
            result = get_dream_diary_path()
            assert result == Path("/tmp/llmem/dream-diary.md")


class TestPaths_ProposedChangesPath:
    def test_returns_proposed_changes(self):
        with patch("llmem.paths.get_llmem_home", return_value=Path("/tmp/llmem")):
            result = get_proposed_changes_path()
            assert result == Path("/tmp/llmem/proposed-changes.md")


class TestPaths_ContextDir:
    def test_returns_context_dir(self):
        with patch("llmem.paths.get_llmem_home", return_value=Path("/tmp/llmem")):
            result = get_context_dir()
            assert result == Path("/tmp/llmem/context")
