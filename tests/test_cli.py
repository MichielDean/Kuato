"""Tests for the llmem CLI entry point."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCli_EntryPoint:
    """Test that the llmem CLI entry point works."""

    def test_cli_help(self):
        """CLI --help outputs usage info with llmem branding."""
        from llmem.cli import main
        import io

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sys.argv = ["llmem", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "llmem" in output.lower()

    def test_cli_prog_name(self):
        """CLI prog name is 'llmem'."""
        from llmem.cli import main
        import io

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sys.argv = ["llmem", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "llmem" in output.lower()


class TestCli_DbPathDefault:
    """Test that CLI uses ~/.config/llmem/memory.db by default."""

    def test_default_db_path(self):
        from llmem.paths import get_db_path

        # Mock the home to a tmp dir where neither .lobsterdog nor .config/llmem exists
        with patch("llmem.paths.Path") as mock_path_cls:
            import tempfile

            fake_home = Path(tempfile.mkdtemp())
            mock_path_cls.home.return_value = fake_home
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw) if a else Path()
            result = get_db_path()
            assert "llmem" in str(result)
            assert result.name == "memory.db"


class TestCli_DbPathEnvOverride:
    """Test that LMEM_HOME overrides DB path."""

    def test_env_override(self):
        import os

        with patch.dict(os.environ, {"LMEM_HOME": "/tmp/llmem-env-test"}):
            from llmem.paths import get_db_path

            result = get_db_path()
            assert result == Path("/tmp/llmem-env-test/memory.db")


class TestCli_TypeValidation:
    """Test that CLI --type rejects unregistered types at runtime."""

    def test_add_unregistered_type_exits_with_error(self, tmp_path):
        """Using --type with an unregistered type prints an error and exits."""
        from llmem.cli import cmd_add
        from llmem.store import _reset_global_registry
        import argparse

        _reset_global_registry()

        db = tmp_path / "test.db"
        # Pre-create a store with disable_vec so the DB is initialized
        from llmem.store import MemoryStore

        MemoryStore(db_path=db, disable_vec=True).close()

        args = argparse.Namespace(
            db=db,
            type="never_registered_type_xyz",
            content="test content",
            file=None,
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        # Patch MemoryStore to use disable_vec=True so sqlite-vec doesn't fail
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            with pytest.raises(SystemExit):
                cmd_add(args)

    def test_add_registered_type_succeeds(self, tmp_path):
        """Using --type with a registered type works."""
        from llmem.cli import cmd_add
        from llmem.store import register_memory_type, _reset_global_registry

        _reset_global_registry()
        register_memory_type("registered_cli_type")

        db = tmp_path / "test.db"
        from llmem.store import MemoryStore

        MemoryStore(db_path=db, disable_vec=True).close()

        import argparse

        args = argparse.Namespace(
            db=db,
            type="registered_cli_type",
            content="test content",
            file=None,
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        # Patch MemoryStore to use disable_vec=True so sqlite-vec doesn't fail
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_add(args)

    def test_type_choices_dynamic_after_registration(self):
        """get_registered_types() reflects newly registered types."""
        from llmem.store import (
            register_memory_type,
            get_registered_types,
            _reset_global_registry,
        )

        _reset_global_registry()
        before = get_registered_types()
        register_memory_type("dynamic_type_test")
        after = get_registered_types()
        assert "dynamic_type_test" not in before
        assert "dynamic_type_test" in after
