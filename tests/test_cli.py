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
