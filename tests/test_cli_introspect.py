"""Tests for llmem introspect and learn CLI commands."""

import argparse
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from llmem.cli import cmd_introspect, cmd_learn


def _make_args(**overrides):
    defaults = {
        "db": None,
        "what_happened": "test failure",
        "category": "ERROR_HANDLING",
        "context": None,
        "caught_by": None,
        "proposed_fix": None,
        "wrong": "did the wrong thing",
        "right": "do the right thing",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _mock_analyzer_unavailable():
    """Create a mock IntrospectionAnalyzer that reports model unavailable."""
    mock_analyzer = MagicMock()
    mock_analyzer.check_available.return_value = False
    return mock_analyzer


class TestCmdIntrospect:
    def test_introspect_stores_self_assessment(self, tmp_path):
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = _make_args(
            db=db,
            what_happened="Called wrong function",
            category="ERROR_HANDLING",
            context="session_hooks.py:on_ending",
            caught_by="self-review",
            proposed_fix="Use introspect_transcript instead",
        )

        with patch("llmem.cli.get_ollama_url", return_value="http://localhost:11434"), \
             patch("llmem.hooks.IntrospectionAnalyzer", return_value=_mock_analyzer_unavailable()):
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                cmd_introspect(args)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

        assert "Introspected:" in output

        store = MemoryStore(db_path=db, disable_vec=True)
        memories = store.list_all(type="self_assessment", valid_only=True, limit=10)
        assert any("Called wrong function" in m["content"] for m in memories)
        store.close()

    def test_introspect_requires_what_happened(self, tmp_path):
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = _make_args(db=db, what_happened=None)
        with pytest.raises(SystemExit):
            cmd_introspect(args)

    def test_introspect_unknown_category_still_stores(self, tmp_path):
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = _make_args(
            db=db,
            what_happened="Unknown category test",
            category="UNKNOWN_CATEGORY_XYZ",
        )

        with patch("llmem.cli.get_ollama_url", return_value="http://localhost:11434"), \
             patch("llmem.hooks.IntrospectionAnalyzer", return_value=_mock_analyzer_unavailable()):
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                cmd_introspect(args)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

        assert "Introspected:" in output

    def test_introspect_without_category(self, tmp_path):
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = _make_args(db=db, what_happened="No category test", category=None)

        with patch("llmem.cli.get_ollama_url", return_value="http://localhost:11434"), \
             patch("llmem.hooks.IntrospectionAnalyzer", return_value=_mock_analyzer_unavailable()):
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                cmd_introspect(args)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

        assert "Introspected:" in output


class TestCmdLearn:
    def test_learn_stores_procedure(self, tmp_path):
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = _make_args(
            db=db,
            wrong="Used foo() instead of bar()",
            right="Use bar() for text-based processing",
            context="module.py:42",
        )

        with patch("llmem.cli.get_ollama_url", return_value="http://localhost:11434"), \
             patch("llmem.hooks.IntrospectionAnalyzer", return_value=_mock_analyzer_unavailable()):
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                cmd_learn(args)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

        assert "Learned:" in output

        store = MemoryStore(db_path=db, disable_vec=True)
        memories = store.list_all(type="procedure", valid_only=True, limit=10)
        assert any("bar()" in m["content"] and "foo()" in m["content"] for m in memories)
        store.close()

    def test_learn_requires_wrong(self, tmp_path):
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = _make_args(db=db, wrong=None)
        with pytest.raises(SystemExit):
            cmd_learn(args)

    def test_learn_requires_right(self, tmp_path):
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = _make_args(db=db, right=None)
        with pytest.raises(SystemExit):
            cmd_learn(args)