"""Tests for the session adapter interface and OpenCodeAdapter."""

import abc
import sqlite3
from pathlib import Path

import pytest

from llmem.adapters.base import SessionAdapter
from llmem.adapters.opencode import OpenCodeAdapter, _create_opencode_schema


class TestSessionAdapter_IsABC:
    """Test that SessionAdapter is an abstract class."""

    def test_is_abc(self):
        assert issubclass(SessionAdapter, abc.ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            SessionAdapter()

    def test_has_abstract_methods(self):
        abstract_methods = {
            name
            for name in dir(SessionAdapter)
            if getattr(
                getattr(SessionAdapter, name, None), "__isabstractmethod__", False
            )
        }
        assert "list_sessions" in abstract_methods
        assert "get_session_transcript" in abstract_methods
        assert "get_session_chunks" in abstract_methods
        assert "session_exists" in abstract_methods
        assert "close" in abstract_methods


class TestOpenCodeAdapter_Ctor:
    """Test OpenCodeAdapter constructor."""

    def test_ctor_with_existing_db(self, tmp_path):
        db_path = tmp_path / "opencode.db"
        conn = sqlite3.connect(str(db_path))
        _create_opencode_schema(conn)
        conn.close()

        adapter = OpenCodeAdapter(db_path=db_path)
        assert adapter._db_path == db_path
        adapter.close()

    def test_ctor_missing_db_raises(self, tmp_path):
        db_path = tmp_path / "nonexistent.db"
        with pytest.raises(FileNotFoundError):
            OpenCodeAdapter(db_path=db_path)


class TestOpenCodeAdapter_NoPipelineDetection:
    """Test that OpenCodeAdapter does not contain pipeline detection logic."""

    def test_no_pipeline_dir_marker(self):
        """OpenCodeAdapter must not have PIPELINE_DIR_MARKER."""
        import llmem.adapters.opencode as opencode_module

        assert not hasattr(opencode_module, "PIPELINE_DIR_MARKER")

    def test_no_classify_session_source(self):
        """OpenCodeAdapter must not have classify_session_source()."""
        import llmem.adapters.opencode as opencode_module

        assert not hasattr(opencode_module, "classify_session_source")

    def test_no_source_filter_param(self):
        """list_sessions() must NOT have a source_filter parameter."""
        import inspect

        sig = inspect.signature(OpenCodeAdapter.list_sessions)
        assert "source_filter" not in sig.parameters


class TestOpenCodeAdapter_Operations:
    """Test OpenCodeAdapter CRUD operations."""

    def _create_test_db(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        _create_opencode_schema(conn)
        # Insert a test session
        conn.execute(
            'INSERT INTO "session" ("id", "title", "slug", "directory", "time_created", "time_updated") '
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "ses_001",
                "Test Session",
                "test-session",
                "/home/user/project",
                1000,
                2000,
            ),
        )
        # Insert a test message
        conn.execute(
            'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("msg_001", "ses_001", "user", '{"role": "user"}', 1100),
        )
        conn.execute(
            'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("msg_002", "ses_001", "assistant", '{"role": "assistant"}', 1200),
        )
        # Insert test parts
        conn.execute(
            'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("part_001", "msg_001", "text", '{"text": "Hello world"}', 1110),
        )
        conn.execute(
            'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("part_002", "msg_002", "text", '{"text": "Hi there"}', 1210),
        )
        conn.commit()
        conn.close()

    def test_list_sessions(self, tmp_path):
        db_path = tmp_path / "opencode.db"
        self._create_test_db(db_path)
        adapter = OpenCodeAdapter(db_path=db_path)
        sessions = adapter.list_sessions()
        assert len(sessions) >= 1
        assert sessions[0]["id"] == "ses_001"
        adapter.close()

    def test_get_session_transcript(self, tmp_path):
        db_path = tmp_path / "opencode.db"
        self._create_test_db(db_path)
        adapter = OpenCodeAdapter(db_path=db_path)
        transcript = adapter.get_session_transcript("ses_001")
        assert transcript is not None
        assert "Hello world" in transcript
        adapter.close()

    def test_session_exists(self, tmp_path):
        db_path = tmp_path / "opencode.db"
        self._create_test_db(db_path)
        adapter = OpenCodeAdapter(db_path=db_path)
        assert adapter.session_exists("ses_001") is True
        assert adapter.session_exists("nonexistent") is False
        adapter.close()
