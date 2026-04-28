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


class TestOpenCodeAdapter_GetSessionChunks:
    """Test OpenCodeAdapter.get_session_chunks() method."""

    def _create_multi_message_db(self, db_path: Path) -> None:
        """Create a database with multiple user/assistant message pairs for chunking."""
        conn = sqlite3.connect(str(db_path))
        _create_opencode_schema(conn)
        conn.execute(
            'INSERT INTO "session" ("id", "title", "slug", "directory", "time_created", "time_updated") '
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ses_chunks", "Chunk Test", "chunk-test", "/proj", 1000, 5000),
        )
        # User message 1
        conn.execute(
            'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("msg_c1", "ses_chunks", "user", '{"role": "user"}', 1100),
        )
        # Assistant message 1
        conn.execute(
            'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("msg_c2", "ses_chunks", "assistant", '{"role": "assistant"}', 1200),
        )
        # User message 2
        conn.execute(
            'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("msg_c3", "ses_chunks", "user", '{"role": "user"}', 1300),
        )
        # Assistant message 2
        conn.execute(
            'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("msg_c4", "ses_chunks", "assistant", '{"role": "assistant"}', 1400),
        )
        # Parts for user msg 1
        conn.execute(
            'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("part_c1", "msg_c1", "text", '{"text": "First question"}', 1110),
        )
        # Parts for asst msg 1
        conn.execute(
            'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("part_c2", "msg_c2", "text", '{"text": "First answer"}', 1210),
        )
        # Parts for user msg 2
        conn.execute(
            'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("part_c3", "msg_c3", "text", '{"text": "Second question"}', 1310),
        )
        # Parts for asst msg 2
        conn.execute(
            'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("part_c4", "msg_c4", "text", '{"text": "Second answer"}', 1410),
        )
        conn.commit()
        conn.close()

    def test_get_session_chunks_returns_chunks(self, tmp_path):
        """get_session_chunks splits a multi-turn session into chunks."""
        db_path = tmp_path / "opencode.db"
        self._create_multi_message_db(db_path)
        adapter = OpenCodeAdapter(db_path=db_path)
        chunks = adapter.get_session_chunks("ses_chunks")
        assert chunks is not None
        assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
        # Each chunk should contain content from one user+assistant turn
        all_content = " ".join(chunks)
        assert "First question" in all_content
        assert "Second question" in all_content
        adapter.close()

    def test_get_session_chunks_nonexistent_session(self, tmp_path):
        """get_session_chunks returns None for nonexistent session."""
        db_path = tmp_path / "opencode.db"
        conn = sqlite3.connect(str(db_path))
        _create_opencode_schema(conn)
        conn.close()
        adapter = OpenCodeAdapter(db_path=db_path)
        result = adapter.get_session_chunks("nonexistent_session")
        assert result is None
        adapter.close()

    def test_get_session_chunks_empty_session(self, tmp_path):
        """get_session_chunks returns empty list for session with no messages."""
        db_path = tmp_path / "opencode.db"
        conn = sqlite3.connect(str(db_path))
        _create_opencode_schema(conn)
        conn.execute(
            'INSERT INTO "session" ("id", "title", "slug", "directory", "time_created", "time_updated") '
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ses_empty", "Empty", "empty", "/proj", 1000, 2000),
        )
        conn.commit()
        conn.close()
        adapter = OpenCodeAdapter(db_path=db_path)
        result = adapter.get_session_chunks("ses_empty")
        assert result == []
        adapter.close()

    def test_get_session_chunks_after_close(self, tmp_path):
        """get_session_chunks returns None after adapter is closed."""
        db_path = tmp_path / "opencode.db"
        self._create_multi_message_db(db_path)
        adapter = OpenCodeAdapter(db_path=db_path)
        adapter.close()
        result = adapter.get_session_chunks("ses_chunks")
        assert result is None

    def test_get_session_chunks_single_user_message(self, tmp_path):
        """get_session_chunks with only one user message returns a single chunk."""
        db_path = tmp_path / "opencode.db"
        conn = sqlite3.connect(str(db_path))
        _create_opencode_schema(conn)
        conn.execute(
            'INSERT INTO "session" ("id", "title", "slug", "directory", "time_created", "time_updated") '
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ses_single", "Single", "single", "/proj", 1000, 2000),
        )
        conn.execute(
            'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("msg_s1", "ses_single", "user", '{"role": "user"}', 1100),
        )
        conn.execute(
            'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
            "VALUES (?, ?, ?, ?, ?)",
            ("part_s1", "msg_s1", "text", '{"text": "Just one message"}', 1110),
        )
        conn.commit()
        conn.close()
        adapter = OpenCodeAdapter(db_path=db_path)
        chunks = adapter.get_session_chunks("ses_single")
        assert chunks is not None
        assert len(chunks) == 1
        assert "Just one message" in chunks[0]
        adapter.close()
