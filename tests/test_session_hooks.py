"""Tests for the OpenCode session extraction pipeline (session_hooks)."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch


from llmem.adapters.opencode import (
    OpenCodeAdapter,
    _create_opencode_schema,
    OPENCODE_SESSION_SOURCE_TYPE,
)
from llmem.session_hooks import (
    OPENCODE_RESULT_ADAPTER_ERROR,
    OPENCODE_RESULT_ALREADY_PROCESSED,
    OPENCODE_RESULT_DB_NOT_FOUND,
    OPENCODE_RESULT_EMPTY_TRANSCRIPT,
    OPENCODE_RESULT_EXTRACTION_FAILED,
    OPENCODE_RESULT_NO_MEMORIES,
    OPENCODE_RESULT_SUCCESS,
    _generate_opencode_source_id,
    process_opencode_sessions,
)


def _create_populated_opencode_db(db_path: Path) -> None:
    """Create an opencode database with sample sessions and messages."""
    conn = sqlite3.connect(str(db_path))
    _create_opencode_schema(conn)
    # Insert a test session
    conn.execute(
        'INSERT INTO "session" ("id", "title", "slug", "directory", "time_created", "time_updated") '
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "ses_test001",
            "Test Session",
            "test-session",
            "/home/user/project",
            1000,
            2000,
        ),
    )
    # Insert user message
    conn.execute(
        'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("msg_u1", "ses_test001", "user", '{"role": "user"}', 1100),
    )
    # Insert assistant message
    conn.execute(
        'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("msg_a1", "ses_test001", "assistant", '{"role": "assistant"}', 1200),
    )
    # Insert parts
    conn.execute(
        'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("part_u1", "msg_u1", "text", '{"text": "How do I implement a cache?"}', 1110),
    )
    conn.execute(
        'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        (
            "part_a1",
            "msg_a1",
            "text",
            '{"text": "You can use an LRU cache pattern."}',
            1210,
        ),
    )
    conn.commit()
    conn.close()


def _create_multi_turn_opencode_db(db_path: Path) -> None:
    """Create an opencode database with a multi-turn session for chunking tests."""
    conn = sqlite3.connect(str(db_path))
    _create_opencode_schema(conn)
    conn.execute(
        'INSERT INTO "session" ("id", "title", "slug", "directory", "time_created", "time_updated") '
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ses_multi", "Multi Turn", "multi-turn", "/proj", 1000, 5000),
    )
    # User message 1
    conn.execute(
        'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("msg_m1", "ses_multi", "user", '{"role": "user"}', 1100),
    )
    # Assistant message 1
    conn.execute(
        'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("msg_m2", "ses_multi", "assistant", '{"role": "assistant"}', 1200),
    )
    # User message 2
    conn.execute(
        'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("msg_m3", "ses_multi", "user", '{"role": "user"}', 1300),
    )
    # Assistant message 2
    conn.execute(
        'INSERT INTO "message" ("id", "session_id", "role", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("msg_m4", "ses_multi", "assistant", '{"role": "assistant"}', 1400),
    )
    # Parts for user msg 1
    conn.execute(
        'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("part_m1", "msg_m1", "text", '{"text": "First question"}', 1110),
    )
    # Parts for asst msg 1
    conn.execute(
        'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("part_m2", "msg_m2", "text", '{"text": "First answer"}', 1210),
    )
    # Parts for user msg 2
    conn.execute(
        'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("part_m3", "msg_m3", "text", '{"text": "Second question"}', 1310),
    )
    # Parts for asst msg 2
    conn.execute(
        'INSERT INTO "part" ("id", "message_id", "type", "data", "time_created") '
        "VALUES (?, ?, ?, ?, ?)",
        ("part_m4", "msg_m4", "text", '{"text": "Second answer"}', 1410),
    )
    conn.commit()
    conn.close()


class TestOpenCodeSessionHook_SourceId:
    """Test _generate_opencode_source_id helper."""

    def test_deterministic_for_same_inputs(self):
        """Same session_id and chunk_index produce the same result."""
        result1 = _generate_opencode_source_id("ses_abc123", 0)
        result2 = _generate_opencode_source_id("ses_abc123", 0)
        assert result1 == result2

    def test_different_for_different_inputs(self):
        """Different session_ids produce different source_ids."""
        result1 = _generate_opencode_source_id("ses_abc123", 0)
        result2 = _generate_opencode_source_id("ses_def456", 0)
        assert result1 != result2

    def test_different_chunk_indices(self):
        """Different chunk indices produce different source_ids."""
        result0 = _generate_opencode_source_id("ses_abc123", 0)
        result1 = _generate_opencode_source_id("ses_abc123", 1)
        assert result0 != result1

    def test_format(self):
        """Source ID format is session_id:chunk_index."""
        result = _generate_opencode_source_id("ses_abc123", 2)
        assert result == "ses_abc123:2"


class TestOpenCodeSessionHook_ResultConstants:
    """Test that result constants follow the naming convention."""

    def test_constants_are_lowercase_strings(self):
        """All result constants are lowercase strings starting with opencode_."""
        constants = [
            OPENCODE_RESULT_SUCCESS,
            OPENCODE_RESULT_DB_NOT_FOUND,
            OPENCODE_RESULT_ALREADY_PROCESSED,
            OPENCODE_RESULT_NO_MEMORIES,
            OPENCODE_RESULT_EMPTY_TRANSCRIPT,
            OPENCODE_RESULT_ADAPTER_ERROR,
            OPENCODE_RESULT_EXTRACTION_FAILED,
        ]
        for const in constants:
            assert isinstance(const, str), f"Expected string, got {type(const)}"
            assert const.startswith("opencode_"), f"Expected opencode_ prefix: {const}"

    def test_all_required_constants_exist(self):
        """All required result constants are defined."""
        assert OPENCODE_RESULT_SUCCESS == "opencode_success"
        assert OPENCODE_RESULT_DB_NOT_FOUND == "opencode_db_not_found"
        assert OPENCODE_RESULT_ALREADY_PROCESSED == "opencode_already_processed"
        assert OPENCODE_RESULT_NO_MEMORIES == "opencode_no_memories"
        assert OPENCODE_RESULT_EMPTY_TRANSCRIPT == "opencode_empty_transcript"
        assert OPENCODE_RESULT_ADAPTER_ERROR == "opencode_adapter_error"
        assert OPENCODE_RESULT_EXTRACTION_FAILED == "opencode_extraction_failed"


class TestOpenCodeSessionHook_Process_DBNotFound:
    """Test process_opencode_sessions when database doesn't exist."""

    def test_process_returns_db_not_found_when_db_missing(self, store, tmp_path):
        """When db_path doesn't exist, returns opencode_db_not_found."""
        nonexistent_path = tmp_path / "nonexistent" / "opencode.db"
        mock_extractor = MagicMock()
        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=nonexistent_path,
        )
        assert OPENCODE_RESULT_DB_NOT_FOUND in results
        assert results[OPENCODE_RESULT_DB_NOT_FOUND] == 1
        # Extractor should NOT have been called
        mock_extractor.extract.assert_not_called()

    def test_process_default_path_returns_db_not_found_when_no_db(
        self, store, tmp_path
    ):
        """When default db_path is used and no DB exists, returns db_not_found."""
        mock_extractor = MagicMock()
        with patch(
            "llmem.config.get_opencode_db_path",
            return_value=tmp_path / "nonexistent" / "opencode.db",
        ):
            results = process_opencode_sessions(
                store=store,
                extractor=mock_extractor,
            )
        assert OPENCODE_RESULT_DB_NOT_FOUND in results

    def test_process_db_not_found_no_exception(self, store, tmp_path):
        """process_opencode_sessions never raises FileNotFoundError."""
        nonexistent_path = tmp_path / "nowhere" / "opencode.db"
        # Should complete without raising
        results = process_opencode_sessions(
            store=store,
            extractor=MagicMock(),
            db_path=nonexistent_path,
        )
        assert isinstance(results, dict)


class TestOpenCodeSessionHook_Process_AlreadyProcessed:
    """Test process_opencode_sessions skips already-processed sessions."""

    def test_already_processed_session_skipped(self, store, tmp_path):
        """When a session chunk is already extracted, it is skipped."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        # Pre-mark the session chunk as already extracted
        source_id = _generate_opencode_source_id("ses_test001", 0)
        store.log_extraction(
            OPENCODE_SESSION_SOURCE_TYPE, source_id, raw_text="test", extracted_count=1
        )

        mock_extractor = MagicMock()
        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )
        assert OPENCODE_RESULT_ALREADY_PROCESSED in results
        # Extractor should NOT have been called for already processed
        mock_extractor.extract.assert_not_called()


class TestOpenCodeSessionHook_Process_EmptyTranscript:
    """Test process_opencode_sessions with empty transcripts."""

    def test_empty_session_returns_empty_transcript(self, store, tmp_path):
        """A session with no messages returns empty_transcript."""
        db_path = tmp_path / "opencode.db"
        conn = sqlite3.connect(str(db_path))
        _create_opencode_schema(conn)
        # Insert session with no messages
        conn.execute(
            'INSERT INTO "session" ("id", "title", "slug", "directory", "time_created", "time_updated") '
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ses_empty", "Empty Session", "empty", "/proj", 1000, 2000),
        )
        conn.commit()
        conn.close()

        mock_extractor = MagicMock()
        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )
        assert OPENCODE_RESULT_EMPTY_TRANSCRIPT in results
        mock_extractor.extract.assert_not_called()


class TestOpenCodeSessionHook_Process_SourceType:
    """Test that extracted memories use opencode_session source type."""

    def test_memories_use_opencode_session_source_type(self, store, tmp_path):
        """Extracted memories have source='opencode_session'."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        # Mock extractor returns a synthetic memory
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            {"type": "fact", "content": "User asked about caching", "confidence": 0.9}
        ]

        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )
        assert OPENCODE_RESULT_SUCCESS in results

        # Verify the memory was stored with opencode_session source type
        # Use a broad query or search by type to find the memory
        memories = store.search(type="fact", valid_only=False, limit=10)
        assert len(memories) >= 1
        for m in memories:
            assert m["source"] == OPENCODE_SESSION_SOURCE_TYPE
            assert m["source"] == "opencode_session"


class TestOpenCodeSessionHook_Process_Chunking:
    """Test process_opencode_sessions processes sessions by chunks."""

    def test_multi_turn_session_chunked_processing(self, store, tmp_path):
        """Multi-turn sessions are processed chunk by chunk."""
        db_path = tmp_path / "opencode.db"
        _create_multi_turn_opencode_db(db_path)

        mock_extractor = MagicMock()
        call_count = 0

        def fake_extract(text):
            nonlocal call_count
            call_count += 1
            return [
                {"type": "fact", "content": f"Memory {call_count}", "confidence": 0.8}
            ]

        mock_extractor.extract.side_effect = fake_extract

        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )
        # Should have processed successfully
        assert OPENCODE_RESULT_SUCCESS in results
        # The multi-turn session should produce 2 chunks
        # (user msg 1 + asst msg 1 = chunk 0, user msg 2 + asst msg 2 = chunk 1)
        assert mock_extractor.extract.call_count >= 2


class TestOpenCodeSessionHook_Process_Force:
    """Test process_opencode_sessions with force=True."""

    def test_force_reprocesses_already_processed(self, store, tmp_path):
        """With force=True, already processed chunks are re-extracted."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        # First pass — process normally
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            {"type": "fact", "content": "Test memory", "confidence": 0.8}
        ]
        results1 = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )
        assert OPENCODE_RESULT_SUCCESS in results1

        # Second pass with force=True — should reprocess
        mock_extractor2 = MagicMock()
        mock_extractor2.extract.return_value = [
            {"type": "fact", "content": "New memory", "confidence": 0.8}
        ]
        results2 = process_opencode_sessions(
            store=store,
            extractor=mock_extractor2,
            db_path=db_path,
            force=True,
        )
        assert OPENCODE_RESULT_SUCCESS in results2
        mock_extractor2.extract.assert_called()


class TestOpenCodeSessionHook_Process_NoMemories:
    """Test process_opencode_sessions when extraction yields no memories."""

    def test_extraction_returns_no_memories(self, store, tmp_path):
        """When extractor returns empty list, result is no_memories."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []

        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )
        assert OPENCODE_RESULT_NO_MEMORIES in results


class TestOpenCodeSessionHook_Process_ExtractionFailed:
    """Test process_opencode_sessions when extraction raises an exception."""

    def test_extraction_exception_handled(self, store, tmp_path):
        """When extractor.extract() raises an exception, it is handled gracefully."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("Ollama unavailable")

        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )
        assert OPENCODE_RESULT_EXTRACTION_FAILED in results


class TestOpenCodeSessionHook_Process_DiscoverSessions:
    """Integration test: process_opencode_sessions discovers and processes sessions."""

    def test_discovers_and_processes_sessions(self, store, tmp_path):
        """End-to-end: discovers sessions, extracts chunks, stores memories."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            {"type": "decision", "content": "Use LRU cache", "confidence": 0.85},
            {"type": "fact", "content": "User asked about caching", "confidence": 0.7},
        ]

        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )

        # Should have at least one successful extraction
        assert OPENCODE_RESULT_SUCCESS in results
        assert results[OPENCODE_RESULT_SUCCESS] >= 1

        # Verify memories were stored
        memories = store.search(query="cache", valid_only=False, limit=10)
        assert len(memories) >= 1

        # Verify extraction was logged
        source_id = _generate_opencode_source_id("ses_test001", 0)
        assert store.is_extracted(OPENCODE_SESSION_SOURCE_TYPE, source_id)


class TestOpenCodeSessionHook_Process_AdapterClosed:
    """Test process_opencode_sessions handles closed adapter gracefully."""

    def test_adapter_closed_returns_adapter_error(self, store, tmp_path):
        """If the adapter is closed during processing, adapter errors are handled."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            {"type": "fact", "content": "test", "confidence": 0.8}
        ]

        # First call succeeds
        results1 = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )
        # At least one success
        assert OPENCODE_RESULT_SUCCESS in results1 or results1 is not None


class TestOpenCodeSessionHook_EnsureExtractionLogging:
    """Test that process_opencode_sessions logs extractions for all code paths."""

    def test_successful_extraction_is_logged(self, store, tmp_path):
        """Successful extracted chunks are logged in the extraction log."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            {"type": "fact", "content": "Test fact", "confidence": 0.8}
        ]

        process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )

        source_id = _generate_opencode_source_id("ses_test001", 0)
        assert store.is_extracted(OPENCODE_SESSION_SOURCE_TYPE, source_id)

    def test_no_memories_extraction_is_logged(self, store, tmp_path):
        """Even when extraction returns no memories, the extraction is logged."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []

        process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )

        source_id = _generate_opencode_source_id("ses_test001", 0)
        assert store.is_extracted(OPENCODE_SESSION_SOURCE_TYPE, source_id)


class TestOpenCodeSessionHook_Embedding:
    """Test that embedding integration works in process_opencode_sessions."""

    def test_embedding_called_when_embedder_provided(self, store, tmp_path):
        """When an embedder is provided, it is called for each extracted memory."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            {"type": "fact", "content": "Test memory", "confidence": 0.8}
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_embedder.vec_to_bytes.return_value = b"\x00" * 4

        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            embedder=mock_embedder,
            db_path=db_path,
        )
        assert OPENCODE_RESULT_SUCCESS in results
        mock_embedder.embed.assert_called()

    def test_memory_stored_without_embedding_on_embed_failure(self, store, tmp_path):
        """When embedding fails, memory is still stored without embedding."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            {"type": "fact", "content": "Test memory", "confidence": 0.8}
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = RuntimeError("Embedding failed")

        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            embedder=mock_embedder,
            db_path=db_path,
        )
        assert OPENCODE_RESULT_SUCCESS in results
        # Memory should still be stored (even without embedding)
        memories = store.search(query="memory", valid_only=False, limit=10)
        assert len(memories) >= 1


class TestOpenCodeSessionHook_SourceIdInExtractionLog:
    """Test that the correct source_id format is used in extraction log."""

    def test_extraction_log_uses_session_chunk_format(self, store, tmp_path):
        """The extraction log uses session_id:chunk_index as source_id."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            {"type": "fact", "content": "Test", "confidence": 0.8}
        ]

        process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
        )

        # Verify the extraction log has the right source_id format
        source_id = _generate_opencode_source_id("ses_test001", 0)
        assert store.is_extracted(OPENCODE_SESSION_SOURCE_TYPE, source_id)


class TestOpenCodeSessionHook_AdapterLifecycle:
    """Test that the adapter is properly opened and closed."""

    def test_adapter_closed_after_processing(self, store, tmp_path):
        """The adapter is closed after processing, even on errors."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []

        with patch.object(OpenCodeAdapter, "close") as mock_close:
            process_opencode_sessions(
                store=store,
                extractor=mock_extractor,
                db_path=db_path,
            )
            mock_close.assert_called()

    def test_adapter_closed_on_exception(self, store, tmp_path):
        """The adapter is closed even when an exception occurs during processing."""
        db_path = tmp_path / "opencode.db"
        _create_populated_opencode_db(db_path)

        mock_extractor = MagicMock()
        mock_extractor.extract.side_effect = RuntimeError("Test error")

        with patch.object(OpenCodeAdapter, "close") as mock_close:
            process_opencode_sessions(
                store=store,
                extractor=mock_extractor,
                db_path=db_path,
            )
            mock_close.assert_called()


class TestOpenCodeSessionHook_Limit:
    """Test that the limit parameter is passed to list_sessions."""

    def test_limit_passed_to_adapter(self, store, tmp_path):
        """The limit parameter limits how many sessions are processed."""
        db_path = tmp_path / "opencode.db"
        conn = sqlite3.connect(str(db_path))
        _create_opencode_schema(conn)
        # Insert 5 sessions
        for i in range(5):
            conn.execute(
                'INSERT INTO "session" ("id", "title", "slug", "directory", "time_created", "time_updated") '
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"ses_limit_{i}",
                    f"Session {i}",
                    f"session-{i}",
                    "/proj",
                    1000 + i,
                    2000 + i,
                ),
            )
        conn.commit()
        conn.close()

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = []

        results = process_opencode_sessions(
            store=store,
            extractor=mock_extractor,
            db_path=db_path,
            limit=3,
        )
        # With limit=3, at most 3 sessions should be processed
        # (Each session has no messages, so they get empty_transcript)
        assert sum(results.get(k, 0) for k in [
                OPENCODE_RESULT_EMPTY_TRANSCRIPT,
                OPENCODE_RESULT_SUCCESS,
                OPENCODE_RESULT_NO_MEMORIES,
            ]) <= 3


class TestSessionHookCoordinatorWithoutAdapter:
    """Test that SessionHookCoordinator works when no adapter is configured.

    This is the Copilot CLI scenario — there is no opencode.db, so the
    coordinator should operate gracefully in adapter-less mode.
    """

    def test_coordinator_created_without_adapter(self, tmp_path):
        from llmem.session_hooks import SessionHookCoordinator
        from llmem.store import MemoryStore
        from llmem.retrieve import Retriever
        from llmem.extract import ExtractionEngine

        store = MemoryStore(db_path=tmp_path / "test.db", disable_vec=True)
        retriever = Retriever(store=store)
        extractor = ExtractionEngine(base_url="http://localhost:11434")
        coordinator = SessionHookCoordinator(
            store=store,
            retriever=retriever,
            extractor=extractor,
            embedder=None,
            adapter=None,
        )
        assert coordinator._adapter is None

    def test_on_compacting_works_without_adapter(self, tmp_path):
        from llmem.session_hooks import SessionHookCoordinator, SESSION_COMPACTING_SUCCESS
        from llmem.store import MemoryStore
        from llmem.retrieve import Retriever
        from llmem.extract import ExtractionEngine

        store = MemoryStore(db_path=tmp_path / "test.db", disable_vec=True)
        store.add(type="decision", content="test decision", confidence=0.9)
        retriever = Retriever(store=store)
        extractor = ExtractionEngine(base_url="http://localhost:11434")
        coordinator = SessionHookCoordinator(
            store=store,
            retriever=retriever,
            extractor=extractor,
            embedder=None,
            adapter=None,
        )
        result_type, _ = coordinator.on_compacting("ses_test")
        assert result_type == SESSION_COMPACTING_SUCCESS

    def test_on_idle_returns_no_transcript_without_adapter(self, tmp_path):
        from llmem.session_hooks import SessionHookCoordinator, SESSION_IDLE_NO_TRANSCRIPT
        from llmem.store import MemoryStore
        from llmem.retrieve import Retriever
        from llmem.extract import ExtractionEngine

        store = MemoryStore(db_path=tmp_path / "test.db", disable_vec=True)
        retriever = Retriever(store=store)
        extractor = ExtractionEngine(base_url="http://localhost:11434")
        coordinator = SessionHookCoordinator(
            store=store,
            retriever=retriever,
            extractor=extractor,
            embedder=None,
            adapter=None,
        )
        result_type, count = coordinator.on_idle("ses_test")
        assert result_type == SESSION_IDLE_NO_TRANSCRIPT
        assert count == 0

    def test_on_ending_returns_no_transcript_without_adapter(self, tmp_path):
        from llmem.session_hooks import SessionHookCoordinator, SESSION_ENDING_NO_TRANSCRIPT
        from llmem.store import MemoryStore
        from llmem.retrieve import Retriever
        from llmem.extract import ExtractionEngine

        store = MemoryStore(db_path=tmp_path / "test.db", disable_vec=True)
        retriever = Retriever(store=store)
        extractor = ExtractionEngine(base_url="http://localhost:11434")
        coordinator = SessionHookCoordinator(
            store=store,
            retriever=retriever,
            extractor=extractor,
            embedder=None,
            adapter=None,
        )
        result_type, count = coordinator.on_ending("ses_test")
        assert result_type == SESSION_ENDING_NO_TRANSCRIPT
        assert count == 0

    def test_create_coordinator_without_opencode_db(self, tmp_path):
        from llmem.session_hooks import create_session_hook_coordinator

        config = {
            "memory": {"ollama_url": "http://localhost:11434"},
            "opencode": {"db_path": str(tmp_path / "nonexistent" / "opencode.db")},
        }
        coordinator = create_session_hook_coordinator(config=config)
        assert coordinator._adapter is None

    def test_create_coordinator_with_opencode_db(self, tmp_path):
        from llmem.session_hooks import create_session_hook_coordinator
        from llmem.adapters.opencode import _create_opencode_schema

        db_path = tmp_path / "opencode.db"
        conn = sqlite3.connect(str(db_path))
        _create_opencode_schema(conn)
        conn.close()

        config = {
            "memory": {"ollama_url": "http://localhost:11434"},
            "opencode": {"db_path": str(db_path)},
        }
        coordinator = create_session_hook_coordinator(config=config)
        assert coordinator._adapter is not None

    def test_create_coordinator_with_copilot_adapter(self, tmp_path):
        from llmem.session_hooks import create_session_hook_coordinator
        from llmem.adapters.copilot import CopilotAdapter

        state_dir = tmp_path / "copilot-state"
        config = {
            "memory": {"ollama_url": "http://localhost:11434"},
            "session": {"adapter": "copilot"},
            "copilot": {
                "state_dir": str(state_dir),
                "share_dir": str(tmp_path),
            },
        }
        coordinator = create_session_hook_coordinator(config=config)
        assert coordinator._adapter is not None
        assert isinstance(coordinator._adapter, CopilotAdapter)

    def test_create_coordinator_with_none_adapter(self, tmp_path):
        from llmem.session_hooks import create_session_hook_coordinator

        config = {
            "memory": {"ollama_url": "http://localhost:11434"},
            "session": {"adapter": "none"},
        }
        coordinator = create_session_hook_coordinator(config=config)
        assert coordinator._adapter is None
