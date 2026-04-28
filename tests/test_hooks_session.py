"""Tests for session event hooks — registry extension and SessionHookCoordinator."""

import inspect
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmem.registry import (
    register_session_hook,
    get_registered_session_hooks,
    _reset_registries,
)
from llmem.session_hooks import (
    SessionHookCoordinator,
    SessionEventManager,
    SESSION_CREATED_SUCCESS,
    SESSION_CREATED_ALREADY_PROCESSED,
    SESSION_CREATED_ERROR,
    SESSION_IDLE_DEBOUNCED,
    SESSION_IDLE_NO_TRANSCRIPT,
    SESSION_COMPACTING_SUCCESS,
    SESSION_COMPACTING_NO_MEMORIES,
    SESSION_COMPACTING_ERROR,
)
from llmem.url_validate import validate_base_url
from llmem.paths import validate_session_id


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset all registries before and after each test."""
    _reset_registries()
    yield
    _reset_registries()


# -- Fixtures --


@pytest.fixture
def mock_store():
    """Create a mock MemoryStore."""
    store = MagicMock()
    store.is_extracted.return_value = False
    store.search.return_value = []
    store.log_extraction.return_value = None
    store.add.return_value = "mem-test-123"
    return store


@pytest.fixture
def mock_retriever():
    """Create a mock Retriever."""
    retriever = MagicMock()
    retriever.format_context.return_value = "- [fact] Test memory content"
    return retriever


@pytest.fixture
def mock_extractor():
    """Create a mock ExtractionEngine."""
    return MagicMock()


@pytest.fixture
def mock_embedder():
    """Create a mock EmbeddingEngine (None by default)."""
    return None


@pytest.fixture
def mock_adapter():
    """Create a mock OpenCodeAdapter."""
    adapter = MagicMock()
    adapter.list_sessions.return_value = [
        {"id": "ses_123", "slug": "my-project", "directory": "/home/user/project"}
    ]
    adapter.get_session_transcript.return_value = (
        "User: How do I do X?\nAssistant: You can do X by..."
    )
    return adapter


@pytest.fixture
def coordinator(
    mock_store, mock_retriever, mock_extractor, mock_embedder, mock_adapter
):
    """Create a SessionHookCoordinator with all mocked dependencies."""
    return SessionHookCoordinator(
        store=mock_store,
        retriever=mock_retriever,
        extractor=mock_extractor,
        embedder=mock_embedder,
        adapter=mock_adapter,
    )


# -- TestRegisterSessionHook --


class TestRegisterSessionHook:
    """Tests for register_session_hook()."""

    def test_register_created_hook(self):
        hook_fn = lambda sid: None
        register_session_hook("created", hook_fn)
        hooks = get_registered_session_hooks()
        assert "created" in hooks
        assert hooks["created"] is hook_fn

    def test_register_idle_hook(self):
        hook_fn = lambda sid: None
        register_session_hook("idle", hook_fn)
        hooks = get_registered_session_hooks()
        assert "idle" in hooks
        assert hooks["idle"] is hook_fn

    def test_register_compacting_hook(self):
        hook_fn = lambda sid: None
        register_session_hook("compacting", hook_fn)
        hooks = get_registered_session_hooks()
        assert "compacting" in hooks
        assert hooks["compacting"] is hook_fn

    def test_invalid_event_type_raises(self):
        with pytest.raises(ValueError, match="invalid event_type"):
            register_session_hook("invalid", lambda sid: None)

    def test_duplicate_event_type_raises(self):
        register_session_hook("created", lambda sid: None)
        with pytest.raises(ValueError, match="already registered"):
            register_session_hook("created", lambda sid: None)

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="must be callable"):
            register_session_hook("created", "not_callable")


# -- TestGetRegisteredSessionHooks --


class TestGetRegisteredSessionHooks:
    """Tests for get_registered_session_hooks()."""

    def test_returns_dict_copy(self):
        """Mutating the returned dict does not affect the internal registry."""
        hook_fn = lambda sid: None
        register_session_hook("created", hook_fn)
        hooks = get_registered_session_hooks()
        hooks["idle"] = lambda sid: None  # Mutate the copy
        assert "idle" not in get_registered_session_hooks()


# -- TestSessionHookCoordinator --


class TestSessionHookCoordinator:
    """Tests for SessionHookCoordinator."""

    def test_created_hook_injects_context(
        self, coordinator, mock_store, mock_retriever, tmp_path
    ):
        """Coordinator calls retriever and writes context file."""
        with patch(
            "llmem.session_hooks.get_context_dir", return_value=tmp_path / "context"
        ):
            result_type, file_path = coordinator.on_created("ses_123")

        assert result_type == SESSION_CREATED_SUCCESS
        assert file_path is not None
        assert file_path.endswith("ses_123.md")
        # Context directory should have been created
        assert (tmp_path / "context").exists()
        # The context file should have been written
        context_file = Path(file_path)
        assert context_file.exists()
        content = context_file.read_text()
        assert "Test memory content" in content

    def test_idle_hook_extracts_memories(self, coordinator, mock_store, mock_adapter):
        """Coordinator calls session hook process_transcript."""
        mock_session_hook = MagicMock()
        mock_session_hook.process_transcript.return_value = ("success", 3)
        coordinator._session_hook = mock_session_hook

        result_type, count = coordinator.on_idle("ses_123")

        assert result_type == "success"
        assert count == 3
        mock_session_hook.process_transcript.assert_called_once_with(
            source_id="ses_123",
            text="User: How do I do X?\nAssistant: You can do X by...",
            source_type="session_idle",
        )

    def test_compacting_hook_injects_key_memories(
        self, coordinator, mock_store, mock_retriever, tmp_path
    ):
        """Coordinator retrieves high-confidence memories."""
        mock_store.search.return_value = [
            {
                "id": "mem-1",
                "type": "decision",
                "content": "Use Python 3.11",
                "confidence": 0.9,
            },
            {
                "id": "mem-2",
                "type": "preference",
                "content": "Prefer TDD",
                "confidence": 0.8,
            },
        ]

        with patch(
            "llmem.session_hooks.get_context_dir", return_value=tmp_path / "context"
        ):
            result_type, file_path = coordinator.on_compacting("ses_123")

        assert result_type == SESSION_COMPACTING_SUCCESS
        assert file_path is not None
        assert file_path.endswith("ses_123-compact.md")
        context_file = Path(file_path)
        assert context_file.exists()
        content = context_file.read_text()
        assert "Use Python 3.11" in content
        assert "Prefer TDD" in content

    def test_created_hook_skips_if_already_processed(
        self, coordinator, mock_store, mock_retriever
    ):
        """Idempotency check: if session was already processed, skip."""
        mock_store.is_extracted.return_value = True

        result_type, file_path = coordinator.on_created("ses_123")

        assert result_type == SESSION_CREATED_ALREADY_PROCESSED
        assert file_path is None
        # Retriever should not have been called
        mock_retriever.format_context.assert_not_called()

    def test_idle_hook_debounces(self, coordinator, mock_store, mock_adapter):
        """Second call within 30s window is skipped."""
        mock_session_hook = MagicMock()
        mock_session_hook.process_transcript.return_value = ("success", 1)
        coordinator._session_hook = mock_session_hook

        # First call succeeds
        result1_type, count1 = coordinator.on_idle("ses_123")
        assert result1_type == "success"

        # Second call within debounce window is skipped
        result2_type, count2 = coordinator.on_idle("ses_123")
        assert result2_type == SESSION_IDLE_DEBOUNCED
        assert count2 == 0

        # process_transcript should only have been called once
        assert mock_session_hook.process_transcript.call_count == 1

    def test_idle_hook_evicts_stale_entries(
        self, coordinator, mock_store, mock_adapter
    ):
        """_last_idle_time entries older than 5 minutes are evicted."""
        mock_session_hook = MagicMock()
        mock_session_hook.process_transcript.return_value = ("success", 1)
        coordinator._session_hook = mock_session_hook

        # Simulate old entries by inserting a stale one directly
        coordinator._last_idle_time["old_session"] = time.monotonic() - 400

        # Call on_idle for a different session — should trigger eviction
        coordinator.on_idle("ses_123")

        # The stale old_session entry should have been evicted
        assert "old_session" not in coordinator._last_idle_time
        # The current session entry should still be present
        assert "ses_123" in coordinator._last_idle_time

    def test_idle_hook_keeps_recent_entries(
        self, coordinator, mock_store, mock_adapter
    ):
        """_last_idle_time entries within the 5-minute window are kept."""
        mock_session_hook = MagicMock()
        mock_session_hook.process_transcript.return_value = ("success", 1)
        coordinator._session_hook = mock_session_hook

        # Insert a recent entry
        coordinator._last_idle_time["recent_session"] = time.monotonic() - 60

        coordinator.on_idle("ses_123")

        # The recent entry should still be present
        assert "recent_session" in coordinator._last_idle_time

    def test_created_hook_returns_error_on_write_failure(
        self, coordinator, mock_store, mock_retriever, tmp_path
    ):
        """on_created returns ('error', None) when writing context file fails."""
        # Make the context dir path exist but be a file, so mkdir fails
        context_path = tmp_path / "blocked_context"
        context_path.write_text("blocking")
        with patch("llmem.session_hooks.get_context_dir", return_value=context_path):
            result_type, file_path = coordinator.on_created("ses_123")

        assert result_type == SESSION_CREATED_ERROR
        assert file_path is None

    def test_idle_hook_returns_no_transcript(
        self, coordinator, mock_store, mock_adapter
    ):
        """on_idle returns ('no_transcript', 0) when adapter has no transcript."""
        mock_adapter.get_session_transcript.return_value = ""

        result_type, count = coordinator.on_idle("ses_123")

        assert result_type == SESSION_IDLE_NO_TRANSCRIPT
        assert count == 0

    def test_compacting_hook_returns_error_on_write_failure(
        self, coordinator, mock_store, mock_retriever, tmp_path
    ):
        """on_compacting returns ('error', None) when writing context file fails."""
        mock_store.search.return_value = [
            {
                "id": "mem-1",
                "type": "decision",
                "content": "Use Python 3.11",
                "confidence": 0.9,
            },
        ]
        # Make the context dir path exist but be a file, so mkdir fails
        context_path = tmp_path / "blocked_compact"
        context_path.write_text("blocking")
        with patch("llmem.session_hooks.get_context_dir", return_value=context_path):
            result_type, file_path = coordinator.on_compacting("ses_123")

        assert result_type == SESSION_COMPACTING_ERROR
        assert file_path is None


class TestSessionEventManager:
    """Tests for SessionEventManager."""

    def test_emit_calls_registered_hooks(self):
        """Event manager calls all registered hooks for event."""
        calls = []

        def on_created(session_id):
            calls.append(("created", session_id))

        register_session_hook("created", on_created)
        manager = SessionEventManager()
        manager.emit("created", "ses_abc")

        assert len(calls) == 1
        assert calls[0] == ("created", "ses_abc")

    def test_emit_ignores_unregistered_events(self):
        """No-op for events with no hooks."""
        manager = SessionEventManager()
        # Should not raise — just silently skip
        manager.emit("created", "ses_abc")


# -- TestValidateBaseUrl --


class TestValidateBaseUrl:
    """Tests for the validate_base_url DRY helper in url_validate.py."""

    def test_valid_url_returns_stripped(self):
        assert validate_base_url("http://localhost:11434/") == "http://localhost:11434"

    def test_valid_url_no_trailing_slash(self):
        assert validate_base_url("http://localhost:11434") == "http://localhost:11434"

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="unsafe Ollama URL"):
            validate_base_url("ftp://example.com", module="extract")

    def test_unsafe_url_raises(self):
        with pytest.raises(ValueError, match="unsafe Ollama URL"):
            validate_base_url("http://192.168.1.1:11434", module="embed")

    def test_module_in_error_message(self):
        with pytest.raises(ValueError, match="introspection"):
            validate_base_url("file:///etc/passwd", module="introspection")


# -- TestDeadCodeRemoval --


class TestDeadCodeRemoval:
    """Tests for dead code and unused import removals from review cycle 3."""

    def test_coordinator_has_no_config_parameter(
        self, mock_store, mock_retriever, mock_extractor, mock_embedder, mock_adapter
    ):
        """SessionHookCoordinator.__init__ should not accept a config parameter.

        The config parameter was stored as self._config but never read,
        making it dead code that triggered unnecessary load_config() calls.
        """
        import inspect

        sig = inspect.signature(SessionHookCoordinator.__init__)
        assert "config" not in sig.parameters, (
            "SessionHookCoordinator.__init__ should not have a 'config' parameter "
            "(was dead code — stored as self._config but never read)"
        )

    def test_coordinator_has_no_config_field(self, coordinator):
        """SessionHookCoordinator should not have a self._config field."""
        assert not hasattr(coordinator, "_config"), (
            "SessionHookCoordinator should not have a _config field "
            "(was dead code — stored but never read after init)"
        )

    def test_embed_no_unused_imports(self):
        """embed.py should not import is_safe_url or _strip_credentials.

        These were left behind after validate_base_url DRY extraction.
        embed.py only needs validate_base_url.
        """
        import llmem.embed as embed_module

        source = inspect.getsource(embed_module)
        # Check that the import line does not contain is_safe_url or _strip_credentials
        import_lines = [
            line
            for line in source.split("\n")
            if line.strip().startswith("from .url_validate import")
        ]
        for line in import_lines:
            assert "is_safe_url" not in line, (
                f"embed.py should not import is_safe_url (unused after DRY extraction): {line}"
            )
            assert "_strip_credentials" not in line, (
                f"embed.py should not import _strip_credentials (unused after DRY extraction): {line}"
            )

    def test_extract_no_unused_imports(self):
        """extract.py should not import is_safe_url or _strip_credentials.

        These were left behind after validate_base_url DRY extraction.
        extract.py only needs validate_base_url.
        """
        import llmem.extract as extract_module

        source = inspect.getsource(extract_module)
        import_lines = [
            line
            for line in source.split("\n")
            if line.strip().startswith("from .url_validate import")
        ]
        for line in import_lines:
            assert "is_safe_url" not in line, (
                f"extract.py should not import is_safe_url (unused after DRY extraction): {line}"
            )
            assert "_strip_credentials" not in line, (
                f"extract.py should not import _strip_credentials (unused after DRY extraction): {line}"
            )


# -- TestValidateSessionId --


class TestValidateSessionId:
    """Tests for validate_session_id path traversal protection."""

    def test_valid_session_id_passes(self):
        """Normal session IDs pass validation unchanged."""
        assert validate_session_id("ses_123") == "ses_123"

    def test_valid_session_id_with_hyphens(self):
        """Session IDs with hyphens and underscores pass."""
        assert validate_session_id("ses-abc_def-456") == "ses-abc_def-456"

    def test_rejects_forward_slash(self):
        """Session IDs containing '/' are rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_session_id("ses/../../etc/passwd")

    def test_rejects_backslash(self):
        """Session IDs containing '\\' are rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_session_id("ses\\..\\..\\etc")

    def test_rejects_dot_dot(self):
        """Session IDs containing '..' are rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_session_id("ses../../etc/passwd")

    def test_rejects_empty_string(self):
        """Empty session IDs are rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            validate_session_id("")

    def test_rejects_dot_dot_in_middle(self):
        """Session IDs with '..' anywhere are rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_session_id("ses_123..hidden")


class TestSessionHookPathTraversal:
    """Tests verifying session hooks reject path-traversal session IDs."""

    def test_created_rejects_slash_in_session_id(self, coordinator):
        """on_created rejects session IDs containing '/'."""
        with pytest.raises(ValueError, match="path traversal"):
            coordinator.on_created("ses/../../etc/passwd")

    def test_created_rejects_dot_dot_in_session_id(self, coordinator):
        """on_created rejects session IDs containing '..'."""
        with pytest.raises(ValueError, match="path traversal"):
            coordinator.on_created("ses..hidden")

    def test_idle_rejects_slash_in_session_id(self, coordinator):
        """on_idle rejects session IDs containing '/'."""
        with pytest.raises(ValueError, match="path traversal"):
            coordinator.on_idle("ses/../../etc/passwd")

    def test_idle_rejects_dot_dot_in_session_id(self, coordinator):
        """on_idle rejects session IDs containing '..'."""
        with pytest.raises(ValueError, match="path traversal"):
            coordinator.on_idle("ses..hidden")

    def test_compacting_rejects_slash_in_session_id(self, coordinator):
        """on_compacting rejects session IDs containing '/'."""
        with pytest.raises(ValueError, match="path traversal"):
            coordinator.on_compacting("ses/../../etc/passwd")

    def test_compacting_rejects_dot_dot_in_session_id(self, coordinator):
        """on_compacting rejects session IDs containing '..'."""
        with pytest.raises(ValueError, match="path traversal"):
            coordinator.on_compacting("ses..hidden")
