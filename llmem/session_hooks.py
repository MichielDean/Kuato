"""Session event hooks for memory injection and extraction.

Provides SessionHookCoordinator that orchestrates memory operations
when OpenCode session lifecycle events occur (created, idle, compacting),
and SessionEventManager that dispatches events to registered hooks.
"""

import logging
import time
from pathlib import Path

from .store import MemoryStore
from .retrieve import Retriever
from .extract import ExtractionEngine
from .embed import EmbeddingEngine
from .adapters.opencode import OpenCodeAdapter
from .hooks import SessionHook
from .config import load_config
from .paths import get_context_dir
from .registry import get_registered_session_hooks

log = logging.getLogger(__name__)

# Result constants for session hook operations
SESSION_CREATED_SUCCESS = "success"
SESSION_CREATED_ALREADY_PROCESSED = "already_processed"
SESSION_CREATED_ERROR = "error"
SESSION_IDLE_DEBOUNCED = "debounced"
SESSION_IDLE_NO_TRANSCRIPT = "no_transcript"
SESSION_COMPACTING_SUCCESS = "success"
SESSION_COMPACTING_NO_MEMORIES = "no_memories"
SESSION_COMPACTING_ERROR = "error"

# Idle debounce window in seconds
_IDLE_DEBOUNCE_SECONDS = 30

# Maximum age for idle tracking entries before eviction (5 minutes)
_IDLE_MAX_AGE_SECONDS = 300

# Key memory types that survive compaction
_COMPACTING_KEY_TYPES = ("decision", "preference", "procedure", "project_state")

# Minimum confidence for compacting hook memories
_COMPACTING_MIN_CONFIDENCE = 0.7


class SessionEventManager:
    """Dispatch session lifecycle events to registered hooks.

    Manages session event callbacks. Accepts event types as constructor
    parameters so any session source (OpenCode, external orchestrators, etc.)
    can emit events.

    Args:
        event_types: The set of valid event types this manager handles.
            Defaults to VALID_SESSION_EVENT_TYPES from the registry module.
    """

    def __init__(self, event_types: frozenset[str] | None = None):
        from .registry import VALID_SESSION_EVENT_TYPES

        self._event_types = (
            event_types if event_types is not None else VALID_SESSION_EVENT_TYPES
        )

    def emit(self, event_type: str, session_id: str) -> None:
        """Emit a session event, calling all registered hooks for that event type.

        Args:
            event_type: The event to emit (e.g. 'created', 'idle', 'compacting').
            session_id: The session ID the event pertains to.
        """
        if event_type not in self._event_types:
            log.debug(
                "llmem: session_hooks: ignoring unregistered event_type '%s' for session %s",
                event_type,
                session_id,
            )
            return

        hooks = get_registered_session_hooks()
        hook_fn = hooks.get(event_type)
        if hook_fn is not None:
            try:
                hook_fn(session_id)
            except Exception as e:
                log.error(
                    "llmem: session_hooks: hook for '%s' failed on session %s: %s",
                    event_type,
                    session_id,
                    e,
                )


class SessionHookCoordinator:
    """Orchestrate memory injection and extraction for OpenCode session events.

    Handles three session lifecycle hooks:
    - session.created: inject relevant memory context into the session
    - session.idle: extract memories from the session transcript
    - session.compacting: inject key memories to preserve during compaction

    Args:
        store: MemoryStore instance for reading/writing memories.
        retriever: Retriever instance for searching/formatting memories.
        extractor: ExtractionEngine for extracting memories from transcripts.
        embedder: Optional EmbeddingEngine for embedding extracted memories.
        adapter: OpenCodeAdapter for reading session content.
        config: Optional config dict. If None, loads via load_config().
    """

    def __init__(
        self,
        store: MemoryStore,
        retriever: Retriever,
        extractor: ExtractionEngine,
        embedder: EmbeddingEngine | None,
        adapter: OpenCodeAdapter,
        config: dict | None = None,
    ):
        self._store = store
        self._retriever = retriever
        self._extractor = extractor
        self._embedder = embedder
        self._adapter = adapter
        self._config = config if config is not None else load_config()
        self._session_hook = SessionHook(
            store=store,
            extractor=extractor,
            embedder=embedder,
        )
        self._last_idle_time: dict[str, float] = {}

    def on_created(self, session_id: str) -> tuple[str, str | None]:
        """Handle session.created event: inject memory context.

        Queries the memory store for relevant memories based on the
        session's working directory and writes a context file.

        Args:
            session_id: The OpenCode session ID.

        Returns:
            Tuple of (result_type, context_file_path).
            - ("success", file_path) on success.
            - ("already_processed", None) if session was already processed.
            - ("error", None) if writing the context file fails.
        """
        if self._store.is_extracted("session_created", session_id):
            log.debug(
                "llmem: session_hooks: on_created: session %s already processed",
                session_id,
            )
            return SESSION_CREATED_ALREADY_PROCESSED, None

        # Try to determine the session's working directory from the adapter
        working_dir = self._get_session_working_dir(session_id)
        query = working_dir or session_id

        context = self._retriever.format_context(query)
        if not context:
            log.debug(
                "llmem: session_hooks: on_created: no relevant memories for session %s",
                session_id,
            )
            # Still write an empty marker to indicate we processed this session
            context = ""

        context_dir = get_context_dir()
        context_file = context_dir / f"{session_id}.md"

        try:
            context_dir.mkdir(parents=True, exist_ok=True)
            context_file.write_text(context)
        except Exception as e:
            log.error(
                "llmem: session_hooks: on_created: failed to write context for session %s: %s",
                session_id,
                e,
            )
            return SESSION_CREATED_ERROR, None

        self._store.log_extraction(
            "session_created", session_id, raw_text=context[:500], extracted_count=0
        )
        log.info(
            "llmem: session_hooks: on_created: injected context for session %s",
            session_id,
        )
        return SESSION_CREATED_SUCCESS, str(context_file)

    def on_idle(self, session_id: str) -> tuple[str, int]:
        """Handle session.idle event: extract memories and run introspection.

        Debounced — if called again within 30 seconds for the same
        session_id, returns ("debounced", 0).

        Args:
            session_id: The OpenCode session ID.

        Returns:
            Tuple of (result_type, count). Count is the number of
            memories extracted on success.
            - ("debounced", 0) if called again within 30s for same session.
            - ("no_transcript", 0) if the adapter returns no transcript.
            - (result_type, count) from process_transcript on success,
              where result_type is typically "success" or "already_processed".
        """
        now = time.monotonic()
        last_time = self._last_idle_time.get(session_id, 0.0)
        if now - last_time < _IDLE_DEBOUNCE_SECONDS:
            log.debug(
                "llmem: session_hooks: on_idle: debounced for session %s",
                session_id,
            )
            return SESSION_IDLE_DEBOUNCED, 0

        self._last_idle_time[session_id] = now

        # Evict stale entries to prevent unbounded growth
        stale_keys = [
            k
            for k, v in self._last_idle_time.items()
            if now - v > _IDLE_MAX_AGE_SECONDS
        ]
        for k in stale_keys:
            del self._last_idle_time[k]

        # Get session transcript from the adapter
        transcript = self._adapter.get_session_transcript(session_id)
        if not transcript:
            log.debug(
                "llmem: session_hooks: on_idle: no transcript for session %s",
                session_id,
            )
            return SESSION_IDLE_NO_TRANSCRIPT, 0

        # Extract memories using SessionHook
        result_type, count = self._session_hook.process_transcript(
            source_id=session_id,
            text=transcript,
            source_type="session_idle",
        )

        log.info(
            "llmem: session_hooks: on_idle: extracted %d memories from session %s (result: %s)",
            count,
            session_id,
            result_type,
        )
        return result_type, count

    def on_compacting(self, session_id: str) -> tuple[str, str | None]:
        """Handle session.compacting event: inject key memories.

        Queries the memory store for high-confidence memories of
        decision, preference, procedure, and project_state types,
        and writes them to the context directory.

        Args:
            session_id: The OpenCode session ID.

        Returns:
            Tuple of (result_type, context_file_path).
            - ("success", file_path) on success.
            - ("no_memories", None) if no key memories found.
            - ("error", None) if writing the context file fails.
        """
        all_key_memories: list[dict] = []
        for mem_type in _COMPACTING_KEY_TYPES:
            results = self._store.search(type=mem_type, valid_only=True, limit=10)
            # Filter by confidence
            for m in results:
                if m.get("confidence", 0) >= _COMPACTING_MIN_CONFIDENCE:
                    all_key_memories.append(m)

        if not all_key_memories:
            log.debug(
                "llmem: session_hooks: on_compacting: no key memories for session %s",
                session_id,
            )
            return SESSION_COMPACTING_NO_MEMORIES, None

        # Deduplicate by ID
        seen_ids: set[str] = set()
        unique_memories: list[dict] = []
        for m in all_key_memories:
            mid = m.get("id", "")
            if mid not in seen_ids:
                seen_ids.add(mid)
                unique_memories.append(m)

        lines = []
        for m in unique_memories:
            line = f"- [{m['type']}] {m['content']}"
            if m.get("summary"):
                line += f" (summary: {m['summary']})"
            lines.append(line)
        context = "\n".join(lines)

        context_dir = get_context_dir()
        context_file = context_dir / f"{session_id}-compact.md"

        try:
            context_dir.mkdir(parents=True, exist_ok=True)
            context_file.write_text(context)
        except Exception as e:
            log.error(
                "llmem: session_hooks: on_compacting: failed to write context for session %s: %s",
                session_id,
                e,
            )
            return SESSION_COMPACTING_ERROR, None

        log.info(
            "llmem: session_hooks: on_compacting: injected %d key memories for session %s",
            len(unique_memories),
            session_id,
        )
        return SESSION_COMPACTING_SUCCESS, str(context_file)

    def _get_session_working_dir(self, session_id: str) -> str | None:
        """Try to determine a session's working directory from the adapter.

        Args:
            session_id: The OpenCode session ID.

        Returns:
            The working directory string, or None if not found.
        """
        try:
            sessions = self._adapter.list_sessions(limit=100)
            for s in sessions:
                if s.get("id") == session_id:
                    return s.get("slug") or s.get("directory") or None
        except Exception as e:
            log.debug(
                "llmem: session_hooks: could not determine working dir for session %s: %s",
                session_id,
                e,
            )
        return None


def create_session_hook_coordinator(
    config: dict | None = None,
) -> SessionHookCoordinator:
    """Factory function for creating a SessionHookCoordinator.

    Wires up MemoryStore, Retriever, ExtractionEngine, EmbeddingEngine,
    and OpenCodeAdapter from config defaults. If config is None, calls
    load_config().

    Args:
        config: Optional pre-loaded config dict.

    Returns:
        A fully-wired SessionHookCoordinator instance.
    """
    from .config import (
        get_db_path,
        get_ollama_url,
        get_opencode_db_path,
    )

    resolved_config = config if config is not None else load_config()
    store = MemoryStore(db_path=get_db_path(config=resolved_config), disable_vec=True)
    ollama_url = get_ollama_url(config=resolved_config)
    extractor = ExtractionEngine(base_url=ollama_url)
    retriever = Retriever(store=store)
    opencode_db = get_opencode_db_path(config=resolved_config)
    adapter = OpenCodeAdapter(db_path=opencode_db)

    return SessionHookCoordinator(
        store=store,
        retriever=retriever,
        extractor=extractor,
        embedder=None,
        adapter=adapter,
        config=resolved_config,
    )
