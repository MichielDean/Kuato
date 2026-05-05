"""Session event hooks for memory injection and extraction.

Provides SessionHookCoordinator that orchestrates memory operations
when OpenCode session lifecycle events occur (created, idle, compacting),
SessionEventManager that dispatches events to registered hooks, and the
process_opencode_sessions batch extraction pipeline.
"""

import logging
import time


from .store import MemoryStore
from .retrieve import Retriever
from .extract import ExtractionEngine
from .embed import EmbeddingEngine
from .adapters.base import SessionAdapter
from .adapters.opencode import OpenCodeAdapter, OPENCODE_SESSION_SOURCE_TYPE
from .hooks import SessionHook, introspect_transcript, create_introspection_analyzer
from .config import load_config
from .paths import get_context_dir, validate_session_id, _validate_write_path
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
SESSION_ENDING_SUCCESS = "ending_success"
SESSION_ENDING_NO_TRANSCRIPT = "ending_no_transcript"

# Result constants for process_opencode_sessions
# Following the pattern of PROCESS_RESULT_* in llmem/hooks.py
OPENCODE_RESULT_SUCCESS = "opencode_success"
OPENCODE_RESULT_DB_NOT_FOUND = "opencode_db_not_found"
OPENCODE_RESULT_ALREADY_PROCESSED = "opencode_already_processed"
OPENCODE_RESULT_NO_MEMORIES = "opencode_no_memories"
OPENCODE_RESULT_EMPTY_TRANSCRIPT = "opencode_empty_transcript"
OPENCODE_RESULT_ADAPTER_ERROR = "opencode_adapter_error"
OPENCODE_RESULT_EXTRACTION_FAILED = "opencode_extraction_failed"

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
    """Orchestrate memory injection and extraction for session events.

    Handles three session lifecycle hooks:
    - session.created: inject relevant memory context into the session
    - session.idle: extract memories from the session transcript
    - session.compacting: inject key memories to preserve during compaction

    Args:
        store: MemoryStore instance for reading/writing memories.
        retriever: Retriever instance for searching/formatting memories.
        extractor: ExtractionEngine for extracting memories from transcripts.
        embedder: Optional EmbeddingEngine for embedding extracted memories.
        adapter: Optional SessionAdapter for reading session content. If None,
            on_idle and on_ending return no_transcript (no session DB available),
            and on_created falls back to session_id as the query.
    """

    def __init__(
        self,
        store: MemoryStore,
        retriever: Retriever,
        extractor: ExtractionEngine,
        embedder: EmbeddingEngine | None,
        adapter: SessionAdapter | None = None,
    ):
        self._store = store
        self._retriever = retriever
        self._adapter = adapter
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

        Raises:
            ValueError: If session_id contains path traversal characters.
        """
        validate_session_id(session_id)
        if self._store.is_extracted("session_created", session_id):
            log.debug(
                "llmem: session_hooks: on_created: session %s already processed",
                session_id,
            )
            return SESSION_CREATED_ALREADY_PROCESSED, None

        # Try to determine the session's working directory from the adapter.
        # Fall back to CWD if no adapter or no session found — this gives
        # project-scoped context rather than searching for a useless UUID.
        working_dir = self._get_session_working_dir(session_id)
        if working_dir:
            query = working_dir
        else:
            import os

            query = os.getcwd()
            log.debug(
                "llmem: session_hooks: on_created: no working directory for session %s, using CWD: %s",
                session_id,
                query,
            )

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
        context_file = _validate_write_path(context_file, "session context")

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

        Raises:
            ValueError: If session_id contains path traversal characters.
        """
        validate_session_id(session_id)
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
        if self._adapter is None:
            log.debug(
                "llmem: session_hooks: on_idle: no adapter configured for session %s",
                session_id,
            )
            return SESSION_IDLE_NO_TRANSCRIPT, 0

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

        Raises:
            ValueError: If session_id contains path traversal characters.
        """
        validate_session_id(session_id)
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
        context_file = _validate_write_path(context_file, "session compact context")

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

    def on_ending(self, session_id: str) -> tuple[str, int]:
        """Handle session.ending event: extract memories and run introspection.

        Unlike on_idle (extraction only), on_ending also runs the
        introspection pass to generate self_assessment memories from
        the session transcript.

        Args:
            session_id: The OpenCode session ID.

        Returns:
            Tuple of (result_type, count). Count is the total number of
            memories extracted + introspected.

        Raises:
            ValueError: If session_id contains path traversal characters.
        """
        validate_session_id(session_id)

        if self._adapter is None:
            log.debug(
                "llmem: session_hooks: on_ending: no adapter configured for session %s",
                session_id,
            )
            return SESSION_ENDING_NO_TRANSCRIPT, 0

        transcript = self._adapter.get_session_transcript(session_id)
        if not transcript:
            log.debug(
                "llmem: session_hooks: on_ending: no transcript for session %s",
                session_id,
            )
            return SESSION_ENDING_NO_TRANSCRIPT, 0

        result_type, count = self._session_hook.process_transcript(
            source_id=session_id,
            text=transcript,
            source_type="session",
        )

        introspection_count = 0
        try:
            analyzer = create_introspection_analyzer()
            if analyzer.check_available():
                introspect_result = introspect_transcript(
                    source_id=f"ending:{session_id}",
                    text=transcript,
                    store=self._store,
                    analyzer=analyzer,
                )
                if introspect_result[0].startswith("introspect_success"):
                    introspection_count = 1
            else:
                log.debug(
                    "llmem: session_hooks: on_ending: introspection model unavailable, skipping for session %s",
                    session_id,
                )
        except Exception as e:
            log.warning(
                "llmem: session_hooks: on_ending: introspection failed for session %s: %s",
                session_id,
                e,
            )

        total = count + introspection_count
        log.info(
            "llmem: session_hooks: on_ending: extracted %d memories + %d introspections for session %s (result: %s)",
            count,
            introspection_count,
            session_id,
            result_type,
        )
        return result_type, total

    def _get_session_working_dir(self, session_id: str) -> str | None:
        """Try to determine a session's working directory from the adapter.

        Args:
            session_id: The session ID.

        Returns:
            The working directory string, or None if not found or no adapter.
        """
        if self._adapter is None:
            return None

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
    and optionally a SessionAdapter from config. If config is None, calls
    load_config(). The adapter type is determined by the ``session.adapter``
    config key (``"opencode"`` or ``"none"``). If the configured adapter's
    data source does not exist, the adapter is set to None and the
    coordinator operates without session transcript access.

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

    adapter = None
    adapter_type = resolved_config.get("session", {}).get("adapter", "opencode")

    if adapter_type == "opencode":
        opencode_db = get_opencode_db_path(config=resolved_config)
        if opencode_db.exists():
            adapter = OpenCodeAdapter(db_path=opencode_db)
        else:
            log.debug(
                "llmem: session_hooks: opencode database not found at %s, running without session adapter",
                opencode_db,
            )
    elif adapter_type == "none":
        pass
    else:
        log.warning(
            "llmem: session_hooks: unknown adapter type %r, running without session adapter",
            adapter_type,
        )

    return SessionHookCoordinator(
        store=store,
        retriever=retriever,
        extractor=extractor,
        embedder=None,
        adapter=adapter,
    )


def _generate_opencode_source_id(session_id: str, chunk_index: int) -> str:
    """Generate a deterministic source_id for an opencode session chunk.

    Uses the session ID and chunk index directly — no hashing needed
    since OpenCode session IDs are already unique strings.

    Args:
        session_id: The opencode session ID (e.g., ses_abc123).
        chunk_index: Zero-based chunk index within the session.

    Returns:
        A source_id string in the form "session_id:chunk_index".
    """
    return f"{session_id}:{chunk_index}"


def process_opencode_sessions(
    store: MemoryStore,
    extractor: ExtractionEngine,
    embedder: EmbeddingEngine | None = None,
    db_path: str | None = None,
    force: bool = False,
    limit: int = 50,
) -> dict[str, int]:
    """Process OpenCode sessions from the SQLite database and extract memories.

    Opens the opencode SQLite database, discovers sessions, chunks them
    by user-message boundaries, and feeds each chunk through the
    extraction pipeline.

    This function is stateless — each call opens a fresh adapter,
    processes sessions, and closes the adapter. No prior initialization
    is required; all dependencies are passed as parameters.

    Args:
        store: The MemoryStore to save extracted memories into.
        extractor: The ExtractionEngine for extracting memories from text.
        embedder: Optional EmbeddingEngine for generating embeddings.
        db_path: Path to the opencode SQLite database. If None, uses
            get_opencode_db_path() from config.
        force: If True, re-process sessions even if already extracted.
        limit: Maximum number of sessions to process.

    Returns:
        A dict mapping result constants to their counts. For example:
        {"opencode_success": 3, "opencode_already_processed": 2}
        Returns {"opencode_db_not_found": 1} if the database file
        doesn't exist (logs warning, never raises).
    """
    if db_path is None:
        from .config import get_opencode_db_path

        db_path = get_opencode_db_path()

    if not db_path.exists():
        log.warning("llmem: session_hooks: opencode database not found: %s", db_path)
        return {OPENCODE_RESULT_DB_NOT_FOUND: 1}

    results: dict[str, int] = {}
    adapter: OpenCodeAdapter | None = None

    try:
        adapter = OpenCodeAdapter(db_path=db_path)
        sessions = adapter.list_sessions(limit=limit)

        for session in sessions:
            session_id = session["id"]

            try:
                chunks = adapter.get_session_chunks(session_id)
            except Exception as e:
                log.warning(
                    "llmem: session_hooks: adapter error for session %s: %s",
                    session_id,
                    e,
                )
                _increment_result(results, OPENCODE_RESULT_ADAPTER_ERROR)
                continue

            if chunks is None:
                # Session not found or adapter closed
                log.info(
                    "llmem: session_hooks: no chunks for session %s (skipped)",
                    session_id,
                )
                _increment_result(results, OPENCODE_RESULT_ADAPTER_ERROR)
                continue

            if not chunks:
                # Session exists but has no messages
                log.info(
                    "llmem: session_hooks: empty transcript for session %s",
                    session_id,
                )
                # Log extraction attempt so we don't retry endlessly
                store.log_extraction(
                    OPENCODE_SESSION_SOURCE_TYPE,
                    _generate_opencode_source_id(session_id, 0),
                    raw_text="",
                    extracted_count=0,
                )
                _increment_result(results, OPENCODE_RESULT_EMPTY_TRANSCRIPT)
                continue

            for chunk_index, chunk_text in enumerate(chunks):
                source_id = _generate_opencode_source_id(session_id, chunk_index)

                if not force and store.is_extracted(
                    OPENCODE_SESSION_SOURCE_TYPE, source_id
                ):
                    _increment_result(results, OPENCODE_RESULT_ALREADY_PROCESSED)
                    continue

                if not chunk_text.strip():
                    log.info(
                        "llmem: session_hooks: empty chunk %d for session %s",
                        chunk_index,
                        session_id,
                    )
                    store.log_extraction(
                        OPENCODE_SESSION_SOURCE_TYPE,
                        source_id,
                        raw_text="",
                        extracted_count=0,
                    )
                    _increment_result(results, OPENCODE_RESULT_EMPTY_TRANSCRIPT)
                    continue

                try:
                    memories = extractor.extract(chunk_text)
                except Exception as e:
                    log.warning(
                        "llmem: session_hooks: extraction failed for %s chunk %d: %s",
                        session_id,
                        chunk_index,
                        e,
                    )
                    _increment_result(results, OPENCODE_RESULT_EXTRACTION_FAILED)
                    continue

                if not memories:
                    store.log_extraction(
                        OPENCODE_SESSION_SOURCE_TYPE,
                        source_id,
                        raw_text=chunk_text[:500],
                        extracted_count=0,
                    )
                    _increment_result(results, OPENCODE_RESULT_NO_MEMORIES)
                    continue

                count = 0
                for m in memories:
                    embedding = None
                    if embedder:
                        try:
                            vec = embedder.embed(m["content"])
                            embedding = embedder.vec_to_bytes(vec)
                        except Exception as exc:
                            log.debug(
                                "llmem: session_hooks: embedding failed for %s chunk %d, storing without embedding: %s",
                                session_id,
                                chunk_index,
                                exc,
                            )
                    store.add(
                        type=m["type"],
                        content=m["content"],
                        confidence=m.get("confidence", 0.8),
                        source=OPENCODE_SESSION_SOURCE_TYPE,
                        embedding=embedding,
                    )
                    count += 1

                store.log_extraction(
                    OPENCODE_SESSION_SOURCE_TYPE,
                    source_id,
                    raw_text=chunk_text[:500],
                    extracted_count=count,
                )
                _increment_result(results, OPENCODE_RESULT_SUCCESS)

    except FileNotFoundError:
        log.warning("llmem: session_hooks: opencode database not found: %s", db_path)
        return {OPENCODE_RESULT_DB_NOT_FOUND: 1}
    except Exception as e:
        log.warning("llmem: session_hooks: unexpected error: %s", e)
        _increment_result(results, OPENCODE_RESULT_ADAPTER_ERROR)
    finally:
        if adapter is not None:
            adapter.close()

    return results


def _increment_result(results: dict[str, int], key: str) -> None:
    """Increment a result counter in the results dict."""
    results[key] = results.get(key, 0) + 1
