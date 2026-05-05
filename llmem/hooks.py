"""Auto-extraction and introspection hooks for session transcripts."""

import json
import logging

from .store import MemoryStore
from .extract import ExtractionEngine, DEFAULT_MODEL, OLLAMA_BASE
from .embed import EmbeddingEngine
from .taxonomy import (
    INTROSPECT_FIELD_LINES,
)

log = logging.getLogger(__name__)

# Result constants for process_transcript
PROCESS_RESULT_SUCCESS = "success"
PROCESS_RESULT_ALREADY_PROCESSED = "already_processed"
PROCESS_RESULT_MODEL_UNAVAILABLE = "model_unavailable"
PROCESS_RESULT_NO_MEMORIES = "no_memories"
PROCESS_RESULT_EMPTY_FILE = "empty_file"
PROCESS_RESULT_FILE_NOT_FOUND = "file_not_found"
PROCESS_RESULT_FILE_TOO_LARGE = "file_too_large"
PROCESS_RESULT_ERROR = "error"

# Introspection result constants
INTROSPECT_RESULT_SUCCESS = "introspect_success"
INTROSPECT_RESULT_MODEL_UNAVAILABLE = "introspect_model_unavailable"
INTROSPECT_RESULT_ALREADY_PROCESSED = "introspect_already_processed"
INTROSPECT_RESULT_ERROR = "introspect_error"
INTROSPECT_RESULT_EMPTY_FILE = "introspect_empty_file"
INTROSPECT_RESULT_FILE_NOT_FOUND = "introspect_file_not_found"
INTROSPECT_RESULT_NO_ASSESSMENT = "introspect_no_assessment"
INTROSPECT_RESULT_FILE_TOO_LARGE = "introspect_file_too_large"

INTROSPECTION_SOURCE_TYPE = "introspection"


class SessionHook:
    """Process session transcript text and extract memories.

    Uses process_transcript() to process transcript strings directly,
    typically sourced from a SessionAdapter (e.g., OpenCodeAdapter) that
    reads sessions from a database.
    """

    def __init__(
        self,
        store: MemoryStore,
        extractor: ExtractionEngine,
        embedder: EmbeddingEngine | None = None,
        force: bool = False,
        max_file_size: int = 10 * 1024 * 1024,
    ):
        self._store = store
        self._extractor = extractor
        self._embedder = embedder
        self._force = force
        self._max_file_size = max_file_size

    def process_transcript(
        self, source_id: str, text: str, source_type: str = "session"
    ) -> tuple[str, int]:
        """Process transcript text directly.

        Args:
            source_id: Identifier for deduplication.
            text: The transcript text to process.
            source_type: Source type label (default: 'session').

        Returns:
            Tuple of (result_type, count).
        """
        if not self._force and self._store.is_extracted(source_type, source_id):
            return PROCESS_RESULT_ALREADY_PROCESSED, 0

        if len(text.encode("utf-8")) > self._max_file_size:
            return PROCESS_RESULT_FILE_TOO_LARGE, 0

        return self._process_text(source_id, text, source_type=source_type)

    def _process_text(
        self, source_id: str, text: str, source_type: str = "session"
    ) -> tuple[str, int]:
        """Internal text processing."""
        if not text.strip():
            return PROCESS_RESULT_EMPTY_FILE, 0

        memories = self._extractor.extract(text)
        if not memories:
            self._store.log_extraction(
                source_type, source_id, raw_text=text[:500], extracted_count=0
            )
            return PROCESS_RESULT_NO_MEMORIES, 0

        count = 0
        for m in memories:
            embedding = None
            if self._embedder:
                try:
                    vec = self._embedder.embed(m["content"])
                    embedding = self._embedder.vec_to_bytes(vec)
                except Exception as e:
                    log.debug(
                        "llmem: hooks: embedding failed for memory in %s: %s",
                        source_id,
                        e,
                    )
            self._store.add(
                type=m["type"],
                content=m["content"],
                confidence=m.get("confidence", 0.8),
                source=source_type,
                embedding=embedding,
            )
            count += 1

        self._store.log_extraction(
            source_type, source_id, raw_text=text[:500], extracted_count=count
        )
        return PROCESS_RESULT_SUCCESS, count


_INTROSPECT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class IntrospectionAnalyzer:
    """Analyze session transcripts for self-assessment patterns."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE):
        from .url_validate import validate_base_url

        self._model = model
        self._base_url = validate_base_url(base_url, module="introspection")

    @property
    def model(self) -> str:
        return self._model

    def check_available(self) -> bool:
        """Check if the introspection model is available."""
        from .url_validate import safe_urlopen

        url = f"{self._base_url}/api/tags"
        try:
            with safe_urlopen(url, allow_remote=True) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                return any(m.startswith(self._model) for m in models)
        except Exception:
            return False


def introspect_transcript(
    source_id: str,
    text: str,
    store: MemoryStore,
    analyzer: IntrospectionAnalyzer,
    embedder: EmbeddingEngine | None = None,
    force: bool = False,
) -> tuple[str, str | None]:
    """Analyze transcript text and store a self_assessment memory."""
    if not force and store.is_extracted(INTROSPECTION_SOURCE_TYPE, source_id):
        return INTROSPECT_RESULT_ALREADY_PROCESSED, None

    if not text.strip():
        return INTROSPECT_RESULT_EMPTY_FILE, None

    return _do_introspect(source_id, text, store, analyzer, embedder, force)


def _do_introspect(
    source_id: str,
    text: str,
    store: MemoryStore,
    analyzer: IntrospectionAnalyzer,
    embedder: EmbeddingEngine | None = None,
    force: bool = False,
) -> tuple[str, str | None]:
    """Internal introspection implementation."""
    if not analyzer.check_available():
        return INTROSPECT_RESULT_MODEL_UNAVAILABLE, None

    prompt = f"""Analyze this coding session transcript and produce a structured self-assessment.

Format each field on its own line as "Field: value":

{INTROSPECT_FIELD_LINES}

Transcript:
{text[:8000]}

Produce a structured self-assessment or state "NO_ASSESSMENT" if nothing notable happened."""

    import urllib.request
    import urllib.error

    from .url_validate import safe_urlopen

    url = f"{analyzer._base_url}/api/generate"
    payload = json.dumps(
        {
            "model": analyzer.model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    try:
        with safe_urlopen(req, allow_remote=True, timeout=300) as resp:
            data = json.loads(resp.read())
            assessment = data.get("response", "").strip()
    except Exception as e:
        log.error("llmem: introspection: analysis failed: %s", e)
        return INTROSPECT_RESULT_ERROR, None

    if not assessment or "NO_ASSESSMENT" in assessment.upper()[:50]:
        return INTROSPECT_RESULT_NO_ASSESSMENT, None

    embedding = None
    if embedder:
        try:
            vec = embedder.embed(assessment)
            embedding = embedder.vec_to_bytes(vec)
        except Exception as e:
            log.debug("llmem: introspection: embedding failed for %s: %s", source_id, e)

    mid = store.add(
        type="self_assessment",
        content=assessment,
        source=INTROSPECTION_SOURCE_TYPE,
        confidence=0.7,
        embedding=embedding,
    )

    store.log_extraction(
        INTROSPECTION_SOURCE_TYPE,
        source_id,
        raw_text=assessment[:500],
        extracted_count=1,
    )
    return INTROSPECT_RESULT_SUCCESS, mid


def create_introspection_analyzer(
    base_url: str = OLLAMA_BASE, model: str = DEFAULT_MODEL
) -> IntrospectionAnalyzer:
    """Factory function for creating an IntrospectionAnalyzer."""
    return IntrospectionAnalyzer(model=model, base_url=base_url)


def create_session_hook(
    store: MemoryStore,
    extractor: ExtractionEngine | None = None,
    embedder: EmbeddingEngine | None = None,
    force: bool = False,
    max_file_size: int = 10 * 1024 * 1024,
) -> SessionHook:
    """Factory function for creating a SessionHook."""
    if extractor is None:
        extractor = ExtractionEngine()
    return SessionHook(
        store=store,
        extractor=extractor,
        embedder=embedder,
        force=force,
        max_file_size=max_file_size,
    )