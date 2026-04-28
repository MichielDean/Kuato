"""Auto-extraction hooks for session transcripts."""

import hashlib
import json
import logging
import re
from pathlib import Path

from .store import MemoryStore
from .extract import ExtractionEngine, DEFAULT_MODEL, OLLAMA_BASE
from .embed import EmbeddingEngine
from .config import (
    get_db_path,
    get_ollama_url,
    is_correction_detection_enabled,
)
from .taxonomy import (
    ERROR_TAXONOMY_KEYS,
    INTROSPECT_FIELD_LINES,
)

log = logging.getLogger(__name__)

# Result constants for process_file
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


def discover_transcript_files(directory: Path, pattern: str = "*.md") -> list[Path]:
    """Find transcript files in a directory.

    Symlinks are excluded to prevent reading arbitrary files via symlink attacks.
    """
    return sorted(
        p for p in directory.glob(pattern) if p.is_file() and not p.is_symlink()
    )


def generate_source_id(file_path: Path) -> str:
    """Generate a deterministic source_id from a file path."""
    abs_path = str(file_path.resolve())
    return hashlib.sha256(abs_path.encode()).hexdigest()[:16]


def is_session_processed(
    store: MemoryStore, source_id: str, source_type: str = "session"
) -> bool:
    """Check if a session has already been processed."""
    return store.is_extracted(source_type, source_id)


class SessionHook:
    """Process session transcript files and extract memories."""

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

    def process_file(self, file_path: Path) -> tuple[str, int]:
        """Process a single transcript file and extract memories.

        Returns:
            Tuple of (result_type, count) where count is the number
            of memories extracted on success.
        """
        if not file_path.exists():
            return PROCESS_RESULT_FILE_NOT_FOUND, 0

        if file_path.stat().st_size == 0:
            return PROCESS_RESULT_EMPTY_FILE, 0

        if file_path.stat().st_size > self._max_file_size:
            return PROCESS_RESULT_FILE_TOO_LARGE, 0

        source_id = generate_source_id(file_path)
        if not self._force and self._store.is_extracted("session", source_id):
            return PROCESS_RESULT_ALREADY_PROCESSED, 0

        try:
            text = file_path.read_text()
        except Exception as e:
            log.error("llmem: hooks: failed to read %s: %s", file_path, e)
            return PROCESS_RESULT_ERROR, 0

        return self._process_text(source_id, text)

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

        # Enforce the same size limit as process_file to prevent OOM
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
                except Exception:
                    pass
            mid = self._store.add(
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

    def process_directory(
        self, directory: Path, pattern: str = "*.md"
    ) -> dict[str, int]:
        """Process all transcript files in a directory.

        Returns:
            Dict mapping result_type to count.
        """
        results: dict[str, int] = {}
        for file_path in discover_transcript_files(directory, pattern):
            result_type, cnt = self.process_file(file_path)
            results[result_type] = results.get(result_type, 0) + (
                cnt if result_type == PROCESS_RESULT_SUCCESS else 1
            )
        return results


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


def introspect_session(
    file_path: Path,
    store: MemoryStore,
    analyzer: IntrospectionAnalyzer,
    embedder: EmbeddingEngine | None = None,
    force: bool = False,
    max_file_size: int = _INTROSPECT_MAX_FILE_SIZE,
) -> tuple[str, str | None]:
    """Analyze a session transcript and store a self_assessment memory."""
    if not file_path.exists():
        return INTROSPECT_RESULT_FILE_NOT_FOUND, None

    try:
        file_size = file_path.stat().st_size
    except OSError:
        return INTROSPECT_RESULT_FILE_NOT_FOUND, None

    if file_size > max_file_size:
        return INTROSPECT_RESULT_FILE_TOO_LARGE, None

    source_id = generate_source_id(file_path)
    if not force and store.is_extracted(INTROSPECTION_SOURCE_TYPE, source_id):
        return INTROSPECT_RESULT_ALREADY_PROCESSED, None

    try:
        text = file_path.read_text()
    except Exception:
        return INTROSPECT_RESULT_FILE_NOT_FOUND, None

    if not text.strip():
        return INTROSPECT_RESULT_EMPTY_FILE, None

    return _do_introspect(source_id, text, store, analyzer, embedder, force)


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
        with safe_urlopen(req, allow_remote=True) as resp:
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
        except Exception:
            pass

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


def process_all_session_sources(
    store: MemoryStore,
    extractor: ExtractionEngine | None = None,
    embedder: EmbeddingEngine | None = None,
    force: bool = False,
) -> dict[str, int]:
    """Process all session sources and extract memories.

    Orchestrates processing across all configured session adapters.
    Currently processes only OpenCode sessions; additional adapters
    can be added in the future following the session_hooks.py pattern.

    Args:
        store: The MemoryStore to save extracted memories into.
        extractor: The ExtractionEngine. Created with defaults if None.
        embedder: Optional EmbeddingEngine for generating embeddings.
        force: If True, re-process sessions even if already extracted.

    Returns:
        A dict mapping result constants to their counts, aggregated
        across all session sources.
    """
    from .session_hooks import process_opencode_sessions

    if extractor is None:
        extractor = ExtractionEngine()

    return process_opencode_sessions(
        store=store,
        extractor=extractor,
        embedder=embedder,
        force=force,
    )
