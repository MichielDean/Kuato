"""Code index module for llmem learn command.

Manages the code_chunks table and code_chunks_vec code_chunks_fts virtual
tables for semantic and full-text search over indexed code.
"""

import logging
import sqlite3
import threading
from pathlib import Path

from .chunking import CodeChunk
from .paths import get_db_path
from .store import _run_migrations

log = logging.getLogger(__name__)


class CodeIndex:
    """Manages the code_chunks table for code indexing.

    Creates/opens the SQLite database, runs migrations, and conditionally
    creates code_chunks_vec and code_chunks_fts virtual tables. Shares the
    same database as MemoryStore for cross-retrieval.

    Args:
        db_path: Path to the SQLite database file. When None, uses
            get_db_path() (~/.config/llmem/memory.db by default).
        vec_dimensions: Dimensionality for the vec0 embedding index.
            Defaults to 768. Must be positive.
        disable_vec: If True, skip loading the sqlite-vec extension.
            Useful in CI or environments without sqlite-vec installed.

    Raises:
        ValueError: If vec_dimensions is not positive.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        vec_dimensions: int = 768,
        disable_vec: bool = False,
    ):
        self.db_path = db_path or get_db_path()
        self._vec_dimensions = int(vec_dimensions)
        if self._vec_dimensions <= 0:
            raise ValueError(
                f"llmem: code_index: vec_dimensions must be positive, got {self._vec_dimensions}"
            )
        self._conn: sqlite3.Connection | None = None
        self._conn_lock = threading.Lock()
        self._vec_available: bool | None = None
        self._disable_vec = disable_vec
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database: run migrations, create tables, init vec/fts."""
        conn = self._connect()
        # Migrations create the code_chunks table (005_add_code_chunks.sql)
        _run_migrations(conn)

        # Ensure the code_chunks table exists (for databases created before
        # migration 005 was applied, this is a no-op since migrations handle it)
        # Init FTS5 virtual table for content search
        self._init_fts_table(conn)

        # Init vec0 virtual table if available
        if self._vec_available:
            self._init_vec_table(conn)

    def _connect(self) -> sqlite3.Connection:
        """Get or create the SQLite connection."""
        if self._conn is None:
            with self._conn_lock:
                if self._conn is None:
                    self._conn = sqlite3.connect(
                        str(self.db_path), check_same_thread=False
                    )
                    self._conn.row_factory = sqlite3.Row
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA foreign_keys=ON")
                    self._load_vec_extension()
        return self._conn

    def _load_vec_extension(self) -> None:
        """Attempt to load the sqlite-vec extension.

        Disables extension loading immediately after use (both success and
        failure paths) to prevent runtime loading of arbitrary shared
        libraries via SQLite.
        """
        if self._disable_vec:
            self._vec_available = False
            return
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            # Disable extension loading immediately after use to prevent
            # runtime loading of arbitrary shared libraries via SQLite.
            self._conn.enable_load_extension(False)
            self._vec_available = True
        except Exception as e:
            # Ensure extension loading is disabled even on failure
            if self._conn:
                try:
                    self._conn.enable_load_extension(False)
                except Exception:
                    pass
            log.warning("llmem: code_index: sqlite-vec extension unavailable: %s", e)
            self._vec_available = False

    def _init_vec_table(self, conn: sqlite3.Connection) -> None:
        """Create the code_chunks_vec virtual table if sqlite-vec is available.

        Follows the same pattern as MemoryStore._init_vec_table() for memories_vec,
        including INSERT/UPDATE/DELETE triggers for automatic sync.
        """
        vec_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='code_chunks_vec'"
        ).fetchone()
        if vec_exists:
            existing_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='code_chunks_vec'"
            ).fetchone()["sql"]
            import re

            dim_match = re.search(r"float\[(\d+)\]", existing_sql)
            if dim_match:
                existing_dim = int(dim_match.group(1))
                if existing_dim != self._vec_dimensions:
                    raise ValueError(
                        f"llmem: code_index: existing code_chunks_vec has "
                        f"dimensions={existing_dim} but vec_dimensions="
                        f"{self._vec_dimensions} was requested. Open with "
                        f"vec_dimensions={existing_dim} to reuse."
                    )
        else:
            conn.execute(
                f'CREATE VIRTUAL TABLE "code_chunks_vec" USING vec0('
                f"rowid INTEGER PRIMARY KEY, embedding float[{self._vec_dimensions}]"
                f" distance_metric=cosine)"
            )
            conn.commit()

        # Rebuild vec index if stale
        vec_count = conn.execute('SELECT count(*) FROM "code_chunks_vec"').fetchone()[0]
        chunk_emb_count = conn.execute(
            'SELECT count(*) FROM "code_chunks" WHERE "embedding" IS NOT NULL'
        ).fetchone()[0]
        if vec_count < chunk_emb_count:
            conn.execute('DELETE FROM "code_chunks_vec"')
            rows = conn.execute(
                'SELECT "rowid", "embedding" FROM "code_chunks" WHERE "embedding" IS NOT NULL'
            ).fetchall()
            if rows:
                for row in rows:
                    conn.execute(
                        'INSERT INTO "code_chunks_vec"("rowid", "embedding") VALUES (?, ?)',
                        (row["rowid"], row["embedding"]),
                    )
                conn.commit()

        # Create triggers for automatic vec sync
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "code_chunks_vec_insert" '
            'AFTER INSERT ON "code_chunks" '
            'WHEN new."embedding" IS NOT NULL '
            "BEGIN "
            'INSERT INTO "code_chunks_vec"("rowid", "embedding") '
            'VALUES (new."rowid", new."embedding"); '
            "END"
        )
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "code_chunks_vec_update" '
            'AFTER UPDATE ON "code_chunks" '
            'WHEN new."embedding" IS NOT NULL '
            "BEGIN "
            'DELETE FROM "code_chunks_vec" WHERE "rowid" = old."rowid"; '
            'INSERT INTO "code_chunks_vec"("rowid", "embedding") '
            'VALUES (new."rowid", new."embedding"); '
            "END"
        )
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "code_chunks_vec_update_null" '
            'AFTER UPDATE ON "code_chunks" '
            'WHEN new."embedding" IS NULL AND old."embedding" IS NOT NULL '
            "BEGIN "
            'DELETE FROM "code_chunks_vec" WHERE "rowid" = old."rowid"; '
            "END"
        )
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "code_chunks_vec_delete" '
            'AFTER DELETE ON "code_chunks" '
            'WHEN old."embedding" IS NOT NULL '
            "BEGIN "
            'DELETE FROM "code_chunks_vec" WHERE "rowid" = old."rowid"; '
            "END"
        )
        conn.commit()

    def _init_fts_table(self, conn: sqlite3.Connection) -> None:
        """Create the code_chunks_fts FTS5 virtual table for content search.

        Uses content=code_chunks and content_rowid=rowid for the
        content-sync pattern, matching the memories_fts pattern.
        Creates INSERT/UPDATE/DELETE triggers for automatic FTS maintenance.
        """
        fts_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='code_chunks_fts'"
        ).fetchone()
        if fts_exists:
            # Check if FTS needs rebuilding (stale content)
            fts_count = conn.execute(
                'SELECT count(*) FROM "code_chunks_fts"'
            ).fetchone()[0]
            chunk_count = conn.execute('SELECT count(*) FROM "code_chunks"').fetchone()[
                0
            ]
            if fts_count == 0 and chunk_count > 0:
                conn.execute(
                    'INSERT INTO "code_chunks_fts"("rowid", "content", "file_path", "language") '
                    'SELECT "rowid", "content", "file_path", "language" FROM "code_chunks" '
                    'WHERE "content" IS NOT NULL'
                )
                conn.commit()
            return

        # Create FTS5 virtual table with content-sync pattern
        conn.execute(
            'CREATE VIRTUAL TABLE "code_chunks_fts" USING fts5('
            '"content", "file_path", "language", '
            'content="code_chunks", content_rowid="rowid")'
        )
        conn.commit()

        # Rebuild FTS index from existing data
        chunk_count = conn.execute('SELECT count(*) FROM "code_chunks"').fetchone()[0]
        if chunk_count > 0:
            conn.execute(
                'INSERT INTO "code_chunks_fts"("rowid", "content", "file_path", "language") '
                'SELECT "rowid", "content", "file_path", "language" FROM "code_chunks" '
                'WHERE "content" IS NOT NULL'
            )
            conn.commit()

        # Create triggers for automatic FTS maintenance
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "code_chunks_fts_insert" '
            'AFTER INSERT ON "code_chunks" BEGIN '
            'INSERT INTO "code_chunks_fts"("rowid", "content", "file_path", "language") '
            'VALUES (new."rowid", new."content", new."file_path", new."language"); '
            "END"
        )
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "code_chunks_fts_update" '
            'AFTER UPDATE ON "code_chunks" BEGIN '
            'INSERT INTO "code_chunks_fts"("code_chunks_fts", "rowid", "content", "file_path", "language") '
            'VALUES (\'delete\', old."rowid", old."content", old."file_path", old."language"); '
            'INSERT INTO "code_chunks_fts"("rowid", "content", "file_path", "language") '
            'VALUES (new."rowid", new."content", new."file_path", new."language"); '
            "END"
        )
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "code_chunks_fts_delete" '
            'AFTER DELETE ON "code_chunks" BEGIN '
            'INSERT INTO "code_chunks_fts"("code_chunks_fts", "rowid", "content", "file_path", "language") '
            'VALUES (\'delete\', old."rowid", old."content", old."file_path", old."language"); '
            "END"
        )
        conn.commit()

    def add_chunk(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        content: str,
        embedding: bytes | None = None,
        language: str | None = None,
        chunk_type: str = "paragraph",
    ) -> str:
        """Insert a code chunk into code_chunks.

        Args:
            file_path: Path of the source file.
            start_line: Starting line number (1-based).
            end_line: Ending line number (1-based, inclusive).
            content: The chunk text content.
            embedding: Optional embedding vector as bytes.
            language: Programming language of the file.
            chunk_type: Type of chunk (e.g., "paragraph" or "fixed_line").

        Returns:
            The generated chunk ID (format: "<file_path>:<start_line>:<end_line>").

        Raises:
            ValueError: If file_path, content are empty, or start_line/end_line
                are not positive.
        """
        if not file_path:
            raise ValueError(
                "llmem: code_index: add_chunk: file_path must not be empty"
            )
        if not content:
            raise ValueError("llmem: code_index: add_chunk: content must not be empty")
        if start_line < 1:
            raise ValueError(
                f"llmem: code_index: add_chunk: start_line must be positive, got {start_line}"
            )
        if end_line < 1:
            raise ValueError(
                f"llmem: code_index: add_chunk: end_line must be positive, got {end_line}"
            )

        chunk_id = f"{file_path}:{start_line}:{end_line}"
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO "code_chunks"
                   ("id", "file_path", "start_line", "end_line", "content",
                    "embedding", "language", "chunk_type")
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    file_path,
                    start_line,
                    end_line,
                    content,
                    embedding,
                    language,
                    chunk_type,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            if str(e).startswith("UNIQUE constraint failed"):
                log.debug(
                    "llmem: code_index: add_chunk: skipping duplicate chunk %s",
                    chunk_id,
                )
            else:
                raise
        return chunk_id

    def add_chunks(self, chunks: list[CodeChunk]) -> list[str]:
        """Batch insert multiple code chunks.

        Delegates to add_chunk() for each chunk. Each chunk is committed
        individually, matching MemoryStore.add() behavior.

        Args:
            chunks: List of CodeChunk named tuples to insert.

        Returns:
            List of chunk IDs for the inserted chunks.
        """
        chunk_ids: list[str] = []
        for chunk in chunks:
            chunk_id = self.add_chunk(
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                embedding=None,  # Embeddings added separately via update
                language=chunk.language,
                chunk_type=chunk.chunk_type,
            )
            chunk_ids.append(chunk_id)
        return chunk_ids

    def remove_by_path(self, file_path: str) -> int:
        """Remove all chunks matching a file path.

        Also deletes corresponding rows from code_chunks_vec if vec is available.

        Args:
            file_path: The file path to remove chunks for.

        Returns:
            The number of rows deleted.
        """
        conn = self._connect()
        cursor = conn.execute(
            'DELETE FROM "code_chunks" WHERE "file_path" = ?',
            (file_path,),
        )
        conn.commit()
        return cursor.rowcount

    def search_by_embedding(
        self,
        query_vec: list[float],
        limit: int = 10,
        threshold: float = 0.5,
        language: str | None = None,
    ) -> list[tuple[dict, float]]:
        """Search code_chunks by embedding similarity.

        Uses the vec0 virtual table when available, falls back to
        brute-force cosine similarity otherwise.

        Args:
            query_vec: The query embedding vector.
            limit: Maximum number of results. Defaults to 10.
            threshold: Minimum cosine similarity. Defaults to 0.5.
            language: Optional language filter.

        Returns:
            List of (chunk_dict, cosine_similarity) tuples sorted by
            similarity descending.
        """
        if self._vec_available:
            return self._search_by_embedding_vec(query_vec, limit, threshold, language)
        return self._search_by_embedding_brute(query_vec, limit, threshold, language)

    def _search_by_embedding_vec(
        self,
        query_vec: list[float],
        limit: int,
        threshold: float,
        language: str | None,
    ) -> list[tuple[dict, float]]:
        """Search using the vec0 virtual table."""
        import struct

        conn = self._connect()
        query_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)

        for multiplier in (3, 10, 50, 0):
            search_limit = (
                max(limit * multiplier, limit + 1)
                if multiplier
                else conn.execute('SELECT count(*) FROM "code_chunks_vec"').fetchone()[
                    0
                ]
            )
            try:
                vec_rows = conn.execute(
                    'SELECT "rowid", "distance" FROM "code_chunks_vec" '
                    'WHERE "embedding" MATCH ? AND k = ? '
                    'ORDER BY "distance"',
                    [query_bytes, search_limit],
                ).fetchall()
            except sqlite3.OperationalError:
                return self._search_by_embedding_brute(
                    query_vec, limit, threshold, language
                )

            if not vec_rows:
                return []

            rowids = [r["rowid"] for r in vec_rows]
            placeholders = ",".join("?" for _ in rowids)
            where = f'"rowid" IN ({placeholders})'
            params: list = list(rowids)
            if language:
                where += ' AND "language" = ?'
                params.append(language)
            # Must explicitly select rowid since SELECT * doesn't include it
            # for tables with TEXT PRIMARY KEY
            chunk_rows = conn.execute(
                f'SELECT "rowid", * FROM "code_chunks" WHERE {where}', params
            ).fetchall()
            matched_rowids = {r["rowid"] for r in chunk_rows}
            # Build a rowid -> id mapping for looking up chunk IDs
            rowid_to_id = {r["rowid"]: r["id"] for r in chunk_rows}
            scored: list[tuple[str, float]] = []
            for r in vec_rows:
                rid = r["rowid"]
                cosine_sim = 1.0 - r["distance"]
                if rid in matched_rowids and cosine_sim >= threshold:
                    scored.append((rowid_to_id[rid], cosine_sim))
            if len(scored) >= limit or multiplier == 0:
                break

        scored = scored[:limit]
        if not scored:
            return []

        # Fetch full rows
        top_ids = [chunk_id for chunk_id, _ in scored]
        id_placeholders = ",".join("?" for _ in top_ids)
        full_rows = conn.execute(
            f'SELECT * FROM "code_chunks" WHERE "id" IN ({id_placeholders})', top_ids
        ).fetchall()
        chunk_map = {r["id"]: dict(r) for r in full_rows}
        results: list[tuple[dict, float]] = []
        for chunk_id, score in scored:
            if chunk_id in chunk_map:
                results.append((chunk_map[chunk_id], score))
        return results

    def _search_by_embedding_brute(
        self,
        query_vec: list[float],
        limit: int,
        threshold: float,
        language: str | None,
    ) -> list[tuple[dict, float]]:
        """Brute-force cosine similarity search fallback."""
        import struct

        conn = self._connect()
        where = 'WHERE "embedding" IS NOT NULL'
        params: list = []
        if language:
            where += ' AND "language" = ?'
            params.append(language)
        rows = conn.execute(
            f'SELECT "id", "embedding" FROM "code_chunks" {where} LIMIT ?',
            params + [10000],
        ).fetchall()
        if not rows:
            return []

        scored: list[tuple[str, float]] = []
        for row in rows:
            emb_bytes = row["embedding"]
            dim = len(emb_bytes) // 4
            vec = list(struct.unpack(f"{dim}f", emb_bytes))
            if len(vec) != len(query_vec):
                continue
            score = self._cosine_sim(query_vec, vec)
            if score >= threshold:
                scored.append((row["id"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored[:limit]:
            return []

        top_ids = [chunk_id for chunk_id, _ in scored[:limit]]
        id_placeholders = ",".join("?" for _ in top_ids)
        full_rows = conn.execute(
            f'SELECT * FROM "code_chunks" WHERE "id" IN ({id_placeholders})', top_ids
        ).fetchall()
        chunk_map = {r["id"]: dict(r) for r in full_rows}
        results: list[tuple[dict, float]] = []
        for chunk_id, score in scored[:limit]:
            if chunk_id in chunk_map:
                results.append((chunk_map[chunk_id], score))
        return results

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 for zero-magnitude vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [-1.0, 1.0] (0.0 for zero-magnitude inputs).
        """
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def search_content(
        self, query: str, limit: int = 20, language: str | None = None
    ) -> list[dict]:
        """FTS5 search on code_chunks content.

        Args:
            query: The search query string.
            limit: Maximum number of results. Defaults to 20.
            language: Optional language filter.

        Returns:
            List of chunk dicts matching the query.
        """
        conn = self._connect()
        # Sanitize the FTS query
        fts_query = self._sanitize_fts_query(query)

        try:
            where_clauses: list[str] = []
            params: list = []

            if language:
                where_clauses.append('c."language" = ?')
                params.append(language)

            where = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""

            rows = conn.execute(
                f'SELECT c.*, -bm25("code_chunks_fts") AS _fts_rank '
                f'FROM "code_chunks_fts" AS fts '
                f'JOIN "code_chunks" AS c ON c."rowid" = fts."rowid" '
                f'WHERE "code_chunks_fts" MATCH ?{where} '
                f"ORDER BY _fts_rank DESC LIMIT ?",
                [fts_query] + params + [limit],
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS not available, fall back to LIKE search
            # Escape LIKE wildcards (% and _) in the user query
            escaped_query = (
                query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            like_pattern = f"%{escaped_query}%"
            where = "WHERE \"content\" LIKE ? ESCAPE '\\'"
            params_like: list = [like_pattern]
            if language:
                where += ' AND "language" = ?'
                params_like.append(language)
            rows = conn.execute(
                f'SELECT * FROM "code_chunks" {where} LIMIT ?',
                params_like + [limit],
            ).fetchall()

        results: list[dict] = []
        for r in rows:
            d = dict(r)
            d.pop("_fts_rank", None)
            results.append(d)
        return results

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize an FTS5 query string.

        Removes FTS5 operators and special characters, keeping only
        word tokens joined with OR.
        """
        import re

        tokens = re.split(r"\s+", query.strip())
        safe_tokens: list[str] = []
        for t in tokens:
            if t.upper() in {"AND", "OR", "NOT", "NEAR"}:
                continue
            clean = re.sub(r"[^\w]", " ", t)
            parts = clean.split()
            safe_tokens.extend(parts)
        if not safe_tokens:
            return '""'
        return " OR ".join(safe_tokens)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
