"""SQLite-backed memory store with vector search."""

import importlib.resources
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .paths import get_db_path

logger = logging.getLogger(__name__)

# Default registered memory types — used to initialize each MemoryStore instance
_DEFAULT_TYPES: frozenset[str] = frozenset(
    {
        "fact",
        "decision",
        "preference",
        "event",
        "project_state",
        "procedure",
        "conversation",
        "self_assessment",
    }
)


def register_memory_type(type_name: str) -> None:
    """Register a new memory type for use with MemoryStore.add().

    Args:
        type_name: The type name to register (e.g., 'custom_type').

    Raises:
        ValueError: If type_name is already registered.

    After registration, store.add(type=type_name, ...) will succeed.
    Before registration, store.add(type=type_name, ...) raises ValueError.

    Note:
        This registers the type globally so all current and future
        MemoryStore instances recognize it. The global registry avoids
        requiring callers to pass a store reference just to check valid
        type names (e.g., from the CLI argparse layer).
    """
    if type_name in _global_registry:
        raise ValueError(
            f"llmem: register_memory_type: type '{type_name}' is already registered"
        )
    _global_registry.add(type_name)


def get_registered_types() -> frozenset[str]:
    """Return the current set of globally registered memory types."""
    return frozenset(_global_registry)


def _reset_global_registry() -> None:
    """Reset the global registry to default types only. For testing."""
    _global_registry.clear()
    _global_registry.update(_DEFAULT_TYPES)


# Global registry — initialized with defaults, extensible via register_memory_type().
# This is read-only from the perspective of MemoryStore instances: each instance
# copies the global set at construction time, but add() validates against the
# instance-local set so that a store opened before a registration does not
# silently accept a type it was not configured for.
_global_registry: set[str] = set(_DEFAULT_TYPES)


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Run all unapplied SQL migration files from the migrations/ directory.

    Reads numbered .sql files from the llmem_migrations package
    (via importlib.resources) and tracks applied migrations in
    the _schema_migrations table.

    Args:
        conn: Active sqlite3.Connection to apply migrations to.
    """
    # Ensure _schema_migrations table exists
    conn.execute(
        'CREATE TABLE IF NOT EXISTS "_schema_migrations" ('
        '"version" INTEGER PRIMARY KEY, '
        "\"applied_at\" TEXT NOT NULL DEFAULT (datetime('now'))"
        ")"
    )
    conn.commit()

    applied = {
        row[0]
        for row in conn.execute('SELECT "version" FROM "_schema_migrations"').fetchall()
    }

    # Discover migration files from the migrations package
    try:
        migrations_pkg = importlib.resources.files("llmem_migrations")
    except ModuleNotFoundError:
        # Fallback: try loading from the llmem package's sibling directory
        migrations_pkg = importlib.resources.files("llmem").joinpath("..", "migrations")

    # Collect and sort migration files
    migration_files: list[tuple[int, str]] = []
    try:
        for item in migrations_pkg.iterdir():
            name = str(item.name) if hasattr(item, "name") else str(item)
            # Match pattern: 001_xxx.sql
            basename = name.rsplit("/", 1)[-1] if "/" in name else name
            match = re.match(r"^(\d+)_.+\.sql$", basename)
            if match:
                version = int(match.group(1))
                migration_files.append((version, basename))
    except (AttributeError, TypeError):
        # If iterdir doesn't work (e.g., namespace package), try reading directly
        migration_files = [
            (1, "001_initial_schema.sql"),
            (2, "002_add_hints.sql"),
            (3, "003_register_default_types.sql"),
        ]

    migration_files.sort()

    for version, filename in migration_files:
        if version in applied:
            continue

        try:
            sql_file = migrations_pkg.joinpath(filename)
            sql_content = sql_file.read_text()
        except (FileNotFoundError, AttributeError) as e:
            logger.warning("llmem: migration: could not read %s: %s", filename, e)
            continue

        # Execute the migration
        try:
            conn.executescript(sql_content)
            conn.execute(
                'INSERT OR IGNORE INTO "_schema_migrations" ("version") VALUES (?)',
                (version,),
            )
            conn.commit()
            logger.info("llmem: migration: applied %s", filename)
        except sqlite3.Error as e:
            logger.error("llmem: migration: failed to apply %s: %s", filename, e)
            conn.rollback()
            raise


class MemoryStore:
    """SQLite-backed memory store with FTS5 full-text and optional vector search.

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
                f"vec_dimensions must be positive, got {self._vec_dimensions}"
            )
        # Defensive copy: each instance snapshots the global registry at
        # construction time. register_memory_type() updates the global
        # registry, which is available to the CLI layer for argparse choices,
        # but existing stores do not retroactively accept new types.
        self._registered_types: set[str] = set(_global_registry)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if str(self.db_path) != ":memory:":
            os.chmod(str(self.db_path.parent), 0o700)
        self._conn: sqlite3.Connection | None = None
        self._conn_lock = threading.Lock()
        self._vec_available: bool | None = None
        self._disable_vec = disable_vec
        self._init_db()
        if str(self.db_path) != ":memory:":
            try:
                os.chmod(str(self.db_path), 0o600)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_db(self):
        conn = self._connect()
        _run_migrations(conn)
        fts_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        ).fetchone()
        if fts_exists:
            fts_count = conn.execute('SELECT count(*) FROM "memories_fts"').fetchone()[
                0
            ]
            mem_count = conn.execute('SELECT count(*) FROM "memories"').fetchone()[0]
            if fts_count == 0 and mem_count > 0:
                conn.execute(
                    'INSERT INTO "memories_fts"("rowid", "content", "summary", "hints") '
                    'SELECT "rowid", "content", "summary", "hints" FROM "memories" WHERE "content" IS NOT NULL'
                )
                conn.commit()
        if self._vec_available:
            self._init_vec_table(conn)
        elif self._disable_vec:
            for trigger in (
                "memories_vec_insert",
                "memories_vec_update",
                "memories_vec_update_null",
                "memories_vec_delete",
            ):
                conn.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
            conn.commit()

    def _init_vec_table(self, conn: sqlite3.Connection):
        vec_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_vec'"
        ).fetchone()
        if vec_exists:
            existing_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories_vec'"
            ).fetchone()["sql"]
            dim_match = re.search(r"float\[(\d+)\]", existing_sql)
            if dim_match:
                existing_dim = int(dim_match.group(1))
                if existing_dim != self._vec_dimensions:
                    raise ValueError(
                        f"Existing vec0 table has dimensions={existing_dim} "
                        f"but vec_dimensions={self._vec_dimensions} was requested. "
                        f"Open the store with vec_dimensions={existing_dim} to reuse the existing index."
                    )
        else:
            conn.execute(
                f'CREATE VIRTUAL TABLE "memories_vec" USING vec0("embedding" float[{self._vec_dimensions}] distance_metric=cosine)'
            )
            conn.commit()
        vec_count = conn.execute('SELECT count(*) FROM "memories_vec"').fetchone()[0]
        mem_emb_count = conn.execute(
            'SELECT count(*) FROM "memories" WHERE "embedding" IS NOT NULL'
        ).fetchone()[0]
        if vec_count < mem_emb_count:
            conn.execute('DELETE FROM "memories_vec"')
            rows = conn.execute(
                'SELECT "rowid", "embedding" FROM "memories" WHERE "embedding" IS NOT NULL'
            ).fetchall()
            if rows:
                for row in rows:
                    conn.execute(
                        'INSERT INTO "memories_vec"("rowid", "embedding") VALUES (?, ?)',
                        (row["rowid"], row["embedding"]),
                    )
                conn.commit()
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "memories_vec_insert" AFTER INSERT ON "memories" '
            'WHEN new."embedding" IS NOT NULL '
            "BEGIN "
            'INSERT INTO "memories_vec"("rowid", "embedding") VALUES (new."rowid", new."embedding"); '
            "END"
        )
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "memories_vec_update" AFTER UPDATE ON "memories" '
            'WHEN new."embedding" IS NOT NULL '
            "BEGIN "
            'DELETE FROM "memories_vec" WHERE "rowid" = old."rowid"; '
            'INSERT INTO "memories_vec"("rowid", "embedding") VALUES (new."rowid", new."embedding"); '
            "END"
        )
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "memories_vec_update_null" AFTER UPDATE ON "memories" '
            'WHEN new."embedding" IS NULL AND old."embedding" IS NOT NULL '
            "BEGIN "
            'DELETE FROM "memories_vec" WHERE "rowid" = old."rowid"; '
            "END"
        )
        conn.execute(
            'CREATE TRIGGER IF NOT EXISTS "memories_vec_delete" AFTER DELETE ON "memories" '
            'WHEN old."embedding" IS NOT NULL '
            "BEGIN "
            'DELETE FROM "memories_vec" WHERE "rowid" = old."rowid"; '
            "END"
        )
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
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

    def _load_vec_extension(self):
        if self._disable_vec:
            self._vec_available = False
            return
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._vec_available = True
        except Exception as e:
            logger.warning("llmem: store: sqlite-vec extension unavailable: %s", e)
            self._vec_available = False

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def add(
        self,
        type: str,
        content: str,
        summary: str | None = None,
        source: str = "manual",
        confidence: float = 0.8,
        valid_from: str | None = None,
        valid_until: str | None = None,
        metadata: dict | None = None,
        embedding: bytes | None = None,
        hints: list[str] | None = None,
        id: str | None = None,
    ) -> str:
        """Add a new memory to the store.

        Args:
            type: Memory type, must be in the registered type set.
            content: The memory content text.
            summary: Optional short summary of the content.
            source: Origin of the memory (default: 'manual').
            confidence: Confidence score 0.0–1.0 (default: 0.8).
            valid_from: ISO timestamp for when the memory becomes valid.
            valid_until: ISO timestamp for when the memory expires.
            metadata: Optional JSON-serializable metadata dict.
            embedding: Optional embedding vector as bytes.
            hints: Optional list of hint strings.
            id: Optional custom ID (auto-generated UUID if None).

        Returns:
            The memory ID string.

        Raises:
            ValueError: If type is not in the registered type set.
        """
        if type not in self._registered_types:
            raise ValueError(
                f"llmem: store: add: unregistered type '{type}'. "
                f"Register it with register_memory_type('{type}') first."
            )
        mem_id = id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        conn.execute(
            """INSERT INTO "memories"
               ("id", "type", "content", "summary", "hints", "source", "confidence",
                "valid_from", "valid_until", "created_at", "updated_at", "metadata", "embedding")
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mem_id,
                type,
                content,
                summary,
                json.dumps(hints or []),
                source,
                confidence,
                valid_from or now,
                valid_until,
                now,
                now,
                json.dumps(metadata or {}),
                embedding,
            ),
        )
        conn.commit()
        return mem_id

    def get(self, mem_id: str, track_access: bool = False) -> dict | None:
        """Retrieve a memory by ID.

        Args:
            mem_id: The memory ID.
            track_access: If True, increment access count and update accessed_at.

        Returns:
            A dict with memory fields, or None if not found.
        """
        conn = self._connect()
        row = conn.execute(
            'SELECT * FROM "memories" WHERE "id" = ?', (mem_id,)
        ).fetchone()
        if not row:
            return None
        d = self._row_to_dict(row)
        if track_access:
            now = datetime.now(timezone.utc).isoformat()
            self._touch(mem_id, now=now)
            d["access_count"] = d.get("access_count", 0) + 1
            d["accessed_at"] = now
        return d

    def get_batch(self, mem_ids: list[str], valid_only: bool = True) -> dict[str, dict]:
        if not mem_ids:
            return {}
        conn = self._connect()
        placeholders = ",".join("?" for _ in mem_ids)
        where = f'"id" IN ({placeholders})'
        if valid_only:
            where += ' AND "valid_until" IS NULL'
        rows = conn.execute(
            f'SELECT * FROM "memories" WHERE {where}',
            mem_ids,
        ).fetchall()
        return {r["id"]: self._row_to_dict(r) for r in rows}

    def update(
        self,
        mem_id: str,
        content: str | None = None,
        summary: str | None = None,
        confidence: float | None = None,
        valid_until: str | None = None,
        metadata: dict | None = None,
        embedding: bytes | None = None,
        clear_embedding: bool = False,
        hints: list[str] | None = None,
    ) -> bool:
        if clear_embedding and embedding is not None:
            raise ValueError("Cannot specify both embedding= and clear_embedding=True")
        conn = self._connect()
        row = conn.execute(
            'SELECT * FROM "memories" WHERE "id" = ?', (mem_id,)
        ).fetchone()
        if not row:
            return False
        now = datetime.now(timezone.utc).isoformat()
        sets = ['"updated_at" = ?']
        vals = [now]
        if content is not None:
            sets.append('"content" = ?')
            vals.append(content)
            if not clear_embedding and embedding is None:
                sets.append('"embedding" = NULL')
        if summary is not None:
            sets.append('"summary" = ?')
            vals.append(summary)
        if confidence is not None:
            sets.append('"confidence" = ?')
            vals.append(confidence)
        if valid_until is not None:
            sets.append('"valid_until" = ?')
            vals.append(valid_until)
        if metadata is not None:
            sets.append('"metadata" = ?')
            vals.append(json.dumps(metadata))
        if hints is not None:
            sets.append('"hints" = ?')
            vals.append(json.dumps(hints))
        if clear_embedding:
            sets.append('"embedding" = NULL')
        elif embedding is not None:
            sets.append('"embedding" = ?')
            vals.append(embedding)
        vals.append(mem_id)
        conn.execute(f'UPDATE "memories" SET {", ".join(sets)} WHERE "id" = ?', vals)
        conn.commit()
        return True

    def invalidate(self, mem_id: str, reason: str | None = None) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        meta_override = {}
        if reason:
            meta_override["invalidation_reason"] = reason
        conn = self._connect()
        row = conn.execute(
            'SELECT "metadata" FROM "memories" WHERE "id" = ?', (mem_id,)
        ).fetchone()
        if not row:
            return False
        meta = json.loads(row["metadata"])
        meta.update(meta_override)
        conn.execute(
            'UPDATE "memories" SET "valid_until" = ?, "metadata" = ?, "embedding" = NULL, "updated_at" = ? WHERE "id" = ?',
            (now, json.dumps(meta), now, mem_id),
        )
        conn.commit()
        return True

    def delete(self, mem_id: str) -> bool:
        conn = self._connect()
        cursor = conn.execute('DELETE FROM "memories" WHERE "id" = ?', (mem_id,))
        conn.commit()
        return cursor.rowcount > 0

    def search(
        self,
        query: str | None = None,
        type: str | None = None,
        valid_only: bool = True,
        limit: int = 20,
        offset: int = 0,
        _include_rank: bool = False,
    ) -> list[dict]:
        conn = self._connect()
        if query:
            clauses = []
            fts_vals = []
            if valid_only:
                clauses.append('m."valid_until" IS NULL')
            if type:
                clauses.append('m."type" = ?')
                fts_vals.append(type)
            where = (" AND " + " AND ".join(clauses)) if clauses else ""
            fts_query = self._sanitize_fts_query(query)
            try:
                rows = conn.execute(
                    f'SELECT m.*, -bm25("memories_fts") AS _fts_rank '
                    f'FROM "memories_fts" AS fts JOIN "memories" AS m ON m."rowid" = fts."rowid" '
                    f'WHERE "memories_fts" MATCH ?{where} '
                    f"ORDER BY _fts_rank DESC LIMIT ? OFFSET ?",
                    [fts_query] + fts_vals + [limit, offset],
                ).fetchall()
            except sqlite3.OperationalError:
                rows = self._fallback_like_search(
                    conn, query, type, valid_only, limit, offset
                )
            results = []
            for r in rows:
                d = self._row_to_dict(r)
                fts_rank = d.pop("_fts_rank", 0.0)
                if _include_rank:
                    d["_fts_rank"] = fts_rank
                results.append(d)
            return results
        clauses = []
        vals = []
        if valid_only:
            clauses.append('"valid_until" IS NULL')
        if type:
            clauses.append('"type" = ?')
            vals.append(type)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = conn.execute(
            f'SELECT * FROM "memories" WHERE {where} ORDER BY "updated_at" DESC LIMIT ? OFFSET ?',
            vals + [limit, offset],
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search_count(
        self,
        query: str | None = None,
        type: str | None = None,
        valid_only: bool = True,
    ) -> int:
        conn = self._connect()
        if query:
            clauses = []
            fts_vals = []
            if valid_only:
                clauses.append('m."valid_until" IS NULL')
            if type:
                clauses.append('m."type" = ?')
                fts_vals.append(type)
            where = (" AND " + " AND ".join(clauses)) if clauses else ""
            fts_query = self._sanitize_fts_query(query)
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) "
                    f'FROM "memories_fts" AS fts JOIN "memories" AS m ON m."rowid" = fts."rowid" '
                    f'WHERE "memories_fts" MATCH ?{where}',
                    [fts_query] + fts_vals,
                ).fetchone()
                return row[0]
            except sqlite3.OperationalError:
                like_clauses = []
                like_vals: list = []
                like_clauses.append(
                    "(m.\"content\" LIKE ? ESCAPE '\\' OR m.\"summary\" LIKE ? ESCAPE '\\' OR m.\"hints\" LIKE ? ESCAPE '\\')"
                )
                like_escaped = self._escape_like(query)
                like_vals.extend(
                    [f"%{like_escaped}%", f"%{like_escaped}%", f"%{like_escaped}%"]
                )
                if valid_only:
                    like_clauses.append('m."valid_until" IS NULL')
                if type:
                    like_clauses.append('m."type" = ?')
                    like_vals.append(type)
                like_where = " AND ".join(like_clauses)
                row = conn.execute(
                    f'SELECT COUNT(*) FROM "memories" AS m WHERE {like_where}',
                    like_vals,
                ).fetchone()
                return row[0]
        clauses = []
        vals = []
        if valid_only:
            clauses.append('"valid_until" IS NULL')
        if type:
            clauses.append('"type" = ?')
            vals.append(type)
        where = " AND ".join(clauses) if clauses else "1=1"
        row = conn.execute(
            f'SELECT COUNT(*) FROM "memories" WHERE {where}',
            vals,
        ).fetchone()
        return row[0]

    def search_by_embedding(
        self,
        query_vec: list[float],
        valid_only: bool = True,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[dict, float]]:
        if self._vec_available:
            return self._search_by_embedding_vec(
                query_vec, valid_only, limit, threshold
            )
        return self._search_by_embedding_brute(query_vec, valid_only, limit, threshold)

    def _search_by_embedding_vec(
        self,
        query_vec: list[float],
        valid_only: bool,
        limit: int,
        threshold: float,
    ) -> list[tuple[dict, float]]:
        import struct

        conn = self._connect()
        query_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)
        for multiplier in (3, 10, 50, 0):
            search_limit = (
                max(limit * multiplier, limit + 1)
                if multiplier
                else conn.execute('SELECT count(*) FROM "memories_vec"').fetchone()[0]
            )
            try:
                vec_rows = conn.execute(
                    'SELECT "rowid", "distance" FROM "memories_vec" '
                    'WHERE "embedding" MATCH ? AND k = ? '
                    'ORDER BY "distance"',
                    [query_bytes, search_limit],
                ).fetchall()
            except sqlite3.OperationalError:
                return self._search_by_embedding_brute(
                    query_vec, valid_only, limit, threshold
                )
            if not vec_rows:
                return []
            rowids = [r["rowid"] for r in vec_rows]
            placeholders = ",".join("?" for _ in rowids)
            where = f'"rowid" IN ({placeholders})'
            if valid_only:
                where += ' AND "valid_until" IS NULL'
            mem_rows = conn.execute(
                f'SELECT "id", "rowid" FROM "memories" WHERE {where}', rowids
            ).fetchall()
            rid_to_memid = {r["rowid"]: r["id"] for r in mem_rows}
            matched_rowids = set(rid_to_memid.keys())
            scored = []
            for r in vec_rows:
                rid = r["rowid"]
                cosine_sim = 1.0 - r["distance"]
                if rid in matched_rowids and cosine_sim >= threshold:
                    scored.append((rid_to_memid[rid], cosine_sim))
            if len(scored) >= limit or multiplier == 0:
                break
        scored = scored[:limit]
        if not scored:
            return []
        top_ids = [mid for mid, _ in scored]
        id_placeholders = ",".join("?" for _ in top_ids)
        full_rows = conn.execute(
            f'SELECT * FROM "memories" WHERE "id" IN ({id_placeholders})', top_ids
        ).fetchall()
        mem_map = {r["id"]: self._row_to_dict(r) for r in full_rows}
        results = []
        for mem_id, score in scored:
            if mem_id in mem_map:
                results.append((mem_map[mem_id], score))
        return results

    def _search_by_embedding_brute(
        self,
        query_vec: list[float],
        valid_only: bool,
        limit: int,
        threshold: float,
    ) -> list[tuple[dict, float]]:
        import struct

        conn = self._connect()
        rows = conn.execute(
            'SELECT "id", "embedding" FROM "memories" WHERE "embedding" IS NOT NULL'
            + (' AND "valid_until" IS NULL' if valid_only else "")
        ).fetchall()
        if not rows:
            return []
        scored = []
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
        top_ids = [mid for mid, _ in scored[:limit]]
        placeholders = ",".join("?" for _ in top_ids)
        rows = conn.execute(
            f'SELECT * FROM "memories" WHERE "id" IN ({placeholders})', top_ids
        ).fetchall()
        row_map = {r["id"]: self._row_to_dict(r) for r in rows}
        results = []
        for mem_id, score in scored[:limit]:
            if mem_id in row_map:
                results.append((row_map[mem_id], score))
        return results

    def list_all(
        self,
        type: str | None = None,
        valid_only: bool = False,
        limit: int = 100,
    ) -> list[dict]:
        return self.search(type=type, valid_only=valid_only, limit=limit)

    def count(self, valid_only: bool = False) -> int:
        conn = self._connect()
        clause = 'WHERE "valid_until" IS NULL' if valid_only else ""
        row = conn.execute(f'SELECT COUNT(*) FROM "memories" {clause}').fetchone()
        return row[0]

    def count_by_type(self, valid_only: bool = False) -> dict[str, int]:
        conn = self._connect()
        clause = 'WHERE "valid_until" IS NULL' if valid_only else ""
        rows = conn.execute(
            f'SELECT "type", COUNT(*) as cnt FROM "memories" {clause} GROUP BY "type" ORDER BY cnt DESC'
        ).fetchall()
        return {r["type"]: r["cnt"] for r in rows}

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        id: str | None = None,
    ) -> str:
        rel_id = id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        conn.execute(
            'INSERT INTO "relations" ("id", "source_id", "target_id", "relation_type", "created_at") VALUES (?,?,?,?,?)',
            (rel_id, source_id, target_id, relation_type, now),
        )
        conn.commit()
        return rel_id

    def get_relations(self, mem_id: str) -> list[dict]:
        conn = self._connect()
        rows = conn.execute(
            'SELECT * FROM "relations" WHERE "source_id" = ? OR "target_id" = ?',
            (mem_id, mem_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_relations_batch(self, mem_ids: list[str]) -> list[dict]:
        if not mem_ids:
            return []
        conn = self._connect()
        placeholders = ",".join("?" for _ in mem_ids)
        rows = conn.execute(
            f'SELECT * FROM "relations" WHERE "source_id" IN ({placeholders}) OR "target_id" IN ({placeholders})',
            mem_ids + mem_ids,
        ).fetchall()
        return [dict(r) for r in rows]

    def traverse_relations(
        self,
        start_ids: list[str],
        max_depth: int = 1,
    ) -> list[dict]:
        """Traverse relation edges from start_ids up to max_depth hops.

        Follows both source_id->target_id and target_id->source_id directions
        so relations are traversed bidirectionally. Uses a recursive CTE.
        Returns deduplicated results with the shortest distance, relation_type,
        and relation_score (decays as 0.5^distance).
        """
        if not start_ids or max_depth < 1:
            return []

        max_depth = min(max_depth, 5)

        conn = self._connect()
        placeholders = ",".join("?" for _ in start_ids)
        exclude = ",".join("?" for _ in start_ids)

        cte_sql = (
            'WITH RECURSIVE "rel_traverse"("node_id", "reached_id", "rel_type", "dist") AS ('
            ' SELECT "source_id", "target_id", "relation_type", 1 FROM "relations"'
            f' WHERE "source_id" IN ({placeholders})'
            " UNION ALL"
            ' SELECT "target_id", "source_id", "relation_type", 1 FROM "relations"'
            f' WHERE "target_id" IN ({placeholders})'
            " UNION ALL"
            ' SELECT r."source_id", r."target_id", r."relation_type", rt."dist" + 1 FROM "relations" r'
            ' INNER JOIN "rel_traverse" rt ON rt."reached_id" = r."source_id"'
            ' WHERE rt."dist" < ?'
            " UNION ALL"
            ' SELECT r."target_id", r."source_id", r."relation_type", rt."dist" + 1 FROM "relations" r'
            ' INNER JOIN "rel_traverse" rt ON rt."reached_id" = r."target_id"'
            ' WHERE rt."dist" < ?'
            ")"
            f' SELECT "reached_id", "rel_type", MIN("dist") as "dist" FROM "rel_traverse"'
            f' WHERE "reached_id" NOT IN ({exclude})'
            ' GROUP BY "reached_id", "rel_type"'
            ' ORDER BY "dist"'
        )
        params = start_ids + start_ids + [max_depth, max_depth] + start_ids
        rows = conn.execute(cte_sql, params).fetchall()

        results = []
        seen = set()
        for row in rows:
            key = (row["reached_id"], row["rel_type"])
            if key in seen:
                continue
            seen.add(key)
            distance = row["dist"]
            results.append(
                {
                    "target_id": row["reached_id"],
                    "relation_type": row["rel_type"],
                    "distance": distance,
                    "relation_score": 0.5**distance,
                }
            )
        return results

    def log_extraction(
        self,
        source_type: str,
        source_id: str,
        raw_text: str | None = None,
        extracted_count: int = 0,
    ):
        conn = self._connect()
        conn.execute(
            """INSERT INTO "extraction_log"
               ("source_type", "source_id", "raw_text", "extracted_count", "created_at")
               VALUES (?,?,?,?,datetime('now'))
               ON CONFLICT("source_type", "source_id") DO UPDATE SET
               "raw_text"=excluded."raw_text",
               "extracted_count"=excluded."extracted_count" """,
            (source_type, source_id, raw_text, extracted_count),
        )
        conn.commit()

    def supersede_by_source(self, source_type: str, source_id: str) -> int:
        conn = self._connect()
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """UPDATE "memories" SET "valid_until" = ?, "updated_at" = ?
               WHERE "source" = ? AND "valid_until" IS NULL
               AND json_extract("metadata", '$.source_id') = ?""",
            (now, now, source_type, source_id),
        )
        conn.commit()
        return cursor.rowcount

    def is_extracted(self, source_type: str, source_id: str) -> bool:
        conn = self._connect()
        row = conn.execute(
            'SELECT 1 FROM "extraction_log" WHERE "source_type" = ? AND "source_id" = ?',
            (source_type, source_id),
        ).fetchone()
        return row is not None

    def remove_extraction_log(self, source_type: str, source_id: str) -> bool:
        conn = self._connect()
        cursor = conn.execute(
            'DELETE FROM "extraction_log" WHERE "source_type" = ? AND "source_id" = ?',
            (source_type, source_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def find_similar(
        self,
        query_vec: list[float] | None = None,
        content: str | None = None,
        threshold: float = 0.85,
        limit: int = 5,
    ) -> list[tuple[dict, float]]:
        if query_vec:
            return self.search_by_embedding(
                query_vec, valid_only=False, limit=limit, threshold=threshold
            )
        if content:
            results = self.search(query=content, valid_only=False, limit=limit)
            return [(r, 0.0) for r in results]
        return []

    def consolidate(
        self, similarity_threshold: float = 0.92, limit: int = 500
    ) -> list[dict]:
        import struct

        conn = self._connect()
        rows = conn.execute(
            'SELECT "id", "content", "embedding" FROM "memories" WHERE "embedding" IS NOT NULL AND "valid_until" IS NULL LIMIT ?',
            (limit,),
        ).fetchall()
        if not rows:
            return []
        pairs = []
        seen = set()
        for i, r1 in enumerate(rows):
            if r1["id"] in seen:
                continue
            emb1 = r1["embedding"]
            dim = len(emb1) // 4
            vec1 = list(struct.unpack(f"{dim}f", emb1))
            for r2 in rows[i + 1 :]:
                if r2["id"] in seen or r2["id"] == r1["id"]:
                    continue
                dim2 = len(r2["embedding"]) // 4
                if dim2 != dim:
                    continue
                vec2 = list(struct.unpack(f"{dim2}f", r2["embedding"]))
                score = self._cosine_sim(vec1, vec2)
                if score >= similarity_threshold:
                    pairs.append(
                        {"source": r1["id"], "target": r2["id"], "score": score}
                    )
                    seen.add(r1["id"])
                    seen.add(r2["id"])
        return pairs

    def export_all(self) -> list[dict]:
        conn = self._connect()
        rows = conn.execute('SELECT * FROM "memories" ORDER BY "created_at"').fetchall()
        return [self._row_to_dict(r) for r in rows]

    def import_memories(self, memories: list[dict]) -> int:
        count = 0
        for m in memories:
            hints = m.get("hints")
            if isinstance(hints, str):
                try:
                    hints = json.loads(hints)
                except (json.JSONDecodeError, TypeError):
                    hints = []
            try:
                self.add(
                    type=m["type"],
                    content=m["content"],
                    summary=m.get("summary"),
                    source=m.get("source", "import"),
                    confidence=m.get("confidence", 0.8),
                    metadata=m.get("metadata"),
                    hints=hints,
                    id=m.get("id"),
                    embedding=m.get("embedding"),
                )
                count += 1
            except sqlite3.IntegrityError as e:
                if str(e).startswith("UNIQUE constraint failed"):
                    continue
                raise
        return count

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        tokens = re.split(r"\s+", query.strip())
        safe_tokens = []
        for t in tokens:
            if t.upper() in {"AND", "OR", "NOT", "NEAR"}:
                continue
            clean = re.sub(r"[^\w]", " ", t)
            parts = clean.split()
            safe_tokens.extend(parts)
        if not safe_tokens:
            return '""'
        return " OR ".join(safe_tokens)

    @staticmethod
    def _escape_like(query: str) -> str:
        return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _fallback_like_search(
        self,
        conn: sqlite3.Connection,
        query: str,
        type: str | None,
        valid_only: bool,
        limit: int,
        offset: int,
    ) -> list[sqlite3.Row]:
        clauses = []
        vals: list = []
        clauses.append(
            "(m.\"content\" LIKE ? ESCAPE '\\' OR m.\"summary\" LIKE ? ESCAPE '\\' OR m.\"hints\" LIKE ? ESCAPE '\\')"
        )
        escaped = self._escape_like(query)
        vals.extend([f"%{escaped}%", f"%{escaped}%", f"%{escaped}%"])
        if valid_only:
            clauses.append('m."valid_until" IS NULL')
        if type:
            clauses.append('m."type" = ?')
            vals.append(type)
        where = " AND ".join(clauses)
        rows = conn.execute(
            f"SELECT m.*, 0.0 AS _fts_rank "
            f'FROM "memories" AS m WHERE {where} '
            f'ORDER BY m."updated_at" DESC LIMIT ? OFFSET ?',
            vals + [limit, offset],
        ).fetchall()
        return rows

    def _touch(self, mem_id: str, *, now: str | None = None):
        now = now or datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        conn.execute(
            'UPDATE "memories" SET "accessed_at" = ?, "access_count" = "access_count" + 1 WHERE "id" = ?',
            (now, mem_id),
        )
        conn.commit()

    def touch(self, mem_id: str) -> bool:
        conn = self._connect()
        row = conn.execute(
            'SELECT 1 FROM "memories" WHERE "id" = ?', (mem_id,)
        ).fetchone()
        if not row:
            return False
        self._touch(mem_id)
        return True

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = {}
        hints_raw = d.get("hints", "[]")
        if isinstance(hints_raw, str):
            try:
                d["hints"] = json.loads(hints_raw)
            except (json.JSONDecodeError, TypeError):
                d["hints"] = []
        elif not isinstance(hints_raw, list):
            d["hints"] = []
        return d

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
