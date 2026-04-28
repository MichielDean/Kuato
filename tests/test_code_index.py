"""Tests for llmem.code_index module — CodeIndex CRUD, search, and vec0 integration."""

import sqlite3
import struct
from pathlib import Path

import pytest

from llmem.code_index import CodeIndex
from llmem.chunking import CodeChunk, ParagraphChunking


class TestCodeIndex_CreateTable:
    """Verify the code_chunks table is created with correct columns and indexes."""

    def test_code_chunks_table_exists(self, tmp_path):
        """The code_chunks table is created after CodeIndex initialization."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        conn = idx._connect()
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='code_chunks'"
        ).fetchone()
        idx.close()
        assert result is not None

    def test_code_chunks_columns(self, tmp_path):
        """code_chunks table has all required columns."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        conn = idx._connect()
        cursor = conn.execute("PRAGMA table_info('code_chunks')")
        columns = {row[1] for row in cursor.fetchall()}
        idx.close()
        expected = {
            "id",
            "file_path",
            "start_line",
            "end_line",
            "content",
            "embedding",
            "language",
            "chunk_type",
            "created_at",
        }
        assert expected.issubset(columns)

    def test_code_chunks_indexes(self, tmp_path):
        """code_chunks table has expected indexes."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        conn = idx._connect()
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        idx.close()
        assert "idx_code_chunks_file_path" in indexes
        assert "idx_code_chunks_language" in indexes


class TestCodeIndex_AddChunk:
    """Test adding a single chunk and verifying round-trip retrieval."""

    def test_add_and_retrieve_chunk(self, tmp_path):
        """Adding a chunk and retrieving it round-trips correctly."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        chunk_id = idx.add_chunk(
            file_path="src/main.py",
            start_line=1,
            end_line=10,
            content="def hello():\n    print('hello')",
            language="python",
            chunk_type="paragraph",
        )
        idx.close()
        assert chunk_id == "src/main.py:1:10"

        # Verify retrieval
        idx = CodeIndex(db_path=db, disable_vec=True)
        conn = idx._connect()
        row = conn.execute(
            'SELECT * FROM "code_chunks" WHERE "id" = ?', (chunk_id,)
        ).fetchone()
        idx.close()
        assert row is not None
        assert row["file_path"] == "src/main.py"
        assert row["start_line"] == 1
        assert row["end_line"] == 10
        assert row["content"] == "def hello():\n    print('hello')"
        assert row["language"] == "python"
        assert row["chunk_type"] == "paragraph"

    def test_add_chunk_validates_file_path(self, tmp_path):
        """add_chunk raises ValueError for empty file_path."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        with pytest.raises(ValueError, match="file_path must not be empty"):
            idx.add_chunk(
                file_path="",
                start_line=1,
                end_line=10,
                content="test",
            )
        idx.close()

    def test_add_chunk_validates_content(self, tmp_path):
        """add_chunk raises ValueError for empty content."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        with pytest.raises(ValueError, match="content must not be empty"):
            idx.add_chunk(
                file_path="test.py",
                start_line=1,
                end_line=1,
                content="",
            )
        idx.close()

    def test_add_chunk_validates_start_line(self, tmp_path):
        """add_chunk raises ValueError for start_line < 1."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        with pytest.raises(ValueError, match="start_line must be positive"):
            idx.add_chunk(
                file_path="test.py",
                start_line=0,
                end_line=1,
                content="test",
            )
        idx.close()

    def test_add_chunk_validates_end_line(self, tmp_path):
        """add_chunk raises ValueError for end_line < 1."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        with pytest.raises(ValueError, match="end_line must be positive"):
            idx.add_chunk(
                file_path="test.py",
                start_line=1,
                end_line=0,
                content="test",
            )
        idx.close()

    def test_add_chunk_with_embedding(self, tmp_path):
        """Adding a chunk with embedding stores the embedding bytes."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        embedding = struct.pack("3f", 0.1, 0.2, 0.3)
        chunk_id = idx.add_chunk(
            file_path="test.py",
            start_line=1,
            end_line=5,
            content="test content",
            embedding=embedding,
        )
        conn = idx._connect()
        row = conn.execute(
            'SELECT "embedding" FROM "code_chunks" WHERE "id" = ?', (chunk_id,)
        ).fetchone()
        idx.close()
        assert row["embedding"] is not None
        assert row["embedding"] == embedding


class TestCodeIndex_AddChunksBatch:
    """Test adding multiple chunks for one file."""

    def test_add_chunks_batch(self, tmp_path):
        """add_chunks inserts multiple chunks and returns their IDs."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        chunker = ParagraphChunking()
        content = "para1 line1\npara1 line2\n\npara2 line1\npara2 line2"
        chunks = chunker.chunk("test.py", content)
        chunk_ids = idx.add_chunks(chunks)
        idx.close()
        assert len(chunk_ids) == len(chunks)
        for cid, chunk in zip(chunk_ids, chunks):
            assert cid == chunk.id

    def test_add_chunks_round_trip(self, tmp_path):
        """Batch-inserted chunks can be retrieved via search_content."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        chunks = [
            CodeChunk(
                id="app.py:1:5",
                file_path="app.py",
                start_line=1,
                end_line=5,
                content="def hello():\n    print('hello world')",
                language="python",
                chunk_type="paragraph",
            ),
            CodeChunk(
                id="app.py:7:12",
                file_path="app.py",
                start_line=7,
                end_line=12,
                content="def goodbye():\n    print('goodbye world')",
                language="python",
                chunk_type="paragraph",
            ),
        ]
        chunk_ids = idx.add_chunks(chunks)
        assert len(chunk_ids) == 2

        results = idx.search_content("hello", limit=10)
        assert len(results) >= 1
        assert any("hello" in r["content"] for r in results)
        idx.close()


class TestCodeIndex_SearchByEmbedding:
    """Test semantic search on code_chunks."""

    @staticmethod
    def _make_embedding(values: list[float]) -> bytes:
        """Pack float values into a bytes embedding."""
        return struct.pack(f"{len(values)}f", *values)

    def test_search_by_embedding_brute_force(self, tmp_path):
        """Brute-force embedding search returns relevant results."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)

        # Add two chunks with different embeddings
        idx.add_chunk(
            file_path="a.py",
            start_line=1,
            end_line=5,
            content="close to query",
            embedding=self._make_embedding([0.1, 0.2, 0.3]),
        )
        idx.add_chunk(
            file_path="b.py",
            start_line=1,
            end_line=5,
            content="far from query",
            embedding=self._make_embedding([0.9, 0.8, 0.7]),
        )

        # Search with a query vector close to the first chunk
        results = idx.search_by_embedding(
            query_vec=[0.1, 0.2, 0.3], limit=5, threshold=0.5
        )
        assert len(results) >= 1
        assert results[0][0]["content"] == "close to query"
        assert results[0][1] >= 0.9  # cosine similarity should be very high
        idx.close()

    def test_search_by_embedding_language_filter(self, tmp_path):
        """Language filter narrows search results."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)

        idx.add_chunk(
            file_path="a.py",
            start_line=1,
            end_line=5,
            content="python code",
            embedding=self._make_embedding([0.1, 0.2, 0.3]),
            language="python",
        )
        idx.add_chunk(
            file_path="b.rs",
            start_line=1,
            end_line=5,
            content="rust code",
            embedding=self._make_embedding([0.1, 0.2, 0.3]),
            language="rust",
        )

        results = idx.search_by_embedding(
            query_vec=[0.1, 0.2, 0.3], limit=5, threshold=0.0, language="python"
        )
        assert len(results) >= 1
        assert all(r["language"] == "python" for r, _ in results)
        idx.close()

    def test_vec_dimensions_validation(self, tmp_path):
        """CodeIndex raises ValueError for non-positive vec_dimensions."""
        db = tmp_path / "test.db"
        with pytest.raises(ValueError, match="vec_dimensions must be positive"):
            CodeIndex(db_path=db, vec_dimensions=0)
        with pytest.raises(ValueError, match="vec_dimensions must be positive"):
            CodeIndex(db_path=db, vec_dimensions=-1)


class TestCodeIndex_SearchByText:
    """Test FTS5 search on code_chunks content."""

    def test_search_content_basic(self, tmp_path):
        """FTS5 search returns chunks matching the query."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        idx.add_chunk(
            file_path="app.py",
            start_line=1,
            end_line=5,
            content="def hello_world():\n    print('hello world')",
            language="python",
        )
        idx.add_chunk(
            file_path="util.py",
            start_line=1,
            end_line=3,
            content="def format_date():\n    return datetime.now()",
            language="python",
        )

        results = idx.search_content("hello", limit=10)
        assert len(results) >= 1
        assert any("hello" in r["content"] for r in results)
        idx.close()

    def test_search_content_language_filter(self, tmp_path):
        """Language filter narrows FTS search results."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        idx.add_chunk(
            file_path="app.py",
            start_line=1,
            end_line=5,
            content="Python hello function",
            language="python",
        )
        idx.add_chunk(
            file_path="app.rs",
            start_line=1,
            end_line=5,
            content="Rust hello function",
            language="rust",
        )

        results = idx.search_content("hello", limit=10, language="python")
        assert len(results) >= 1
        assert all(r["language"] == "python" for r in results)
        idx.close()

    def test_search_content_returns_chunk_dicts(self, tmp_path):
        """search_content returns dicts with expected keys."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        idx.add_chunk(
            file_path="test.py",
            start_line=1,
            end_line=3,
            content="print('hello')",
            language="python",
        )

        results = idx.search_content("hello", limit=10)
        assert len(results) >= 1
        r = results[0]
        assert "id" in r
        assert "file_path" in r
        assert "content" in r
        assert "language" in r
        assert "start_line" in r
        assert "end_line" in r
        assert "chunk_type" in r
        idx.close()


class TestCodeIndex_RemoveByFilePath:
    """Test removing all chunks for a given file path."""

    def test_remove_by_path(self, tmp_path):
        """remove_by_path deletes all chunks for a file path."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        idx.add_chunk(
            file_path="app.py",
            start_line=1,
            end_line=5,
            content="app code",
        )
        idx.add_chunk(
            file_path="app.py",
            start_line=10,
            end_line=15,
            content="more app code",
        )
        idx.add_chunk(
            file_path="other.py",
            start_line=1,
            end_line=3,
            content="other code",
        )

        removed = idx.remove_by_path("app.py")
        assert removed == 2

        # Verify only other.py chunks remain
        results = idx.search_content("code", limit=10)
        assert len(results) >= 1
        assert all(r["file_path"] != "app.py" for r in results)
        idx.close()

    def test_remove_by_path_returns_zero_for_nonexistent(self, tmp_path):
        """remove_by_path returns 0 when no matching chunks exist."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        removed = idx.remove_by_path("nonexistent.py")
        assert removed == 0
        idx.close()


class TestCodeIndex_Vec0Integration:
    """Integration test for vec0 virtual table (requires sqlite-vec)."""

    @pytest.fixture
    def vec_index(self, tmp_path):
        """Create a CodeIndex with vec enabled (requires sqlite-vec)."""
        pytest.importorskip("sqlite_vec")
        db = tmp_path / "vec_test.db"
        idx = CodeIndex(db_path=db, vec_dimensions=3, disable_vec=False)
        yield idx
        idx.close()

    @staticmethod
    def _make_embedding(values: list[float]) -> bytes:
        """Pack float values into a bytes embedding."""
        return struct.pack(f"{len(values)}f", *values)

    def test_vec0_table_creation(self, vec_index, tmp_path):
        """Verify code_chunks_vec virtual table is created."""
        conn = vec_index._connect()
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='code_chunks_vec'"
        ).fetchone()
        assert row is not None
        sql = row["sql"]
        assert "rowid" in sql.lower() or "INTEGER PRIMARY KEY" in sql

    def test_search_by_embedding_with_vec(self, vec_index):
        """search_by_embedding works with real sqlite-vec extension."""
        vec_index.add_chunk(
            file_path="a.py",
            start_line=1,
            end_line=5,
            content="close to query",
            embedding=self._make_embedding([0.1, 0.2, 0.3]),
        )
        vec_index.add_chunk(
            file_path="b.py",
            start_line=1,
            end_line=5,
            content="far from query",
            embedding=self._make_embedding([0.9, 0.8, 0.7]),
        )
        results = vec_index.search_by_embedding(
            query_vec=[0.1, 0.2, 0.3], limit=5, threshold=0.5
        )
        assert len(results) >= 1
        assert results[0][0]["content"] == "close to query"
        assert results[0][1] >= 0.9


class TestCodeIndex_Close:
    """Test that close() properly closes the connection."""

    def test_close_sets_conn_to_none(self, tmp_path):
        """After close(), _conn is None."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        idx.close()
        assert idx._conn is None

    def test_close_idempotent(self, tmp_path):
        """Calling close() multiple times does not raise."""
        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        idx.close()
        idx.close()  # Should not raise
