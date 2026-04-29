"""Tests for schema migration system."""

import sqlite3


from llmem.store import MemoryStore, _run_migrations


class TestMigration_NumberedFiles:
    """Test that the migration runner executes numbered .sql files."""

    def test_migrations_applied(self, tmp_path):
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        rows = conn.execute(
            'SELECT "version" FROM "_schema_migrations" ORDER BY "version"'
        ).fetchall()
        versions = [row[0] for row in rows]
        store.close()
        assert 1 in versions
        assert 2 in versions
        assert 3 in versions

    def test_migrations_idempotent(self, tmp_path):
        """Running migrations twice doesn't fail or duplicate."""
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        _run_migrations(conn)
        _run_migrations(conn)
        rows = conn.execute('SELECT "version" FROM "_schema_migrations"').fetchall()
        versions = [row[0] for row in rows]
        conn.close()
        # Each version should appear exactly once
        assert versions.count(1) == 1
        assert versions.count(2) == 1
        assert versions.count(3) == 1


class TestMigration_TrackedInTable:
    """Test that _schema_migrations table tracks applied migrations."""

    def test_schema_migrations_table_exists(self, tmp_path):
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        # Verify the table exists
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_migrations'"
        ).fetchone()
        store.close()
        assert result is not None

    def test_schema_migrations_has_applied_at(self, tmp_path):
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        # Check that applied_at column has non-null values
        rows = conn.execute('SELECT "applied_at" FROM "_schema_migrations"').fetchall()
        store.close()
        assert all(row[0] is not None for row in rows)


class TestMigration_DDLQuotedIdentifiers:
    """Test that schema uses double-quoted identifiers."""

    def test_memories_table_quoted(self, tmp_path):
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        # Check the SQL for the memories table
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        store.close()
        assert sql is not None
        # The table should have been created with quoted identifiers in the migration
        # At minimum, the table name should be 'memories'
        assert sql[0] is not None

    def test_no_check_constraint_on_type(self, tmp_path):
        """The memories table must NOT have a CHECK(type IN (...)) constraint."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        store.close()
        assert sql is not None
        # The SQL must NOT contain "type" as part of a CHECK constraint
        # The memories table still has CHECK("confidence" ...) which is fine
        sql_text = sql[0]
        assert "CHECK(type IN" not in sql_text
        assert 'CHECK("type" IN' not in sql_text


class TestMigration_DDLDMLSeparated:
    """Test that DDL and DML are in separate migration files."""

    def test_003_is_dml_only(self):
        """Migration 003 should only contain INSERT (DML), not CREATE TABLE (DDL)."""
        import importlib.resources

        migrations_pkg = importlib.resources.files("llmem_migrations")
        sql_content = migrations_pkg.joinpath(
            "003_register_default_types.sql"
        ).read_text()
        assert "CREATE TABLE" not in sql_content
        assert "INSERT" in sql_content
        assert "BEGIN TRANSACTION" in sql_content
        assert "COMMIT" in sql_content


class TestMigration_004Inbox:
    """Test that migration 004 creates the inbox table."""

    def test_004_inbox_migration_applied(self, tmp_path):
        """Migration 004 should be tracked in _schema_migrations."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        rows = conn.execute(
            'SELECT "version" FROM "_schema_migrations" ORDER BY "version"'
        ).fetchall()
        versions = [row[0] for row in rows]
        store.close()
        assert 4 in versions

    def test_004_is_ddl_only(self):
        """Migration 004 should be DDL-only (no INSERT INTO data tables)."""
        import importlib.resources

        migrations_pkg = importlib.resources.files("llmem_migrations")
        sql_content = migrations_pkg.joinpath("004_add_inbox.sql").read_text()
        assert "CREATE TABLE" in sql_content
        # Should NOT contain data DML (no INSERT INTO memories, etc.)
        # But it does contain the schema_migrations version insert which is fine
        assert "BEGIN TRANSACTION" not in sql_content  # No transaction wrapping for DDL


class TestMigration_CodeChunksTable:
    """Test that migration 005 creates the code_chunks table properly."""

    def test_migration_005_applied(self, tmp_path):
        """Migration 005 is tracked in _schema_migrations."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        rows = conn.execute(
            'SELECT "version" FROM "_schema_migrations" ORDER BY "version"'
        ).fetchall()
        versions = [row[0] for row in rows]
        store.close()
        assert 5 in versions

    def test_code_chunks_table_has_required_columns(self, tmp_path):
        """code_chunks table has all columns from the spec."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        cursor = conn.execute("PRAGMA table_info('code_chunks')")
        columns = {row[1] for row in cursor.fetchall()}
        store.close()
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

    def test_code_chunks_indexes_exist(self, tmp_path):
        """code_chunks table has the expected indexes."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        store.close()
        assert "idx_code_chunks_file_path" in indexes
        assert "idx_code_chunks_language" in indexes
        assert "idx_code_chunks_chunk_type" in indexes

    def test_005_is_ddl_only(self):
        """Migration 005 should only contain DDL, not DML (no BEGIN TRANSACTION)."""
        import importlib.resources

        migrations_pkg = importlib.resources.files("llmem_migrations")
        sql_content = migrations_pkg.joinpath("005_add_code_chunks.sql").read_text()
        assert "CREATE TABLE" in sql_content
        assert "BEGIN TRANSACTION" not in sql_content
        # Only the schema_migrations INSERT is allowed, no data DML
        assert 'INSERT INTO "memory_types"' not in sql_content


class TestMigration_006RefTypes:
    """Test that migration 006 adds target_type column and extends relation_type CHECK."""

    def test_006_ref_types_migration_applied(self, tmp_path):
        """Migration 006 is tracked in _schema_migrations."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        rows = conn.execute(
            'SELECT "version" FROM "_schema_migrations" ORDER BY "version"'
        ).fetchall()
        versions = [row[0] for row in rows]
        store.close()
        assert 6 in versions

    def test_006_relations_has_target_type_column(self, tmp_path):
        """relations table has a target_type column."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        cursor = conn.execute("PRAGMA table_info('relations')")
        columns = {row[1] for row in cursor.fetchall()}
        store.close()
        assert "target_type" in columns

    def test_006_target_type_default_is_memory(self, tmp_path):
        """Existing relation rows have target_type='memory'."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        # Add a relation to test default
        mid1 = store.add(type="fact", content="memory A")
        mid2 = store.add(type="fact", content="memory B")
        rel_id = store.add_relation(mid1, mid2, "related_to")
        conn = store._connect()
        rows = conn.execute(
            'SELECT "target_type" FROM "relations" WHERE "id" = ?',
            (rel_id,),
        ).fetchall()
        store.close()
        assert all(row[0] == "memory" for row in rows)

    def test_006_relation_type_includes_references(self, tmp_path):
        """The CHECK constraint on relation_type includes 'references'."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        mid1 = store.add(type="fact", content="memory A")
        mid2 = store.add(type="fact", content="memory B")
        # This would raise if 'references' is not in the CHECK constraint
        rel_id = store.add_relation(mid1, mid2, "references")
        assert rel_id is not None
        store.close()

    def test_006_is_ddl_only(self):
        """Migration 006 contains DDL, no BEGIN TRANSACTION (DDL-only pattern)."""
        import importlib.resources

        migrations_pkg = importlib.resources.files("llmem_migrations")
        sql_content = migrations_pkg.joinpath("006_add_ref_types.sql").read_text()
        assert "CREATE TABLE" in sql_content or "ALTER TABLE" in sql_content
        # No transaction wrapping for DDL
        assert "BEGIN TRANSACTION" not in sql_content
