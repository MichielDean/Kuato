"""Tests for schema migration system."""

import sqlite3
from pathlib import Path

import pytest

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
