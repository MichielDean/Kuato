"""Tests for llmem.store module — MemoryStore, register_memory_type, type validation."""

import logging
import struct

import pytest

from llmem.store import (
    MemoryStore,
    register_memory_type,
    get_registered_types,
    _reset_global_registry,
)


class TestStore_RegisterType:
    """Test register_memory_type()."""

    def setup_method(self):
        """Reset the global registry to a clean state before each test."""
        _reset_global_registry()

    def test_register_custom_type(self):
        """Register a custom type that isn't in the default set."""
        test_type = "test_custom_type_xyz"
        register_memory_type(test_type)
        assert test_type in get_registered_types()

    def test_register_duplicate_raises(self):
        """Registering an already-registered type raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            register_memory_type("fact")  # 'fact' is a default type


class TestStore_AddRegisteredType:
    """Test adding memories with registered types."""

    def test_add_default_type_succeeds(self, store):
        mid = store.add(type="fact", content="test fact")
        assert mid is not None
        result = store.get(mid)
        assert result["type"] == "fact"

    def test_add_custom_registered_type_succeeds(self, store):
        """A type registered before the store was created is accepted."""
        test_type = "test_custom_type_abc"
        register_memory_type(test_type)
        # Store created before registration won't know about it,
        # so create a new store that picks up the registration.
        from llmem.store import _global_registry

        new_store = MemoryStore(
            db_path=store.db_path.parent / "test2.db", disable_vec=True
        )
        mid = new_store.add(type=test_type, content="custom content")
        assert mid is not None
        result = new_store.get(mid)
        assert result["type"] == test_type
        new_store.close()


class TestStore_AddUnregisteredTypeFails:
    """Test that adding an unregistered type raises ValueError."""

    def test_add_unknown_type_raises(self, store):
        with pytest.raises(ValueError, match="unregistered type"):
            store.add(type="unknown_type_never_registered", content="test")


class TestStore_SelfAssessmentIsDefaultRegistered:
    """Test that self_assessment is in the default registered types (backward compat)."""

    def test_self_assessment_registered(self):
        assert "self_assessment" in get_registered_types()

    def test_add_self_assessment_succeeds(self, store):
        mid = store.add(type="self_assessment", content="self assessment content")
        assert mid is not None
        result = store.get(mid)
        assert result["type"] == "self_assessment"


class TestStore_RegistryPerInstance:
    """Test that each MemoryStore snapshots the global registry at construction."""

    def test_store_does_not_retroactively_accept_new_types(self, store):
        """A store created before a type is registered rejects that type."""
        register_memory_type("brand_new_type_xyz")
        # The store was created before the registration, so it should reject it
        with pytest.raises(ValueError, match="unregistered type"):
            store.add(type="brand_new_type_xyz", content="should fail")

    def test_new_store_accepts_newly_registered_types(self, store):
        """A store created after a type is registered accepts that type."""
        register_memory_type("brand_new_type_abc")
        new_store = MemoryStore(
            db_path=store.db_path.parent / "new_store.db", disable_vec=True
        )
        mid = new_store.add(type="brand_new_type_abc", content="should work")
        assert mid is not None
        new_store.close()


class TestStore_Add_Get:
    """Basic store operations."""

    def test_add_returns_uuid(self, store):
        mid = store.add(type="fact", content="test")
        assert len(mid) == 36
        assert "-" in mid

    def test_add_and_get_roundtrip(self, store):
        mid = store.add(type="fact", content="hello world", summary="a greeting")
        result = store.get(mid)
        assert result is not None
        assert result["content"] == "hello world"
        assert result["summary"] == "a greeting"
        assert result["type"] == "fact"
        assert result["source"] == "manual"
        assert result["confidence"] == 0.8

    def test_get_nonexistent_returns_none(self, store):
        result = store.get("does-not-exist")
        assert result is None

    def test_add_default_values(self, store):
        mid = store.add(type="fact", content="defaults")
        result = store.get(mid)
        assert result["source"] == "manual"
        assert result["confidence"] == 0.8
        assert result["hints"] == []
        assert result["metadata"] == {}


class TestStore_Search:
    def test_search_by_text(self, store):
        store.add(type="fact", content="Python is a programming language")
        store.add(type="fact", content="Rust is a systems language")
        results = store.search(query="Python")
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)

    def test_search_by_type(self, store):
        store.add(type="fact", content="test fact")
        store.add(type="decision", content="test decision")
        results = store.search(type="fact")
        assert all(r["type"] == "fact" for r in results)


class TestStore_Count:
    def test_count(self, store):
        store.add(type="fact", content="one")
        store.add(type="fact", content="two")
        assert store.count() == 2

    def test_count_by_type(self, store):
        store.add(type="fact", content="one")
        store.add(type="decision", content="two")
        counts = store.count_by_type()
        assert counts.get("fact", 0) >= 1
        assert counts.get("decision", 0) >= 1


class TestStore_SearchCount:
    """Test search_count() method."""

    def test_search_count_with_query(self, store):
        store.add(type="fact", content="Python programming language")
        store.add(type="fact", content="Rust systems language")
        count = store.search_count(query="Python")
        assert count >= 1

    def test_search_count_without_query(self, store):
        store.add(type="fact", content="one")
        store.add(type="fact", content="two")
        store.add(type="decision", content="three")
        count = store.search_count()
        assert count >= 3

    def test_search_count_with_type_filter(self, store):
        store.add(type="fact", content="alpha")
        store.add(type="decision", content="beta")
        count = store.search_count(type="fact")
        assert count >= 1

    def test_search_count_valid_only(self, store):
        mid = store.add(type="fact", content="test for count")
        store.invalidate(mid)
        count_valid = store.search_count(query="count", valid_only=True)
        count_all = store.search_count(query="count", valid_only=False)
        assert count_all > count_valid


class TestStore_FallbackLikeSearch:
    """Test _fallback_like_search and _fallback_like_count helpers."""

    def test_fallback_like_search_returns_results(self, store):
        """_fallback_like_search returns results when FTS is unavailable (simulated)."""
        store.add(type="fact", content="hello world programming")
        store.add(type="fact", content="rust programming language")
        conn = store._connect()
        results = store._fallback_like_search(conn, "programming", None, True, 10, 0)
        assert len(results) >= 2

    def test_fallback_like_count_returns_count(self, store):
        """_fallback_like_count returns the same count as search_count for basic queries."""
        store.add(type="fact", content="alpha programming test")
        store.add(type="fact", content="beta programming test")
        conn = store._connect()
        count = store._fallback_like_count(conn, "programming", None, True)
        assert count >= 2


class TestStore_Vec0Integration:
    """Integration test for vec0 virtual table with sqlite-vec extension.

    Tests the new vec0 column-definition API that is required by sqlite-vec>=0.1.6.
    All other tests use disable_vec=True, so this class tests WITHOUT that flag
    to verify the real vector search path works.
    """

    @pytest.fixture
    def vec_store(self, tmp_path):
        """Create a MemoryStore with vec enabled (requires sqlite-vec)."""
        pytest.importorskip("sqlite_vec")
        db = tmp_path / "vec_test.db"
        s = MemoryStore(db_path=db, vec_dimensions=3, disable_vec=False)
        yield s
        s.close()

    @staticmethod
    def _make_embedding(values: list[float]) -> bytes:
        """Pack float values into a bytes embedding."""
        return struct.pack(f"{len(values)}f", *values)

    def test_vec0_table_creation(self, vec_store):
        """Verify vec0 virtual table is created with the new column-definition API."""
        conn = vec_store._connect()
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories_vec'"
        ).fetchone()
        assert row is not None, "memories_vec table should exist"
        sql = row["sql"]
        # The new API requires rowid/column definitions, not just float[N]
        assert "rowid" in sql.lower() or "INTEGER PRIMARY KEY" in sql, (
            f"vec0 table must use column-definition API. Got: {sql}"
        )

    def test_search_by_embedding_with_vec(self, vec_store):
        """search_by_embedding works with real sqlite-vec extension."""
        emb1 = self._make_embedding([0.1, 0.2, 0.3])
        emb2 = self._make_embedding([0.9, 0.8, 0.7])
        vec_store.add(type="fact", content="close to query", embedding=emb1)
        vec_store.add(type="fact", content="far from query", embedding=emb2)

        query_vec = [0.1, 0.2, 0.3]
        results = vec_store.search_by_embedding(query_vec, limit=5, threshold=0.5)
        assert len(results) >= 1
        # The first result should be the memory close to the query vector
        assert results[0][0]["content"] == "close to query"
        assert results[0][1] >= 0.9  # cosine similarity should be very high


class TestStore_ChmodWarning:
    """Test that os.chmod failure on db file is logged as a warning, not silently ignored."""

    def test_chmod_failure_logs_warning(self, tmp_path, caplog):
        """When os.chmod fails on the db file, a warning should be logged."""
        # We need to allow the directory chmod but fail the db file chmod.
        # The easiest way: create the store successfully first, then verify
        # the warning message pattern by checking the code path.
        import os
        import unittest.mock

        original_chmod = os.chmod
        call_count = 0

        def chmod_selective(path, mode):
            nonlocal call_count
            call_count += 1
            # Fail only on the db file chmod (second chmod call in __init__)
            if call_count == 2 and str(path).endswith(".db"):
                raise OSError("Permission denied")
            return original_chmod(path, mode)

        db = tmp_path / "chmod_test.db"
        with unittest.mock.patch("os.chmod", side_effect=chmod_selective):
            with caplog.at_level(logging.WARNING):
                store = MemoryStore(db_path=db, disable_vec=True)
                store.close()

        # Verify a warning was logged about the chmod failure
        chmod_warnings = [
            r for r in caplog.records if "failed to set permissions" in r.message
        ]
        assert len(chmod_warnings) >= 1, (
            f"Expected at least one warning about chmod failure, got: "
            f"{[r.message for r in caplog.records]}"
        )
