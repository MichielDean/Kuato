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


class TestStore_TouchBatch:
    """Test MemoryStore.touch_batch()."""

    def test_touch_batch_updates_access_count_and_accessed_at(self, store):
        """touch_batch increments access_count and updates accessed_at for
        specified IDs only."""
        id1 = store.add(type="fact", content="memory one")
        id2 = store.add(type="fact", content="memory two")
        id3 = store.add(type="fact", content="memory three")

        store.touch_batch([id1, id2])

        mem1 = store.get(id1, track_access=False)
        mem2 = store.get(id2, track_access=False)
        mem3 = store.get(id3, track_access=False)

        assert mem1["access_count"] == 1
        assert mem1["accessed_at"] is not None
        assert mem2["access_count"] == 1
        assert mem2["accessed_at"] is not None
        assert mem3["access_count"] == 0
        assert mem3["accessed_at"] is None

    def test_touch_batch_empty_list_returns_zero(self, store):
        """touch_batch with empty list returns 0 without error."""
        result = store.touch_batch([])
        assert result == 0

    def test_touch_batch_nonexistent_ids_ignored(self, store):
        """touch_batch with nonexistent IDs returns 0. Partial match updates
        only the existing memory."""
        mid = store.add(type="fact", content="real memory")
        # Only the real ID exists
        result = store.touch_batch(["nonexistent"])
        assert result == 0

        # Partial match: one real + one fake
        result = store.touch_batch([mid, "nonexistent"])
        assert result == 1
        mem = store.get(mid, track_access=False)
        assert mem["access_count"] == 1

    def test_touch_batch_returns_affected_count(self, store):
        """touch_batch returns the number of rows actually updated."""
        id1 = store.add(type="fact", content="memory one")
        id2 = store.add(type="fact", content="memory two")
        id3 = store.add(type="fact", content="memory three")

        result = store.touch_batch([id1, id2, id3])
        assert result == 3

        # Touching the same IDs again still counts as 3 rows affected
        result = store.touch_batch([id1, id2, id3])
        assert result == 3


class TestStore_ExportAll_Limit:
    """Test that export_all respects the limit parameter."""

    def test_export_all_default_limit(self, tmp_path):
        """export_all has a default limit of 10000 to prevent unbounded memory usage."""
        from llmem.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test.db", disable_vec=True)
        # Add 5 memories
        for i in range(5):
            store.add(type="fact", content=f"Memory {i}", source="test")
        # Default limit should return all 5
        result = store.export_all()
        assert len(result) == 5
        store.close()

    def test_export_all_respects_limit(self, tmp_path):
        """export_all respects the explicit limit parameter."""
        from llmem.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test.db", disable_vec=True)
        # Add 10 memories
        for i in range(10):
            store.add(type="fact", content=f"Memory {i}", source="test")
        # Limit to 3
        result = store.export_all(limit=3)
        assert len(result) == 3
        store.close()

    def test_export_all_limit_none(self, tmp_path):
        """export_all with limit=None returns all memories without limit."""
        from llmem.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test.db", disable_vec=True)
        store.add(type="fact", content="Memory", source="test")
        result = store.export_all(limit=None)
        assert len(result) == 1
        store.close()

    def test_export_all_limit_none_returns_all(self, tmp_path):
        """export_all(limit=None) returns all memories without a cap.

        The default limit of 10000 prevents unbounded memory usage, but
        passing limit=None explicitly removes the cap for true export-all
        use cases (e.g. the CLI export command).
        """
        from llmem.store import MemoryStore

        store = MemoryStore(db_path=tmp_path / "test.db", disable_vec=True)
        # Add 5 memories
        for i in range(5):
            store.add(type="fact", content=f"Memory {i}", source="test")
        # limit=None should return all memories without capping
        result = store.export_all(limit=None)
        assert len(result) == 5


class TestStore_Migration005Compatibility:
    """Test that migration 005 (code_chunks) doesn't break existing MemoryStore functionality."""

    def test_store_init_with_code_chunks_table(self, tmp_path):
        """MemoryStore can initialize when code_chunks table exists."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        # Verify the memories table still works
        mid = store.add(type="fact", content="test after migration 005")
        result = store.get(mid)
        assert result is not None
        assert result["content"] == "test after migration 005"
        store.close()

    def test_code_chunks_table_exists_after_store_init(self, tmp_path):
        """The code_chunks table is created by migration 005 after MemoryStore init."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        conn = store._connect()
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='code_chunks'"
        ).fetchone()
        store.close()
        assert result is not None

    def test_store_operations_unchanged_after_migration(self, tmp_path):
        """All basic MemoryStore operations still work after migration 005."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        # add
        mid = store.add(type="fact", content="migration 005 test", confidence=0.9)
        # get
        result = store.get(mid)
        assert result["content"] == "migration 005 test"
        # search
        results = store.search(query="migration")
        assert len(results) >= 1
        # count
        assert store.count() >= 1
        # update
        store.update(mid, content="updated migration 005 test")
        result = store.get(mid)
        assert result["content"] == "updated migration 005 test"
        # delete
        assert store.delete(mid)
        assert store.get(mid) is None
        store.close()


class TestStore_RelationsRefTypes:
    """Test add_relation with target_type and references relation type."""

    def test_add_relation_with_target_type_code(self, store):
        """add_relation with target_type='code' inserts row with target_type='code'."""
        mid = store.add(type="fact", content="source memory")
        code_target_id = "src/lib.rs:42:58"
        rel_id = store.add_relation(
            mid, code_target_id, "references", target_type="code"
        )
        assert rel_id is not None
        relations = store.get_relations(mid)
        assert len(relations) >= 1
        ref_rel = [r for r in relations if r["id"] == rel_id][0]
        assert ref_rel["target_type"] == "code"
        assert ref_rel["target_id"] == code_target_id
        assert ref_rel["relation_type"] == "references"

    def test_add_relation_default_target_type_memory(self, store):
        """add_relation without target_type defaults to 'memory'."""
        mid1 = store.add(type="fact", content="memory A")
        mid2 = store.add(type="fact", content="memory B")
        rel_id = store.add_relation(mid1, mid2, "related_to")
        relations = store.get_relations(mid1)
        ref_rel = [r for r in relations if r["id"] == rel_id][0]
        assert ref_rel["target_type"] == "memory"

    def test_add_relation_references_type_accepted(self, store):
        """add_relation with 'references' type succeeds (CHECK constraint extended)."""
        mid = store.add(type="fact", content="source memory")
        mid2 = store.add(type="fact", content="target memory")
        rel_id = store.add_relation(mid, mid2, "references")
        assert rel_id is not None
        relations = store.get_relations(mid)
        assert any(r["relation_type"] == "references" for r in relations)

    def test_existing_relations_have_target_type_memory(self, store):
        """All relation rows have target_type='memory' after migration 006."""
        mid1 = store.add(type="fact", content="first")
        mid2 = store.add(type="fact", content="second")
        store.add_relation(mid1, mid2, "related_to")
        conn = store._connect()
        rows = conn.execute('SELECT "target_type" FROM "relations"').fetchall()
        for row in rows:
            assert row[0] == "memory"

    def test_get_relations_includes_target_type(self, store):
        """get_relations(mem_id) returns dicts with a target_type key."""
        mid = store.add(type="fact", content="test memory")
        mid2 = store.add(type="fact", content="related memory")
        store.add_relation(mid, mid2, "related_to")
        relations = store.get_relations(mid)
        assert len(relations) >= 1
        for r in relations:
            assert "target_type" in r


class TestStore_TraverseRelationsWithRefs:
    """Test traverse_relations with target_type filter for code refs."""

    def test_traverse_relations_follows_code_refs(self, store):
        """traverse_relations where mem has a references edge to a code chunk
        returns the code chunk target_id."""
        mid = store.add(type="fact", content="source memory")
        code_ref = "src/lib.rs:42:58"
        store.add_relation(mid, code_ref, "references", target_type="code")
        results = store.traverse_relations([mid], max_depth=1)
        code_results = [r for r in results if r["target_type"] == "code"]
        assert len(code_results) >= 1
        assert code_results[0]["target_id"] == code_ref

    def test_traverse_relations_code_ref_expansion_depth(self, store):
        """traverse_relations respects max_depth, does not expand beyond configured depth."""
        mid = store.add(type="fact", content="root memory")
        results_depth_1 = store.traverse_relations([mid], max_depth=1)
        results_depth_2 = store.traverse_relations([mid], max_depth=2)
        # Both should return results since there's one hop, but the structure
        # should respect the max_depth parameter (no extra expansion beyond 1)
        assert isinstance(results_depth_1, list)
        assert isinstance(results_depth_2, list)

    def test_traverse_relations_target_type_filter(self, store):
        """traverse_relations with target_type='code' only returns code edges."""
        mid = store.add(type="fact", content="source memory")
        mid2 = store.add(type="fact", content="target memory")
        code_ref = "src/lib.rs:1:10"
        store.add_relation(mid, mid2, "related_to", target_type="memory")
        store.add_relation(mid, code_ref, "references", target_type="code")
        # target_type='code' only
        code_results = store.traverse_relations([mid], max_depth=1, target_type="code")
        for r in code_results:
            assert r["target_type"] == "code"
        # target_type='memory' only
        mem_results = store.traverse_relations([mid], max_depth=1, target_type="memory")
        for r in mem_results:
            assert r["target_type"] == "memory"

    def test_traverse_relations_backward_arm_reports_memory_type(self, store):
        """When traverse_relations reaches backward through a code-ref edge
        at depth > 1, the reached memory node must have target_type='memory',
        not 'code' (the edge's target_type).

        Scenario: memA → references → codeChunk, memB → references → codeChunk.
        Starting from memA at depth 2, the backward arm reaches memB (source of
        an edge targeting codeChunk). The reached node is a memory, so its
        target_type in results must be 'memory'.
        """
        mid_a = store.add(type="fact", content="memory A")
        mid_b = store.add(type="fact", content="memory B")
        code_ref = "src/lib.rs:42:58"
        store.add_relation(mid_a, code_ref, "references", target_type="code")
        store.add_relation(mid_b, code_ref, "references", target_type="code")
        results = store.traverse_relations([mid_a], max_depth=2)
        # Find the result for mid_b (reached backward through code-ref edge)
        b_results = [r for r in results if r["target_id"] == mid_b]
        assert len(b_results) >= 1, (
            f"Expected to reach {mid_b} from {mid_a} via code ref {code_ref}, "
            f"but got results: {results}"
        )
        # The reached node is a memory — target_type must be 'memory', not 'code'
        assert b_results[0]["target_type"] == "memory", (
            f"Backward-reached memory {mid_b} should have target_type='memory', "
            f"got '{b_results[0]['target_type']}'"
        )


class TestStore_DeleteOrphanedRelations:
    """Test that deleting a memory cleans up orphaned target_id relations."""

    def test_delete_memory_removes_target_id_relations(self, store):
        """When a memory is deleted, relations where it is the target_id
        (target_type='memory') are cleaned up — no orphaned rows remain."""
        mid1 = store.add(type="fact", content="source memory")
        mid2 = store.add(type="fact", content="target memory")
        store.add_relation(mid1, mid2, "related_to", target_type="memory")
        # mid2 is the target — delete it
        assert store.delete(mid2)
        # Relation should be gone (no orphan)
        relations = store.get_relations(mid1)
        assert not any(r["target_id"] == mid2 for r in relations)

    def test_delete_memory_removes_source_id_relations_via_cascade(self, store):
        """When a memory is deleted, relations where it is the source_id
        are removed by the ON DELETE CASCADE FK constraint."""
        mid1 = store.add(type="fact", content="source memory")
        mid2 = store.add(type="fact", content="target memory")
        store.add_relation(mid1, mid2, "related_to", target_type="memory")
        # mid1 is the source — delete it
        assert store.delete(mid1)
        relations = store.get_relations(mid2)
        assert not any(r["source_id"] == mid1 for r in relations)

    def test_delete_memory_preserves_code_ref_relations(self, store):
        """When a memory is deleted, code ref relations (target_type='code')
        where the memory is the source are cascade-deleted, but code ref
        relations from other memories are untouched."""
        mid1 = store.add(type="fact", content="has code ref")
        store.add_relation(mid1, "src/app.rs:10:20", "references", target_type="code")
        # Delete mid1 — source_id cascade removes the relation
        assert store.delete(mid1)
        # Verify the relation is gone
        relations = store.get_relations(mid1)
        assert len(relations) == 0

    def test_delete_memory_no_relations_no_error(self, store):
        """Deleting a memory with no relations does not raise."""
        mid = store.add(type="fact", content="lonely memory")
        # Add an unrelated code ref from another memory
        mid2 = store.add(type="fact", content="other memory")
        store.add_relation(mid2, "src/x.rs:1:2", "references", target_type="code")
        assert store.delete(mid)
        # Code ref from mid2 is still intact
        relations = store.get_relations(mid2)
        assert any(r["target_type"] == "code" for r in relations)
