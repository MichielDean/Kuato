"""Tests for llmem.store module — MemoryStore, register_memory_type, type validation."""

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
