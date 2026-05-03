"""Tests for Dreamer decoupling — no skill_patch in core."""

import inspect
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llmem.dream import Dreamer
from llmem.store import MemoryStore
from llmem.registry import (
    register_dream_hook,
    _reset_registries,
)


@pytest.fixture(autouse=True)
def _clean_hooks():
    """Reset dream hook registry before and after each test."""
    _reset_registries()
    yield
    _reset_registries()


class TestDream_NoSkillPatchInCore:
    """Core Dreamer must not reference skill_patch_threshold."""

    def test_dreamer_ctor_has_no_skill_patch_threshold_param(self):
        """Dreamer.__init__ must not accept skill_patch_threshold."""
        sig = inspect.signature(Dreamer.__init__)
        assert "skill_patch_threshold" not in sig.parameters

    def test_dream_run_does_not_reference_skill_patch(self):
        """Dreamer.run() source must not mention skill_patch."""
        source = inspect.getsource(Dreamer.run)
        assert "skill_patch" not in source

    def test_dream_module_has_no_skill_patch_constant(self):
        """llmem.dream must not define DEFAULT_SKILL_PATCH_THRESHOLD."""
        import llmem.dream as dream_mod

        assert not hasattr(dream_mod, "DEFAULT_SKILL_PATCH_THRESHOLD")

    def test_dreamer_has_no_skill_patch_field(self):
        """Dreamer instances must not have _skill_patch_threshold field."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        dreamer = Dreamer(store=store)
        assert not hasattr(dreamer, "_skill_patch_threshold")
        store.close()


class TestDream_HookIntegration:
    """Dream hooks run after each phase."""

    def test_dream_hook_called_after_phase(self, tmp_path):
        """A registered dream hook is called during Dreamer.run()."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        dreamer = Dreamer(store=store)
        calls = []

        def hook_fn(d, r, a):
            calls.append(("light", a))

        register_dream_hook("light", hook_fn)
        dreamer.run(apply=False, phase="light")
        assert len(calls) == 1
        assert calls[0] == ("light", False)
        store.close()

    def test_dream_hook_error_does_not_crash(self, tmp_path):
        """A faulty dream hook is logged but does not crash Dreamer.run()."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        dreamer = Dreamer(store=store)

        def bad_hook(d, r, a):
            raise RuntimeError("hook crashed")

        register_dream_hook("deep", bad_hook)
        # Must not raise
        result = dreamer.run(apply=False, phase="deep")
        assert result.deep is not None
        store.close()

    def test_no_hooks_still_works(self, tmp_path):
        """Dreamer.run() works fine with no hooks registered."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        dreamer = Dreamer(store=store)
        result = dreamer.run(apply=False)
        assert result is not None
        store.close()


class TestDream_DeepPhasePromotesInbox:
    """Dream deep phase with apply=True promotes inbox items."""

    def test_dream_deep_phase_promotes_inbox_items(self):
        """When apply=True and inbox has items, deep phase promotes them."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        # Add items to inbox
        store.add_to_inbox(content="important observation", attention_score=0.8)
        store.add_to_inbox(content="tentative note", attention_score=0.3)

        dreamer = Dreamer(store=store)
        result = dreamer.run(apply=True, phase="deep")

        # Items with attention_score >= min_score (default 0.5) should be promoted
        assert result.deep is not None
        assert result.deep.promoted_count >= 1
        # Inbox should be empty after consolidation (all items either promoted or evicted)
        assert store.inbox_count() == 0
        # The high-score item should now be in long-term memory
        memories = store.search(query="important observation")
        assert len(memories) >= 1
        store.close()

    def test_dream_deep_phase_dry_run_does_not_promote(self):
        """When apply=False, deep phase does not actually promote inbox items."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        store.add_to_inbox(content="tentative note", attention_score=0.7)

        dreamer = Dreamer(store=store)
        dreamer.run(apply=False, phase="deep")

        # dry_run=True should not clear the inbox
        assert store.inbox_count() == 1
        store.close()


class TestDream_DeepPhaseAutoLink:
    """Test auto-linking in dream deep phase with consolidate_duplicates."""

    def test_dream_deep_auto_links_similar_memories(self):
        """With auto_link_threshold and memories with high cosine similarity,
        running deep phase with apply=True creates a related_to relation edge."""
        import struct

        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        # Add two memories with near-identical embeddings
        emb = struct.pack("3f", 0.9, 0.1, 0.1)
        mid1 = store.add(type="fact", content="python programming", embedding=emb)
        store.add(type="fact", content="python programming lang", embedding=emb)

        dreamer = Dreamer(store=store, auto_link_threshold=0.85)
        result = dreamer.run(apply=True, phase="deep")

        assert result.deep is not None
        # With identical embeddings, consolidate_duplicates should find them
        # and auto-link should create a relation
        assert result.deep.auto_linked_count >= 1

        # Verify the relation exists
        relations = store.get_relations(mid1)
        related_to = [r for r in relations if r["relation_type"] == "related_to"]
        assert len(related_to) >= 1
        store.close()

    def test_dream_deep_auto_link_respects_threshold(self):
        """Memories with cosine similarity < threshold do NOT get auto-linked."""
        import struct

        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        # Very different embeddings
        emb1 = struct.pack("3f", 1.0, 0.0, 0.0)
        emb2 = struct.pack("3f", 0.0, 1.0, 0.0)
        store.add(type="fact", content="completely different A", embedding=emb1)
        store.add(type="fact", content="completely different B", embedding=emb2)

        # High threshold means they shouldn't be linked
        dreamer = Dreamer(store=store, auto_link_threshold=0.99)
        result = dreamer.run(apply=True, phase="deep")

        assert result.deep is not None
        assert result.deep.auto_linked_count == 0
        store.close()

    def test_dream_deep_auto_link_dry_run_does_not_create_relations(self):
        """With apply=False, no auto-link relations are created."""
        import struct

        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        emb = struct.pack("3f", 0.9, 0.1, 0.1)
        store.add(type="fact", content="python programming", embedding=emb)
        store.add(type="fact", content="python programming lang", embedding=emb)

        dreamer = Dreamer(store=store, auto_link_threshold=0.85)
        result = dreamer.run(apply=False, phase="deep")

        assert result.deep is not None
        # Dry run should not create any relations
        assert result.deep.auto_linked_count == 0
        store.close()

    def test_dream_deep_auto_link_default_threshold(self):
        """Dreamer auto_link_threshold defaults to 0.85."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        dreamer = Dreamer(store=store)
        assert dreamer._auto_link_threshold == 0.85
        store.close()


class TestDream_DecayUsesCreatedAt:
    """Decay uses created_at, not updated_at. Memories accessed within
    the decay interval (accessed_at is recent) are exempt from decay."""

    def test_decay_uses_created_at_not_updated_at(self, tmp_path):
        """A memory with old created_at but recent updated_at should still decay
        if it hasn't been accessed recently."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        old_ts = "2025-01-01T00:00:00+00:00"
        mem_id = store.add(
            type="fact",
            content="stale memory",
            confidence=0.7,
        )
        conn = store._connect()
        conn.execute(
            'UPDATE "memories" SET "created_at" = ?, "updated_at" = ? WHERE "id" = ?',
            (old_ts, "2026-05-01T00:00:00+00:00", mem_id),
        )
        conn.commit()

        dreamer = Dreamer(
            store=store,
            decay_interval_days=1,
            decay_rate=0.2,
            decay_floor=0.1,
            confidence_floor=0.3,
        )
        result = dreamer.run(apply=True, phase="deep")

        assert result.deep is not None
        assert result.deep.decayed_count >= 1

        mem = store.get(mem_id)
        assert mem is not None
        assert mem["confidence"] < 0.7
        store.close()

    def test_recent_memory_does_not_decay(self, tmp_path):
        """A memory with recent created_at should NOT decay."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        store.add(type="fact", content="fresh memory", confidence=0.7)

        dreamer = Dreamer(
            store=store,
            decay_interval_days=30,
            decay_rate=0.2,
            decay_floor=0.1,
            confidence_floor=0.3,
        )
        result = dreamer.run(apply=True, phase="deep")

        assert result.deep is not None
        assert result.deep.decayed_count == 0
        store.close()

    def test_recently_accessed_memory_immune_to_decay(self, tmp_path):
        """An old memory accessed within the decay interval should NOT decay."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        old_ts = "2025-01-01T00:00:00+00:00"
        recent_ts = datetime.now(timezone.utc).isoformat()
        mem_id = store.add(
            type="fact",
            content="old but recently accessed",
            confidence=0.7,
        )
        conn = store._connect()
        conn.execute(
            'UPDATE "memories" SET "created_at" = ?, "accessed_at" = ? WHERE "id" = ?',
            (old_ts, recent_ts, mem_id),
        )
        conn.commit()

        dreamer = Dreamer(
            store=store,
            decay_interval_days=30,
            decay_rate=0.2,
            decay_floor=0.1,
            confidence_floor=0.3,
        )
        result = dreamer.run(apply=True, phase="deep")

        assert result.deep.decayed_count == 0

        mem = store.get(mem_id)
        assert mem["confidence"] >= 0.7
        store.close()

    def test_old_unaccessed_memory_decays(self, tmp_path):
        """An old memory not accessed within the decay interval should decay."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        old_ts = "2025-01-01T00:00:00+00:00"
        stale_access = "2024-06-01T00:00:00+00:00"
        mem_id = store.add(
            type="fact",
            content="old and stale",
            confidence=0.8,
        )
        conn = store._connect()
        conn.execute(
            'UPDATE "memories" SET "created_at" = ?, "accessed_at" = ? WHERE "id" = ?',
            (old_ts, stale_access, mem_id),
        )
        conn.commit()

        dreamer = Dreamer(
            store=store,
            decay_interval_days=30,
            decay_rate=0.2,
            decay_floor=0.1,
            confidence_floor=0.3,
        )
        result = dreamer.run(apply=True, phase="deep")

        assert result.deep.decayed_count >= 1

        mem = store.get(mem_id)
        assert mem["confidence"] < 0.8
        store.close()

    def test_old_memory_without_accessed_at_decays(self, tmp_path):
        """An old memory never accessed (NULL accessed_at) should decay."""
        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        old_ts = "2025-01-01T00:00:00+00:00"
        mem_id = store.add(
            type="fact",
            content="old never accessed",
            confidence=0.8,
        )
        conn = store._connect()
        conn.execute(
            'UPDATE "memories" SET "created_at" = ? WHERE "id" = ?',
            (old_ts, mem_id),
        )
        conn.commit()

        dreamer = Dreamer(
            store=store,
            decay_interval_days=30,
            decay_rate=0.2,
            decay_floor=0.1,
            confidence_floor=0.3,
        )
        result = dreamer.run(apply=True, phase="deep")

        assert result.deep.decayed_count >= 1
        store.close()


class TestDream_DeepPhaseMerge:
    """Test merge of near-duplicates in dream deep phase."""

    def test_merge_near_duplicates_in_deep_phase(self):
        """Light phase finds duplicates; deep phase merges them (invalidate loser, add supersedes relation)."""
        import struct

        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        emb = struct.pack("3f", 0.9, 0.1, 0.1)
        # Higher-confidence memory
        mid_high = store.add(
            type="fact", content="python is great", confidence=0.9, embedding=emb
        )
        # Lower-confidence duplicate
        mid_low = store.add(
            type="fact", content="python is awesome", confidence=0.7, embedding=emb
        )

        dreamer = Dreamer(
            store=store,
            similarity_threshold=0.85,
            decay_interval_days=999,  # Skip decay for this test
            boost_threshold=999,  # Skip boost for this test
        )
        result = dreamer.run(apply=True)

        assert result.deep is not None
        assert result.deep.merged_count >= 1

        # Verify the loser was invalidated
        all_memories = store.search(valid_only=False, limit=10, query="python")
        valid = [m for m in all_memories if m.get("valid_until") is None]
        # Only the keeper should remain valid
        assert len(valid) <= 1

        # Verify supersedes relation was created
        relations = store.get_relations(mid_high)
        supersedes = [r for r in relations if r["relation_type"] == "supersedes"]
        assert len(supersedes) >= 1
        store.close()

    def test_merge_dry_run_does_not_invalidate(self):
        """With apply=False, merge should not invalidate memories."""
        import struct

        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        emb = struct.pack("3f", 0.9, 0.1, 0.1)
        store.add(type="fact", content="python is great", confidence=0.9, embedding=emb)
        store.add(type="fact", content="python is awesome", confidence=0.7, embedding=emb)

        dreamer = Dreamer(
            store=store,
            similarity_threshold=0.85,
            decay_interval_days=999,
            boost_threshold=999,
        )
        result = dreamer.run(apply=False)

        assert result.deep is not None
        assert result.deep.merged_count >= 1

        # But no memories should actually be invalidated
        all_valid = store.search(valid_only=True, limit=10, query="python")
        assert len(all_valid) >= 2
        store.close()

    def test_merge_prefers_higher_confidence_as_keeper(self):
        """When merging, the higher-confidence memory should be kept."""
        import struct

        store = MemoryStore(db_path=Path(":memory:"), disable_vec=True)
        emb = struct.pack("3f", 0.9, 0.1, 0.1)
        mid_low = store.add(
            type="fact", content="python is great", confidence=0.6, embedding=emb
        )
        mid_high = store.add(
            type="fact", content="python is awesome", confidence=0.95, embedding=emb
        )

        dreamer = Dreamer(
            store=store,
            similarity_threshold=0.85,
            decay_interval_days=999,
            boost_threshold=999,
        )
        result = dreamer.run(apply=True)

        assert result.deep.merged_count >= 1

        # The higher-confidence memory should still be valid
        high_mem = store.get(mid_high)
        assert high_mem is not None
        assert high_mem.get("valid_until") is None

        # The lower-confidence memory should be invalidated
        low_mem = store.get(mid_low)
        assert low_mem is not None
        assert low_mem.get("valid_until") is not None
        store.close()
