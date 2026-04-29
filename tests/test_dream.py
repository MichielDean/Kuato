"""Tests for Dreamer decoupling — no skill_patch in core."""

import inspect
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
