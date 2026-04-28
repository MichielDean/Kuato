"""Tests for the llmem.registry module — extension point registry."""

import argparse

import pytest

from llmem.adapters.base import SessionAdapter
from llmem.registry import (
    register_session_adapter,
    get_registered_adapters,
    get_adapter_class,
    register_dream_hook,
    get_registered_dream_hooks,
    register_cli_plugin,
    get_registered_cli_plugins,
    get_cli_plugin_setup_fn,
    _reset_registries,
)


# -- Fixtures --


class StubAdapter(SessionAdapter):
    """Minimal concrete SessionAdapter for testing."""

    def __init__(self, **kwargs):
        pass

    def list_sessions(self, limit=50):
        return []

    def get_session_transcript(self, session_id):
        return None

    def get_session_chunks(self, session_id):
        return []

    def session_exists(self, session_id):
        return False

    def close(self):
        pass


class AnotherStubAdapter(SessionAdapter):
    """Second minimal adapter for duplicate-name tests."""

    def __init__(self, **kwargs):
        pass

    def list_sessions(self, limit=50):
        return []

    def get_session_transcript(self, session_id):
        return None

    def get_session_chunks(self, session_id):
        return []

    def session_exists(self, session_id):
        return False

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset all registries before and after each test."""
    _reset_registries()
    yield
    _reset_registries()


# -- register_session_adapter --


class TestRegistry_RegisterSessionAdapter:
    """Tests for register_session_adapter()."""

    def test_register_adapter_adds_to_registry(self):
        register_session_adapter("stub", StubAdapter)
        assert "stub" in get_registered_adapters()
        assert get_adapter_class("stub") is StubAdapter

    def test_register_duplicate_raises(self):
        register_session_adapter("stub", StubAdapter)
        with pytest.raises(ValueError, match="already registered"):
            register_session_adapter("stub", AnotherStubAdapter)

    def test_get_registered_adapters_returns_frozenset(self):
        result = get_registered_adapters()
        assert isinstance(result, frozenset)

    def test_get_adapter_class_not_found_returns_none(self):
        assert get_adapter_class("nonexistent") is None

    def test_register_non_session_adapter_subclass_raises(self):
        with pytest.raises(TypeError, match="not a subclass of SessionAdapter"):
            register_session_adapter("bad", dict)

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_session_adapter("", StubAdapter)

    def test_register_non_class_raises(self):
        with pytest.raises(TypeError, match="must be a class"):
            register_session_adapter("bad", "not_a_class")

    def test_reset_registry_clears_custom_adapters(self):
        register_session_adapter("stub", StubAdapter)
        assert "stub" in get_registered_adapters()
        _reset_registries()
        assert "stub" not in get_registered_adapters()

    def test_frozenset_immutability(self):
        """get_registered_adapters() returns frozenset; caller cannot mutate registry."""
        register_session_adapter("stub", StubAdapter)
        adapters = get_registered_adapters()
        assert isinstance(adapters, frozenset)
        # frozenset has no .add() — confirm it's genuinely immutable
        assert not hasattr(adapters, "add") or not callable(
            getattr(adapters, "add", None)
        )


# -- register_dream_hook --


class TestRegistry_RegisterDreamHook:
    """Tests for register_dream_hook()."""

    def test_register_hook_adds_to_registry(self):
        hook_fn = lambda dreamer, result, apply: None
        register_dream_hook("light", hook_fn)
        hooks = get_registered_dream_hooks()
        assert "light" in hooks
        assert hooks["light"] is hook_fn

    def test_register_hook_with_invalid_phase_raises(self):
        with pytest.raises(ValueError, match="invalid phase"):
            register_dream_hook("invalid_phase", lambda d, r, a: None)

    def test_register_hook_duplicate_phase_raises(self):
        register_dream_hook("deep", lambda d, r, a: None)
        with pytest.raises(ValueError, match="already registered"):
            register_dream_hook("deep", lambda d, r, a: None)

    def test_get_registered_hooks_returns_dict(self):
        hooks = get_registered_dream_hooks()
        assert isinstance(hooks, dict)

    def test_reset_registry_clears_custom_hooks(self):
        register_dream_hook("rem", lambda d, r, a: None)
        assert "rem" in get_registered_dream_hooks()
        _reset_registries()
        assert "rem" not in get_registered_dream_hooks()

    def test_register_hook_non_callable_raises(self):
        with pytest.raises(TypeError, match="must be callable"):
            register_dream_hook("light", "not_callable")

    def test_hooks_for_all_phases(self):
        """All three phases can be registered independently."""
        register_dream_hook("light", lambda d, r, a: "light_hook")
        register_dream_hook("deep", lambda d, r, a: "deep_hook")
        register_dream_hook("rem", lambda d, r, a: "rem_hook")
        hooks = get_registered_dream_hooks()
        assert len(hooks) == 3
        assert set(hooks.keys()) == {"light", "deep", "rem"}

    def test_hooks_dict_is_copy(self):
        """get_registered_dream_hooks() returns a copy; modifying it doesn't affect the registry."""
        register_dream_hook("light", lambda d, r, a: None)
        hooks = get_registered_dream_hooks()
        hooks["deep"] = lambda d, r, a: None  # Mutate the copy
        assert "deep" not in get_registered_dream_hooks()


# -- register_cli_plugin --


class TestRegistry_RegisterCliPlugin:
    """Tests for register_cli_plugin()."""

    def test_register_plugin_adds_to_registry(self):
        setup_fn = lambda subparsers: None
        register_cli_plugin("my-extension", setup_fn)
        assert "my-extension" in get_registered_cli_plugins()

    def test_register_duplicate_plugin_raises(self):
        register_cli_plugin("my-extension", lambda sp: None)
        with pytest.raises(ValueError, match="already registered"):
            register_cli_plugin("my-extension", lambda sp: None)

    def test_get_registered_plugins_returns_frozenset(self):
        result = get_registered_cli_plugins()
        assert isinstance(result, frozenset)

    def test_reset_registry_clears_custom_plugins(self):
        register_cli_plugin("my-extension", lambda sp: None)
        assert "my-extension" in get_registered_cli_plugins()
        _reset_registries()
        assert "my-extension" not in get_registered_cli_plugins()

    def test_register_plugin_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_cli_plugin("", lambda sp: None)

    def test_register_plugin_non_callable_raises(self):
        with pytest.raises(TypeError, match="must be callable"):
            register_cli_plugin("bad", "not_callable")

    def test_get_cli_plugin_setup_fn(self):
        setup_fn = lambda subparsers: None
        register_cli_plugin("my-extension", setup_fn)
        assert get_cli_plugin_setup_fn("my-extension") is setup_fn

    def test_get_cli_plugin_setup_fn_not_found(self):
        assert get_cli_plugin_setup_fn("nonexistent") is None
