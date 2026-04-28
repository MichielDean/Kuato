"""Extension registry for llmem.

Provides registration functions for session adapters, dream hooks,
and CLI plugins. Each registry uses the same pattern as
register_memory_type() in store.py: module-level mutable state
with defensive copies at usage sites.

Register functions are called by extensions to plug in domain-specific
behavior without modifying core code.
"""

import logging
from typing import Callable

from .adapters.base import SessionAdapter

log = logging.getLogger(__name__)

# Valid dream phases for hook registration.
_VALID_DREAM_PHASES = frozenset({"light", "deep", "rem"})

# --- Module-level registries ---
# Populated by explicit register_X() calls only.
# Each consumer makes a defensive copy at usage time.

_adapter_registry: dict[str, type[SessionAdapter]] = {}
_dream_hook_registry: dict[str, Callable] = {}
_cli_plugin_registry: dict[str, Callable] = {}


def register_session_adapter(name: str, adapter_class: type[SessionAdapter]) -> None:
    """Register a session adapter class under a unique name.

    Args:
        name: Unique adapter name (e.g., 'opencode', 'cistern').
        adapter_class: A subclass of SessionAdapter.

    Raises:
        ValueError: If name is already registered.
        TypeError: If adapter_class is not a subclass of SessionAdapter.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            "llmem: registry: register_session_adapter: name must be a non-empty string"
        )
    if not isinstance(adapter_class, type):
        raise TypeError(
            f"llmem: registry: register_session_adapter: "
            f"adapter_class must be a class, got {type(adapter_class)!r}"
        )
    if not issubclass(adapter_class, SessionAdapter):
        raise TypeError(
            f"llmem: registry: register_session_adapter: "
            f"{adapter_class!r} is not a subclass of SessionAdapter"
        )
    if name in _adapter_registry:
        raise ValueError(
            f"llmem: registry: register_session_adapter: "
            f"adapter '{name}' is already registered"
        )
    _adapter_registry[name] = adapter_class


def get_registered_adapters() -> frozenset[str]:
    """Return the set of registered adapter names.

    Returns:
        An immutable frozenset of adapter name strings.
    """
    return frozenset(_adapter_registry.keys())


def get_adapter_class(name: str) -> type[SessionAdapter] | None:
    """Return the adapter class registered under the given name.

    Args:
        name: The adapter name to look up.

    Returns:
        The adapter class, or None if not found.
    """
    return _adapter_registry.get(name)


def register_dream_hook(
    phase: str,
    hook_fn: Callable,
) -> None:
    """Register a dream-phase hook function.

    The hook_fn is called by Dreamer.run() after the phase's built-in
    logic completes, with (dreamer, result, apply) arguments.

    Args:
        phase: One of 'light', 'deep', or 'rem'.
        hook_fn: Callable with signature (Dreamer, DreamResult, bool) -> None.

    Raises:
        ValueError: If phase is not a valid dream phase name.
        ValueError: If a hook is already registered for that phase.
    """
    if phase not in _VALID_DREAM_PHASES:
        raise ValueError(
            f"llmem: registry: register_dream_hook: "
            f"invalid phase '{phase}', must be one of {sorted(_VALID_DREAM_PHASES)}"
        )
    if not callable(hook_fn):
        raise TypeError(
            f"llmem: registry: register_dream_hook: "
            f"hook_fn must be callable, got {type(hook_fn)!r}"
        )
    if phase in _dream_hook_registry:
        raise ValueError(
            f"llmem: registry: register_dream_hook: "
            f"a hook is already registered for phase '{phase}'"
        )
    _dream_hook_registry[phase] = hook_fn


def get_registered_dream_hooks() -> dict[str, Callable]:
    """Return a copy of the dream hook registry.

    Returns:
        A dict mapping phase name to hook function. Callers cannot
        mutate the internal registry.
    """
    return dict(_dream_hook_registry)


def register_cli_plugin(
    name: str,
    setup_fn: Callable,
) -> None:
    """Register a CLI plugin setup function.

    The setup_fn receives an argparse._SubParserGroup and adds its
    own subcommands to it.

    Args:
        name: Unique plugin name (e.g., 'lobsterdog').
        setup_fn: Callable with signature (argparse._SubParserGroup) -> None.

    Raises:
        ValueError: If name is already registered.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            "llmem: registry: register_cli_plugin: name must be a non-empty string"
        )
    if not callable(setup_fn):
        raise TypeError(
            f"llmem: registry: register_cli_plugin: "
            f"setup_fn must be callable, got {type(setup_fn)!r}"
        )
    if name in _cli_plugin_registry:
        raise ValueError(
            f"llmem: registry: register_cli_plugin: "
            f"plugin '{name}' is already registered"
        )
    _cli_plugin_registry[name] = setup_fn


def get_registered_cli_plugins() -> frozenset[str]:
    """Return the set of registered CLI plugin names.

    Returns:
        An immutable frozenset of plugin name strings.
    """
    return frozenset(_cli_plugin_registry.keys())


def get_cli_plugin_setup_fn(name: str) -> Callable | None:
    """Return the setup function for a CLI plugin.

    Args:
        name: The plugin name to look up.

    Returns:
        The setup function, or None if not found.
    """
    return _cli_plugin_registry.get(name)


def _reset_registries() -> None:
    """Reset all registries to empty. For testing only.

    Follows the pattern of _reset_global_registry() in store.py.
    """
    _adapter_registry.clear()
    _dream_hook_registry.clear()
    _cli_plugin_registry.clear()
