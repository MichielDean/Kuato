"""Session adapter interface for llmem."""

from .base import SessionAdapter
from .opencode import OpenCodeAdapter

__all__ = ["SessionAdapter", "OpenCodeAdapter"]


def get_registered_adapters():
    """Convenience re-export of llmem.registry.get_registered_adapters()."""
    from llmem.registry import get_registered_adapters as _get

    return _get()


__all__.append("get_registered_adapters")
