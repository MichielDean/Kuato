"""Session adapter interface for llmem."""

from .base import SessionAdapter
from .opencode import OpenCodeAdapter

__all__ = ["SessionAdapter", "OpenCodeAdapter"]
