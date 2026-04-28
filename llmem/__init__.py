"""LLMem — structured memory with semantic search."""

from .store import MemoryStore, register_memory_type
from .paths import (
    get_llmem_home,
    get_config_path,
    get_db_path,
    get_dream_diary_path,
    get_proposed_changes_path,
    get_context_dir,
    migrate_from_lobsterdog,
)
from .config import load_config
from .adapters.base import SessionAdapter
from .adapters.opencode import OpenCodeAdapter

__all__ = [
    "MemoryStore",
    "register_memory_type",
    "get_llmem_home",
    "get_config_path",
    "get_db_path",
    "get_dream_diary_path",
    "get_proposed_changes_path",
    "get_context_dir",
    "migrate_from_lobsterdog",
    "load_config",
    "SessionAdapter",
    "OpenCodeAdapter",
]
