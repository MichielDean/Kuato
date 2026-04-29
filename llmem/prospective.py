"""Prospective memory indexing — generate hints for memories."""

import json
import logging

from .store import MemoryStore
from .ollama import _call_ollama_generate

log = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def generate_hints(
    content: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE,
) -> list[str]:
    """Generate hints for a memory using Ollama.

    Args:
        content: The memory content to generate hints for.
        model: Ollama model to use.
        base_url: Ollama base URL.

    Returns:
        A list of hint strings. Empty list on error.
    """
    prompt = f"""Generate 3-5 short, keyword-style hints for this memory. Return a JSON array of strings.

Memory: {content[:500]}

Hints:"""

    response = _call_ollama_generate(model, prompt, base_url=base_url)
    if not response:
        return []

    try:
        hints = json.loads(response)
        if isinstance(hints, list):
            return [str(h) for h in hints]
    except json.JSONDecodeError:
        pass

    return []


class ProspectiveIndexer:
    """Generate and attach hints to memories that lack them."""

    def __init__(
        self,
        store: MemoryStore,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE,
    ):
        self._store = store
        self._model = model
        self._base_url = base_url

    def index_all(self, limit: int = 100) -> int:
        """Add hints to memories that lack them.

        Returns:
            Number of memories updated.
        """
        count = 0
        memories = self._store.search(valid_only=True, limit=limit)
        for m in memories:
            hints = m.get("hints", [])
            if not hints or hints == []:
                generated = generate_hints(
                    m["content"], model=self._model, base_url=self._base_url
                )
                if generated:
                    self._store.update(m["id"], hints=generated)
                    count += 1
        return count
