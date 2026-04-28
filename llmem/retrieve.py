"""Retrieval and ranking engine for memory search."""

import json
import logging
from datetime import datetime, timezone

from .store import MemoryStore
from .embed import EmbeddingEngine

log = logging.getLogger(__name__)

RECENCY_DAYS = 30
RECENCY_BOOST = 1.3
CONFIDENCE_WEIGHT = 0.4
RECENCY_WEIGHT = 0.3
ACCESS_WEIGHT = 0.2
TYPE_WEIGHT = 0.1

TYPE_PRIORITY = {
    "decision": 1.2,
    "preference": 1.1,
    "procedure": 1.1,
    "fact": 1.0,
    "project_state": 1.0,
    "self_assessment": 1.0,
    "event": 0.9,
    "conversation": 0.7,
}


class Retriever:
    """Retrieve and rank memories by relevance."""

    def __init__(self, store: MemoryStore, embedder: EmbeddingEngine | None = None):
        self._store = store
        self._embedder = embedder

    def search(
        self,
        query: str,
        limit: int = 20,
        type_filter: str | None = None,
        traverse_relations: bool = False,
        relation_depth: int = 1,
    ) -> list[dict]:
        """Search memories with ranking.

        Args:
            query: Search query string.
            limit: Maximum results.
            type_filter: Filter by memory type.
            traverse_relations: Include related memories.
            relation_depth: Max relation traversal depth.

        Returns:
            List of memory dicts sorted by relevance.
        """
        results = self._store.search(
            query=query, type=type_filter, limit=limit, _include_rank=True
        )

        if traverse_relations and results:
            mem_ids = [r["id"] for r in results]
            related = self._store.traverse_relations(mem_ids, max_depth=relation_depth)
            related_ids = [
                r["target_id"]
                for r in related
                if r["target_id"] not in {m["id"] for m in results}
            ]
            if related_ids:
                related_memories = self._store.get_batch(related_ids)
                for mid in related_ids[:limit]:
                    if mid in related_memories:
                        results.append(related_memories[mid])

        return results[:limit]

    def format_context(
        self, query: str, budget: int = 4000, type_filter: str | None = None
    ) -> str:
        """Format search results as context for LLM injection."""
        results = self.search(query, limit=20, type_filter=type_filter)
        if not results:
            return ""
        lines = []
        for m in results:
            line = f"- [{m['type']}] {m['content']}"
            if m.get("summary"):
                line += f" (summary: {m['summary']})"
            lines.append(line)
        context = "\n".join(lines)
        return context[:budget]

    def format_detailed(
        self,
        query: str,
        limit: int = 20,
        traverse_relations: bool = False,
        relation_depth: int = 1,
    ) -> str:
        """Format search results with full detail."""
        results = self.search(
            query,
            limit=limit,
            traverse_relations=traverse_relations,
            relation_depth=relation_depth,
        )
        if not results:
            return ""
        output = []
        for m in results:
            output.append(f"[{m['type']}] {m['content']}")
            if m.get("summary"):
                output.append(f"  Summary: {m['summary']}")
            output.append(f"  Confidence: {m.get('confidence', 0):.2f}")
            output.append("")
        return "\n".join(output)
