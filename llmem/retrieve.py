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

DEFAULT_ALPHA = 0.7
DEFAULT_RRF_K = 60


def _rrf_score(
    semantic_ranks: dict[str, int],
    fts_ranks: dict[str, int],
    alpha: float = DEFAULT_ALPHA,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion scores from semantic and FTS rank maps.

    Args:
        semantic_ranks: Mapping of memory ID to its rank in semantic search (1-based).
        fts_ranks: Mapping of memory ID to its rank in FTS5 search (1-based).
        alpha: Weight for semantic scores (0.0 = pure FTS, 1.0 = pure semantic).
            Defaults to 0.7.
        k: RRF constant to dampen the contribution of high ranks. Defaults to 60.

    Returns:
        List of (id, score) tuples sorted by descending score, ties broken by
        ascending ID for determinism. Empty inputs return an empty list.
    """
    if not semantic_ranks and not fts_ranks:
        return []

    all_ids = set(semantic_ranks) | set(fts_ranks)
    # IDs missing from one list get a default rank len(their_present_list) + 1
    # (spec: "rank len(that_list) + 1 for the missing side").
    n_semantic = len(semantic_ranks)
    n_fts = len(fts_ranks)

    scored: list[tuple[str, float]] = []
    for mid in all_ids:
        # If the ID is missing from semantic_ranks, it gets default rank
        # len(semantic_ranks) + 1 (i.e., it would be ranked last in that list).
        semantic_rank = semantic_ranks.get(mid, n_semantic + 1)
        # If the ID is missing from fts_ranks, it gets default rank
        # len(fts_ranks) + 1 (i.e., it would be ranked last in that list).
        fts_rank = fts_ranks.get(mid, n_fts + 1)
        score = alpha * (1 / (k + semantic_rank)) + (1 - alpha) * (1 / (k + fts_rank))
        scored.append((mid, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


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

    def hybrid_search(
        self,
        query: str,
        limit: int = 20,
        type_filter: str | None = None,
        alpha: float = DEFAULT_ALPHA,
        search_mode: str = "hybrid",
    ) -> list[dict]:
        """Search memories using hybrid RRF fusion of FTS5 and semantic results.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            type_filter: Filter by memory type.
            alpha: Weight for semantic scores (0.0 = pure FTS, 1.0 = pure
                semantic). Defaults to 0.7.
            search_mode: One of "hybrid", "fts", or "semantic".
                - "hybrid": Run both FTS5 and semantic search, merge via RRF.
                - "fts": Run FTS5 only.
                - "semantic": Run semantic search only (requires embedder).

        Returns:
            List of memory dicts sorted by descending RRF score. Each dict
            includes an ``_rrf_score`` key. Raises ValueError if
            search_mode="semantic" and no embedder is configured.

        Raises:
            ValueError: If search_mode="semantic" and self._embedder is None.
        """
        if not query:
            return []

        valid_modes = {"hybrid", "fts", "semantic"}
        if search_mode not in valid_modes:
            raise ValueError(
                f"llmem: retrieve: invalid search_mode {search_mode!r}, "
                f"must be one of {sorted(valid_modes)}"
            )

        if search_mode == "fts":
            fts_results = self._store.search(
                query=query, type=type_filter, limit=limit, _include_rank=True
            )
            # Build FTS-only rank map and compute RRF scores
            fts_rank_map: dict[str, int] = {
                r["id"]: i + 1 for i, r in enumerate(fts_results)
            }
            scored = _rrf_score({}, fts_rank_map, alpha=0.0)
            score_by_id = dict(scored)
            results: list[dict] = []
            for r in fts_results:
                mem = {k: v for k, v in r.items() if k != "_fts_rank"}
                mem["_rrf_score"] = score_by_id.get(r["id"], 0.0)
                results.append(mem)
            return results[:limit]

        if search_mode == "semantic":
            if self._embedder is None:
                raise ValueError(
                    "llmem: retrieve: semantic search requires an embedder"
                )
            query_vec = self._embedder.embed(query)
            semantic_results = self._store.search_by_embedding(query_vec, limit=limit)
            # Build semantic-only rank map and compute RRF scores
            semantic_rank_map: dict[str, int] = {
                mem["id"]: i + 1 for i, (mem, _score) in enumerate(semantic_results)
            }
            scored = _rrf_score(semantic_rank_map, {}, alpha=1.0)
            score_by_id = dict(scored)
            results: list[dict] = []
            for rank, (mem, _score) in enumerate(semantic_results, start=1):
                r = dict(mem)
                r["_rrf_score"] = score_by_id.get(mem["id"], 0.0)
                results.append(r)
            # Apply type_filter (same post-filter pattern as hybrid path)
            if type_filter:
                results = [r for r in results if r.get("type") == type_filter]
            return results[:limit]

        # search_mode == "hybrid"
        if self._embedder is None:
            log.warning(
                "llmem: retrieve: embedder not configured, falling back to FTS5-only"
            )
            return self.hybrid_search(
                query=query,
                limit=limit,
                type_filter=type_filter,
                alpha=alpha,
                search_mode="fts",
            )
        fts_results = self._store.search(
            query=query, type=type_filter, limit=limit, _include_rank=True
        )
        # Strip internal _fts_rank from FTS results before merging (issue: metadata leak)
        fts_results_clean = [
            {k: v for k, v in r.items() if k != "_fts_rank"} for r in fts_results
        ]
        fts_ranks: dict[str, int] = {r["id"]: i + 1 for i, r in enumerate(fts_results)}

        # Run semantic search
        try:
            query_vec = self._embedder.embed(query)
            semantic_results = self._store.search_by_embedding(query_vec, limit=limit)
            semantic_ranks: dict[str, int] = {
                mem["id"]: i + 1 for i, (mem, _score) in enumerate(semantic_results)
            }
        except Exception:
            log.warning(
                "llmem: retrieve: semantic search failed, falling back to FTS5-only"
            )
            return self.hybrid_search(
                query=query,
                limit=limit,
                type_filter=type_filter,
                alpha=alpha,
                search_mode="fts",
            )

        # Compute RRF scores
        scored = _rrf_score(semantic_ranks, fts_ranks, alpha=alpha)

        # Merge result dicts, deduplicating by memory ID
        all_results: dict[str, dict] = {}
        for r in fts_results_clean:
            all_results[r["id"]] = r
        for mem, _score in semantic_results:
            if mem["id"] not in all_results:
                all_results[mem["id"]] = mem

        # Apply type_filter
        if type_filter:
            all_results = {
                mid: r for mid, r in all_results.items() if r.get("type") == type_filter
            }

        # Build final sorted list with RRF scores
        results: list[dict] = []
        for mid, rrf_score in scored:
            if mid in all_results:
                mem = dict(all_results[mid])
                mem["_rrf_score"] = rrf_score
                results.append(mem)

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
