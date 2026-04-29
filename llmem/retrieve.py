"""Retrieval and ranking engine for memory search."""

import logging
import math
from datetime import datetime, timezone

from .store import MemoryStore
from .embed import EmbeddingEngine

try:
    from memory.providers import EmbedProvider
except ImportError:
    EmbedProvider = None  # type: ignore[assignment,misc]

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


def _compute_rerank_signals(
    memory: dict, type_priority: dict[str, float], now: datetime
) -> dict[str, float]:
    """Compute per-memory reranking signals from memory dict fields.

    Takes a memory dict (with keys confidence, accessed_at, access_count,
    created_at, type), a type priority mapping, and the current UTC time.
    Returns a dict with keys "confidence", "recency", "access", "type"
    containing individual normalized signal scores. Returns 0.0 for missing
    fields. Never raises — uses .get() with defaults.

    Args:
        memory: A memory dict with optional keys confidence, accessed_at,
            access_count, created_at, type.
        type_priority: Mapping of memory type name to priority weight.
        now: Current UTC datetime for time-based calculations.

    Returns:
        Dict with confidence, recency, access, and type signal floats.
    """
    # Confidence signal: direct use of confidence field, default 0.0
    confidence = float(memory.get("confidence", 0.0) or 0.0)

    # Recency signal: exp(-0.01 * days_since_access), default 0.0
    accessed_at_raw = memory.get("accessed_at")
    recency = 0.0
    if accessed_at_raw:
        try:
            accessed_at = datetime.fromisoformat(accessed_at_raw)
            if accessed_at.tzinfo is None:
                accessed_at = accessed_at.replace(tzinfo=timezone.utc)
            days_since = (now - accessed_at).days
            recency = math.exp(-0.01 * days_since)
        except (ValueError, TypeError):
            log.debug("llmem: retrieve: unparseable accessed_at: %r", accessed_at_raw)
            recency = 0.0

    # Access frequency signal: log(1 + access_count / max(age_days, 1))
    access_count = int(memory.get("access_count", 0) or 0)
    created_at_raw = memory.get("created_at")
    age_days = 1
    if created_at_raw:
        try:
            created_at = datetime.fromisoformat(created_at_raw)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_days = max((now - created_at).days, 1)
        except (ValueError, TypeError):
            log.debug("llmem: retrieve: unparseable created_at: %r", created_at_raw)
            age_days = 1
    access = math.log(1 + access_count / age_days) if access_count > 0 else 0.0

    # Type priority signal: lookup in type_priority, default 1.0
    mem_type = memory.get("type", "")
    type_signal = float(type_priority.get(mem_type, 1.0))

    return {
        "confidence": confidence,
        "recency": recency,
        "access": access,
        "type": type_signal,
    }


def _compute_weighted_signal(signals: dict[str, float]) -> float:
    """Combine confidence, recency, access, and type signals using weights.

    Takes the signal dict from _compute_rerank_signals and returns the
    weighted sum: CONFIDENCE_WEIGHT * confidence + RECENCY_WEIGHT * recency
    + ACCESS_WEIGHT * access + TYPE_WEIGHT * type.

    Args:
        signals: Dict with confidence, recency, access, and type floats.

    Returns:
        Weighted signal float in [0.0, max_weight_sum] range.
    """
    return (
        CONFIDENCE_WEIGHT * signals["confidence"]
        + RECENCY_WEIGHT * signals["recency"]
        + ACCESS_WEIGHT * signals["access"]
        + TYPE_WEIGHT * signals["type"]
    )


class Retriever:
    """Retrieve and rank memories by relevance."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: "EmbeddingEngine | EmbedProvider | None" = None,
        blend: float = 0.3,
    ):
        """Initialize the Retriever.

        Args:
            store: The memory store to search against.
            embedder: Optional embedding engine or provider for semantic search.
                Accepts either an EmbeddingEngine (legacy) or an EmbedProvider
                from the memory.providers module.
            blend: Blend factor for reranking (0.0 = pure RRF, 1.0 = pure
                signals). Defaults to 0.3.

        Raises:
            ValueError: If blend is not in [0.0, 1.0].
        """
        if not 0.0 <= blend <= 1.0:
            raise ValueError(
                f"llmem: retrieve: blend factor {blend!r} out of range [0.0, 1.0]"
            )
        self._store = store
        self._embedder = embedder
        self._blend = blend

    def search(
        self,
        query: str,
        limit: int = 20,
        type_filter: str | None = None,
        traverse_relations: bool = False,
        relation_depth: int = 1,
        traverse_refs: bool = False,
        max_ref_depth: int = 3,
        track_access: bool = True,
    ) -> list[dict]:
        """Search memories with ranking.

        Args:
            query: Search query string.
            limit: Maximum results.
            type_filter: Filter by memory type.
            traverse_relations: Include related memories.
            relation_depth: Max relation traversal depth.
            traverse_refs: If True, follow references edges (target_type='code')
                from result memory IDs and resolve code refs via
                refs.resolve_code_ref(), appending resolved code dicts to
                results. Defaults to False.
            max_ref_depth: Max ref expansion depth (1-5, default 3).
                Controls how many hops to follow when traversing code refs.
            track_access: If True, increment access_count and update
                accessed_at for each result. Defaults to True.

        Returns:
            List of memory dicts sorted by relevance.
        """
        results = self._store.search(
            query=query, type=type_filter, limit=limit, _include_rank=True
        )

        # Track access for each result (best-effort, never raises)
        self._track_access(results, track_access=track_access)

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

        if traverse_refs and results:
            from .refs import resolve_code_ref

            effective_depth = min(max(max_ref_depth, 1), 5)
            mem_ids = [r["id"] for r in results if r.get("id")]
            code_refs = self._store.traverse_relations(
                mem_ids,
                max_depth=effective_depth,
                target_type="code",
            )
            seen_refs = set()
            for ref in code_refs:
                ref_id = ref["target_id"]
                if ref_id in seen_refs:
                    continue
                seen_refs.add(ref_id)
                resolved = resolve_code_ref(ref_id)
                if resolved is not None:
                    resolved["_source"] = "code"
                    results.append(resolved)

        return results[:limit]

    def hybrid_search(
        self,
        query: str,
        limit: int = 20,
        type_filter: str | None = None,
        alpha: float = DEFAULT_ALPHA,
        search_mode: str = "hybrid",
        track_access: bool = True,
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
            track_access: If True, increment access_count and update
                accessed_at for each result. Defaults to True.

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

            # Apply reranking
            now = datetime.now(timezone.utc)
            results = self._apply_reranking(results, now)

            # Track access for each result (best-effort, never raises)
            self._track_access(results, track_access=track_access)

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

            # Apply reranking
            now = datetime.now(timezone.utc)
            results = self._apply_reranking(results, now)

            # Track access for each result (best-effort, never raises)
            self._track_access(results, track_access=track_access)

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
                track_access=track_access,
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
                track_access=track_access,
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

        # Apply reranking
        now = datetime.now(timezone.utc)
        results = self._apply_reranking(results, now)

        # Track access for each result (best-effort, never raises)
        self._track_access(results, track_access=track_access)

        return results[:limit]

    def _apply_reranking(self, results: list[dict], now: datetime) -> list[dict]:
        """Apply multi-signal reranking to search results.

        Modifies each result dict to add a _rerank_score key and re-sorts
        the results by _rerank_score descending (ties broken by ascending ID).

        Args:
            results: List of result dicts, each with _rrf_score key.
            now: Current UTC datetime for time-based calculations.

        Returns:
            Results re-sorted by _rerank_score descending.
        """
        if not results or self._blend == 0.0:
            # blend=0.0 means pure RRF — rerank_score equals rrf_score
            for r in results:
                r["_rerank_score"] = r.get("_rrf_score", 0.0)
            return results

        for r in results:
            signals = _compute_rerank_signals(r, TYPE_PRIORITY, now)
            weighted = _compute_weighted_signal(signals)
            rrf_score = r.get("_rrf_score", 0.0)
            r["_rerank_score"] = rrf_score * (1 - self._blend) + weighted * self._blend

        results.sort(key=lambda x: (-x.get("_rerank_score", 0.0), x.get("id", "")))
        return results

    def _track_access(self, results: list[dict], track_access: bool = True) -> None:
        """Track access for each result by calling store.touch_batch().

        Best-effort: errors are caught and logged, never propagated.
        When track_access is False, returns immediately without touching.

        Args:
            results: List of result dicts, each with an 'id' key.
            track_access: If True, increment access_count and update
                accessed_at for each result. If False, skip access tracking.
        """
        if not track_access:
            return
        if not results:
            return
        try:
            ids = [r["id"] for r in results]
            self._store.touch_batch(ids)
        except Exception:
            log.debug(
                "llmem: retrieve: failed to track access for %d results",
                len(results),
            )

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
