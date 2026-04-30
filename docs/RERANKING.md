# LLMem Search Reranking

Multi-signal reranking for hybrid search results. [Back to README](../README.md)

## Multi-Signal Reranking

After RRF fusion, search results are automatically reranked using a blend of the RRF score and four weighted signals:

```
final_score = rrf_score * (1 - blend) + weighted_signal * blend
```

**Default blend factor: 0.3** (70% RRF, 30% signals). Configure via `Retriever(store, embedder, blend=...)`. Range: 0.0 (pure RRF) to 1.0 (pure signals). Out-of-range values raise `ValueError`.

### Signals and Weights

| Signal | Weight | Formula |
|--------|--------|---------|
| Confidence | 0.4 | Direct use of `confidence` field (0.0–1.0, default 0.0 for missing) |
| Recency | 0.3 | `exp(-0.01 * days_since_access)` (0.0 if never accessed) |
| Access frequency | 0.2 | `log(1 + access_count / max(age_days, 1))` (0.0 if never accessed) |
| Type priority | 0.1 | Lookup in `TYPE_PRIORITY` dict (default 1.0 for unknown types) |

### Type Priority

| Type | Priority | | Type | Priority |
|------|----------|-|------|----------|
| decision | 1.2 | | fact | 1.0 |
| preference | 1.1 | | project_state | 1.0 |
| procedure | 1.1 | | self_assessment | 1.0 |
| | | | event | 0.9 |
| | | | conversation | 0.7 |

Search results include both `_rrf_score` (raw RRF fusion score) and `_rerank_score` (blended final score). Results are sorted by `_rerank_score` descending, with ties broken by ascending memory ID. Search operations (`Retriever.search()` and `Retriever.hybrid_search()`) automatically track access — each returned result's `access_count` and `accessed_at` are updated (best-effort), keeping the recency and access frequency signals current. This Hebbian reinforcement is on by default (`track_access=True`); pass `track_access=False` to skip access tracking (useful for analytics queries that shouldn't inflate counts).