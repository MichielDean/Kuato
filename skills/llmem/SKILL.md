---
name: llmem
description: >
  Manage LLMem — structured memory system with SQLite-backed factual memory,
  semantic search, relation traversal (via --traverse-relations flag on CLI),
  extraction, consolidation, and background dreaming (decay, boost, promote,
  merge). Use when the user wants to: (1) add, search, update, or delete memories,
  (2) extract memories from text, (3) consolidate duplicates, (4) generate context
  for injection, (5) check memory stats, (6) search with relation traversal to find
  related memories (llmem search "query" --traverse-relations), (7) run background
  consolidation/dream. Triggers on: "memory", "remember", "recall", "llmem",
  "memories", "forget", "consolidate memories", "dream".
license: MIT
---

# LLMem

LLMem is a structured memory system. It stores factual memories in SQLite with optional embedding-based semantic search via Ollama (nomic-embed-text). Semantic search uses a sqlite-vec HNSW ANN index for sub-linear retrieval, with automatic fallback to brute-force cosine similarity if the sqlite-vec extension is not available. Memories can be connected via typed relations (`related_to`, `supersedes`, `contradicts`, `depends_on`, `derived_from`), and searches can traverse these relation edges to surface related memories. A background dreaming/consolidation system (light, deep, REM phases) runs automatically via systemd timer to decay idle memories, boost frequent ones, promote high-value ones, and merge near-duplicates.

## Installation

- **CLI:** `~/.local/bin/llmem`
- **Config:** `~/.config/llmem/config.yaml`
- **DB:** `~/.config/llmem/memory.db`
- **Ollama:** `http://localhost:11434` (local, for embeddings)

## Session Adapter Configuration

LLMem uses a session adapter to read conversation transcripts for memory extraction. Set `session.adapter` in config.yaml:

- `opencode` (default) — reads from `~/.local/share/opencode/opencode.db`. Auto-selected if the database exists.
- `copilot` — reads from `~/.copilot/session-state/`. Auto-selected if only Copilot session data exists. Full transcripts require `--share`.
- `none` — no adapter. Context injection still works, but transcript extraction returns `no_transcript`.

```yaml
# For Copilot CLI:
session:
  adapter: copilot
copilot:
  state_dir: ~/.copilot/session-state
  share_dir: .
```

`llmem init` auto-detects the adapter type based on which session state directory exists.

## Memory Types

| Type | Use for |
|------|---------|
| fact | Objective truths, definitions, state of the world |
| decision | Choices made and their rationale |
| preference | User preferences, style choices |
| event | Things that happened at a point in time |
| project_state | Current status of a project or system |
| procedure | How-to knowledge, step sequences |
| conversation | Notable conversation outcomes or commitments |
| self_assessment | Structured introspective records — error patterns, behavioral corrections, recurring mistakes, proposed procedural updates |

## Error Taxonomy

Self-assessment memories are categorized using a standard error taxonomy. Each category identifies a class of mistake that the introspection system tracks for pattern detection:

| Category | Description |
|----------|-------------|
| NULL_SAFETY | Missing null/None/undefined checks before property access or method calls |
| ERROR_HANDLING | Missing try/except, bare except, swallowed errors, unhandled promise rejections |
| OFF_BY_ONE | Boundary errors, wrong loop bounds, fencepost errors |
| RACE_CONDITION | Concurrency issues, async/await problems, missing locks |
| AUTH_BYPASS | Missing auth checks, SSRF, injection vulnerabilities, security oversights |
| DATA_INTEGRITY | Stale derived fields, out-of-sync caches/embeddings/indexes, source-of-truth divergence |
| MISSING_VERIFICATION | Skipped test steps, unverified outputs, assumed-it-works |
| EDGE_CASE | Unhandled empty input, unexpected types, boundary values |
| PERFORMANCE | N+1 queries, unnecessary recomputation, memory leaks |
| DESIGN | Architectural issues, wrong abstraction level, coupling problems |
| REVIEW_PASSED | Clean review with no findings — positive outcome for tracking purposes |

## Structured self_assessment Format

Self-assessment memories follow a structured format with nine fields:

| Field | Required | Description |
|-------|----------|-------------|
| Category | Yes | Taxonomy category from the Error Taxonomy above (e.g. `NULL_SAFETY`, `ERROR_HANDLING`) |
| Context | No | Where and when — file, task, session date |
| What_happened | Yes | Behavioral description, not narrative |
| Outcomes | No | What were the results? Did things work on first try or require iterations? |
| What_caught_it | No | How the error was discovered (`self-review`, `test`, `user`, `CI`) |
| Estimates_vs_actual | No | Was the complexity assessment accurate? Did tasks take more or less effort? |
| Recurring | No | `yes` or `no`; if yes, reference past self_assessment IDs |
| Proposed_update | No | Specific procedural directive to prevent recurrence |
| Iteration_count | No | How many attempts before success (integer). 1 = first try, 2 = one retry, etc. |

## Key Commands

```bash
# Add a memory
llmem add "content" --type fact
llmem add "prefer dark theme" --type preference --confidence 0.9

# Add a memory with a relation
llmem add --type fact "python 3.12 is the latest version" --relation supersedes --relation-to <old-memory-id>

# Search memories (hybrid RRF fusion by default)
llmem search "query"
llmem search "query" --type decision --limit 5
llmem search "query" --context --budget 4000

# Search mode flags (mutually exclusive)
llmem search "query" --fts-only        # FTS5 keyword search only (no embedder needed)
llmem search "query" --semantic-only    # Semantic (embedding) search only (requires embedder)

# Search with relation traversal
llmem search "query" --traverse-relations

# List all
llmem list
llmem list --type fact --limit 20

# Get specific memory (read-only, does not update access stats)
llmem get <id>

# Update a memory
llmem update <id> --content "new content"

# Invalidate (soft-delete — marks as expired)
llmem invalidate <id> --reason "no longer relevant"

# Delete permanently
llmem delete <id>

# Stats
llmem stats

# Extract memories from text using local LLM
llmem extract "text to process"
llmem extract --file path/to/file --source-id unique-source-id
llmem extract --dry-run "text"  # preview without saving

# Auto-extract from session transcripts (idle hook)
llmem hook idle <session_id>        # trigger memory extraction and introspection for a session

# Consolidate near-duplicates
llmem consolidate --threshold 0.92

# Dream — background consolidation (decay, boost, promote, merge)
llmem dream                                                # Preview all phases (dry-run)
llmem dream --apply                                        # Execute all phases
llmem dream --phase light                                  # Run only the light phase
llmem dream --phase deep                                   # Run only the deep phase
llmem dream --phase rem                                    # Run only the REM phase
llmem dream --apply --phase deep                           # Apply only the deep phase
llmem dream --apply --skill-patch-threshold 5              # Generate skill patches only for categories with 5+ occurrences
llmem dream --apply --skill-patch-threshold 0              # Disable skill patch generation entirely

# Generate embeddings for memories missing them
llmem embed

# Merge two memories (source invalidated, target kept)
llmem merge <source-id> <target-id>

# Export/import
llmem export --output memories.json
llmem import memories.json

# Inject memory context for a new session
llmem context <session_id>

# Inject key memories during session compaction
llmem context --compacting <session_id>

# Add a structured self_assessment memory (introspection, manual mode)
llmem introspect --category NULL_SAFETY --what-happened "missing null check before property access" --context "handler.py:42" --caught-by self-review --proposed-update "always check for None before accessing .field"
llmem introspect --category ERROR_HANDLING --what-happened "swallowed exception in finally block" --recurring yes --recurring-ref mem-abc123
llmem introspect --category DESIGN --what-happened "tight coupling between modules" --outcomes "required 3 iterations to fix" --estimates-vs-actual "estimated 1 hour, took 4"
llmem introspect --category NULL_SAFETY --what-happened "forgot None check" --iteration-count 2

# Auto-analyze a session transcript (introspection, auto mode)
llmem introspect --auto --session ~/.local/share/opencode/sessions/2026-01-15.json
llmem introspect --auto --session transcript.md --force          # Re-analyze even if already processed
llmem introspect --auto --session transcript.md --no-embed      # Skip embedding generation

# Auto-analyze and confirm before storing (introspection, interactive mode)
llmem introspect --interactive --session ~/.local/share/opencode/sessions/2026-01-15.json

# Track review findings as self_assessment memories (automatic post-review hook)
llmem track-review --finding-file /tmp/review-findings.json --context "handler.py"        Batch mode: persist findings from a JSON file
llmem track-review --category NULL_SAFETY --what-happened "missing null check" --context "handler.py:42" --severity Required --caught-by self-review  Single finding
llmem track-review --context "handler.py"                                                   Clean review (no findings) → creates REVIEW_PASSED memory

# List error taxonomy categories for a severity tier
llmem suggest-categories Required          # → NULL_SAFETY, ERROR_HANDLING, MISSING_VERIFICATION, EDGE_CASE
llmem suggest-categories Blocking         # → AUTH_BYPASS, RACE_CONDITION, DATA_INTEGRITY
llmem suggest-categories Passed           # → REVIEW_PASSED
```

## Relations

Memories can be linked by typed relations: `supersedes`, `contradicts`, `depends_on`, `related_to`, `derived_from`.

```bash
# Create a relation when adding a memory
llmem add --type fact "new info" --relation supersedes --relation-to <old-id>

# Python API: add a relation
store.add_relation(source_id, target_id, "related_to")

# Python API: get all relations for a memory
store.get_relations(mem_id)

# Python API: batch-get relations for multiple memories
store.get_relations_batch([id1, id2, id3])
```

### Relation Traversal in Search

When `traverse_relations=True` is passed to `Retriever.search()`, related memories are surfaced alongside direct matches:

- **related_to / depends_on / derived_from** — related memories appear in results with a `relation_score` that decays as `0.5^distance` (distance-1: 0.5, distance-2: 0.25)
- **supersedes** — traversal-surfaced memories that are superseded by another result are removed; the superseding memory gets a `supersedes` field listing the IDs it replaces. Direct search matches are never removed.
- **contradicts** — contradicted memories are flagged with a `contradicted_by` field (list of source IDs) but not removed from results

**CLI usage:**

```bash
# Search with 1-hop relation traversal
llmem search "query" --traverse-relations

# Search with 2-hop traversal
llmem search "query" --traverse-relations --relation-depth 2

# Combined with type filter and JSON output
llmem search "query" --traverse-relations --type decision --json
```

**Python API:**

```python
from llmem.store import MemoryStore
from llmem.retrieve import Retriever

store = MemoryStore()
retriever = Retriever(store)

# Hybrid search (default: fuses FTS5 + semantic via RRF, alpha=0.7, blend=0.3)
results = retriever.hybrid_search("python", limit=10)

# FTS5-only or semantic-only
results = retriever.hybrid_search("python", search_mode="fts")
results = retriever.hybrid_search("python", search_mode="semantic")

# Control semantic/keyword weight (0.0 = pure FTS, 1.0 = pure semantic)
results = retriever.hybrid_search("python", alpha=0.5)

# Control reranking blend (0.0 = pure RRF, 1.0 = pure signal-based)
retriever = Retriever(store, blend=0.5)
results = retriever.hybrid_search("python", limit=10)

# Search with traversal (default 1-hop)
results = retriever.search("python", traverse_relations=True)

# Deeper traversal (2 hops)
results = retriever.search("python", traverse_relations=True, relation_depth=2)

# Direct store traversal
traversal = store.traverse_relations(["id1", "id2"], max_depth=2)
# Returns: [{"target_id": ..., "relation_type": ..., "distance": 1, "relation_score": 0.5}, ...]
```

Results with traversal include these extra fields:
- `relation_score` (float) — 0.5^distance for traversal-surfaced memories, 0.0 for direct matches
- `relation_type` (str) — the relation type that surfaced this memory
- `supersedes` (list[str]) — IDs this memory supersedes (only present when applicable)
- `contradicted_by` (list[str]) — IDs that contradict this memory (only present when applicable)

### Multi-Signal Reranking

After RRF fusion, search results are automatically reranked using a blend of the RRF score and four weighted signals:

```
final_score = rrf_score * (1 - blend) + weighted_signal * blend
```

**Default blend factor: 0.3** (70% RRF, 30% signals). Configure via `Retriever(store, embedder, blend=0.5)`. Range: 0.0 (pure RRF) to 1.0 (pure signals). Out-of-range values raise `ValueError`.

**Signals and weights:**

| Signal | Weight | Formula |
|--------|--------|---------|
| Confidence | 0.4 | Direct use of `confidence` field (0.0–1.0, default 0.0) |
| Recency | 0.3 | `exp(-0.01 * days_since_access)` (0.0 if never accessed) |
| Access frequency | 0.2 | `log(1 + access_count / max(age_days, 1))` (0.0 if never accessed) |
| Type priority | 0.1 | Lookup in `TYPE_PRIORITY` dict (default 1.0 for unknown types) |

**Type priority weights:**

| Type | Priority |
|------|----------|
| decision | 1.2 |
| preference | 1.1 |
| procedure | 1.1 |
| fact | 1.0 |
| project_state | 1.0 |
| self_assessment | 1.0 |
| event | 0.9 |
| conversation | 0.7 |

Search results include both `_rrf_score` (raw RRF fusion score) and `_rerank_score` (blended final score). Results are sorted by `_rerank_score` descending, with ties broken by ascending memory ID.

**Python API:**

```python
from llmem.retrieve import Retriever

# Default blend (0.3): 70% RRF + 30% signals
retriever = Retriever(store, embedder=embedder)

# Pure RRF (disable reranking)
retriever = Retriever(store, embedder=embedder, blend=0.0)

# Equal weight RRF and signals
retriever = Retriever(store, embedder=embedder, blend=0.5)
```

## Important Notes

- **Invalidate, don't delete** unless the memory was wrong. Invalidated memories stay for reference but aren't returned in searches.
- **Embeddings** require Ollama running with `nomic-embed-text` pulled. If Ollama is down, `--no-embed` skips embeddings.
- **ANN vector index** — semantic search uses sqlite-vec (`vec0` virtual table) for fast ANN retrieval. In environments without sqlite-vec, pass `disable_vec=True` to `MemoryStore` to fall back to brute-force search.
- **Dimension configuration** — `MemoryStore(db_path=..., vec_dimensions=768)` sets the embedding dimensions for the vec0 index. Changing dimensions on an existing database raises `ValueError`. Default is 768 (nomic-embed-text).
- **Clearing embeddings** — use `MemoryStore.update(id, clear_embedding=True)` to remove an embedding (sets to NULL and syncs the vec0 index). Passing both `embedding=` and `clear_embedding=True` raises `ValueError`.
- **Extraction** uses `qwen2.5:1.5b` by default. It runs locally via Ollama.
- **Confidence** is 0.0-1.0. Higher = more certain. Facts from the user directly should be 0.9+, auto-extracted should be 0.7.
- **Context generation** is what gets injected into the system prompt for context. Use `llmem context` to preview what gets injected.
- **Auto-extraction** uses `llmem hook idle <session_id>` to extract memories from session transcripts. The idle hook processes the session's transcript, extracts memories, and runs introspection automatically. It uses the `extraction_log` table with `source_type='session'` to prevent re-extraction. The `auto_extract` flag in `~/.config/llmem/config.yaml` controls whether `llmem hook` runs extraction (defaults to true).
- **Auto-introspection** runs by default with `llmem hook idle`, producing `self_assessment` memories from each session transcript. For standalone introspection, use `llmem introspect --auto` or `--interactive` to analyze a single file. Both use `qwen2.5:1.5b` via Ollama and the `extraction_log` table with `source_type='introspection'` for deduplication. Use `--force` to re-analyze an already-processed file. Interactive mode presents the LLM analysis for user confirmation before storing.
- **Session hook** can be called from an agent's instructions or cron to make memory collection ambient — no need to remember `llmem extract` after every session.
- **Access tracking** — `llmem get` is read-only and does not update `access_count` or `accessed_at`. Use the Python API `MemoryStore.get(id, track_access=True)` or `MemoryStore.touch(id)` to record access. **Search operations (`Retriever.search()` and `Retriever.hybrid_search()`) now automatically track access** — each returned result's `access_count` and `accessed_at` are updated (best-effort, errors are logged not raised). This ensures the recency and access frequency reranking signals stay current.
- **Iteration count** — use `llmem introspect --iteration-count N` to record how many attempts before success. This feeds calibration tracking: if average iteration counts for a category decrease after a behavioral adaptation, the adaptation is marked effective.
- **Calibration status metadata** — procedure memories created by behavioral insights receive `calibration_status` (trend: `decreasing`, `stable`, or `increasing`) and `calibrated_at` metadata when calibration runs. Stale procedures get `stale_procedure: true` and `stale_at` metadata. These are visible via `llmem get <id>`.
- **Review outcome tracking** — `llmem track-review` persists review findings as `self_assessment` memories automatically. It is the mechanical post-review hook for adversarial code reviews. Three modes: single finding (`--category` + `--what-happened`), batch (`--finding-file` with JSON array), and clean review (no flags → `REVIEW_PASSED`). `--category` and `--finding-file` are mutually exclusive. Every review invocation MUST produce at least one memory — clean reviews create a `REVIEW_PASSED` memory automatically. Use `llmem suggest-categories <TIER>` to see valid categories for a severity tier (Blocking, Required, Strong Suggestions, Noted, Passed).


## Dream — Background Consolidation

The dream system is an automated memory maintenance pipeline that runs three phases:

- **Light** — finds near-duplicates using cosine similarity (configurable threshold, default 0.92). Produces merge candidates for the deep phase.
- **Deep** — decays idle memories (confidence decreases over time), boosts frequently accessed memories, promotes high-scoring memories, and merges near-duplicates using LLM-assisted merge with fallback to concatenation.
- **REM** — extracts themes and clusters from memory, writes a human-readable dream diary to `~/.config/llmem/dream-diary.md`. Self-assessment memories are grouped by error taxonomy category (e.g. "2 self_assessment memories about NULL_SAFETY") for pattern detection. When a category has 3+ occurrences (configurable via `skill_patch_threshold`), the REM phase generates three outputs: (1) a **procedural memory** (Tier 1 — automatic, low confidence), (2) a **behavioral insight** entry in `proposed-changes.md` (Tier 2 — human review), and (3) a **skill patch** entry in `proposed-changes.md` marked with `[SKILL PATCH]` (Tier 3 — human review). Skill patches are structured markdown snippets with Detection Rule, Checklist, Pitfall, and Verification sections that can be appended to existing skills or used as mini-skills. They are NOT auto-applied — they require human review and deployment. When behavioral insights are generated, **calibration tracking** compares self_assessment error rates (or average iteration counts) before and after each adaptation was introduced, marking them as effective (decreasing) or ineffective (stable/increasing). Procedure memories that are never accessed and older than `stale_procedure_days` are aggressively decayed (confidence reduced at double the normal decay rate). The dream diary includes a `### Calibration` section with per-category effectiveness and stale procedure counts.

**Default mode is dry-run** — use `--apply` to actually make changes. Without it, `llmem dream` only previews what would happen.

**Scheduling**: A systemd user timer (`llmem-dream.timer`) runs `llmem dream --apply` nightly at 3am. See `harness/llmem-dream.service` and `harness/llmem-dream.timer`.

**Dream config** lives under the `dream:` key in `~/.config/llmem/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `similarity_threshold` | 0.92 | Cosine similarity for near-duplicate detection |
| `decay_rate` | 0.05 | Confidence reduction per decay interval |
| `decay_interval_days` | 30 | Days per decay interval |
| `decay_floor` | 0.3 | Minimum confidence after decay |
| `confidence_floor` | 0.3 | Memories at or below this are invalidated |
| `boost_threshold` | 5 | Access count that triggers confidence boost |
| `boost_amount` | 0.05 | Confidence boost amount |
| `min_score` | 0.5 | Minimum promotion score |
| `min_recall_count` | 3 | Minimum access count for promotion |
| `min_unique_queries` | 1 | Minimum hint queries for promotion |
| `boost_on_promote` | 0.1 | Confidence boost when promoted |
| `merge_model` | qwen2.5:1.5b | Ollama model for LLM-assisted merge |
| `skill_patch_threshold` | 3 | Minimum self_assessment occurrences in a category to generate a skill patch (0 to disable) |
| `diary_path` | ~/.config/llmem/dream-diary.md | Path to dream diary file |
| `behavioral_threshold` | 3 | Minimum self_assessment occurrences to trigger behavioral insight |
| `behavioral_lookback_days` | 30 | Days of self_assessment memories for behavioral insights |
| `proposed_changes_path` | ~/.config/llmem/proposed-changes.md | Path for proposed-changes output |
| `calibration_enabled` | true | Whether to compute calibration metrics during REM phase |
| `stale_procedure_days` | 30 | Days without access before a procedure memory is considered stale |
| `calibration_lookback_days` | 90 | How far back to look for self_assessment memories when computing calibration |
