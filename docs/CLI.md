# LLMem CLI Reference

Complete reference for all `llmem` CLI commands and options. [Back to README](../README.md)

## CLI Reference

```
llmem [OPTIONS] COMMAND

Options:
  --db PATH    Path to memory database (default: ~/.config/llmem/memory.db)
  --json       Output results as JSON

Commands:
  add                 Add a memory
  get                 Get a memory by ID
  search              Search memories
  list                List memories
  stats               Show memory statistics
  update              Update a memory
  invalidate          Invalidate a memory (mark expired/wrong)
  delete              Delete a memory
  export              Export all memories to JSON
  import              Import memories from a JSON file
  init                Initialize the llmem memory system
  metrics             Report embedding quality metrics
  dream               Run the dream consolidation cycle
  introspect          Analyze a failure and store self_assessment memory
  learn               Learn a lesson from a wrong→right correction
  track-review        Persist review findings as self_assessment memories
  context             Inject relevant memory context for a session
  hook                Handle session lifecycle hook events
```

**Python-only commands** (not yet in the Go CLI): `register-type`, `types`, `note`, `inbox`, `consolidate`, `embed`, `learn` (codebase indexing), `suggest-categories`.

> The Go CLI uses `metrics` instead of `embed`, and `learn` is for wrong→right corrections (not codebase indexing).

### `llmem add`

```bash
llmem add --type TYPE --content TEXT [--summary TEXT] [--source SOURCE] \
  [--confidence FLOAT] [--valid-until TIMESTAMP] [--metadata JSON] \
  [--file PATH]
```

- `--type`: Memory type (default: `fact`).
- `--content` or `--file`: The memory text, or read content from a file. Files in protected system directories (e.g. `/etc`, `/bin`, `/usr/bin`) are blocked to prevent arbitrary file reads.
- `--summary`: Memory summary.
- `--source`: Source of the memory (default: `manual`).
- `--confidence`: Confidence score 0–1 (default: 0.8).
- `--valid-until`: ISO 8601 timestamp for validity expiration.
- `--metadata`: JSON metadata string.
- `--file`: Read content from a file path. The path is resolved and validated against system directory traversal.

### `llmem search`

```bash
llmem search QUERY [--type TYPE] [--limit N] [--json] [--valid-only] \
  [--fts-only | --semantic-only]
```

Hybrid search combining FTS5 keyword search and vector semantic search via Reciprocal Rank Fusion (RRF).

- `--type`: Filter by memory type.
- `--limit`: Maximum results (default: 20).
- `--json`: Output results as JSON.
- `--valid-only`: Only show valid (not invalidated) memories.
- `--fts-only`: Use FTS5 keyword search only (no embedder needed).
- `--semantic-only`: Use semantic (embedding) search only (requires an embedder).

Without `--fts-only` or `--semantic-only`, the default mode is FTS5-only (since the Go CLI currently sets `DisableVec: true`). Python supports full hybrid mode with `--include-code`, `--traverse-refs`, and `--provider` flags.

### `llmem list`

```bash
llmem list [--type TYPE] [--all] [--limit N]
```

By default, excludes expired memories. Use `--all` to include them. `--limit` defaults to 100.

### `llmem stats`

Shows total, active, and expired memory counts, plus a breakdown by type. Use `--json` for JSON output.

### `llmem update`

```bash
llmem update ID [--content TEXT] [--summary TEXT] [--confidence FLOAT] \
  [--valid-until TIMESTAMP] [--metadata JSON]
```

Only fields with `--` flags are updated; omitted fields remain unchanged.

### `llmem invalidate`

```bash
llmem invalidate ID [--reason TEXT]
```

Marks a memory as no longer valid, with an optional reason.

### `llmem delete`

```bash
llmem delete ID
```

Permanently removes a memory (and its embeddings, FTS index entries, relations).

### `llmem export` / `llmem import`

```bash
llmem export [--output FILE] [--limit N]
llmem import FILE
```

Export produces a JSON array of all memories (default limit: 10,000). Import validates that each entry has `type` and `content` string fields before inserting.

- `--output FILE`: Write export to a file. The output path is validated against traversal attacks.
- `--limit N`: Maximum memories to export (default: 10000).
- Import `FILE`: Path to a JSON file. Files are validated against system directory traversal. The file must be under 10 MiB.

### `llmem metrics`

```bash
llmem metrics
```

Report the count of embeddings stored in the database. In the Go CLI, this command outputs the number of embedded memories. For full embedding quality metrics (anisotropy, similarity range, discrimination gap), use the Python CLI's `llmem embed` command.

### `llmem introspect`

#### Manual mode

```bash
llmem introspect --what-happened TEXT [--category CATEGORY] [--context CONTEXT] \
  [--caught-by WHO] [--proposed-fix FIX] [--model MODEL] [--base-url URL]
```

Analyze a failure and store a `self_assessment` memory. Uses LLM expansion via Ollama when available, with graceful degradation to storage-only mode when Ollama is unavailable.

- `--what-happened` (required): Description of what went wrong.
- `--category`: Error taxonomy category (e.g., `NULL_SAFETY`, `ERROR_HANDLING`). See `taxonomy.ErrorTaxonomy` for all categories.
- `--context`: Context where the failure occurred (e.g., `handler.go:42`).
- `--caught-by`: How the finding was discovered (e.g., `self-review`, `CI`).
- `--proposed-fix`: Proposed fix for the issue.
- `--model`: LLM model for introspection (default: `glm-5.1:cloud`).
- `--base-url`: Ollama base URL for introspection (default: `http://localhost:11434`).

#### Automatic mode

```bash
llmem introspect --auto --session SESSION_ID [--model MODEL] [--base-url URL]
llmem introspect --auto --text TEXT [--model MODEL] [--base-url URL]
```

Automatically introspect a session transcript or arbitrary text and store a `self_assessment` memory. When Ollama is available, uses the LLM to expand the introspection into a richer assessment; when unavailable, stores the raw structured fields directly (graceful degradation).

- `--auto`: Enable automatic introspection mode.
- `--text`: Text to introspect. Use with `--auto`. When both `--text` and `--session` are provided, `--text` takes precedence.
- `--session`: Session ID to read transcript from (requires the OpenCode adapter). Use with `--auto`. The session ID is validated against path traversal.
- `--model`: LLM model for introspection (default: `glm-5.1:cloud`).
- `--base-url`: Ollama base URL for introspection (default: `http://localhost:11434`).

At least one of `--text` or `--session` is required when using `--auto`.

### `llmem learn`

```bash
llmem learn --wrong TEXT --right TEXT [--context CONTEXT]
```

Learn a lesson from a wrong→right correction and store it as a `procedure` memory. Uses LLM expansion via Ollama when available, with graceful degradation to storage-only mode.

- `--wrong` (required): What was wrong.
- `--right` (required): What is correct.
- `--context`: Additional context for the correction.

> **Note:** In the Python CLI, `llmem learn` ingests a codebase directory into the code index. In the Go CLI, `llmem learn` is for wrong→right lesson corrections.

### `llmem dream`

```bash
llmem dream [--apply] [--phase light|deep|rem] [--report PATH]
```

Run the dream consolidation cycle, which performs automated memory maintenance in three phases:

- **Light phase:** Sort and deduplicate near-duplicate memories (cosine similarity ≥ `dream.similarity_threshold`).
- **Deep phase:** Score, promote, decay, and merge memories. Also promotes inbox items to long-term memory (items with attention_score ≥ `dream.min_score` become permanent; lower-scored items are evicted). Decays confidence on idle memories. Boosts frequently accessed memories. Performs LLM-assisted merging of similar pairs. Auto-links memories with high cosine similarity (≥ `dream.auto_link_threshold`, default 0.85).
- **REM phase:** Extract themes from memory clusters and write a dream diary (read-only reflection). When Ollama is available, generates actionable behavioral insights via LLM with "Do" directives, "Verify" steps, and `[SKILL PATCH]` sections (Detection Rule, Checklist, Pitfall, Verification); falls back to count-based summaries when Ollama is unavailable. When run with `--apply`, also appends behavioral insight and skill patch sections to `proposed-changes.md` at `~/.config/llmem/proposed-changes.md` (or `LMEM_HOME/proposed-changes.md`). Each dream run's entries are separated by a timestamp header. The file is append-only — existing content is preserved.

Without `--apply`, the dream cycle runs as a **dry run** — output is prefixed with `[DRY RUN]` and no changes are written to the database.

Flags:

- `--apply`: Apply changes. Without this flag, the dream cycle runs as a dry run (no modifications to the database).
- `--phase`: Run a specific dream phase only. Choices: `light`, `deep`, `rem`. Default: all phases.
- `--report PATH`: Write an HTML dream report to the given path. The path is validated — it must not target a protected system directory (e.g. `/etc`, `/var`), contain `..` traversal, or be a symlink. Paths outside the llmem home directory are allowed (e.g. custom report output locations). On validation failure, prints an error to stderr and exits with code 1.

All dream configuration (thresholds, model, schedule, etc.) is read from the `dream:` section of `config.yaml` (see [Configuration](../docs/CONFIGURATION.md)). The `dream.ollama_url` and `dream.model` fields control Ollama connectivity for behavioral insight generation; they fall back to `memory.ollama_url` and `memory.extract_model` respectively if not set.

Output is printed to stdout. On `--report` path validation errors, the error message is printed to stderr.

> **Python-only commands:** The Python CLI includes additional commands not yet in the Go CLI: `register-type`, `types`, `note`, `inbox`, `consolidate`, `embed` (full embedding quality metrics), `learn` (codebase indexing), and `suggest-categories`.

### `llmem context`

```bash
llmem context --session-id ID [--compacting]
```

Inject relevant memory context for a session. Used by session hooks to inject memories into a new or compacting session.

- `--session-id` (required): The session ID to inject context for. Validated against path traversal attacks.
- `--compacting`: If set, inject key memories for compaction instead of session start context. High-confidence memories of types `decision`, `preference`, `procedure`, and `project_state` (confidence ≥ 0.7) are selected.

### `llmem hook`

```bash
llmem hook --type TYPE --session-id ID [--model MODEL] [--base-url URL]
```

Handle session lifecycle hook events. Supports four hook types:

- `--type` (required): Hook type. Choices: `created`, `idle`, `compacting`, `ending`.
- `--session-id` (required): Session ID for the hook event. Validated against path traversal attacks.
- `--model`: LLM model for introspection (default: `glm-5.1:cloud`). Used by the `ending` hook for automatic introspection.
- `--base-url`: Ollama base URL for introspection (default: `http://localhost:11434`). Used by the `ending` hook for automatic introspection.

The `idle` hook processes the session's transcript, extracts memories via the extraction pipeline (chunk → dedup → LLM extract → embed → store), and generates embedding vectors for each extracted memory. It uses a debounce mechanism (via `extraction_log` table) to prevent re-extraction. When `ExtractionEngine` is not configured, extraction is skipped gracefully.

The `ending` hook extracts memories from the transcript (same pipeline as `idle`), then runs `IntrospectTranscript` to produce a session-end `self_assessment` memory. When the LLM is unavailable, `IntrospectTranscript` falls back to a degraded plain-text summary of the session (no LLM call attempted).

The `ending` hook performs automatic introspection on the session transcript. It reads the transcript via the configured adapter, generates a `self_assessment` memory using `IntrospectAuto`, and outputs the result type and memory ID. If no adapter is configured or the transcript is empty, it returns `no_transcript`. If introspection fails but the transcript was read, it logs a warning and returns success without crashing the ending event.

### `llmem track-review`

```bash
llmem track-review --single --findings PATH
llmem track-review --batch --findings PATH
llmem track-review --clean
```

Persist review findings as `self_assessment` memories. Three modes:

1. **Single finding** (`--single`): Store one finding from a findings file or stdin.
2. **Batch** (`--batch`): Store multiple findings from a findings file or stdin. Each line is parsed as `Category: value` and stored.
3. **Clean** (`--clean`): Invalidate all existing `self_assessment` memories with `source=track-review`.

- `--single`: Store a single finding.
- `--batch`: Store multiple findings.
- `--clean`: Invalidate all track-review memories.
- `--findings`: Path to findings file (or stdin). The path is validated against system directory traversal.

Every invocation with `--single` or `--batch` parses lines from input. Lines with unknown categories still produce memories with the parsed category. Empty input or unknown categories produce a `REVIEW_PASSED` memory.