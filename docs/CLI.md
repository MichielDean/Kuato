# LLMem CLI Reference

Complete reference for all `llmem` CLI commands and options. [Back to README](../README.md)

## CLI Reference

```
llmem [OPTIONS] COMMAND

Options:
  --db PATH    Path to memory database (default: ~/.config/llmem/memory.db)

Commands:
  add                 Add a memory
  get                 Get a memory by ID
  search              Search memories (hybrid RRF fusion of FTS5 + semantic)
  list                List memories
  stats               Show memory statistics
  update              Update a memory
  invalidate          Invalidate a memory (mark expired/wrong)
  delete              Delete a memory
  export              Export all memories to JSON
  import              Import memories from a JSON file
  register-type       Register a new memory type
  types               List registered memory types
  note                Add a note to the working memory inbox
  inbox               List items in the working memory inbox
  embed               Report embedding quality metrics
  consolidate         Promote inbox items to long-term memory
  dream               Run the dream consolidation cycle
  init                Initialize the llmem memory system
  learn               Ingest a codebase into the code index
  context             Inject relevant memory context for a session
  hook                Handle session lifecycle hook events
  track-review        Persist review findings as self_assessment memories
  suggest-categories  List error taxonomy categories for a severity tier
```

Plugins registered via [`register_cli_plugin`](#cli-plugin-registry) add additional subcommands at runtime.

**Backward compatibility:** The `lobmem` command is installed as a symlink to `llmem`. When invoked as `lobmem`, a deprecation warning is printed to stderr: `warning: 'lobmem' is deprecated, use 'llmem'`. This wrapper will be removed in a future version.

### `llmem add`

```bash
llmem add --type TYPE --content TEXT [--summary TEXT] [--source SOURCE] \
  [--confidence FLOAT] [--valid-until TIMESTAMP] [--metadata JSON] \
  [--relation TYPE --relation-to ID]
```

- `--type` (required): Memory type. Use `llmem types` to list registered types, or `llmem register-type` to add new ones.
- `--content` or `--file`: The memory text (or read from a file). Files in protected system directories (e.g. `/etc`, `/bin`, `/usr/bin`) are blocked to prevent arbitrary file reads.
- `--source`: Source of the memory (default: `manual`). Valid: `manual`, `session`, `heartbeat`, `extraction`, `import`.
- `--confidence`: Confidence score 0–1 (default: 0.8).
- `--relation` / `--relation-to`: Create a relation to another memory after adding.

### `llmem search`

```bash
llmem search QUERY [--type TYPE] [--limit N] [--json] [--include-code] \
  [--traverse-refs] [--max-ref-depth N] [--fts-only | --semantic-only] [--provider PROVIDER]
```

Hybrid search combining FTS5 keyword search and vector semantic search via Reciprocal Rank Fusion (RRF), followed by multi-signal reranking. By default, both search modes are merged with `alpha=0.7` (favoring semantic results while keeping keyword relevance), then reranked with `blend=0.3` (70% semantic, 30% confidence/recency/access/type signals).

- `--provider PROVIDER`: Override the embedding provider for this search. Choices: `ollama`, `openai`, `local`, `none`. When specified, `resolve_provider` is called with this provider as default, bypassing config-based selection. Falls back to FTS5-only search if the provider fails.
- `--fts-only`: Use FTS5 keyword search only (no embedder needed).
- `--semantic-only`: Use semantic (embedding) search only (requires an embedder). Raises an error if no embedder is available.
- `--include-code`: Include indexed code chunks in search results alongside memories. Code results are interleaved with memory results using RRF scoring. Code results display with a `[code]` prefix and show `file=` and `lines=` instead of type and confidence. Requires running `llmem learn` first to populate the code index.
- `--traverse-refs`: Follow code reference edges from search results. When a memory has a `references` relation to a code chunk (`target_type='code'`), the referenced file content is resolved and included in the results. Code ref results have `_source: "code"` and include `file_path`, `start_line`, `end_line`, and `content` keys. Security: code refs must use relative paths (no `/` prefix or `..` traversal) and resolve under the current working directory.
- `--max-ref-depth N`: Maximum number of hops when traversing code reference edges (1–5, default: 3). Higher values follow ref chains deeper (e.g., depth 2 finds memories sharing a code ref, depth 3 finds their code refs).
- Without either `--fts-only` / `--semantic-only` flag: hybrid mode — runs both FTS5 and semantic search, fuses results via RRF, then applies reranking. Falls back to FTS5-only if no embedder is configured.

With `--json`, outputs raw JSON (each result includes `_rrf_score` and `_rerank_score` keys, plus a `_source` key of `"memory"` or `"code"` when `--include-code` is used); otherwise, a human-readable table with an `rrf=` score column.

### `llmem list`

```bash
llmem list [--type TYPE] [--all] [--limit N]
```

By default, excludes expired memories. Use `--all` to include them.

### `llmem stats`

Shows total, active, and expired memory counts, plus a breakdown by type.

### `llmem update`

```bash
llmem update ID [--content TEXT] [--summary TEXT] [--confidence FLOAT] \
  [--valid-until TIMESTAMP] [--metadata JSON]
```

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
llmem export [--output FILE]
llmem import FILE
```

Export produces a JSON array of all memories. Import validates that each entry has `type` and `content` string fields before inserting.

- `--output FILE`: Write export to a file. The output path is validated against traversal attacks, system directories, and symlinks.
- Import `FILE`: Path to a JSON file. Files in protected system directories are blocked, and the file must be under 10 MiB.

### `llmem note`

```bash
llmem note TEXT [--source SOURCE] [--attention-score FLOAT] [--metadata JSON]
```

Add an ephemeral note to the working memory inbox. Notes are staged and not yet in long-term memory.

- `TEXT` (required): Note content text.
- `--source`: Origin of the note. One of `note`, `learn`, `extract`, `consolidation`. Default: `note`.
- `--attention-score`: Float in [0.0, 1.0]. Higher scores survive eviction and are prioritized during consolidation. Default: `0.5`.
- `--metadata`: Optional JSON metadata string.

The inbox has a configurable capacity (default: 7, Miller's 7±2). When full, the item with the lowest attention score is evicted (tiebreak: earliest `created_at`).

### `llmem inbox`

```bash
llmem inbox [--limit N] [--json]
```

List items in the working memory inbox, ordered by attention score descending.

- `--limit`: Maximum items to display. Default: 20.
- `--json`: Output as JSON (full item details).

Without `--json`, prints a human-readable table with ID, source, score, content preview, and timestamp.

### `llmem embed`

```bash
llmem embed
```

Report embedding quality metrics for existing embeddings. Computes anisotropy, similarity range, and discrimination gap from the embeddings already stored in the database. Does **not** generate new embeddings — only analyses existing ones.

Since the sole purpose of the `embed` subcommand is to report metrics, they are always reported (no flag needed).

Output includes:

- **Anisotropy**: Measures vector uniformity. Lower is better (more isotropic). Value in [0.0, 1.0].
- **Similarity range**: Spread between max and min pairwise cosine similarity. Higher is better.
- **Discrimination gap**: Average inter-class cosine distance minus average intra-class cosine distance. Higher is better.

Warnings are printed to stderr when anisotropy > 0.5 or similarity range < 0.1, suggesting embeddings may be poor quality and a different model should be considered.

On large stores (>10,000 embeddings), metrics are computed on a capped sample to prevent O(n²) CPU hangs. A note is printed showing the capped count.

### `llmem consolidate`

```bash
llmem consolidate [--min-score FLOAT] [--dry-run] [--metrics]
```

Promote inbox items to long-term memory. Items with attention score ≥ `min_score` become permanent memories (with `source=consolidation` and `confidence=attention_score`). Items below the threshold are evicted. After consolidation, the inbox is empty.

- `--min-score`: Minimum attention score threshold for promotion. Items below this are evicted. Default: `0.0` (promote everything).
- `--dry-run`: Show what would happen without making changes. Promoted items won't have a `memory_id`, and no rows are inserted or deleted.
- `--metrics`: Compute and report embedding quality metrics (anisotropy, similarity range, discrimination gap) after consolidation. Useful for checking embedding health after memories have been promoted.

### `llmem dream`

```bash
llmem dream [--apply] [--phase light|deep|rem] [--report PATH]
```

Run the dream consolidation cycle, which performs automated memory maintenance in three phases:

- **Light phase:** Sort and deduplicate near-duplicate memories (cosine similarity ≥ `dream.similarity_threshold`).
- **Deep phase:** Score, promote, decay, and merge memories. Also promotes inbox items to long-term memory (items with attention_score ≥ `dream.min_score` become permanent; lower-scored items are evicted). Decays confidence on idle memories. Boosts frequently accessed memories. Performs LLM-assisted merging of similar pairs. Auto-links memories with high cosine similarity (≥ `dream.auto_link_threshold`, default 0.85).
- **REM phase:** Extract themes from memory clusters and write a dream diary (read-only reflection).

Without `--apply`, the dream cycle runs as a **dry run** — output is prefixed with `[DRY RUN]` and no changes are written to the database.

Flags:

- `--apply`: Apply changes. Without this flag, the dream cycle runs as a dry run (no modifications to the database).
- `--phase`: Run a specific dream phase only. Choices: `light`, `deep`, `rem`. Default: all phases.
- `--report PATH`: Write an HTML dream report to the given path. The path is validated — it must not target a protected system directory (e.g. `/etc`, `/var`), contain `..` traversal, or be a symlink. Paths outside the llmem home directory are allowed (e.g. custom report output locations). On validation failure, prints an error to stderr and exits with code 1.

All dream configuration (thresholds, model, schedule, etc.) is read from the `dream:` section of `config.yaml` (see [Configuration](../docs/CONFIGURATION.md)). The `ollama_url` is read from the `memory:` section, not `dream:`.

Output is printed to stdout. On `--report` path validation errors, the error message is printed to stderr.

### `llmem register-type` / `llmem types`

```bash
llmem register-type my_custom_type
llmem types
```

The default types are: `fact`, `decision`, `preference`, `event`, `project_state`, `procedure`, `conversation`, `self_assessment`. Register additional types at runtime via `register-type`. Type names must match `^[a-z][a-z0-9_]*$` and be at most 64 characters.

### `llmem init`

```bash
llmem init [--ollama-url URL] [--non-interactive] [--force]
```

Initialize the llmem memory system. Creates `~/.config/llmem/` (or `LMEM_HOME`) with `config.yaml` and initializes the SQLite database (`memory.db`). Detects available LLM providers in order of precedence: Ollama (if reachable) > OpenAI (if `OPENAI_API_KEY` is set) > Anthropic (if `ANTHROPIC_API_KEY` is set) > local (if `sentence-transformers` is installed) > none.

- `--ollama-url URL`: Override the Ollama base URL (default: `http://localhost:11434`). Must be a valid `http://` or `https://` URL.
- `--non-interactive`: Skip all prompts and use defaults. Useful for scripting and CI.
- `--force`: Overwrite an existing `config.yaml`. Without this flag, init is idempotent — running it twice prints a message and exits without error.

In interactive mode, you'll be prompted for:

1. **Ollama URL** — press Enter to accept the detected/default URL, or type a custom URL (validated for safety).
2. **Dream cycle** — enable or disable the background dream cycle (default: enabled).

If `~/.lobsterdog/` exists (legacy path), init automatically migrates data to `~/.config/llmem/`.

### `llmem learn`

```bash
llmem learn PATH [--strategy paragraph|fixed] [--window-size N] [--overlap N] \
  [--no-embed] [--max-file-size N] [--max-depth N] [--ollama-url URL]
```

Ingest a codebase directory into the code index. Walks the directory tree respecting `.gitignore` files, chunks each source file, generates embeddings, and stores the results in a `code_chunks` table (shared database with memories). Running `llmem learn` on the same directory is idempotent — stale chunks for each file are removed before re-inserting.

- `PATH` (required): Root directory to ingest.
- `--strategy`: Chunking strategy — `paragraph` (split at blank-line boundaries, merges short paragraphs) or `fixed` (sliding window with overlap). Default: `paragraph`.
- `--window-size`: Window size in lines for `fixed` chunking strategy. Default: 50.
- `--overlap`: Overlap in lines between consecutive chunks for `fixed` strategy. Must be less than `--window-size`. Default: 10.
- `--no-embed`: Skip embedding generation. Chunks are stored with text only (searchable via FTS5, but not via semantic/vec search). Useful for quick indexing without an embedding provider.
- `--max-file-size`: Maximum file size in bytes to index. Files exceeding this limit are skipped. Default: 1048576 (1 MiB).
- `--max-depth`: Maximum directory recursion depth. Prevents stack overflow from deeply nested trees. Default: 50.
- `--ollama-url`: Ollama base URL for embedding generation. Default: `http://localhost:11434`.

**Skipped files:** Binary files (images, fonts, archives, compiled objects), common non-code directories (`.git`, `__pycache__`, `node_modules`, `.venv`), credential files (`.env`, `.env.*`, `.pem`, `.key`, `id_rsa`, `id_dsa`, `id_ed25519`, `id_ecdsa`, `.netrc`, `.htpasswd`, `.npmrc`, `.pypirc`), and symlinks (to prevent path traversal).

**Language detection:** File extensions are mapped to language names automatically (e.g., `.py` → `python`, `.rs` → `rust`, `.go` → `go`). Unknown extensions produce `None`.

**Output:** `Ingested N chunks from M files`

### `llmem context`

```bash
llmem context SESSION_ID [--compacting]
```

Inject relevant memory context for a session. Used by session hooks (e.g., Copilot CLI, OpenCode plugins) to inject memories into a new or compacting session. Writes a context file and prints its content to stdout.

- `SESSION_ID` (required): The session ID to inject context for. Validated against path traversal attacks (rejects `/`, `\`, `..`).
- `--compacting`: If set, inject key memories for compaction instead of session start context. High-confidence memories of types `decision`, `preference`, `procedure`, and `project_state` (confidence ≥ 0.7) are selected.

On success, prints the context content to stdout. On error, prints to stderr and exits with code 1. If the session was already processed, prints nothing (for session start) or the existing context file (for re-invocations).

### `llmem hook`

```bash
llmem hook idle SESSION_ID
```

Handle session lifecycle hook events. Currently supports only the `idle` hook type, which triggers memory extraction and introspection for a session.

- `hook_type` (required): The hook type to dispatch. Only `idle` is supported.
- `SESSION_ID` (required): The session ID for the hook event. Validated against path traversal attacks.

The idle hook processes the session's transcript, extracts memories, and runs introspection automatically. It uses the `extraction_log` table with `source_type='session'` to prevent re-extraction. If the session was recently processed (debounce), it logs a debug message and exits normally.

### `llmem track-review`

```bash
llmem track-review --context CONTEXT  [--category CATEGORY --what-happened WHAT] [--severity SEVERITY] [--caught-by WHO]
llmem track-review --finding-file FILE --context CONTEXT
llmem track-review --context CONTEXT
```

Persist review findings as `self_assessment` memories. This is the mechanical post-review hook for adversarial code reviews. Three modes:

1. **Single finding**: `--category` + `--what-happened` (optionally `--severity`, `--caught-by`)
2. **Batch from file**: `--finding-file` (JSON array of finding objects, each with `category`, `what_happened`, optional `severity` and `caughtBy`)
3. **Clean review**: no flags → creates a `REVIEW_PASSED` memory

- `--context` (recommended): File or task identifier (e.g., `handler.py:42`).
- `--category`: Error taxonomy category for a single finding (e.g., `NULL_SAFETY`, `ERROR_HANDLING`). See `llmem suggest-categories` for categories per severity tier.
- `--what-happened`: Behavioral description of the finding.
- `--severity`: Severity tier (`Blocking`, `Required`, `Strong Suggestions`, `Noted`).
- `--caught-by`: How the finding was discovered (e.g., `self-review`, `CI`). Defaults to `self-review` for single findings, and `self-review` for batch findings where `caughtBy` is not specified in the JSON.
- `--finding-file`: Path to a JSON file containing an array of finding objects. Mutually exclusive with `--category`.

`--category` and `--finding-file` are mutually exclusive. Every invocation MUST produce at least one memory — clean reviews create a `REVIEW_PASSED` memory automatically. An empty JSON array in batch mode also creates a `REVIEW_PASSED` memory.

### `llmem suggest-categories`

```bash
llmem suggest-categories TIER
```

List the error taxonomy categories applicable to a severity tier. Useful for determining which categories to use when tracking review findings.

- `TIER` (required): Severity tier. Choices: `Blocking`, `Required`, `Strong Suggestions`, `Noted`, `Passed`.

Output: One category per line. Example:

```
$ llmem suggest-categories Required
NULL_SAFETY
ERROR_HANDLING
MISSING_VERIFICATION
EDGE_CASE
```