# LLMem — Structured Memory with Semantic Search

[![CI](https://github.com/MichielDean/LLMem/actions/workflows/ci.yml/badge.svg)](https://github.com/MichielDean/LLMem/actions/workflows/ci.yml)

LLMem is a SQLite-backed memory store for LLM agents. It provides structured storage with full-text search (FTS5), optional vector similarity search (via sqlite-vec), an extensible type system, and a provider abstraction layer for LLM embeddings and text generation. A background dreaming cycle consolidates, decays, and merges memories over time.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and coding conventions.

## Installation

```bash
pip install llmem
```

Requires Python 3.11+ and [PyYAML](https://pypi.org/project/PyYAML/).

This copies all 10 skills into `~/.agents/skills/`, where OpenCode discovers them automatically.

## Skills

| Skill | Description |
|-------|-------------|
| **llmem** | Manage LLMem memories — add, search, consolidate, dream, introspect, and track review outcomes. |
| **introspection** | Operational reference for the introspection framework — self-assessment, sampajanna checks, error taxonomy. |
| **introspection-review-tracker** | Reference for the automated ReviewOutcomeTracker hook that persists review findings as self_assessment memories. |
| **git-sync** | Sync git repos before editing. Fetch, rebase, stash, and detect conflicts. |
| **task-intake** | Discover project stack, test commands, and conventions before making changes. |
| **test-and-verify** | Run quality gates (format, lint, typecheck, test) after code changes. |
| **branch-strategy** | Enforce consistent branching, commit messages, and push strategies. |
| **critical-code-reviewer** | Adversarial code reviews with zero tolerance for mediocrity. |
| **pre-pr-review** | Pre-PR code review via isolated subagent before pushing to GitHub. |
| **visual-explainer** | Generate self-contained HTML diagrams, reviews, and slide decks. |

## Templates

The `templates/` directory contains generic, personality-agnostic template files that you can copy and customize for your own agent setup:

| Template | Purpose |
|----------|---------|
| **templates/rules.md** | Generic workflow rules — no personal or tool-specific references |
| **templates/identity.md** | Agent identity scaffold — fill in your agent's name, personality, and boundaries |
| **templates/user.md** | User profile scaffold — fill in your name, timezone, and preferences |

To use them, copy the templates into your `harness/` directory and fill them in:

```bash
cp templates/identity.md harness/identity.md
cp templates/user.md harness/user.md
cp templates/rules.md harness/rules.md
# Then edit each file to personalize for your setup
```

The `opencode.json` configuration loads `harness/identity.md`, `harness/user.md`, and `harness/rules.md` — so after copying and customizing, your agent will use your personalized versions.

## Verification

After installation, verify skills are discoverable:

For vector similarity search, install with the `vec` extra:

```bash
pip install llmem[vec]
```

This pulls in `sqlite-vec>=0.1.6`.

For local embedding without any server (uses `sentence-transformers`), install with the `local` extra:

```bash
pip install llmem[local]
```

This pulls in `sentence-transformers>=2.2.0`. You can combine extras:

```bash
pip install llmem[vec,local]
```

For development:

```bash
pip install ".[dev]"
```

### Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally for extraction, embedding, and dreaming features — **or** install `llmem[local]` for local embedding without any server, **or** set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` for cloud providers

## Provider Abstraction Layer

LLMem decouples embedding and text generation from any specific LLM backend through two abstract base classes:

| Protocol | Methods | Purpose |
|----------|---------|---------|
| `EmbedProvider` | `embed()`, `embed_batch()`, `check_available()`, `dimension()` | Vector embeddings for semantic search |
| `GenerateProvider` | `generate()`, `check_available()` | Text generation from prompts |

### Concrete Providers

| Provider | Embed | Generate | API Key | Default Base URL |
|----------|-------|----------|---------|-------------------|
| `OllamaProvider` | Yes | Yes | None | `http://localhost:11434` |
| `OpenAIProvider` | Yes | Yes | `OPENAI_API_KEY` or constructor arg | `https://api.openai.com` |
| `AnthropicProvider` | No | Yes | `ANTHROPIC_API_KEY` or constructor arg | `https://api.anthropic.com` |
| `SentenceTransformersProvider` | Yes | No | None | N/A (local) |
| `NoneProvider` | Yes (zeros) | Yes (empty string) | None | N/A |

### Graceful Degradation

`resolve_provider(config)` returns the best available `(embed_provider, generate_provider)` pair by trying providers in order:

**Embed fallback chain:** Ollama → OpenAI → SentenceTransformers (local) → NoneProvider (zero vectors, FTS5-only mode)

**Generate fallback chain:** Ollama → OpenAI → Anthropic → NoneProvider (empty string)

Each provider's `check_available()` returns `False` on any error (never raises), so degradation is automatic.

### Dimension Reporting

All `EmbedProvider` subclasses implement `dimension() -> int`, which returns the output vector dimensionality without making API calls or loading models:

| Provider | Default Dimension | Known Model Overrides |
|----------|-------------------|----------------------|
| `OllamaProvider` | 768 | `mxbai-embed-large` → 1024, `all-minilm` → 384, `snowflake-arctic-embed` → 1024 |
| `OpenAIProvider` | 1536 | `text-embedding-3-large` → 3072, `text-embedding-ada-002` → 1536 |
| `SentenceTransformersProvider` | 384 | `all-mpnet-base-v2` → 768, `all-roberta-large-v1` → 1024, and 6 more (see `_KNOWN_LOCAL_DIMENSIONS`) |
| `NoneProvider` | 768 | Configurable via `embed_dimensions` constructor arg |

```python
provider = SentenceTransformersProvider(model_name="all-MiniLM-L6-v2")
dim = provider.dimension()  # 384 — no model loading, no API call
```

### Quick Start

```python
from memory.providers import resolve_provider

embed, generate = resolve_provider(config={})
# With Ollama running: OllamaProvider for both
# Without Ollama, with OPENAI_API_KEY set: OpenAIProvider for both
# Without any server, with sentence-transformers installed: SentenceTransformersProvider for embed
# Without any provider: NoneProvider for both

vec = embed.embed("hello world")
dim = embed.dimension()  # e.g. 768 for nomic-embed-text, 384 for all-MiniLM-L6-v2
text = generate.generate("Summarize this document")
```

### Direct Construction

```python
from memory.providers import OllamaProvider, OpenAIProvider, AnthropicProvider, SentenceTransformersProvider, NoneProvider

# Ollama (local, no API key needed)
ollama = OllamaProvider(
    embed_model="nomic-embed-text",
    generate_model="qwen2.5:1.5b",
    base_url="http://localhost:11434",
    timeout=60,
)

# OpenAI
openai = OpenAIProvider(
    embed_model="text-embedding-3-small",
    generate_model="gpt-4o-mini",
    api_key="sk-...",  # or set OPENAI_API_KEY env var
    base_url="https://api.openai.com",
)

# Anthropic (generation only, no embedding API)
anthropic = AnthropicProvider(
    model="claude-sonnet-4-20250514",
    api_key="sk-ant-...",  # or set ANTHROPIC_API_KEY env var
)

# SentenceTransformers (local, no server needed)
local = SentenceTransformersProvider(
    model_name="all-MiniLM-L6-v2",  # default, runs locally
)
vec = local.embed("hello world")
dim = local.dimension()  # 384 for all-MiniLM-L6-v2

# NoneProvider (FTS5-only fallback)
none = NoneProvider(embed_dimensions=768)
```

### Per-Operation Overrides

The embed and generate providers can be configured independently:

```yaml
provider:
  default: ollama
  embed:
    provider: openai
    model: text-embedding-3-small
  generate:
    provider: anthropic
    model: claude-sonnet-4-20250514
  local:
    model: all-MiniLM-L6-v2
```

This yields `OpenAIProvider` for embeddings and `AnthropicProvider` for generation. Setting `provider: local` for embed uses `SentenceTransformersProvider` for local embedding without any server dependency.

## Provider Configuration

Provider-related keys in `config.yaml`:

```yaml
provider:
  default: ollama              # ollama | openai | anthropic | local | none
  ollama:
    base_url: http://localhost:11434
  openai:
    api_key: sk-...             # or set OPENAI_API_KEY env var
    base_url: https://api.openai.com
    embed_model: text-embedding-3-small
    generate_model: gpt-4o-mini
  anthropic:
    api_key: sk-ant-...         # or set ANTHROPIC_API_KEY env var
    base_url: https://api.anthropic.com
    generate_model: claude-sonnet-4-20250514
  local:
    model: all-MiniLM-L6-v2    # sentence-transformers model name (local, no server)
```

Both config-based and environment variable API keys are supported; config keys take precedence.

When `provider: local` is selected for embeddings, LLMem uses `SentenceTransformersProvider` which runs models locally via the `sentence-transformers` library — no server required. Install with `pip install llmem[local]`.

## Quick Start

```bash
# Initialize the memory system (creates config, database, detects providers)
llmem init

# Non-interactive mode (skip prompts, use defaults)
llmem init --non-interactive

# Add a memory
llmem add --type fact --content "Project uses pytest for testing"

# Search memories
llmem search "testing"
llmem search "testing" --type fact --limit 5 --json
llmem search "testing" --include-code --json

# Index a codebase
llmem learn ./src
llmem learn ./src --strategy fixed --window-size 30 --overlap 5
llmem learn ./src --no-embed

# List all memories
llmem list
llmem list --type decision --all

# Get a specific memory
llmem get <memory-id>

# Show statistics
llmem stats

# Register a custom memory type
llmem register-type my_custom_type
llmem types

# Working memory inbox
llmem note "Important observation from today's session"
llmem note "Tentative insight" --attention-score 0.3
llmem inbox
llmem consolidate --dry-run
llmem consolidate --min-score 0.5

# Dream cycle (automated memory maintenance)
llmem dream                   # Dry run — preview changes only
llmem dream --apply           # Apply changes
llmem dream --phase deep      # Run only the deep phase
llmem dream --report dream.html  # Generate HTML dream report

# Check embedding quality
llmem embed
llmem consolidate --metrics

# Export and import
llmem export --output backup.json
llmem import backup.json
```

## General Configuration

LLMem looks for configuration at `~/.config/llmem/config.yaml`. If this file doesn't exist, sensible defaults are used.

### Path Resolution

| Path | Default | Override |
|------|---------|----------|
| Home directory | `~/.config/llmem/` | `LMEM_HOME` env var |
| Database | `~/.config/llmem/memory.db` | `config.yaml: memory.db` |
| Config file | `~/.config/llmem/config.yaml` | — |
| Dream diary | `~/.config/llmem/dream-diary.md` | `config.yaml: dream.diary_path` |
| Proposed changes | `~/.config/llmem/proposed-changes.md` | `config.yaml: dream.proposed_changes_path` |

**Backward compatibility:** If `~/.lobsterdog/` exists and `~/.config/llmem/` doesn't, LLMem will use the legacy path. Call `migrate_from_lobsterdog()` to copy data to the new location.

**`LMEM_HOME` env var:** Set this to override the home directory entirely. The path is validated against directory traversal, system directories, and symlink attacks.

### config.yaml Reference

```yaml
memory:
  context_budget: 4000
  auto_extract: true
  max_file_size: 10485760     # 10MB
  inbox_capacity: 7            # Miller's 7±2; items above this count are evicted

dream:
  enabled: true
  schedule: "*-*-* 03:00:00"
  similarity_threshold: 0.92
  decay_rate: 0.05
  decay_interval_days: 30
  decay_floor: 0.3
  confidence_floor: 0.3
  boost_threshold: 5
  boost_amount: 0.05
  min_score: 0.5
  min_recall_count: 3
  min_unique_queries: 1
  boost_on_promote: 0.1
  merge_model: qwen2.5:1.5b
  diary_path: null            # Auto-resolved from get_dream_diary_path()
  report_path: ~/.agent/diagrams/dream-report.html  # Must not target system dirs, '..' traversal, or symlinks
  behavioral_threshold: 3
  behavioral_lookback_days: 30
  proposed_changes_path: null # Auto-resolved from get_proposed_changes_path()
  calibration_enabled: true
  stale_procedure_days: 30
  calibration_lookback_days: 90
  auto_link_threshold: 0.85  # Cosine similarity threshold for auto-linking related memories

opencode:
  context_dir: null           # Auto-resolved from get_context_dir()
  db_path: ~/.local/share/opencode/opencode.db

correction_detection:
  enabled: true
```

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
- **Deep phase:** Score, promote, decay, and merge memories. Also promotes inbox items to long-term memory (items with attention_score ≥ `dream.min_score` become permanent; lower-scored items are evicted). Decays confidence on idle memories, boosts frequently accessed memories, performs LLM-assisted merging of similar pairs, and auto-links memories with high cosine similarity (≥ `dream.auto_link_threshold`, default 0.85).
- **REM phase:** Extract themes from memory clusters and write a dream diary (read-only reflection).

Without `--apply`, the dream cycle runs as a **dry run** — output is prefixed with `[DRY RUN]` and no changes are written to the database.

Flags:

- `--apply`: Apply changes. Without this flag, the dream cycle runs as a dry run (no modifications to the database).
- `--phase`: Run a specific dream phase only. Choices: `light`, `deep`, `rem`. Default: all phases.
- `--report PATH`: Write an HTML dream report to the given path. The path is validated — it must not target a protected system directory (e.g. `/etc`, `/var`), contain `..` traversal, or be a symlink. Paths outside the llmem home directory are allowed (e.g. custom report output locations). On validation failure, prints an error to stderr and exits with code 1.

All dream configuration (thresholds, model, schedule, etc.) is read from the `dream:` section of `config.yaml` (see [Configuration](#general-configuration)). The `ollama_url` is read from the `memory:` section, not `dream:`.

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

## Python API

```python
from llmem import (
    MemoryStore,
    register_memory_type,
    get_config_path,
    get_db_path,
    get_llmem_home,
    migrate_from_lobsterdog,
    load_config,
    validate_session_id,
    SessionAdapter,
    OpenCodeAdapter,
    register_session_adapter,
    register_session_hook,
    get_registered_session_hooks,
    register_dream_hook,
    register_cli_plugin,
)
from llmem.retrieve import Retriever, _rrf_score, DEFAULT_ALPHA, DEFAULT_RRF_K
from llmem.metrics import (
    compute_metrics,
    anisotropy,
    similarity_range,
    discrimination_gap,
    cosine_similarity,
    bytes_to_vec,
    EmbeddingMetrics,
    ANISOTROPY_WARNING_THRESHOLD,
    SIMILARITY_RANGE_WARNING_THRESHOLD,
    METRICS_MAX_EMBEDDINGS,
)
from llmem.config import write_config_yaml
from llmem.ollama import ProviderDetector, is_ollama_running

# Open a store
store = MemoryStore()  # uses default path ~/.config/llmem/memory.db

# Add a memory
mid = store.add(type="fact", content="Project uses SQLite with WAL mode")

# FTS5 search (classic)
results = store.search("SQLite", limit=10)

# Hybrid search (FTS5 + semantic, RRF fusion)
from llmem.retrieve import Retriever
from llmem.embed import EmbeddingEngine

embedder = EmbeddingEngine()
retriever = Retriever(store=store, embedder=embedder)

# Or use any EmbedProvider (e.g. from resolve_provider):
from memory.providers import resolve_provider
embed_provider, _ = resolve_provider({"provider": {"default": "local"}})
retriever = Retriever(store=store, embedder=embed_provider)

# Default: hybrid mode (alpha=0.7, favors semantic), reranking blend=0.3
results = retriever.hybrid_search("Python async patterns", limit=10)

# FTS5-only (no embedder needed)
results = retriever.hybrid_search("Python async patterns", search_mode="fts")

# Semantic-only (requires embedder)
results = retriever.hybrid_search("Python async patterns", search_mode="semantic")

# Control semantic vs. keyword weight (0.0 = pure FTS, 1.0 = pure semantic)
results = retriever.hybrid_search("query", alpha=0.5)

# Control reranking blend (0.0 = pure RRF, 1.0 = pure signal-based)
# blend=0.3 default: 70% RRF score + 30% weighted signals (confidence, recency, access, type)
retriever = Retriever(store=store, embedder=embedder, blend=0.5)

# Restrict code ref resolution to specific directories (default: [Path.cwd()])
retriever = Retriever(store=store, embedder=embedder, allowed_paths=[Path("./project")])

# Skip access tracking (don't increment access_count for this query)
results = retriever.search("analytics query", limit=10, track_access=False)
results = retriever.hybrid_search("analytics query", limit=10, track_access=False)

# Follow code reference edges from search results
# When traverse_refs=True, memories with 'references' relations to code chunks
# will have the referenced file content resolved and appended to results.
results = retriever.search("auth logic", limit=10, traverse_refs=True)

# Control ref expansion depth (1-5, default 3)
results = retriever.search("auth logic", limit=10, traverse_refs=True, max_ref_depth=2)

# Each result dict includes "_rrf_score" (RRF fusion score) and "_rerank_score" (blended final score)

# Get by ID
mem = store.get(mid)

# Update
store.update(mid, content="Updated content")

# Invalidate (soft delete)
store.invalidate(mid, reason="No longer relevant")

# --- Working Memory Inbox ---
# The inbox is a capacity-limited staging area for ephemeral information.
# Items enter via add_to_inbox() and are promoted to long-term memory via
# consolidate() or the dream deep phase.

# Add a note to the inbox (default attention_score=0.5, source=note)
inbox_id = store.add_to_inbox(content="Important observation", attention_score=0.8)

# Add with explicit source and metadata
inbox_id = store.add_to_inbox(
    content="Learned something",
    source="learn",  # note | learn | extract | consolidation
    attention_score=0.7,
    metadata={"context": "session-abc"},
)

# Retrieve an inbox item
item = store.get_from_inbox(inbox_id)
# item = {"id": ..., "content": ..., "source": ..., "attention_score": ..., ...}

# List inbox items (ordered by attention_score DESC, created_at ASC)
items = store.list_inbox(limit=20)

# Get inbox count
count = store.inbox_count()

# Update attention score
store.update_inbox_attention_score(inbox_id, 0.9)

# Remove an inbox item
store.remove_from_inbox(inbox_id)

# Consolidate inbox → long-term memory
# Items with attention_score >= min_score become memories (source=consolidation,
# confidence=attention_score). Items below are evicted. Inbox is empty after.
result = store.consolidate(min_score=0.5)
# result = {"promoted": [...], "evicted": [...]}

# Dry run (shows what would happen without changes)
result = store.consolidate(min_score=0.5, dry_run=True)

# Batch access tracking (efficient single UPDATE for multiple IDs)
# Increments access_count and updates accessed_at for each listed memory.
# Returns the number of rows actually updated (non-existent IDs are silently ignored).
affected = store.touch_batch([id1, id2, id3])

# List with filters
memories = store.list_all(type="fact", valid_only=True, limit=50)

# Relations (memory-to-memory and memory-to-code)
store.add_relation(mem_id_a, mem_id_b, "supersedes")
store.add_relation(mem_id_a, "src/lib.rs:42:58", "references", target_type="code")
relations = store.get_relations(mem_id_a)
related = store.traverse_relations(mem_id_a, relation_type="supersedes", max_depth=3)
code_refs = store.traverse_relations(mem_id_a, max_depth=2, target_type="code")

# Export / Import
data = store.export_all()          # default limit: 10,000 memories
data = store.export_all(limit=None)  # export all memories without limit
count = store.import_memories(data)

# Type registry
register_memory_type("custom_type")
types = get_registered_types()

# Close (or use as context manager)
store.close()

with MemoryStore() as store:
    store.add(type="fact", content="Context-managed store")

# Migration from lobsterdog
migrated = migrate_from_lobsterdog()  # Returns True if anything was copied

# Config
config = load_config()
home = get_llmem_home()
db_path = get_db_path()
config_path = get_config_path()

# Programmatically write config.yaml
written = write_config_yaml(
    config_path,
    {"memory": {"ollama_url": "http://localhost:11434", "embed_model": "nomic-embed-text"}},
    force=False,  # Set True to overwrite existing
)

# Detect available LLM providers
detector = ProviderDetector()
result = detector.detect(ollama_url="http://localhost:11434")
# result["provider"] → "ollama" | "openai" | "anthropic" | "none"
# result["ollama_url"]

# Check if Ollama is running
if is_ollama_running("http://localhost:11434"):
    print("Ollama is reachable")
```

### Embedding Quality Metrics

The `llmem.metrics` module provides functions to detect poor-quality embeddings:

```python
from llmem.metrics import (
    compute_metrics,
    anisotropy,
    similarity_range,
    discrimination_gap,
    cosine_similarity,
    bytes_to_vec,
    EmbeddingMetrics,
    ANISOTROPY_WARNING_THRESHOLD,
    SIMILARITY_RANGE_WARNING_THRESHOLD,
    METRICS_MAX_EMBEDDINGS,
)

# Compute all metrics at once (convenience wrapper)
m = compute_metrics(embeddings, labels=labels)
# m.anisotropy        → float in [0.0, 1.0]; lower is better
# m.similarity_range  → float; higher is better
# m.discrimination_gap → float | None; higher is better (None if no labels)

# Individual metric functions
aniso = anisotropy(embeddings)             # Average pairwise cosine similarity, clamped [0, 1]
sim_range = similarity_range(embeddings)   # Max - min pairwise cosine similarity
disc_gap = discrimination_gap(embeddings, labels)  # Inter-class vs intra-class separation

# Utility functions
sim = cosine_similarity(vec_a, vec_b)  # Cosine similarity, 0.0 for zero vectors
vec = bytes_to_vec(emb_bytes)          # Decode packed float32 bytes to list[float]

# Fetch embeddings from store (for metrics computation)
rows = store.get_embeddings_with_types(limit=10000)  # (embedding_bytes, type) tuples
count = store.count_embeddings()  # Count of valid embedded memories
```

**Warning thresholds:** `ANISOTROPY_WARNING_THRESHOLD = 0.5` (anisotropy above this may indicate poor embeddings), `SIMILARITY_RANGE_WARNING_THRESHOLD = 0.1` (similarity range below this may indicate poor embeddings).

**Performance safeguard:** `METRICS_MAX_EMBEDDINGS = 10000` — metrics computations are O(n²) pairwise, so `compute_metrics()` and `get_embeddings_with_types()` cap the number of vectors to prevent CPU hangs and OOM on large stores.

### `safe_urlopen`

The `safe_urlopen` function is the safe replacement for `urllib.request.urlopen()`. It validates URLs against SSRF, blocks redirects, and re-resolves hostnames before opening:

```python
from llmem.url_validate import safe_urlopen

# Default: allow_remote is inferred from the URL
response = safe_urlopen("http://localhost:11434/api/generate")

# Explicit allow_remote for remote endpoints
response = safe_urlopen("https://api.openai.com/v1/models", allow_remote=True)
```

The `allow_remote` parameter controls whether non-loopback URLs are permitted. If `None` (default), it's inferred from the URL — loopback URLs default to `False`, all others default to `False` as well (fail-closed). Pass `allow_remote=True` explicitly for known-remote endpoints.

### `get_server_auth_token`

```python
from llmem.config import get_server_auth_token

token = get_server_auth_token()
# Returns None if no token configured
# Raises ValueError if token is set but < 16 characters (too weak)
```

### Session Adapters

`SessionAdapter` is an abstract base class for reading session transcripts. `OpenCodeAdapter` is the built-in implementation that reads from the OpenCode SQLite database.

```python
from llmem.adapters import OpenCodeAdapter

adapter = OpenCodeAdapter(db_path=Path("~/.local/share/opencode/opencode.db"))
sessions = adapter.list_sessions(limit=10)
transcript = adapter.get_session_transcript(session_id)
chunks = adapter.get_session_chunks(session_id)
exists = adapter.session_exists(session_id)
adapter.close()
```

`OpenCodeAdapter.__init__` validates `db_path` for security: it rejects paths containing `..` traversal, paths targeting system directories (`/etc`, `/var`, etc.), and symlink paths. Paths that cannot be accessed (e.g. permission denied) also raise `ValueError`.

### Session Extraction Pipeline

The `session_hooks` module provides `process_opencode_sessions()` — a complete pipeline that discovers OpenCode sessions from the SQLite database, chunks them, and feeds each chunk through the extraction engine:

```python
from llmem.session_hooks import process_opencode_sessions, OPENCODE_RESULT_SUCCESS
from llmem.store import MemoryStore
from llmem.extract import ExtractionEngine
from llmem.embed import EmbeddingEngine

store = MemoryStore()
extractor = ExtractionEngine()
results = process_opencode_sessions(
    store=store,
    extractor=extractor,
    embedder=EmbeddingEngine(),
    force=False,       # skip already-processed sessions
    limit=50,          # max sessions to process
)
# results = {"opencode_success": 3, "opencode_already_processed": 2, ...}
```

Result constants: `OPENCODE_RESULT_SUCCESS`, `OPENCODE_RESULT_DB_NOT_FOUND`, `OPENCODE_RESULT_ALREADY_PROCESSED`, `OPENCODE_RESULT_NO_MEMORIES`, `OPENCODE_RESULT_EMPTY_TRANSCRIPT`, `OPENCODE_RESULT_ADAPTER_ERROR`, `OPENCODE_RESULT_EXTRACTION_FAILED`.

The `process_all_session_sources()` function in `llmem/hooks` orchestrates all session sources, currently delegating to `process_opencode_sessions`:

```python
from llmem.hooks import process_all_session_sources
from llmem.store import MemoryStore

store = MemoryStore()
results = process_all_session_sources(store=store, force=False)
# Returns aggregated result counts from all session sources
```

To implement a custom adapter, subclass `SessionAdapter`:

```python
from llmem.adapters.base import SessionAdapter

class MyAdapter(SessionAdapter):
    def list_sessions(self, limit=50):
        ...

    def get_session_transcript(self, session_id):
        ...

    def get_session_chunks(self, session_id):
        ...

    def session_exists(self, session_id):
        ...

    def close(self):
        ...
```

### Session Hooks

Session hooks inject relevant memories when an OpenCode session lifecycle event occurs, and extract memories when a session goes idle. Three events are supported:

| Event | Hook | Behavior |
|-------|------|----------|
| `session.created` | `on_created(session_id)` | Queries the memory store for relevant memories and writes a context file (`{session_id}.md`). Returns `("success", file_path)`, `("already_processed", None)`, or `("error", None)`. |
| `session.idle` | `on_idle(session_id)` | Extracts memories from the session transcript with 30-second debounce. Returns `("success", count)`, `("debounced", 0)`, or `("no_transcript", 0)`. |
| `session.compacting` | `on_compacting(session_id)` | Injects high-confidence key memories (`decision`, `preference`, `procedure`, `project_state` with confidence ≥ 0.7) to preserve context during compaction. Returns `("success", file_path)` or `("no_memories", None)`. |

`SessionHookCoordinator` orchestrates the three hooks:

```python
from llmem.session_hooks import create_session_hook_coordinator

coordinator = create_session_hook_coordinator()  # uses default config
# or with custom config:
coordinator = create_session_hook_coordinator(config=my_config)

result_type, path = coordinator.on_created("session-abc123")
result_type, count = coordinator.on_idle("session-abc123")
result_type, path = coordinator.on_compacting("session-abc123")
```

`SessionEventManager` dispatches events to registered hooks:

```python
from llmem.session_hooks import SessionEventManager

manager = SessionEventManager()
manager.emit("created", "session-abc123")  # calls registered "created" hook
manager.emit("idle", "session-abc123")     # calls registered "idle" hook
manager.emit("compacting", "session-abc123")  # calls registered "compacting" hook
```

`validate_session_id()` rejects session IDs containing `/`, `\`, or `..` to prevent path traversal attacks on context file paths:

```python
from llmem import validate_session_id

validate_session_id("abc123")    # returns "abc123"
validate_session_id("../etc/passwd")  # raises ValueError
validate_session_id("foo/bar")   # raises ValueError
```

### Code Indexing

The `CodeIndex` class manages the `code_chunks` table for semantic and full-text search over indexed code. It shares the same SQLite database as `MemoryStore` for cross-retrieval.

```python
from llmem.code_index import CodeIndex
from llmem.chunking import ParagraphChunking, FixedLineChunking, detect_language, walk_code_files

# Open the code index (uses the same database as MemoryStore)
code_index = CodeIndex()  # defaults to ~/.config/llmem/memory.db

# Add a single chunk
chunk_id = code_index.add_chunk(
    file_path="src/main.py",
    start_line=1,
    end_line=42,
    content="def main():\n    ...",
    language="python",
    chunk_type="paragraph",
)

# Batch add chunks from CodeChunk named tuples
chunks = chunker.chunk("src/main.py", content, language="python")
chunk_ids = code_index.add_chunks(chunks)

# Remove all chunks for a file (useful before re-indexing)
removed = code_index.remove_by_path("src/main.py")

# Full-text search
results = code_index.search_content("async def", limit=10)

# Semantic search (requires sqlite-vec and embeddings)
results = code_index.search_by_embedding(query_vec, limit=10, threshold=0.5)

code_index.close()
```

**Chunking strategies:**

```python
from llmem.chunking import ParagraphChunking, FixedLineChunking

# Paragraph chunking: splits at blank-line boundaries (default)
chunker = ParagraphChunking(min_lines=1, max_lines=200)
chunks = chunker.chunk("src/app.py", content, language="python")

# Fixed-line chunking: sliding window with overlap
chunker = FixedLineChunking(window_size=50, overlap=10)
chunks = chunker.chunk("src/app.py", content, language="python")
```

**Directory walking:**

```python
from llmem.chunking import walk_code_files, parse_gitignore

# Walk a directory respecting .gitignore
code_files = walk_code_files(Path("./my-project"))

# With custom size/depth limits
code_files = walk_code_files(
    Path("./my-project"),
    max_file_size=2 * 1024 * 1024,  # 2 MiB
    max_depth=30,
)
```

`walk_code_files` skips symlinks, binary files, credential files (`.env`, `.pem`, `.key`, SSH keys), and common non-code directories. `detect_language(file_path)` returns a language string from the file extension, or `None` for unknown extensions.

## Extension Points

LLMem provides a registry system that allows harnesses and external tools to plug in domain-specific behavior without modifying core code. All registry functions validate their inputs and raise `ValueError` or `TypeError` on invalid arguments.

### Session Adapter Registry

Register a custom session adapter so that other parts of the system can discover it by name:

```python
from llmem import register_session_adapter
from llmem.adapters.base import SessionAdapter

class MyAdapter(SessionAdapter):
    # ... implement abstract methods ...
    pass

register_session_adapter("my_adapter", MyAdapter)
```

List or look up registered adapters:

```python
from llmem.registry import get_registered_adapters, get_adapter_class

names = get_registered_adapters()        # frozenset of adapter names
cls = get_adapter_class("my_adapter")     # the adapter class, or None
```

### Dream Hook Registry

Register a function to run after a dream phase completes. Hooks are called with `(Dreamer instance, DreamResult, apply: bool)` and errors are logged without crashing the dream cycle.

```python
from llmem import register_dream_hook

def my_light_hook(dreamer, result, apply):
    # Post-light-phase logic here
    pass

register_dream_hook("light", my_light_hook)
```

Valid phases: `"light"`, `"deep"`, `"rem"`. Only one hook per phase is allowed; registering a duplicate raises `ValueError`.

### Session Hook Registry

Register a callback function for session lifecycle events. When `SessionEventManager.emit()` is called, the corresponding hook is invoked with the session ID.

```python
from llmem import register_session_hook, get_registered_session_hooks

def on_session_created(session_id):
    print(f"Session {session_id} was created")

register_session_hook("created", on_session_created)
```

Valid event types: `"created"`, `"idle"`, `"compacting"`. Only one hook per event type is allowed; registering a duplicate raises `ValueError`. The hook function must be callable; otherwise `TypeError` is raised.

List registered hooks:

```python
hooks = get_registered_session_hooks()  # dict mapping event type to hook function
```

### CLI Plugin Registry

Register a setup function that adds subcommands to the `llmem` CLI. The setup function receives an `argparse._SubParserGroup` and can add its own subparsers. Errors in plugin setup are logged but do not crash the CLI.

```python
from llmem import register_cli_plugin

def my_plugin_setup(subparsers):
    p = subparsers.add_parser("my-cmd", help="My custom command")
    p.add_argument("--flag", help="A flag")
    p.set_defaults(func=my_cmd_handler)

register_cli_plugin("my_plugin", my_plugin_setup)
```

After registration, `llmem my-cmd --flag value` becomes available. List registered plugins:

```python
from llmem.registry import get_registered_cli_plugins

names = get_registered_cli_plugins()  # frozenset of plugin names
```

## Database

LLMem uses SQLite with WAL mode and numbered SQL migrations (stored in the `llmem_migrations` package). Migrations are tracked in a `_schema_migrations` table and run automatically when the database is opened.

### Vector Search

When `sqlite-vec` is available, LLMem creates a `memories_vec` virtual table for cosine similarity search. If the extension isn't installed, vector search is gracefully disabled and the store falls back to FTS-only search.

The embedding dimension defaults to 768 (matching `nomic-embed-text`), configurable via `vec_dimensions`.

### Code Index

LLMem also provides a code indexing system via the `code_chunks` table, created by migration 004. This table stores chunked source code with embeddings for cross-retrieval alongside memories.

The `code_chunks` table schema:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | Format: `<file_path>:<start_line>:<end_line>` |
| `file_path` | TEXT NOT NULL | Relative path of the source file |
| `start_line` | INTEGER NOT NULL | Starting line (1-based) |
| `end_line` | INTEGER NOT NULL | Ending line (1-based, inclusive) |
| `content` | TEXT NOT NULL | Chunk text content |
| `embedding` | BLOB | Embedding vector bytes (nullable) |
| `language` | TEXT | Detected programming language |
| `chunk_type` | TEXT NOT NULL | Chunking strategy (`paragraph` or `fixed_line`) |
| `created_at` | TEXT NOT NULL | ISO timestamp |

When `sqlite-vec` is available, a `code_chunks_vec` virtual table enables semantic similarity search over code chunk embeddings, with INSERT/UPDATE/DELETE triggers for automatic synchronization. An FTS5 `code_chunks_fts` virtual table provides full-text search over chunk content, file paths, and language names.

The `--include-code` flag on `llmem search` interleaves code chunk results with memory results using the same RRF scoring formula, enabling unified search across both knowledge stores.

### Code Reference Edges

The `relations` table supports two `target_type` values: `'memory'` (the default, linking two memories) and `'code'` (linking a memory to a code chunk). When `target_type='code'`, the `target_id` uses the format `path:start_line:end_line` (e.g., `src/lib.rs:42:58`) referencing a file location rather than a memory UUID.

The `references` relation type (added to the `relation_type` CHECK constraint) creates edges from memories to code chunks. This enables `--traverse-refs` in search, which follows reference edges from result memories and resolves the referenced file content at query time.

Code ref paths must be relative (no leading `/`) and must not contain `..` traversal. Refs are resolved against an `allowed_paths` allowlist that defaults to `[Path.cwd()]`, preventing arbitrary file reads.

## Dream Cycle

The dream cycle performs automated memory maintenance during idle periods. It can be invoked manually via `llmem dream` (see [CLI Reference](#llmem-dream)) or run automatically by the `lobsterdog-dream.service` systemd timer.

- **Light phase:** Sort and deduplicate near-duplicate memories (cosine similarity ≥ threshold).
- **Deep phase:** Score, promote, decay, and merge memories. Also promotes inbox items to long-term memory (items with attention_score ≥ `dream.min_score` become permanent memories; lower-scored items are evicted). Decays confidence on idle memories. Boosts frequently accessed memories. LLM-assisted merging of similar pairs. Auto-links memories with high cosine similarity (≥ `dream.auto_link_threshold`, default 0.85) by creating `related_to` relations between them.
- **REM phase:** Extract themes from memory clusters and write a dream diary (read-only reflection).

Configuration is under the `dream:` key in `config.yaml`. Set `dream.enabled: false` to disable.

Extension hooks can be registered to run custom logic after each dream phase. See [Extension Points — Dream Hook Registry](#dream-hook-registry) for details.

## Extraction and Hooks

The `hooks` module provides automatic extraction from session transcripts:

- `process_file()`: Extract memories from a transcript file.
- `process_session()`: Extract from an OpenCode session ID.
- `process_all_session_sources()`: Process all session sources (delegates to `session_hooks.process_opencode_sessions`).
- Self-assessment extraction with structured error taxonomy.
- Correction detection for identifying mistakes.

The `session_hooks` module provides `process_opencode_sessions()` — the full pipeline that reads OpenCode sessions from the SQLite database, chunks them by message boundaries, and runs extraction and embedding.

The `extract` module uses Ollama (default: `qwen2.5:1.5b`) to extract structured memories from text. The `embed` module generates embeddings using Ollama (default: `nomic-embed-text`).

## OpenCode Custom Tools

LLMem ships six type-safe OpenCode tools that replace raw `llmem` CLI calls with described, schema-validated tool invocations. Tools run `llmem` as a subprocess with a 60-second timeout and return strings — errors are prefixed with `Error:`.

| Tool | CLI Equivalent | Description |
|------|----------------|-------------|
| `llmem-search` | `llmem search <query> --json` | Search memories via FTS5; returns JSON array |
| `llmem-add` | `llmem add --type T --content C` | Add a new memory; returns ID and type |
| `llmem-context` | `llmem search <query> --json --limit 20` | Retrieve formatted context for a topic; returns a context block truncated to a character budget |
| `llmem-invalidate` | `llmem invalidate <ID>` | Invalidate a memory by ID with optional reason |
| `llmem-stats` | `llmem stats` | Show memory statistics (total, active, expired, by type) |
| `llmem-hook` | `llmem hook` | Run the extraction hook on session transcripts |

### `llmem-search`

```
llmem-search(query: string, type?: string, limit?: number)
```

- `query` (required): Search query for FTS5 full-text search.
- `type` (optional): Filter by memory type (fact, decision, preference, etc.).
- `limit` (optional, min 1): Maximum number of results (default: 20).

Returns a JSON string of matching memory objects on success, or an `Error:` string on failure (CLI error, invalid JSON, non-array response).

### `llmem-add`

```
llmem-add(type: string, content: string, source?: string, confidence?: number)
```

- `type` (required): Memory type (`fact`, `decision`, `preference`, `event`, `project_state`, `procedure`, `conversation`, `self_assessment`).
- `content` (required): Memory content text.
- `source` (optional): Source of memory (default: `manual`).
- `confidence` (optional, 0–1): Confidence score (default: 0.8).

Returns the memory ID and type on success, or an `Error:` string on failure (invalid type, CLI error).

### `llmem-context`

```
llmem-context(query: string, budget?: number)
```

- `query` (required): Topic or query to recall context for.
- `budget` (optional, min 0): Character budget for the returned context block (default: 4000).

Searches memories and formats results as a context block suitable for LLM injection:

```
- [fact] Project uses SQLite with WAL mode
- [decision] Use pytest for testing (summary: Testing framework choice)
```

Returns `"No memories found."` if no results match, or an `Error:` string on failure. Truncation is Unicode-safe (splits on code points, not UTF-16 code units).

### `llmem-invalidate`

```
llmem-invalidate(id: string, reason?: string)
```

- `id` (required): Memory ID to invalidate.
- `reason` (optional): Reason for invalidation.

Returns a confirmation string on success, or an `Error:` string if the memory is not found or the CLI fails.

### `llmem-stats`

```
llmem-stats()
```

Takes no arguments. Returns a formatted statistics string (total, active, expired counts, and breakdown by type), or an `Error:` string on failure.

### `llmem-hook`

```
llmem-hook(force?: boolean, noEmbed?: boolean, noIntrospect?: boolean, file?: string, directory?: string)
```

- `force` (optional): Force re-extraction of already-processed sessions.
- `noEmbed` (optional): Skip embedding generation (faster, no Ollama required).
- `noIntrospect` (optional): Skip introspection for trivial sessions.
- `file` (optional): Path to a specific transcript file. Resolved relative to `context.directory`; paths escaping the directory scope are rejected.
- `directory` (optional): Path to a directory of transcript files. Same containment check as `file`.

Returns the extraction result on success, or an `Error:` string on failure (path traversal, CLI error).

### Error Handling

All tools follow a consistent contract:

- **Success**: returns the CLI output as a string.
- **Error**: returns a string starting with `Error:`.
- **CLI not found**: returns `"Error: llmem CLI not found on PATH"` (exit code 127).
- **Timeout**: returns `"Error: llmem <cmd> timed out after 60000ms"` (exit code 124).
- Tools never throw — all errors are returned as strings.

## OpenCode Integration (npm)

The `opencode-llmem` npm package provides a JavaScript plugin that integrates LLMem session hooks into OpenCode via its plugin interface.

### Installation

```bash
npm install opencode-llmem
```

The `postinstall` script copies the hook source files to `~/.agents/plugins/llmem/` so OpenCode can discover and load them.

### Usage

Register all three session lifecycle hooks on an OpenCode session object:

```javascript
const llmem = require("opencode-llmem");

// In your OpenCode plugin setup:
llmem.register(session);
```

This registers handlers for the `session.created`, `session.idle`, and `session.compacting` events. Each handler:

- Validates the session ID against path traversal attacks.
- Rate-limits process spawning to prevent flooding (1-second cooldown, max 3 concurrent processes).
- Calls the `llmem` CLI via `execFileSync` (no shell, no injection risk).
- Writes the resulting context to the LLMem context directory.
- Degrades gracefully — errors are logged but never block the session.

### Configuration

The JavaScript hooks read configuration from `LMEM_HOME/config.yaml` (or `~/.config/llmem/config.yaml` by default). The `opencode.context_dir` key controls where context files are written.

## Copilot CLI Integration (npm)

The `copilot-llmem` npm package provides a Copilot CLI plugin that integrates LLMem session hooks into GitHub Copilot CLI via declarative hooks. It calls the same `llmem` CLI under the hood as the OpenCode plugin.

### Installation

```bash
copilot plugin install MichielDean/llmem-plugin
```

Or via npm:

```bash
npm install copilot-llmem
```

The `postinstall` script copies the skill directories (`llmem`, `introspection`, `introspection-review-tracker`) and the `memory-assistant` agent to `~/.agents/` so they are discoverable at runtime.

### How It Works

The plugin registers three session lifecycle hooks in `hooks.json`:

| Hook | CLI Command | Behavior |
|------|-------------|----------|
| `sessionStart` | `llmem context <session_id>` | Injects relevant memory context at session start |
| `agentStop` | `llmem hook idle <session_id>` | Extracts memories and runs introspection when an agent stops |
| `sessionCompacting` | `llmem context --compacting <session_id>` | Injects key memories during session compaction |

Each hook extracts the `session_id` from the incoming JSON via `python3 -c`, passes it to the `llmem` CLI, and degrades gracefully on errors.

### Plugin Structure

```
copilot-llmem/
  plugin.json              # Plugin manifest
  hooks.json               # Declarative hook definitions
  install.js               # Postinstall script (copies skills/agents to ~/.agents/)
  package.json             # npm package config
  agents/
    memory-assistant.agent.md  # AI assistant agent definition
  skills/
    llmem/SKILL.md             # LLMem skill reference
    introspection/SKILL.md     # Introspection framework skill
    introspection-review-tracker/SKILL.md  # Review outcome tracking skill
```

### Verification

After installation, verify the skills are discoverable:

```bash
ls ~/.agents/skills/llmem ~/.agents/skills/introspection ~/.agents/skills/introspection-review-tracker
ls ~/.agents/agents/memory-assistant.agent.md
```

Run the bundled tests:

```bash
npm test
```

## Security

- `LMEM_HOME` is validated against path traversal, system directories, and symlink attacks.
- Write paths are validated against system directories and symbolic links. `is_symlink()` checks are wrapped in `try/except OSError` to handle inaccessible paths gracefully.
- System directory blocking uses a shared `_BLOCKED_PATH_PREFIXES` tuple (with prefix + `/` matching to avoid false positives) across `_validate_home_path`, `_validate_write_path`, and `OpenCodeAdapter.__init__` to prevent DRY violations.
- `validate_session_id()` rejects session IDs containing `/`, `\`, or `..` to prevent path traversal when constructing context file paths.
- **CLI path validation**: `llmem add --file`, `llmem import`, and `llmem export --output` block access to protected system directories (e.g. `/etc`, `/bin`, `/usr/bin`, `/sbin`, `/usr/sbin`, `/dev`, `/proc`, `/sys`, `/var`, `/boot`, `/root`). These checks use prefix + `/` matching to avoid false positives (e.g. `/binary_search` is not blocked as `/bin`).
- **Import file size limit**: `llmem import` rejects files larger than 10 MiB.
- URL validation (`is_safe_url`) blocks private/reserved IPs and SSRF vectors, including percent-encoded IP hostnames (e.g. `%31%32%37%2e%30%2e%30%2e%31` is decoded before IP checks). When `allow_remote=False` (the default), only loopback addresses on the Ollama default port are permitted — all other IPs including public addresses are rejected. `safe_urlopen` enforces URL validation, blocks all redirects (via `_NoRedirectHandler`), mitigates DNS rebinding by re-resolving the hostname immediately before the request, strips credentials from error messages, and requires an explicit `allow_remote` parameter (defaults to `False`) for non-loopback addresses. It accepts both string URLs and `urllib.request.Request` objects, and applies a default 30-second timeout to prevent indefinite hangs.
- OpenCode session extraction validates the database path: rejects path traversal (`..`), system directories, symlinks, and URI injection (`?` and `#` characters). `get_opencode_db_path()` validates via `_validate_home_path` before returning.
- API keys are masked in `__repr__` on provider instances (`***masked***`).
- API keys are refused over plain HTTP to non-loopback hosts. `OpenAIProvider` and `AnthropicProvider` raise `ValueError` if `base_url` is `http://` and the hostname is not a loopback address (checked via exact string match and `ipaddress` for IPv6-mapped addresses like `::ffff:127.0.0.1`). Substring matches like `localhost.evil.com` are blocked.
- A warning is logged when API keys are sent to a non-default base URL to alert the user of potential credential exfiltration risk.
- Validation error messages use `_strip_credentials()` to remove userinfo from URLs — never embed user-supplied URL credentials in error messages or logs.
- `_strip_credentials()` is used consistently across `is_safe_url()`, `safe_urlopen()`, provider error messages, and config URL validation to prevent credential leaking.
- Embedding and generation inputs are validated against size limits: `MAX_TEXT_LENGTH` (100,000 characters per text) and `MAX_BATCH_SIZE` (2,048 texts per batch) to prevent OOM and resource exhaustion.
- Embedding dimension validation in `MemoryStore.add()` rejects vectors whose dimension doesn't match `vec_dimensions`, preventing dimension mismatch bugs from silently corrupting the vector index.
- All SQL queries use parameterized statements (no injection risk).
- **Code reference path validation**: `validate_code_ref_path()` rejects absolute paths (leading `/`) and directory traversal (`..`) in code ref target_ids. `resolve_code_ref()` enforces an `allowed_paths` allowlist (defaulting to `[Path.cwd()]`) and blocks resolved paths targeting system directories (`/etc`, `/var`, etc.). Code refs must use the relative format `path:start_line:end_line`.
- `add_relation()` validates code ref `target_id` paths at insertion time — unsafe paths are rejected with `ValueError`.
- SQLite extension loading is disabled immediately after `sqlite-vec` loads, preventing runtime loading of arbitrary shared libraries.
- Database files are created with `umask(0o177)` before creation, then `chmod(0o600)` applied to the DB file and its WAL/SHM sidecars (prevents a race window where sensitive memory content is world-readable on multi-user systems). Parent directories use `0o700`.
- `config.yaml` is written with `0o600` file permissions (owner-only read/write) to protect API keys and secrets from other users on shared systems.
- **Server auth token strength**: `server.auth_token` in `config.yaml` must be at least 16 characters. Short tokens are rejected with a hint to generate a strong token.
- `import_memories()` validates entry IDs (string, max 256 chars), embeddings (bytes, max 1 MB), and confidence (numeric) before insertion. Invalid entries are skipped with warnings rather than crashing.
- `export_all()` defaults to a limit of 10,000 memories to prevent unbounded memory consumption; pass `limit=None` to export all.
- `_search_by_embedding_brute()` uses a `LIMIT` clause (10,000 rows max) to prevent OOM on large databases.
- **Embedding metrics computation capping**: `get_embeddings_with_types(limit=)` applies a SQL `LIMIT` clause (default: 10,000) and `compute_metrics(max_embeddings=10000)` truncates input vectors. These caps prevent O(n²) pairwise metrics computations from causing OOM or CPU hangs on large stores.
- `process_transcript()` enforces the same size limit as `process_file()` to prevent OOM from large session transcripts.
- **Dream diary locking**: On platforms with `fcntl` (Linux/macOS), dream diary writes use an exclusive file lock to prevent corruption from concurrent dream cycles.
- OpenCode tool invocations (`_llmem.ts`) prepend `--` before user arguments to prevent argparse flag injection.
- JavaScript hooks use `execFileSync` (not shell-based `execSync`) and `validateSessionId()` for path traversal protection, with `canSpawnProcess()` rate limiting and `MAX_CONCURRENT=3` process cap to prevent resource exhaustion.
- Prototype pollution protection in `_parseSimpleYaml`: keys `__proto__`, `constructor`, and `prototype` are filtered from parsed YAML to prevent Object prototype mutation.
- ProviderDetector.detect() only returns `provider` and `ollama_url` — no API key presence is exposed.
- Migration from `~/.lobsterdog/` skips symlinks (using `follow_symlinks=False`).

**Code indexing security:**

- `walk_code_files()` skips all symlinks (both file and directory) to prevent path traversal and data exposure.
- Default file size limit of 1 MiB (`--max-file-size`) prevents memory exhaustion from large files.
- Default directory depth limit of 50 (`--max-depth`) prevents stack overflow from deeply nested trees.
- Credential files are excluded from indexing: `.env`, `.env.*` variants, `.pem`, `.key`, SSH private keys (`id_rsa`, `id_dsa`, `id_ed25519`, `id_ecdsa`), `.netrc`, `.htpasswd`, `.npmrc`, `.pypirc`.
- `.gitignore` patterns are respected at every directory level, with correct handling of anchored patterns (leading `/`), negation patterns (`!`), and directory-only patterns (trailing `/`).

## Module Reference

| Module | Description |
|--------|-------------|
| `memory.providers` | Abstract base classes, concrete providers (`OllamaProvider`, `OpenAIProvider`, `AnthropicProvider`, `SentenceTransformersProvider`, `NoneProvider`), `resolve_provider()`, `dimension()`, `_is_loopback_hostname()`, `_validate_embed_inputs()`, `_strip_credentials()` |
| `memory.ollama` | `check_ollama_model()`, `_call_ollama_generate()` |
| `memory.url_validate` | `is_safe_url()`, `safe_urlopen()`, `_strip_credentials()`, `_NoRedirectHandler()`, `validate_base_url()`, `SafeRedirectHandler` |
| `memory.config` | Configuration loading, defaults, typed accessors (e.g. `get_provider_config()`, `get_ollama_url()` with SSRF validation) |
| `llmem.session_hooks` | `SessionHookCoordinator`, `SessionEventManager`, `create_session_hook_coordinator()`, result constants |
| `llmem.url_validate` | `is_safe_url()`, `safe_urlopen()`, `_strip_credentials()`, `validate_base_url()`, `_NoRedirectHandler`, `_extract_url_string()` (mirrors `memory.url_validate`), DNS rebinding protection |
| `llmem.paths` | `validate_session_id()`, `get_context_dir()`, `_validate_write_path()`, `BLOCKED_SYSTEM_PREFIXES`, home/write path checks |
| `llmem.registry` | `register_session_hook()`, `get_registered_session_hooks()`, `VALID_SESSION_EVENT_TYPES` |
| `llmem.taxonomy` | `ERROR_TAXONOMY`, `REVIEW_SEVERITY_TAXONOMY`, `SELF_ASSESSMENT_FIELDS`, `ERROR_TAXONOMY_KEYS` |
| `llmem.metrics` | `compute_metrics()`, `anisotropy()`, `similarity_range()`, `discrimination_gap()`, `cosine_similarity()`, `bytes_to_vec()`, `EmbeddingMetrics` dataclass, warning thresholds, `METRICS_MAX_EMBEDDINGS` |
| `llmem.store` | `MemoryStore` with `export_all(limit=)`, `import_memories()` validation, brute-force/embedding caps, dimension validation, inbox methods (`add_to_inbox`, `get_from_inbox`, `list_inbox`, `remove_from_inbox`, `update_inbox_attention_score`, `consolidate`), capacity eviction, `get_embeddings_with_types(limit=)`, `count_embeddings()` |
| `llmem.code_index` | `CodeIndex` — manages `code_chunks` table, FTS5/vec virtual tables, add/search/remove operations |
| `llmem.refs` | `resolve_code_ref()`, `validate_code_ref_path()` — code reference resolution for memory-to-code-chunk edges |
| `llmem.chunking` | `ParagraphChunking`, `FixedLineChunking`, `detect_language()`, `walk_code_files()`, `parse_gitignore()`, `is_ignored()` |

## Running Tests

```bash
python -m pytest
```

1181 Python tests and 65 JavaScript tests (copilot-llmem) covering all providers, URL validation, configuration, security, session hooks, CLI commands (context, hook, track-review, suggest-categories), and edge cases.

## License

MIT