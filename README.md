# LLMem — Structured Memory with Semantic Search

LLMem is a SQLite-backed memory store for LLM agents. It provides structured storage with full-text search (FTS5), optional vector similarity search (via sqlite-vec), an extensible type system, and a provider abstraction layer for LLM embeddings and text generation. A background dreaming cycle consolidates, decays, and merges memories over time.

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

For development:

```bash
pip install ".[dev]"
```

### Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally for extraction, embedding, and dreaming features

## Provider Abstraction Layer

LLMem decouples embedding and text generation from any specific LLM backend through two abstract base classes:

| Protocol | Methods | Purpose |
|----------|---------|---------|
| `EmbedProvider` | `embed()`, `embed_batch()`, `check_available()` | Vector embeddings for semantic search |
| `GenerateProvider` | `generate()`, `check_available()` | Text generation from prompts |

### Concrete Providers

| Provider | Embed | Generate | API Key | Default Base URL |
|----------|-------|----------|---------|-------------------|
| `OllamaProvider` | Yes | Yes | None | `http://localhost:11434` |
| `OpenAIProvider` | Yes | Yes | `OPENAI_API_KEY` or constructor arg | `https://api.openai.com` |
| `AnthropicProvider` | No | Yes | `ANTHROPIC_API_KEY` or constructor arg | `https://api.anthropic.com` |
| `NoneProvider` | Yes (zeros) | Yes (empty string) | None | N/A |

### Graceful Degradation

`resolve_provider(config)` returns the best available `(embed_provider, generate_provider)` pair by trying providers in order:

**Embed fallback chain:** Ollama → OpenAI → NoneProvider (zero vectors, FTS5-only mode)

**Generate fallback chain:** Ollama → OpenAI → Anthropic → NoneProvider (empty string)

Each provider's `check_available()` returns `False` on any error (never raises), so degradation is automatic.

### Quick Start

```python
from memory.providers import resolve_provider

embed, generate = resolve_provider(config={})
# With Ollama running: OllamaProvider for both
# Without Ollama, with OPENAI_API_KEY set: OpenAIProvider for both
# Without any provider: NoneProvider for both

vec = embed.embed("hello world")
text = generate.generate("Summarize this document")
```

### Direct Construction

```python
from memory.providers import OllamaProvider, OpenAIProvider, AnthropicProvider, NoneProvider

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
```

This yields `OpenAIProvider` for embeddings and `AnthropicProvider` for generation.

## Provider Configuration

Provider-related keys in `config.yaml`:

```yaml
provider:
  default: ollama              # ollama | openai | anthropic | none
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
```

Both config-based and environment variable API keys are supported; config keys take precedence.

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
  db: null                    # Auto-resolved from get_db_path()
  ollama_url: http://localhost:11434
  embed_model: nomic-embed-text
  extract_model: qwen2.5:1.5b
  prospective_model: qwen2.5:1.5b
  context_budget: 4000
  auto_extract: true
  max_file_size: 10485760     # 10MB
  session_dirs:
    - ~/.local/share/opencode/sessions

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
  report_path: ~/.agent/diagrams/dream-report.html
  behavioral_threshold: 3
  behavioral_lookback_days: 30
  proposed_changes_path: null # Auto-resolved from get_proposed_changes_path()
  calibration_enabled: true
  stale_procedure_days: 30
  calibration_lookback_days: 90

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
  add             Add a memory
  get             Get a memory by ID
  search          Search memories (hybrid RRF fusion of FTS5 + semantic)
  list            List memories
  stats           Show memory statistics
  update          Update a memory
  invalidate      Invalidate a memory (mark expired/wrong)
  delete          Delete a memory
  export          Export all memories to JSON
  import          Import memories from a JSON file
  register-type   Register a new memory type
  types           List registered memory types
  init            Initialize the llmem memory system
```

Plugins registered via [`register_cli_plugin`](#cli-plugin-registry) add additional subcommands at runtime.

### `llmem add`

```bash
llmem add --type TYPE --content TEXT [--summary TEXT] [--source SOURCE] \
  [--confidence FLOAT] [--valid-until TIMESTAMP] [--metadata JSON] \
  [--relation TYPE --relation-to ID]
```

- `--type` (required): Memory type. Use `llmem types` to list registered types, or `llmem register-type` to add new ones.
- `--content` or `--file`: The memory text (or read from a file).
- `--source`: Source of the memory (default: `manual`). Valid: `manual`, `session`, `heartbeat`, `extraction`, `import`.
- `--confidence`: Confidence score 0–1 (default: 0.8).
- `--relation` / `--relation-to`: Create a relation to another memory after adding.

### `llmem search`

```bash
llmem search QUERY [--type TYPE] [--limit N] [--json] [--fts-only | --semantic-only]
```

Hybrid search combining FTS5 keyword search and vector semantic search via Reciprocal Rank Fusion (RRF), followed by multi-signal reranking. By default, both search modes are merged with `alpha=0.7` (favoring semantic results while keeping keyword relevance), then reranked with `blend=0.3` (70% semantic, 30% confidence/recency/access/type signals).

- `--fts-only`: Use FTS5 keyword search only (no embedder needed).
- `--semantic-only`: Use semantic (embedding) search only (requires an embedder). Raises an error if no embedder is available.
- Without either flag: hybrid mode — runs both FTS5 and semantic search, fuses results via RRF, then applies reranking. Falls back to FTS5-only if no embedder is configured.

With `--json`, outputs raw JSON (each result includes `_rrf_score` and `_rerank_score` keys); otherwise, a human-readable table with an `rrf=` score column.

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

Initialize the llmem memory system. Creates `~/.config/llmem/` (or `LMEM_HOME`) with `config.yaml` and initializes the SQLite database (`memory.db`). Detects available LLM providers in order of precedence: Ollama (if reachable) > OpenAI (if `OPENAI_API_KEY` is set) > Anthropic (if `ANTHROPIC_API_KEY` is set).

- `--ollama-url URL`: Override the Ollama base URL (default: `http://localhost:11434`). Must be a valid `http://` or `https://` URL.
- `--non-interactive`: Skip all prompts and use defaults. Useful for scripting and CI.
- `--force`: Overwrite an existing `config.yaml`. Without this flag, init is idempotent — running it twice prints a message and exits without error.

In interactive mode, you'll be prompted for:

1. **Ollama URL** — press Enter to accept the detected/default URL, or type a custom URL (validated for safety).
2. **Dream cycle** — enable or disable the background dream cycle (default: enabled).

If `~/.lobsterdog/` exists (legacy path), init automatically migrates data to `~/.config/llmem/`.

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

Search results include both `_rrf_score` (raw RRF fusion score) and `_rerank_score` (blended final score). Results are sorted by `_rerank_score` descending, with ties broken by ascending memory ID. Search operations (`Retriever.search()` and `Retriever.hybrid_search()`) automatically track access — each returned result's `access_count` and `accessed_at` are updated (best-effort), keeping the recency and access frequency signals current.

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

# Each result dict includes "_rrf_score" (RRF fusion score) and "_rerank_score" (blended final score)

# Get by ID
mem = store.get(mid)

# Update
store.update(mid, content="Updated content")

# Invalidate (soft delete)
store.invalidate(mid, reason="No longer relevant")

# List with filters
memories = store.list_all(type="fact", valid_only=True, limit=50)

# Relations
store.add_relation(mem_id_a, mem_id_b, "supersedes")
relations = store.get_relations(mem_id_a)
related = store.traverse_relations(mem_id_a, relation_type="supersedes", max_depth=3)

# Export / Import
data = store.export_all()                    # default cap: 50,000 rows
data = store.export_all(limit=1000)          # custom limit
count = store.import_memories(data)          # validates id, embedding, confidence

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
# result["ollama_url"], result["openai_key_found"], result["anthropic_key_found"]

# Check if Ollama is running
if is_ollama_running("http://localhost:11434"):
    print("Ollama is reachable")
```

### Session Adapters

`SessionAdapter` is an abstract base class for reading session transcripts. `OpenCodeAdapter` is the built-in implementation that reads from the OpenCode SQLite database.

```python
from llmem.adapters import OpenCodeAdapter

adapter = OpenCodeAdapter()
sessions = adapter.list_sessions(limit=10)
transcript = adapter.get_session_transcript(session_id)
chunks = adapter.get_session_chunks(session_id)
exists = adapter.session_exists(session_id)
adapter.close()
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

## Dream Cycle

The dream cycle performs automated memory maintenance during idle periods:

- **Light phase:** Sort and deduplicate near-duplicate memories (cosine similarity ≥ threshold).
- **Deep phase:** Score, promote, decay, and merge memories. Decays confidence on idle memories. Boosts frequently accessed memories. LLM-assisted merging of similar pairs.
- **REM phase:** Extract themes from memory clusters and write a dream diary (read-only reflection).

Configuration is under the `dream:` key in `config.yaml`. Set `dream.enabled: false` to disable.

Extension hooks can be registered to run custom logic after each dream phase. See [Extension Points — Dream Hook Registry](#dream-hook-registry) for details.

## Extraction and Hooks

The `hooks` module provides automatic extraction from session transcripts:

- `process_file()`: Extract memories from a transcript file.
- `process_session()`: Extract from an OpenCode session ID.
- Self-assessment extraction with structured error taxonomy.
- Correction detection for identifying mistakes.

The `extract` module uses Ollama (default: `qwen2.5:1.5b`) to extract structured memories from text. The `embed` module generates embeddings using Ollama (default: `nomic-embed-text`).

## OpenCode Custom Tools

LLMem ships six type-safe OpenCode tools that replace raw `llmem` CLI calls with described, schema-validated tool invocations. Tools run `llmem` as a subprocess and return strings — errors are prefixed with `Error:`.

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
- Rate-limits process spawning to prevent flooding (1-second cooldown).
- Calls the `llmem` CLI via `execFileSync` (no shell, no injection risk).
- Writes the resulting context to the LLMem context directory.
- Degrades gracefully — errors are logged but never block the session.

### Configuration

The JavaScript hooks read configuration from `LMEM_HOME/config.yaml` (or `~/.config/llmem/config.yaml` by default). The `opencode.context_dir` key controls where context files are written.

## Security

- `LMEM_HOME` is validated against path traversal, system directories, and symlink attacks.
- Write paths are validated against system directories and symbolic links.
- `validate_session_id()` rejects session IDs containing `/`, `\`, or `..` to prevent path traversal when constructing context file paths.
- URL validation (`is_safe_url`) blocks private/reserved IPs and SSRF vectors, including percent-encoded IP hostnames (e.g. `%31%32%37%2e%30%2e%30%2e%31` is decoded before IP checks). `safe_urlopen` enforces URL validation, blocks redirects, mitigates DNS rebinding, and strips credentials from error messages. It accepts both string URLs and `urllib.request.Request` objects, and requires an explicit `allow_remote` parameter (defaults to `False`) for non-loopback addresses.
- API keys are masked in `__repr__` on provider instances (`***masked***`).
- API keys are refused over plain HTTP to non-loopback hosts. `OpenAIProvider` and `AnthropicProvider` raise `ValueError` if `base_url` is `http://` and the hostname is not an exact loopback address (`localhost`, `127.0.0.1`, or `::1`). Substring matches like `localhost.evil.com` are blocked.
- A warning is logged when API keys are sent to a non-default base URL to alert the user of potential credential exfiltration risk.
- Validation error messages use generic strings (never embed user-supplied URLs).
- All SQL queries use parameterized statements (no injection risk).
- Database files are created with `umask(0o177)` before creation, then `chmod(0o600)` applied to the DB file and its WAL/SHM sidecars (prevents a race window where sensitive memory content is world-readable on multi-user systems). Parent directories use `0o700`.
- `import_memories()` validates entry IDs (string, max 256 chars), embeddings (bytes, max 1 MB), and confidence (numeric) before insertion. Invalid entries are skipped with warnings rather than crashing.
- `export_all()` caps results at 50,000 rows by default to prevent OOM on large databases.
- `_search_by_embedding_brute()` uses a `LIMIT` clause (10,000 rows max) to prevent OOM on large databases.
- `process_transcript()` enforces the same size limit as `process_file()` to prevent OOM from large session transcripts.
- JavaScript hooks use `execFileSync` (not shell-based `execSync`) and `validateSessionId()` for path traversal protection, with `canSpawnProcess()` rate limiting.
- Migration from `~/.lobsterdog/` skips symlinks (using `follow_symlinks=False`).

## Module Reference

| Module | Description |
|--------|-------------|
| `memory.providers` | Abstract base classes, concrete providers, `resolve_provider()`, `_is_loopback_hostname()` |
| `memory.ollama` | `check_ollama_model()`, `_call_ollama_generate()` |
| `memory.url_validate` | `is_safe_url()`, `safe_urlopen()`, `sanitize_url_for_log()`, `validate_base_url()` |
| `memory.config` | Configuration loading, defaults, typed accessors (e.g. `get_provider_config()`) |
| `llmem.session_hooks` | `SessionHookCoordinator`, `SessionEventManager`, `create_session_hook_coordinator()`, result constants |
| `llmem.url_validate` | `is_safe_url()`, `safe_urlopen()`, `validate_base_url()`, `_extract_url_string()` (mirrors `memory.url_validate`) |
| `llmem.paths` | `validate_session_id()`, `get_context_dir()`, `_validate_write_path()` |
| `llmem.registry` | `register_session_hook()`, `get_registered_session_hooks()`, `VALID_SESSION_EVENT_TYPES` |
| `llmem.store` | `MemoryStore` with `export_all(limit=)`, `import_memories()` validation, brute-force/embedding caps |

## Running Tests

```bash
python -m pytest
```

660 Python tests and 53 JavaScript tests covering all providers, URL validation, configuration, security, session hooks, and edge cases.

## License

MIT