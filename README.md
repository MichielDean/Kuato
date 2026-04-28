# LLMem — Structured Memory with Semantic Search

LLMem is a SQLite-backed memory store for LLM agents. It provides structured storage with full-text search (FTS5), optional vector similarity search (via sqlite-vec), and an extensible type system. A background dreaming cycle consolidates, decays, and merges memories over time.

## Installation

```bash
pip install llmem
```

This copies all 7 skills into `~/.agents/skills/`, where OpenCode discovers them automatically.

## Skills

| Skill | Description |
|-------|-------------|
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

### Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally for extraction, embedding, and dreaming features

## Quick Start

```bash
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

## Configuration

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
  search          Search memories (FTS + optional vector)
  list            List memories
  stats           Show memory statistics
  update          Update a memory
  invalidate      Invalidate a memory (mark expired/wrong)
  delete          Delete a memory
  export          Export all memories to JSON
  import          Import memories from a JSON file
  register-type   Register a new memory type
  types           List registered memory types
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
llmem search QUERY [--type TYPE] [--limit N] [--json]
```

Full-text search across content, summary, and hints. With `--json`, outputs raw JSON; otherwise, a human-readable table.

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
    SessionAdapter,
    OpenCodeAdapter,
    register_session_adapter,
    register_dream_hook,
    register_cli_plugin,
)

# Open a store
store = MemoryStore()  # uses default path ~/.config/llmem/memory.db

# Add a memory
mid = store.add(type="fact", content="Project uses SQLite with WAL mode")

# Search
results = store.search("SQLite", limit=10)

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
data = store.export_all()
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

## Security

- `LMEM_HOME` is validated against path traversal, system directories, and symlink attacks.
- Write paths are validated against system directories and symbolic links.
- URL validation (`is_safe_url`) blocks private/reserved IPs and SSRF vectors.
- All SQL queries use parameterized statements (no injection risk).
- Database files are created with mode `0600`; parent directories with `0o700`.
- Migration from `~/.lobsterdog/` skips symlinks (using `follow_symlinks=False`).

## License

MIT