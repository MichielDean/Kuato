# LLMem Installation

How to install and set up LLMem. [Back to README](../README.md)

## Python Installation

LLMem is not yet published on PyPI or npm. Install from source.

### Quick install (one command)

```bash
curl -sSL https://raw.githubusercontent.com/MichielDean/LLMem/main/setup.sh | bash
```

Or clone and run locally:

```bash
git clone https://github.com/MichielDean/LLMem.git
cd LLMem && ./setup.sh
```

The setup script clones the repo, installs the Python package, installs plugins, runs `llmem init --non-interactive`, and verifies everything works. It's idempotent — safe to re-run. Run `./setup.sh --help` for options (`--extras vec,local`, `--plugin opencode|copilot|both|none`, `--repo PATH`).

### Option 1: From source (CLI + library)

This gives you the `llmem` command-line tool and the Python library.

```bash
git clone https://github.com/MichielDean/LLMem.git
cd LLMem
pip install .
```

**With optional extras:**

```bash
# Vector similarity search (sqlite-vec)
pip install ".[vec]"

# Local embedding without any server (sentence-transformers)
pip install ".[local]"

# Both + dev dependencies
pip install ".[vec,local,dev]"
```

**Initialize and verify:**

```bash
llmem init
llmem stats
```

`llmem init` creates `~/.config/llmem/` with `config.yaml` and initializes the SQLite database. It detects available providers (Ollama, OpenAI, Anthropic, local) automatically.

### Option 2: OpenCode plugin (skills + session hooks)

This installs the LLMem skills (llmem, llmem-setup, introspection, introspection-review-tracker) and session hooks into OpenCode.

```bash
cd LLMem/opencode-llmem
npm install
```

The `postinstall` script copies skill directories to `~/.agents/skills/` where OpenCode discovers them automatically. You still need the CLI installed (Option 1) for the hooks to call.

### What to install

| Goal | What to install |
|------|----------------|
| Use the `llmem` CLI or Python library | Option 1 |
| Use LLMem with OpenCode | Option 1 + Option 2 |
| Just the skill files (no CLI, no hooks) | Copy `skills/` to `~/.agents/skills/` manually |

### Requirements

- Python 3.11+
- Node.js 20+ (for plugin installation only)
- [Ollama](https://ollama.com) running locally for extraction, embedding, and dreaming — **or** install `llmem[local]` for local embedding without any server — **or** set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` for cloud providers

## Go Installation

The Go implementation provides the core `MemoryStore` as a pure-Go library with no CGo dependency. It uses `modernc.org/sqlite` (pure-Go SQLite driver) and `pressly/goose` for schema migrations.

### Requirements

- Go 1.26.1+

### Install as a dependency

```bash
go get github.com/MichielDean/LLMem
```

### Build from source

```bash
git clone https://github.com/MichielDean/LLMem.git
cd LLMem
make build
```

### Run tests

```bash
make test
# or
go test ./...
```

### What's included

| Package | Description |
|---------|-------------|
| `internal/store` | Core `MemoryStore` — add, get, update, invalidate, delete, search, relations, import/export, vector search |
| `internal/config` | Configuration loading from YAML, path resolution, defaults |
| `internal/dream` | Dream consolidation cycle (light, deep, REM phases) |
| `internal/extract` | LLM-based memory extraction via Ollama |
| `internal/introspect` | Failure analysis (`IntrospectFailure`) and lesson learning (`LearnLesson`) |
| `internal/ollama` | Ollama `/api/generate` and `/api/tags` client |
| `internal/paths` | Path resolution, validation, migration from legacy `~/.lobsterdog/` |
| `internal/session` | Session lifecycle hooks (`OnCreated`, `OnIdle`, `OnCompacting`, `OnEnding`) |
| `internal/systemd` | Systemd service/timer unit generation for dream cycle |
| `internal/taxonomy` | Error taxonomy constants for self_assessment memories |
| `cmd/llmem` | CLI entrypoint — 17 subcommands (add, get, search, list, stats, update, invalidate, delete, export, import, init, metrics, dream, introspect, learn, track-review, context, hook) |
| `migrations/` | 7 embedded SQL migrations (shared schema with Python) |
| `Makefile` | Build, test, lint, clean targets |

### Key differences from Python

| Feature | Python | Go |
|---------|--------|-----|
| SQLite driver | Built-in `sqlite3` | `modernc.org/sqlite` (pure Go, no CGo) |
| Vector search | `sqlite-vec` Python package | `vec0` virtual table (pure Go via modernc) |
| Migrations | Manual numbered SQL files | `pressly/goose` with embedded SQL files |
| CLI | Full-featured (24 commands) | Core commands (17 commands) |
| Embeddings/Providers | Ollama, OpenAI, Anthropic, local | Ollama (`/api/generate` and `/api/embeddings`) |
| Reranking | RRF + multi-signal | FTS5-only (hybrid coming) |
| Session hooks | Full lifecycle | Full lifecycle (Go implementation) |
| Dream cycle | Full (light, deep, REM) | Full (light, deep, REM) |

The Go `MemoryStore` shares the **exact same database schema** as Python. You can use them interchangeably — a database created by Python is readable by Go, and vice versa.