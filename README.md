# LLMem — Structured Memory with Semantic Search

[![CI](https://github.com/MichielDean/LLMem/actions/workflows/ci.yml/badge.svg)](https://github.com/MichielDean/LLMem/actions/workflows/ci.yml)

LLMem is a SQLite-backed memory store for LLM agents. It provides structured storage with full-text search (FTS5), optional vector similarity search (via sqlite-vec), an extensible type system, and a provider abstraction layer for LLM embeddings and text generation. A background dreaming cycle consolidates, decays, and merges memories over time.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and coding conventions.

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/INSTALLATION.md) | Install from source, optional extras, plugin setup |
| [Providers](docs/PROVIDERS.md) | Embedding/generation providers, fallback chains, configuration |
| [CLI Reference](docs/CLI.md) | All `llmem` commands and options |
| [Python API](docs/API.md) | MemoryStore, Retriever, extension points, database schema, module reference |
| [Integrations](docs/INTEGRATIONS.md) | OpenCode, Copilot CLI, custom tools, session hooks |
| [Configuration](docs/CONFIGURATION.md) | config.yaml reference, path resolution, dream settings |
| [Search Reranking](docs/RERANKING.md) | Multi-signal reranking, signal weights, type priority |
| [Dream Cycle & Extraction](docs/DREAM.md) | Dream phases, extraction pipeline, session hooks |
| [Security](docs/SECURITY.md) | Path validation, SSRF protection, credential handling, code indexing security |

## Installation

Quick install:

```bash
curl -sSL https://raw.githubusercontent.com/MichielDean/LLMem/main/setup.sh | bash
```

Or clone and run locally:

```bash
git clone https://github.com/MichielDean/LLMem.git
cd LLMem && ./setup.sh
```

For detailed installation options (extras, plugins, requirements), see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Skills

LLMem ships four skills focused on memory management. Agent workflow skills (git-sync, task-intake, test-and-verify, branch-strategy, critical-code-reviewer, pre-pr-review, visual-explainer) are distributed separately as part of agent harnesses.

| Skill | Description |
|-------|-------------|
| **llmem** | Manage LLMem memories — add, search, consolidate, dream, introspect, and track review outcomes. |
| **llmem-setup** | Install and configure LLMem for an agent harness — provider setup, skill registration, harness integration. |
| **introspection** | Operational reference for the introspection framework — self-assessment, sampajanna checks, error taxonomy. |
| **introspection-review-tracker** | Reference for the automated ReviewOutcomeTracker hook that persists review findings as self_assessment memories. |

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

After installing from source, verify everything works:

```bash
# Check the CLI is available
llmem --help

# Initialize config and database
llmem init

# Confirm the store is working
llmem stats

# Verify skills are discoverable (OpenCode plugin only)
ls ~/.agents/skills/llmem ~/.agents/skills/introspection ~/.agents/skills/git-sync
```

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

## Running Tests

```bash
python -m pytest
```

1349 Python tests and 142 JavaScript tests covering all providers, session adapters (OpenCode, Copilot, none), URL validation, configuration, security, session hooks, CLI commands, and edge cases.

## License

MIT