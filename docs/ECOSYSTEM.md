# LLMem Ecosystem — OpenCode Integration

LLMem integrates with [OpenCode](https://opencode.ai) through the `opencode-llmem` npm package, providing session lifecycle hooks that inject relevant memories and extract new ones automatically.

## Installation

```bash
npm install -g opencode-llmem
```

After installation, the `postinstall` script registers LLMem's session hooks into your OpenCode configuration. No manual setup is required — the hooks are discovered automatically on the next OpenCode session.

## Session Hooks

The `opencode-llmem` package registers three session lifecycle hooks:

| Event | Hook | Behavior |
|-------|------|----------|
| `session.created` | `on_created(session_id)` | Queries the memory store for relevant memories and writes a context file for the new session. |
| `session.idle` | `on_idle(session_id)` | Extracts memories from the session transcript after a 30-second debounce period. |
| `session.compacting` | `on_compacting(session_id)` | Injects high-confidence key memories to preserve context during compaction. |

These hooks run the `llmem` CLI under the hood, so the Python package must be installed (`pip install llmem`) and available on your `PATH`.

## Configuring Providers

LLMem uses a provider abstraction layer to decouple embedding and text generation from any specific LLM backend. Provider configuration is managed in `~/.config/llmem/config.yaml`.

See the [Provider Configuration](../README.md#provider-configuration) section of the README for the full YAML reference, including settings for Ollama, OpenAI, Anthropic, and local (sentence-transformers) providers.

### Quick Provider Setup

The simplest way to configure providers is to run:

```bash
llmem init
```

This interactive command detects available providers (Ollama, OpenAI, Anthropic, local) and writes a sensible `config.yaml`. For non-interactive setup:

```bash
llmem init --non-interactive
```

## OpenCode Tools

LLMem also ships six type-safe OpenCode tools that provide direct access to the memory store from within OpenCode sessions:

| Tool | Description |
|------|-------------|
| `llmem-search` | Search memories via FTS5 full-text search |
| `llmem-add` | Add a new memory with type, content, and confidence |
| `llmem-context` | Retrieve formatted context for a topic within a character budget |
| `llmem-invalidate` | Invalidate a memory by ID with optional reason |
| `llmem-stats` | Show memory statistics (total, active, expired, by type) |
| `llmem-hook` | Run the extraction hook on session transcripts |

These tools are installed alongside the root `llmem` npm package (the skills distribution package) and are discovered by OpenCode automatically.

## Architecture

```
OpenCode Session
    │
    ├── session.created ──► on_created() ──► llmem CLI ──► MemoryStore (SQLite)
    │                                                      ├─ FTS5 search
    │                                                      └─ Vector search
    ├── session.idle ────► on_idle() ────► llmem CLI ──► ExtractionEngine
    │                                                      └─ Embed + store
    └── session.compacting ► on_compacting() ► llmem CLI ► High-confidence memories
```

The `opencode-llmem` package acts as a thin adapter between OpenCode's plugin system and the `llmem` Python CLI. All memory operations go through the Python package, which provides the full-featured store, search, and extraction pipeline.
