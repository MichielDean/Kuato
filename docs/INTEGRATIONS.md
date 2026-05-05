# LLMem Integrations

Integration guides for OpenCode, Copilot CLI, and custom tool implementations. [Back to README](../README.md)

## OpenCode Integration (Ecosystem)

LLMem integrates with [OpenCode](https://opencode.ai) through the `opencode-llmem` npm package, providing session lifecycle hooks that inject relevant memories and extract new ones automatically.

### Installation

#### Quick install (one command)

```bash
curl -sSL https://raw.githubusercontent.com/MichielDean/LLMem/main/setup.sh | bash
```

#### Manual install

If you installed from source instead of using the setup script, run `npm install` from the `opencode-llmem/` directory inside the LLMem repo. After installation, the `postinstall` script copies skill directories to `~/.agents/skills/` where OpenCode discovers them. No manual setup is required — the hooks are discovered automatically on the next OpenCode session.

### Session Hooks

The `opencode-llmem` package registers three session lifecycle hooks:

| Event | Hook | Behavior |
|-------|------|----------|
| `session.created` | `on_created(session_id)` | Queries the memory store for relevant memories and writes a context file for the new session. |
| `session.idle` | `on_idle(session_id)` | Extracts memories from the session transcript after a 30-second debounce period. |
| `session.compacting` | `on_compacting(session_id)` | Injects high-confidence key memories to preserve context during compaction. |

These hooks run the `llmem` CLI under the hood, so the Python package must be installed from source and available on your `PATH`. Install it from the repo root:

```bash
git clone https://github.com/MichielDean/LLMem.git
cd LLMem
pip install .
```

### Configuring Providers

LLMem uses a provider abstraction layer to decouple embedding and text generation from any specific LLM backend. Provider configuration is managed in `~/.config/llmem/config.yaml`.

See the [Provider Configuration](PROVIDERS.md#provider-configuration) section for the full YAML reference, including settings for Ollama, OpenAI, Anthropic, and local (sentence-transformers) providers.

#### Quick Provider Setup

The simplest way to configure providers is to run:

```bash
llmem init
```

This interactive command detects available providers (Ollama, OpenAI, Anthropic, local) and writes a sensible `config.yaml`. For non-interactive setup:

```bash
llmem init --non-interactive
```

### OpenCode Tools

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

### Architecture

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

### OpenCode Custom Tools

LLMem ships six type-safe OpenCode tools that replace raw `llmem` CLI calls with described, schema-validated tool invocations. Tools run `llmem` as a subprocess with a 60-second timeout and return strings — errors are prefixed with `Error:`.

| Tool | CLI Equivalent | Description |
|------|----------------|-------------|
| `llmem-search` | `llmem search <query> --json` | Search memories via FTS5; returns JSON array |
| `llmem-add` | `llmem add --type T --content C` | Add a new memory; returns ID and type |
| `llmem-context` | `llmem search <query> --json --limit 20` | Retrieve formatted context for a topic; returns a context block truncated to a character budget |
| `llmem-invalidate` | `llmem invalidate <ID>` | Invalidate a memory by ID with optional reason |
| `llmem-stats` | `llmem stats` | Show memory statistics (total, active, expired, by type) |
| `llmem-hook` | `llmem hook` | Run the extraction hook on session transcripts |

#### `llmem-search`

```
llmem-search(query: string, type?: string, limit?: number)
```

- `query` (required): Search query for FTS5 full-text search.
- `type` (optional): Filter by memory type (fact, decision, preference, etc.).
- `limit` (optional, min 1): Maximum number of results (default: 20).

Returns a JSON string of matching memory objects on success, or an `Error:` string on failure (CLI error, invalid JSON, non-array response).

#### `llmem-add`

```
llmem-add(type: string, content: string, source?: string, confidence?: number)
```

- `type` (required): Memory type (`fact`, `decision`, `preference`, `event`, `project_state`, `procedure`, `conversation`, `self_assessment`).
- `content` (required): Memory content text.
- `source` (optional): Source of memory (default: `manual`).
- `confidence` (optional, 0–1): Confidence score (default: 0.8).

Returns the memory ID and type on success, or an `Error:` string on failure (invalid type, CLI error).

#### `llmem-context`

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

#### `llmem-invalidate`

```
llmem-invalidate(id: string, reason?: string)
```

- `id` (required): Memory ID to invalidate.
- `reason` (optional): Reason for invalidation.

Returns a confirmation string on success, or an `Error:` string if the memory is not found or the CLI fails.

#### `llmem-stats`

```
llmem-stats()
```

Takes no arguments. Returns a formatted statistics string (total, active, expired counts, and breakdown by type), or an `Error:` string on failure.

#### `llmem-hook`

```
llmem-hook(force?: boolean, noEmbed?: boolean, noIntrospect?: boolean, file?: string, directory?: string)
```

- `force` (optional): Force re-extraction of already-processed sessions.
- `noEmbed` (optional): Skip embedding generation (faster, no Ollama required).
- `noIntrospect` (optional): Skip introspection for trivial sessions.
- `file` (optional): Path to a specific transcript file. Resolved relative to `context.directory`; paths escaping the directory scope are rejected.
- `directory` (optional): Path to a directory of transcript files. Same containment check as `file`.

Returns the extraction result on success, or an `Error:` string on failure (path traversal, CLI error).

#### Error Handling

All tools follow a consistent contract:

- **Success**: returns the CLI output as a string.
- **Error**: returns a string starting with `Error:`.
- **CLI not found**: returns `"Error: llmem CLI not found on PATH"` (exit code 127).
- **Timeout**: returns `"Error: llmem <cmd> timed out after 60000ms"` (exit code 124).
- Tools never throw — all errors are returned as strings.

### OpenCode Integration (npm)

The `opencode-llmem` npm package provides a JavaScript plugin that integrates LLMem session hooks into OpenCode via its plugin interface.

#### Installation

```bash
npm install opencode-llmem
```

The `postinstall` script copies the hook source files to `~/.agents/plugins/llmem/` so OpenCode can discover and load them.

#### Usage

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

#### Configuration

The JavaScript hooks read configuration from `LMEM_HOME/config.yaml` (or `~/.config/llmem/config.yaml` by default). The `opencode.context_dir` key controls where context files are written.

## Future Integrations

After installation, verify the skills are discoverable:

```bash
ls ~/.agents/skills/llmem ~/.agents/skills/introspection ~/.agents/skills/introspection-review-tracker
ls ~/.agents/agents/memory-assistant.agent.md
```

Run the bundled tests:

```bash
npm test
```