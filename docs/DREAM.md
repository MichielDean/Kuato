# LLMem Dream Cycle & Extraction

Automated memory maintenance and session extraction pipelines. [Back to README](../README.md)

## Dream Cycle

The dream cycle performs automated memory maintenance during idle periods. It can be invoked manually via `llmem dream` (see [CLI Reference](CLI.md#llmem-dream)) or run automatically by a systemd timer.

- **Light phase:** Sort and deduplicate near-duplicate memories (cosine similarity ≥ threshold).
- **Deep phase:** Score, promote, decay, and merge memories. Also promotes inbox items to long-term memory (items with attention_score ≥ `dream.min_score` become permanent memories; lower-scored items are evicted). Decays confidence on idle memories. Boosts frequently accessed memories. LLM-assisted merging of similar pairs. Auto-links memories with high cosine similarity (≥ `dream.auto_link_threshold`, default 0.85) by creating `related_to` relations between them.
- **REM phase:** Extract themes from memory clusters and write a dream diary (read-only reflection).

Configuration is under the `dream:` key in `config.yaml`. Set `dream.enabled: false` to disable.

Extension hooks can be registered to run custom logic after each dream phase. See [Extension Points — Dream Hook Registry](API.md#dream-hook-registry) for details.

## Extraction and Hooks

The `hooks` module provides automatic extraction from session transcripts:

- `process_file()`: Extract memories from a transcript file.
- `process_session()`: Extract from an OpenCode session ID.
- `process_all_session_sources()`: Process all session sources (delegates to `session_hooks.process_opencode_sessions`).
- Self-assessment extraction with structured error taxonomy.
- Correction detection for identifying mistakes.

The `session_hooks` module provides `process_opencode_sessions()` — the full pipeline that reads OpenCode sessions from the SQLite database, chunks them by message boundaries, and runs extraction and embedding.

The `extract` module uses Ollama (default: `qwen2.5:1.5b`) to extract structured memories from text. The `embed` module generates embeddings using Ollama (default: `nomic-embed-text`).