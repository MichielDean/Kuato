# LLMem Providers

Provider abstraction layer for embedding and text generation, plus configuration reference. [Back to README](../README.md)

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

When `provider: local` is selected for embeddings, LLMem uses `SentenceTransformersProvider` which runs models locally via the `sentence-transformers` library — no server required. Install with `pip install ".[local]"`.