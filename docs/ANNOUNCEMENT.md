# Announcing LLMem — Structured Memory with Semantic Search for LLM Agents

We're excited to announce LLMem, an open-source SQLite-backed memory store designed for LLM agents. LLMem gives your agents persistent, searchable memory that goes far beyond simple key-value storage — combining full-text search (FTS5), optional vector similarity search, an extensible type system, and a provider abstraction layer for LLM embeddings and text generation.

LLMem ships with a background dreaming cycle that automatically consolidates, decays, and merges memories over time — keeping your agent's knowledge base clean and relevant without manual curation. The provider abstraction supports Ollama, OpenAI, Anthropic, and local sentence-transformers models, with graceful fallback so your agent always works even when individual providers are unavailable. A working memory inbox with attention scoring lets agents stage ephemeral observations before promoting them to long-term storage.

Getting started is simple: `pip install llmem` for the Python package, or `npm install -g opencode-llmem` for OpenCode session integration. LLMem runs on Python 3.11+ with a single runtime dependency (PyYAML) and uses SQLite for zero-config local storage. Optional extras add vector search (`pip install llmem[vec]`) and local embedding without any server (`pip install llmem[local]`).

Check out the [GitHub repository](https://github.com/MichielDean/LLMem) for full documentation, the [CONTRIBUTING.md](https://github.com/MichielDean/LLMem/blob/main/CONTRIBUTING.md) guide for how to get involved, and the [ecosystem page](https://github.com/MichielDean/LLMem/blob/main/docs/ECOSYSTEM.md) for OpenCode integration details. We welcome contributions — whether it's bug fixes, new features, or documentation improvements.
