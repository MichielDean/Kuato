# LLMem Installation

How to install and set up LLMem. [Back to README](../README.md)

## Installation

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

This installs the LLMem skills (llmem, introspection, git-sync, etc.) and session hooks into OpenCode.

```bash
cd LLMem/opencode-llmem
npm install
```

The `postinstall` script copies skill directories to `~/.agents/skills/` where OpenCode discovers them automatically. You still need the CLI installed (Option 1) for the hooks to call.

### Option 3: Copilot CLI plugin (skills + session hooks)

```bash
copilot plugin install MichielDean/LLMem:copilot-llmem
```

Or, if you've cloned the repo:

```bash
cd LLMem/copilot-llmem
npm install
```

The plugin copies skills and the `memory-assistant` agent to `~/.agents/`. You still need the CLI installed (Option 1) for the hooks to call.

### What to install

| Goal | What to install |
|------|----------------|
| Use the `llmem` CLI or Python library | Option 1 |
| Use LLMem with OpenCode | Option 1 + Option 2 |
| Use LLMem with Copilot CLI | Option 1 + Option 3 |
| Use LLMem with both OpenCode and Copilot | Option 1 + Option 2 + Option 3 |
| Just the skill files (no CLI, no hooks) | Copy `skills/` to `~/.agents/skills/` manually |

### Requirements

- Python 3.11+
- Node.js 20+ (for plugin installation only)
- [Ollama](https://ollama.com) running locally for extraction, embedding, and dreaming — **or** install `llmem[local]` for local embedding without any server — **or** set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` for cloud providers