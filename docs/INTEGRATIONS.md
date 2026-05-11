# LLMem Integrations

Integration guides for OpenCode, Claude Code, Copilot CLI, and custom tool implementations. [Back to README](../README.md)

## Architecture: Plugin-First, Zero-Config

LLMem uses a **plugin-first architecture**. The plugin handles automatic memory lifecycle hooks — context injection at session start, memory extraction on idle/end, and context preservation during compaction. **No manual instruction editing required.**

```
Agent Session
    │
    ├── Plugin (auto, no instructions needed)
    │   ├── session.created/start → llmem stats + search → inject context
    │   ├── session.idle/end      → llmem hook idle/ending → extract + introspect
    │   └── session.compacting    → llmem context --compacting → preserve memories
    │
    ├── Skills (on-demand, loaded by trigger)
    │   ├── llmem                      → CLI reference, memory types, commands
    │   ├── llmem-setup                → Install and configure LLMem
    │   ├── introspection              → Self-assessment framework, error taxonomy
    │   └── introspection-review-tracker → Review outcome tracking
    │
    └── Custom Tools (structural, zero-instruction)
        ├── llmem-search   → Search memories
        ├── llmem-add      → Add a memory
        ├── llmem-context  → Get context for a topic
        ├── llmem-invalidate → Soft-delete a memory
        ├── llmem-stats    → Show memory statistics
        └── llmem-hook     → Run extraction hook
```

### Why Plugin-First?

- **No instruction pollution.** The plugin injects context automatically. Skills load on-demand. Your AGENTS.md/CLAUDE.md stays clean.
- **Platform-agnostic core.** Same Go binary, same skills, same CLI across OpenCode, Claude Code, and Copilot CLI. Only the thin adapter plugin differs.
- **Single install command.** `npm install` deploys skills, plugins, and tools for your platform.
- **No per-platform instruction docs to maintain.** The plugin handles behavioral injection, not 80-line instruction blocks.

## OpenCode Integration

### Installation

#### Quick install (one command)

```bash
curl -sSL https://raw.githubusercontent.com/MichielDean/LLMem/main/setup.sh | bash
```

#### Manual install

```bash
git clone https://github.com/MichielDean/LLMem.git
cd LLMem && ./setup.sh
```

The setup script installs the Go binary, runs `npm install` (which deploys skills, plugins, and tools), and initializes the database.

### Plugin

The OpenCode plugin (`plugins/opencode/llmem.js`) handles:

| Event | Action |
|-------|--------|
| `session.created` | Runs `llmem stats` + `llmem search behavioral/proposed` — injects results as log context |
| `session.idle` | Runs `llmem hook idle <session_id>` — extracts memories from transcript |
| `session.ending` | (not yet wired — agent-driven via skills) |
| `experimental.session.compacting` | Runs `llmem context --compacting` — preserves key memories |

The plugin is deployed to `~/.config/opencode/plugins/llmem.js` by the install script. No manual configuration needed — OpenCode auto-discovers plugins in this directory.

To explicitly add it to your `opencode.json`:

```json
{
  "plugin": ["llmem"]
}
```

### Custom Tools

The `.opencode/tools/` directory contains six type-safe tools that the agent can invoke directly without loading a skill:

| Tool | CLI Equivalent | Description |
|------|----------------|-------------|
| `llmem-search` | `llmem search <query> --json` | Search memories; returns JSON array |
| `llmem-add` | `llmem add --type T --content C` | Add a memory; returns ID and type |
| `llmem-context` | `llmem search <query> --limit 20` | Retrieve formatted context for a topic |
| `llmem-invalidate` | `llmem invalidate <ID>` | Invalidate a memory by ID |
| `llmem-stats` | `llmem stats` | Show memory statistics |
| `llmem-hook` | `llmem hook <type>` | Run the extraction hook |

### Skills

Four skills ship with LLMem and are installed to `~/.agents/skills/`:

| Skill | Description |
|-------|-------------|
| **llmem** | Full CLI reference, memory types, commands, dream config |
| **llmem-setup** | Install, configure, and integrate LLMem into a harness |
| **introspection** | Self-assessment framework, error taxonomy, vigilance checks |
| **introspection-review-tracker** | Review outcome tracking for code reviews |

### Optional: AGENTS.md Pointer

If you want a persistent reminder (not required — the plugin handles context injection), add this minimal line:

```markdown
## Memory

Plugin-managed. Search when uncertain: `llmem search "topic"`. Add when you learn: `llmem add --type fact --content "..."`.
```

### Configuring Providers

LLMem uses a provider abstraction layer. The simplest configuration:

```bash
llmem init
```

This detects available providers (Ollama, OpenAI, Anthropic, local) and writes `config.yaml`. For non-interactive setup:

```bash
llmem init --non-interactive
```

See [Provider Configuration](PROVIDERS.md) for the full YAML reference.

## Claude Code / Copilot CLI Integration

The agent plugin (`plugins/agent/`) uses the standard `.claude-plugin/` format and is compatible with both Claude Code and GitHub Copilot CLI.

### Plugin Structure

```
plugins/agent/
├── .claude-plugin/
│   └── plugin.json          # Plugin manifest
├── hooks/
│   └── hooks.json           # Session lifecycle hooks
└── skills/
    ├── llmem/SKILL.md
    ├── llmem-setup/SKILL.md
    ├── introspection/SKILL.md
    └── introspection-review-tracker/SKILL.md
```

### Installation

```bash
claude plugin install ~/.claude/plugins/llmem
# Or for testing:
claude --plugin-dir ./plugins/agent
```

The `hooks.json` declares:

| Event | Action |
|-------|--------|
| `SessionStart` | Runs `llmem stats` + behavioral + proposed searches — stdout injected as context |
| `SessionEnd` | Runs `llmem hook ending` — extracts memories and runs introspection |
| `PreCompact` | Runs `llmem context --compacting` — preserves key memories |

The `SessionStart` hook's **stdout is added as context that Claude can see and act on** — this is the key mechanism for zero-config integration. The agent sees the memory context at session start without any AGENTS.md modifications.

### Skills

Skills are namespaced under the plugin: `/llmem:llmem`, `/llmem:llmem-setup`, etc. They're discovered automatically when the plugin is enabled.

### Optional: CLAUDE.md Pointer

If you want a persistent reminder (not required — the `SessionStart` hook handles context injection):

```markdown
## Memory

Plugin-managed. Search when uncertain: `llmem search "topic"`. Add when you learn: `llmem add --type fact --content "..."`.
```

## Future Integrations

After installation, verify the skills are discoverable:

```bash
ls ~/.agents/skills/llmem ~/.agents/skills/introspection ~/.agents/skills/introspection-review-tracker

# OpenCode plugin
ls ~/.config/opencode/plugins/llmem.js

# Claude Code / Copilot plugin
ls ~/.claude/plugins/llmem/.claude-plugin/plugin.json
```

Run the bundled tests:

```bash
npm test
```