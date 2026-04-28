# opencode-kuato

Generic skills for OpenCode — git-sync, task-intake, test-and-verify, branch-strategy, critical-code-reviewer, pre-pr-review, and visual-explainer.

## Installation

```bash
npm install -g opencode-kuato
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

```bash
npm test
```

Or manually check that the skill directories exist:

```bash
ls ~/.agents/skills/
```

## License

MIT