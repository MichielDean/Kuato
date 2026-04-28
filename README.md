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