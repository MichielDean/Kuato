---
name: branch-strategy
description: Enforce consistent branching, commit messages, and push strategies across all repositories. Prevents inconsistent commit styles, wrong push targets, and orphaned work. ALWAYS consult before deciding whether to branch, push directly, or create a PR. Triggers on: "make a commit", "push this", "create a branch", "open a PR", or implicitly before any git commit/push operation in a repository.
license: MIT
---

# Branch Strategy

**ALWAYS use the right push strategy for the repo. No exceptions.**

Inconsistent commits, wrong-branch pushes, and orphaned work come from guessing. The strategy is deterministic based on repo type — there's never a judgment call.

## When to Run

- Before creating a commit
- Before creating a branch
- Before pushing to a remote
- Before opening a PR

## Repo Classification

Every repo falls into one of two categories. The category determines the entire strategy.

### Personal Config Repos — Direct Push

Repos that are single-user, configuration/personal, or have no CI:

- Personal config repos (dotfiles, scripts)
- Dotfiles, personal scripts
- Any repo without branch protection or CI

**Strategy:** Work on `main`. Commit. Push directly.

```bash
git add -A
git commit -m "<type>: <concise description>"
git push origin main
```

### Project/Shared Repos — Branch + PR + Automerge

Repos with CI, collaborators, or branch protection:

- Shared project repos with CI
- `ScaledTest` (shared project)
- Any repo with `.github/workflows/` or branch protection

**Strategy:** Create a feature branch. Push. Open PR. Enable automerge.

```bash
git checkout -b <type>/<short-description>
# ... make changes ...
git add -A
git commit -m "<type>: <concise description>"
git push -u origin <branch-name>
gh pr create --title "<type>: <description>" --body "$(cat <<'EOF'
## Summary
<what changed and why>
EOF
)"
gh pr merge <number> --repo <repo> --squash --auto
```

If you don't have `gh` or PR access, push the branch and tell the user to create the PR.

## Commit Message Convention

### Format

```
<type>: <concise description in imperative mood>
```

### Types

| Type | Use for |
|------|---------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code restructure, no behavior change |
| `test` | Adding or fixing tests |
| `docs` | Documentation only |
| `chore` | Build, deps, config, tooling |
| `style` | Formatting, whitespace (no logic change) |
| `perf` | Performance improvement |

### Rules

- **Imperative mood**: "add feature" not "added feature" or "adds feature"
- **Lowercase start**: "fix: resolve null pointer in handler" not "Fix: Resolve..."
- **No period at end**
- **Concise**: 50 chars or less for the subject line
- **Body if needed**: Blank line after subject, then explain WHY not WHAT
- **One logical change per commit**: Don't bundle unrelated fixes

### Examples

```
feat: add retry logic to castellarius pipe runner
fix: handle empty config in aqueduct initialization
refactor: extract state machine transitions into separate module
test: add coverage for cataractae error paths
chore: update go dependencies
docs: add troubleshooting guide for docker setup
```

### Anti-patterns

```
updated stuff                    # What stuff? Why?
fix bug                          # Which bug? Fixed how?
WIP                              # Never commit WIP to shared repos
feat: I added a new feature     # Not imperative mood
fix: fixed the null pointer.    # Period, past tense
```

## Branch Naming

Only for shared/project repos. Personal repos use `main` directly.

### Format

```
<type>/<short-description>
```

### Types

| Prefix | Use for |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `refactor/` | Restructuring |
| `chore/` | Maintenance |
| `docs/` | Documentation |

### Rules

- Lowercase, hyphen-separated
- Descriptive but short (3-5 words max after prefix)
- Match the commit type

### Examples

```
feat/retry-logic
fix/null-pointer-handler
refactor/state-machine
chore/update-deps
```

## PR Rules

1. **Always enable automerge** after creating a PR:
   ```bash
   gh pr merge <n> --repo <r> --squash --auto
   ```

2. **Never abandon an open PR.** If CI is green, it should merge. If you don't have merge access, tell the user.

3. **Admin merge** only when you have authority and CI is green:
   ```bash
   gh pr merge <n> --squash --admin
   ```
   Say explicitly that you used `--admin`.

4. **PR body** must include:
   - What changed and why
   - Any known limitations or skipped suggestions from review

5. **Squash merge** preferred. Keeps main history clean.

## Handling Edge Cases

- **Unsure which category?** Check for `.github/workflows/` or `CODEOWNERS`. If either exists, treat as shared.
- **Hotfix in a shared repo?** Branch from main as `fix/hotfix-description`, PR, automerge. Even hotfixes go through CI.
- **Multiple related changes?** One branch, multiple commits on that branch, one PR. Don't open a PR per commit.
- **Direct push rejected by branch protection?** That confirms it's a shared repo. Switch to branch + PR strategy.