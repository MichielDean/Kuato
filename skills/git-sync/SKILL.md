---
name: git-sync
description: Sync the current git repo with its remote before starting any code work. Prevents merge conflicts caused by stale branches. ALWAYS run before editing code in any repository. Handles: fetching remote, rebasing feature branches onto main, pulling latest on main, stashing dirty worktree, and conflict detection. Triggers on: "git sync", "sync repo", "sync branch", or implicitly before any code modification task in a git repository.
license: MIT
---

# Git Sync

**ALWAYS sync the repository before editing any files in a git repo. No exceptions.**

Merge conflicts from stale branches are the #1 source of wasted time. Sync first, edit second. Every time.

## Non-Interactive Environment

Agents run in non-interactive shells. Git must never open an interactive editor — it will hang forever.

**ALWAYS set these before any git operations in a non-interactive context:**

```bash
export GIT_EDITOR=true
export GIT_SEQUENCE_EDITOR=true
```

`true` is the /usr/bin/true command — it exits 0 immediately, accepting any editor invocation without modification. This prevents `git rebase --continue`, `git commit --amend`, and similar commands from opening Vim.

For global config (personal machines):

```bash
git config --global core.editor true
git config --global sequence.editor true
```

## When to Run

- Before ANY file edit in ANY git repository
- Before starting a new task that involves code changes
- Before creating a new branch
- When switching between tasks in the same repo

## Procedure

### Step 1: Fetch

```bash
git fetch --all --prune
```

Update all remote refs. The `--prune` removes stale remote-tracking branches.

### Step 2: Check for Dirty Working Tree

```bash
git status --porcelain
```

If the working tree has uncommitted changes (output is non-empty), stash them:

```bash
git stash push -m "auto-sync-$(date +%s)"
```

Note the stash ref for later — but `git stash pop` will use the top of stack by default.

### Step 3: Sync Based on Branch

**If on `main` or `master`:**

```bash
git pull --rebase origin main
```

Use rebase to maintain a clean, linear history. If on `master`, substitute accordingly.

**If on a feature branch:**

```bash
git rebase origin/main
```

Rebase the feature branch onto the latest origin/main. This keeps the branch up-to-date and produces clean merge commits later.

### Step 4: Restore Stashed Changes

If you stashed in Step 2:

```bash
git stash pop
```

If the pop fails due to conflicts, the stash is preserved. Resolve conflicts, then `git stash drop` once resolved.

### Step 5: Verify Clean State

```bash
git status
git log --oneline -3
```

Confirm:
- Working tree is clean (or has expected modifications)
- Current branch is correct
- Recent commits match expectations

## Conflict Handling

If a rebase or pull produces conflicts:

1. Check which files are conflicted: `git status`
2. Resolve each conflict in the file
3. Stage resolved files: `git add <file>`
4. Continue the rebase: `GIT_SEQUENCE_EDITOR=true git rebase --continue`
5. If the rebase is hopelessly broken: `git rebase --abort` and alert the user

**NEVER force push to main/master.**

## Edge Cases

- **Detached HEAD:** If `git rev-parse --abbrev-ref HEAD` returns `HEAD`, the repo is in detached HEAD state. Alert the user before proceeding.
- **No remote:** If there's no remote configured, skip the fetch and pull steps. Just verify the working tree is clean.
- **Multiple remotes:** `git fetch --all` handles this. The rebase/pull should target `origin/main` unless the project convention is different.
- **Shallow clone:** If `git rev-parse --is-shallow-repository` returns `true`, consider `git fetch --unshallow` if full history is needed for rebase. Otherwise, standard fetch works.

## Landing Changes — ALWAYS Push

**Work is NOT done until it is on the remote. No exceptions.**

Changes that exist only on disk are orphaned. An agent that edits files and doesn't push is worse than an agent that didn't edit at all — the next session won't know the work happened.

### After Making Changes

1. **Commit**: Stage and commit all modified/added files with a descriptive message
2. **Push**: `git push origin <branch>`
3. **Verify**: `git status` shows clean, `git log` shows your commit on the remote

### For Feature Work

Create a pull request and enable automerge:

```bash
git checkout -b <descriptive-branch-name>
# ... make changes, commit ...
git push -u origin <branch-name>
gh pr create --title "description" --body "$(cat <<'EOF'
## Summary
<what and why>
EOF
)"
gh pr merge <n> --repo <repo> --squash --auto
```

Automerge means: once CI is green, it merges without waiting. No abandoned PRs.

**If CI requires admin override** and you have authority:

```bash
gh pr merge <n> --squash --admin
```

Say explicitly that you used `--admin`.

### For Direct Push (When Allowed)

If the repo allows direct pushes to main (e.g., personal projects, personal config):

```bash
git add -A
git commit -m "descriptive message"
git push origin main
```

Still verify with `git status` and `git log` that it landed.

### Anti-Patterns

- **"Ready to push when you are"** — No. Push it. Always.
- **Leaving changes uncommitted** — The next session sees a dirty tree and has no idea why.
- **Opening a PR and walking away** — Enable automerge or stay until it merges.
- **Push failing and not retrying** — Resolve and push again. A failed push = work not saved.

## Quick Reference

```bash
# Full sync sequence
git fetch --all --prune
git stash push -m "auto-sync"  # only if dirty
git pull --rebase origin main  # if on main
# OR
git rebase origin/main         # if on feature branch
git stash pop                  # only if stashed
git status                     # verify
```