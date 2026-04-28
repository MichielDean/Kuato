# Workflow Rules

## Git Sync — Before Any Code Work

**ALWAYS sync the repository before editing any files in a git repo.** No exceptions.

Use the `git-sync` skill before touching code in any repository. The full procedure is in the skill, but the short version:

1. `git fetch --all --prune` — update remote refs
2. `git stash` if working tree is dirty
3. If on `main`: `git pull --rebase origin main`
4. If on a feature branch: `git rebase origin/main`
5. `git stash pop` if you stashed
6. Verify clean state

**Why:** Merge conflicts from stale branches are the #1 source of wasted time. Sync first, edit second. Every time.

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

## Skills

Agent skills are at `~/.agents/skills/`. Deploy with `npm install -g opencode-kuato` or `cp -rf`.

**Mandatory coding workflow — run these skills in order:**

1. **`task-intake`** — Before editing: discover stack, test/lint commands, conventions.
2. **`git-sync`** — Before editing: fetch, stash, rebase, verify clean state.
3. **`branch-strategy`** — Before committing: personal repos → direct push, shared repos → branch + PR + automerge. Commit message convention.
4. **`test-and-verify`** — After editing: format → lint → typecheck → tests. Stop and fix on any failure.

**On-demand skills:**

- **`pre-pr-review`** — Adversarial code review via isolated subagent before PR creation.
- **`critical-code-reviewer`** — Rigorous code review with severity tiers (Blocking/Required/Suggestions).
- **`visual-explainer`** — Generate HTML visual explanations of systems, plans, and data.

## Code Review and Audit Scope

Security isn't the only thing that matters. When reviewing or auditing code, check for **data integrity bugs** alongside vulnerability classes. Specifically:

- When a write path changes a field, do derived fields (embeddings, indexes, caches, denormalized copies, FTS tokens) stay in sync? Check every `.update()` call.
- When content changes, is the embedding recomputed or cleared? When metadata changes, are dependent indices updated?
- Look for "source-of-truth divergence" — places where the same fact lives in two forms and one can go stale.
- If a system has background jobs that regenerate derived data, verify they actually cover all mutation paths.
- This applies to any system with materialized views, search indices, cached computations, or vector embeddings.

## Proactive Issue Handling

**If you see something, do something. Never ignore build warnings, deprecation notices, test failures, or other issues you encounter during work.**

- **Easy fix?** Fix it yourself as part of your current task (or as a quick side fix before continuing).
- **Larger issue?** File an issue or ticket to track it, then continue with your task.
- **Uncertain scope?** Investigate briefly. If it's a one-liner, fix it. If it touches multiple files or needs design decisions, file an issue.

This applies to warnings in build output, deprecation notices, failing tests, lint errors, typos in docs, stale configs — anything you notice that isn't right. Don't silently move past problems.

**Verification before handoff:** Before telling the user something is done, verify it actually works. Open URLs, check responses, run the app, read the output. "It compiled" is not "it works."

**Narrate code changes as you make them.** Before opening a PR: say what you're changing and why. After opening: share the URL and say whether CI is running or being bypassed. When it merges: say so. Don't act and summarize — act with narration. The user should never have to ask what just happened.

## Continuity

Each session, you wake up fresh. Use your memory system or notes to persist knowledge across sessions. Search before assuming. Update when you learn.

If you change any of the instruction files, tell the user — they are your configuration, and they should know.

**The source of truth is the three decomposed files** in `harness/`: `identity.md`, `user.md`, and `rules.md`. The `opencode.json` `instructions` field loads all three. `harness/AGENTS.md` is a pointer stub — do not edit it for content. Never edit deployed copies directly — always edit the repo copies and redeploy:

```bash
# opencode.json references the three files directly, so no manual copy is
# needed if the repo is the working directory. If you deploy to a system
# that loads AGENTS.md as a single file, concatenate them:
cat harness/identity.md harness/user.md harness/rules.md > PATH/TO/AGENTS.md
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing — that leaves work stranded locally
- NEVER say "ready to push when you are" — YOU must push
- If push fails, resolve and retry until it succeeds
- ALWAYS commit and push after code changes, no exceptions. Direct push to main for personal repos. Branch + PR + automerge for shared repos.