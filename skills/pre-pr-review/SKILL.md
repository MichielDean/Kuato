---
name: pre-pr-review
description: >
  Adversarial pre-PR code review with author isolation and introspection-illusion
  detection. ALWAYS run before creating any pull request. Use when about to call
  gh pr create, when asked to review code before pushing, or any time code changes
  need a second opinion before going to GitHub. Spawns an isolated reviewer subagent
  (no author context) using the critical-code-reviewer methodology. Blocks PR
  creation until Blocking and Required issues are addressed. Triggers on:
  "run pre-pr-review", "review before PR", "check the code", "self-review",
  or any intent to create a pull request.
license: MIT
---

# Pre-PR Review Workflow

Adversarial code review via isolated subagent. Author isolation is the key mechanism:
the reviewer subagent receives only the diff, modified file contents, and originals —
no session history, no knowledge it wrote the code.

## When to Run

- Before every `gh pr create` call
- When explicitly asked to review a branch or diff
- When asked to review an existing PR number

## Step 1: Gather the Diff

**For current branch (pre-PR):**
```bash
cd <repo> && git diff main...HEAD
```

**For an existing PR:**
```bash
export GH_TOKEN=$(pass github/gh-token 2>/dev/null)
gh pr diff <number>
```

Also collect the full content of each modified file (after change):
```bash
git diff main...HEAD --name-only | xargs -I{} sh -c 'echo "=== {} ===" && cat {}'
```

## Step 2: Gather Original Files (Behavioral Regression Context)

For each modified file, also collect the **original version** before the change:
```bash
# For current branch:
git diff main...HEAD --name-only | xargs -I{} sh -c 'echo "=== {} ===" && git show main:{}'
# For an existing PR:
gh pr diff <number> --name-only | xargs -I{} sh -c 'echo "=== {} ===" && git show HEAD~1:{}'
```

This is critical — the reviewer must compare old vs new to catch behavioral regressions
(removed timeouts, dropped logging, missing error context, etc.).

## Step 3: Spawn the Reviewer Subagent

Spawn a subagent with complete author isolation. The subagent gets: diff + original files + modified files. No session history.

Use this task template (fill in {DIFF}, {ORIGINAL_FILES}, {FILES}):

```
You are a senior engineer conducting an adversarial PR review.
You did NOT write this code. Your job is to find every bug, security issue,
design flaw, and inconsistency. Be constructively brutal.

Severity tiers: Blocking | Required | Strong Suggestions | Noted
Output format: file:line reference, issue, why it matters, fix.

## Core Checklist — verify ALL explicitly:

### Logic & Type Safety
- Unhandled error paths, silent swallows with no logging
- Type safety violations, `any` abuse, overly broad Record/map types
- Import-time side effects (constructors that throw, eager singletons)
- API-specific message ordering (e.g. Anthropic requires `user` as first role)
- Async methods that never await — lying signatures

### Operational Concerns (never skip)
- Every network call: timeout present? What happens on indefinite hang?
- Every caught error: logged at warn/error before swallowing?
- Every external API wrapper: unit tests present? No tests = **Required**
- Retry behavior: present or explicitly documented as absent?

### Behavioral Regressions (compare originals vs modified)
- What did the original code do that the new code silently dropped?
- Timeouts, logging, error context, metrics — any regression = **Blocking**
- Stale comments/docstrings describing removed or changed behavior = Required

### Naming & Structure
- Misleading type aliases, lying import aliases, dead parameters
- Methods named as LLM calls that do pure local computation
- `async` on methods with no await

### Introspection Illusion (outside-view principle)
- Introspection illusion: Is the author trusting their own reasoning without evidence? Are there claims ("this handles X") that no test or log output verifies?
- Self-serving assumptions: Is the author giving themselves passes they'd flag in someone else's code? Are there "good enough" shortcuts, assumed intents, or skipped edge cases?
- Outside-view verification: What would an adversarial reviewer see that the author doesn't? Check specifically for: error paths with no logging, success assumptions in failure-prone code, and "obviously correct" logic that hasn't been tested.

## Diff:
<diff>
{DIFF}
</diff>

## Original Files (before change) — regression detection:
<originals>
{ORIGINAL_FILES}
</originals>

## Modified Files (after change):
<files>
{FILES}
</files>

Return ONLY the structured review. No preamble.
```

## Step 4: Evaluate Findings

- **Blocking** → must fix before PR. Do not call `gh pr create`.
- **Required** → fix before PR unless the author explicitly waives.
- **Strong Suggestions** → fix if time permits; note in PR description if skipped.
- **Noted** → mention in PR description, no action required.

## Step 5: Fix and Re-Review

If Blocking or Required issues exist:
1. Fix them in the codebase
2. Rebuild and verify tests pass
3. Re-run this skill (re-diff, re-spawn reviewer)
4. Only proceed when reviewer returns no Blocking/Required items

## Step 6: Create the PR

Only after a clean review pass:
```bash
gh pr create --title "..." --body "..."
```

Include in the PR body:
- Any Strong Suggestions deliberately skipped (with rationale)
- "Pre-PR adversarial review passed: no Blocking/Required issues"

## Notes

- Keep subagent task under ~8k chars; truncate diff if needed (prioritize changed lines over package-lock)
- For large PRs, split by file and run multiple subagent passes
- Reviewer subagent needs no tools — text-only response