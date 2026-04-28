---
name: test-and-verify
description: Discover and run the correct test, lint, typecheck, and format commands for a project before and after making code changes. Prevents shipping broken code by enforcing quality gates as a mandatory step. ALWAYS run after editing code in any repository. Triggers on: "run tests", "verify changes", "check if it passes", or implicitly after any code modification in a git repository.
license: MIT
---

# Test and Verify

**ALWAYS verify your changes before pushing. No exceptions.**

Unverified code is broken code until proven otherwise. Running quality gates is not optional — it's the difference between a working change and a regression you'll have to fix later.

## When to Run

- After ANY code change, before committing
- After pulling/merging if CI status is unknown
- When asked to verify, validate, or check anything
- Before opening a PR (quality gates must pass)

## Procedure

### Step 1: Recover Project Quality Gates

If you recently ran `task-intake`, you already know the commands. Otherwise:

Check project notes for known test/lint commands. If notes have the commands, use them. If not, run `task-intake` first — you can't verify what you haven't discovered.

### Step 2: Run Quality Gates in Order

Run these sequentially. **Stop and fix if any fail.** Do not skip a failing gate and continue to the next.

#### 1. Format Check

Formatting issues are noise in diffs. Fix them first so subsequent output is clean.

```bash
# Python
ruff format --check . 2>/dev/null || black --check . 2>/dev/null
# Go
gofmt -d . 2>/dev/null
# TypeScript/JavaScript
npm run lint -- --fix 2>/dev/null || npx prettier --check . 2>/dev/null
```

If format check fails, fix it:
```bash
# Python
ruff format . || black .
# Go
gofmt -w .
# TypeScript/JavaScript
npm run lint -- --fix || npx prettier --write .
```

#### 2. Lint

```bash
# Python
ruff check . 2>/dev/null || flake8 . 2>/dev/null
# Go
go vet ./... && golangci-lint run 2>/dev/null
# TypeScript/JavaScript
npm run lint 2>/dev/null || npx eslint . 2>/dev/null
```

If lint fails, fix the reported issues and re-run.

#### 3. Type Check

```bash
# Python
mypy . 2>/dev/null || pyright 2>/dev/null
# Go (built into compiler, but verify)
go build ./...
# TypeScript
npx tsc --noEmit 2>/dev/null || npm run typecheck 2>/dev/null
```

If type check fails, fix the type errors and re-run.

#### 4. Tests

```bash
# Python
pytest || python -m pytest
# Go
go test ./...
# Node.js/TypeScript
npm test 2>/dev/null || npx vitest run 2>/dev/null || npx jest 2>/dev/null
# Makefile-driven
make test
```

If tests fail:
1. Read the failure output carefully
2. Determine if your change caused it or if it was already failing
3. Fix your change or fix the test — do not leave failing tests
4. Re-run the full test suite

### Step 3: Verify No Regressions

After all gates pass:

1. **Check git diff**: Review your changes one more time
   ```bash
   git diff
   ```

2. **Check for unintended changes**: Make sure you didn't modify files outside your intended scope
   ```bash
   git diff --stat
   ```

3. **Check for debug artifacts**: No `console.log`, `print("DEBUG")`, `fmt.Println("HERE")`, or temporary files left behind

### Step 4: Record Results

If this is a new project and you discovered the gate commands, save them to project notes or memory for future sessions.

### Step 5: Record Outcomes

After running all quality gates, record the outcome for future reference. This is not optional — every test-and-verify invocation should produce a record for calibration and introspection.

**After a successful run (all gates pass on first attempt):**

Record: which gates passed, the context (file paths changed), and that it passed on first attempt.

**After a failing run:**

Record: which gate failed, the number of failures, what happened, and the context (file paths).

**With iteration count (if you needed multiple attempts):**

Record: which gate passed, the context, and how many attempts it took.

Record an outcome for **each gate** you ran. For example, if you ran format, lint, typecheck, and tests, and all passed on first try, record all four outcomes.

This data helps track iteration counts over time — identifying recurring weak spots and measuring improvement.

## Failure Handling

**A failing quality gate is not a suggestion to skip — it's a bug you just introduced or exposed.**

- **Format/lint failures in code you changed**: Fix your code.
- **Type errors in code you changed**: Fix your types.
- **Test failures in code you changed**: Fix your code or your test.
- **Pre-existing failures you didn't cause**: Document them. Don't silently ignore. Tell the user: "Tests X, Y were already failing before my change. My change does not affect them."

## Short-Circuit

If you're making a trivial change (documentation only, config comment, etc.), you may skip tests. But lint/format/typecheck still apply — they're cheap and catch embarrassing mistakes.

## Anti-Patterns

- **Running only tests, skipping lint/typecheck.** Lint catches bugs tests miss. Type errors are bugs waiting for runtime.
- **"All tests pass" without running them.** If you didn't see the output, it didn't happen.
- **Fixing the test instead of the code.** A test that always passes is worse than no test.
- **Leaving debug output.** `print("here")` in production code is a failure of verification.
- **Not saving discovered commands.** Next session, you'll rediscover them the hard way. Again.