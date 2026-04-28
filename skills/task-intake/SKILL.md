---
name: task-intake
description: Discover project conventions, test commands, stack, and structure before making any code changes. ALWAYS run before editing code in an unfamiliar or partially-familiar repository. Prevents wrong assumptions about frameworks, test runners, linting tools, and project layout. Triggers on: starting a new task in a repo, working in a repo for the first time, "what's the stack", "how do I run tests here", or implicitly before any code modification in a repo not recently intaked.
license: MIT
---

# Task Intake

**ALWAYS understand the project before changing it. No exceptions.**

Editing code without knowing the stack, conventions, or quality gates is guesswork dressed up as help. You wouldn't walk into a stranger's kitchen and start reorganizing — don't do it with a codebase.

## When to Run

- First time working in a repo
- First time working in a repo in a while (memory may be stale)
- When switching between repos mid-session
- When asked to add a feature or fix a bug in a repo you haven't recently intaked

## Procedure

### Step 1: Search Memory

Search existing notes or memory for project conventions:

- Check for saved notes about the repo's stack, test commands, and lint commands
- If memory already has this information and it's recent, skip to Step 5 to confirm it's still accurate. Otherwise, continue.

### Step 2: Identify the Stack

Check for these files in the repo root (listed in priority order):

| File | Stack |
|------|-------|
| `go.mod` | Go |
| `pyproject.toml` / `setup.py` / `requirements.txt` | Python |
| `package.json` | Node.js / TypeScript |
| `Cargo.toml` | Rust |
| `Gemfile` / `*.gemspec` | Ruby |
| `Makefile` | Build-driven (check contents for actual language) |
| `Dockerfile` | Check `FROM` and `RUN` lines for stack |

Multiple files may coexist (e.g., `go.mod` + `Dockerfile` + `Makefile`). The language module file (`go.mod`, `pyproject.toml`, etc.) is the primary stack identifier.

### Step 3: Discover Quality Gates

Run these in order. Skip any that fail — not every project has all of them.

**Python:**
```bash
# Tests
pytest --co -q 2>/dev/null || python -m pytest --co -q 2>/dev/null
# Lint
ruff check . 2>/dev/null || flake8 . 2>/dev/null || pylint */ 2>/dev/null
# Type check
mypy . 2>/dev/null || pyright 2>/dev/null
# Format check
black --check . 2>/dev/null || ruff format --check . 2>/dev/null
```

**Go:**
```bash
# Tests
go test ./...
# Vet
go vet ./...
# Lint
golangci-lint run 2>/dev/null
# Format
gofmt -d .
```

**Node.js/TypeScript:**
```bash
# Check scripts
cat package.json | grep -A20 '"scripts"'
# Tests (try common patterns)
npm test 2>/dev/null || npx vitest 2>/dev/null || npx jest 2>/dev/null
# Lint
npm run lint 2>/dev/null || npx eslint . 2>/dev/null
# Type check
npx tsc --noEmit 2>/dev/null
```

**Makefile-driven:**
```bash
# Check available targets
make -pn 2>/dev/null | grep -E '^[a-z].*:' | head -20
# Or just read the Makefile
```

### Step 4: Check for Project Conventions

Look for, in order of importance:

1. **`AGENTS.md`** — Agent-specific instructions (mandatory to read if present)
2. **`CONTRIBUTING.md`** — Human contribution guidelines
3. **`README.md`** — Project overview, setup instructions
4. **`.github/`** — CI workflows reveal test/lint commands actually used
5. **`.editorconfig`** / **`.prettierrc`** / **`pyproject.toml [tool.ruff]`** — Formatting rules
6. **`renovate.json`** / **`dependabot.yml`** — Dependency management

### Step 5: Write Findings to Memory

If this is a new or updated intake, save it to project notes or memory for future sessions:

- Record: stack, test command, lint command, typecheck command, format command, key conventions
- This ensures the NEXT session doesn't need to rediscover everything.

### Step 6: Summarize Before Acting

Before writing any code, confirm you know:

- What language/framework the project uses
- How to run the tests
- How to run the linter/formatter
- Whether there are type checks
- Any project-specific conventions from AGENTS.md/CONTRIBUTING.md

If you can't answer all four, you're not ready. Keep looking.

## Anti-Patterns

- **Assuming the stack from the directory name.** `backend/` is Go, not Python. `scripts/` is Python, not Go. Check the files.
- **Running `npm test` in a Go project.** You just installed Node for no reason and the test failed for the wrong reason.
- **Skipping the README.** The README often contains setup steps (env vars, database migrations, seed data) that are prerequisites to running anything.
- **Not checking AGENTS.md.** If the project has one, it overrides general assumptions. Always.
- **Forgetting to save to memory.** Discovery that isn't recorded is discovery you'll repeat next session.