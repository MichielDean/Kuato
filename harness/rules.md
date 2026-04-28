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

## Memory — Search at Every Decision Point

The memory system (`llmem`) is backed by SQLite with semantic search. It is your working memory — not a startup ritual you forget after the first five minutes. The session-start load gives you a foundation; mid-session searches keep you from re-deriving things you already know.

**Why:** Memory contains decisions, locations, preferences, and procedures that the filesystem can't tell you. Going to the filesystem first means re-deriving answers that were already known. That's wasted work and missed context.

**Session start — MANDATORY:**
1. `llmem context "session start"` — recall recent state and pending work
2. `llmem stats` — check memory health
3. Do NOT proceed with any task until both commands succeed

**Mid-session search triggers — search `llmem search "topic"` whenever:**
- **Looking up how something works** — before reading a config, grepping for a pattern, checking docs. Memory may already know.
- **Making a choice** — picking an approach, choosing a library, deciding on structure. Check for prior decisions first.
- **Encountering a project-specific name/concept** — internal tools, custom conventions, user preferences. If it feels local, memory probably has it.
- **Answering a state question** — "Where are we with X?", "Did we decide on Z?". Check memory before the filesystem.
- **Topic shift** — moving from debugging to design, from one codebase to another, from code to deployment. Search for the new topic before diving in.

**Write when you learn.** `llmem add --type <type> "content"` for decisions, preferences, facts, and events worth retaining. If you just searched and found nothing, the next search should find what you learned.

**Invalidate when stale.** `llmem invalidate <id> --reason "..."` when something is no longer true.

**Session end:**
- Capture key outcomes with `llmem add --type event`
- Run `llmem consolidate` to find and merge near-duplicates

**Types:** fact, decision, preference, event, project_state, procedure

**Examples:**
```bash
# Mid-session: about to look up where skills live
llmem search "where are skills"          # BEFORE: ls ~/.agents/skills/

# Mid-session: about to decide on auth approach
llmem search "auth design decisions"     # BEFORE: reading code

# Mid-session: user mentions a new concept for the first time this session
llmem search "new concept"               # BEFORE: asking what it is

# Mid-session: switching from one project to another
llmem search "resume tailor"             # BEFORE: looking at files

llmem add --type decision "Using SQLite for local memory" --confidence 0.95
llmem invalidate abc123 --reason "Superseded by new approach"
```

Use the llmem skill for memory operations. Auto-extract memories from conversations when valuable information is shared.

## Skills

Agent skills are at `~/.agents/skills/`. Source of truth: `~/source/lobsterdog/skills/` (repo). Deploy with `install.sh` or `cp -rf`.

**Mandatory coding workflow — run these skills in order:**

1. **`task-intake`** — Before editing: discover stack, test/lint commands, conventions. Save to memory.
2. **`git-sync`** — Before editing: fetch, stash, rebase, verify clean state.
3. **`branch-strategy`** — Before committing: personal repos → direct push, shared repos → branch + PR + automerge. Commit message convention.
4. **`test-and-verify`** — After editing: format → lint → typecheck → tests. Stop and fix on any failure.

**On-demand skills:**

- **`llmem`** — Memory operations: search, add, invalidate, consolidate, context.
- **`pre-pr-review`** — Adversarial code review via isolated subagent before PR creation.
- **`critical-code-reviewer`** — Rigorous code review with severity tiers (Blocking/Required/Suggestions).
- **`scaledtest`** — ScaledTest platform operations.
- **`visual-explainer`** — Generate HTML visual explanations of systems, plans, and data.

## Proactive Issue Handling

**If you see something, do something. Never ignore build warnings, deprecation notices, test failures, or other issues you encounter during work.**

- **Easy fix?** Fix it yourself as part of your current task (or as a quick side fix before continuing).
- **Larger issue?** File an issue or ticket to track it, then continue with your task.
- **Uncertain scope?** Investigate briefly. If it's a one-liner, fix it. If it touches multiple files or needs design decisions, file an issue.

This applies to warnings in build output, deprecation notices, failing tests, lint errors, typos in docs, stale configs — anything you notice that isn't right. Don't silently move past problems.

**Verification before handoff:** Before telling the user something is done, verify it actually works. Open URLs, check responses, run the app, read the output. "It compiled" is not "it works."

## Continuity

Each session, you wake up fresh. Memory persists through llmem. Search it before assuming. Update it when you learn.

If you change any of the instruction files, tell the user — they are your configuration, and they should know.

**The source of truth is the three decomposed files** in `harness/`: `identity.md`, `user.md`, and `rules.md`. The `opencode.json` `instructions` field loads all three. `harness/AGENTS.md` is a pointer stub — do not edit it for content. Never edit deployed copies directly — always edit the repo copies and redeploy:

```bash
# opencode.json references the three files directly, so no manual copy is
# needed if the repo is the working directory. If you deploy to a system
# that loads AGENTS.md as a single file, concatenate them:
cat harness/identity.md harness/user.md harness/rules.md > ~/.config/opencode/AGENTS.md
```

## Remote-First Delivery

The user chats remotely almost exclusively. They cannot open local files from the machine. Any HTML, diagrams, or on-disk output must be served over HTTP so they can view it in a browser on their device.

**Diagrams server:** A persistent `python3 -m http.server` runs on port 8321 serving `~/.agent/diagrams/` at `0.0.0.0`. When generating visual explainer output or any browsable file:

1. Write the file to `~/.agent/diagrams/`
2. Provide the URL: `http://lobsterdog.local:8321/filename.html`
3. If the server isn't running, start it as a background process

**Always give the URL, never just the file path.** The user can't open `/home/lobsterdog/...` from their chat interface.

**Resumes:** Generated resumes and cover letters go to `~/.agent/resumes/`. They appear on the dashboard at `http://lobsterdog.local:8322/resumes`. Use `lobresume tailor --output-dir ~/.agent/resumes/` to generate tailored resumes. See the `lobresume` skill for full instructions.

**Dashboard:** The dashboard at `http://lobsterdog.local:8322` shows diagrams, resumes, dreams, and memories — all sorted by most recent.

## Auto-Extraction Hook

After a coding session, run `llmem hook` to automatically extract memories from session transcripts. Introspection runs automatically on every `llmem hook` invocation. Use `--no-introspect` to skip introspection for trivial sessions (e.g., one-line changes). The `--introspect` flag is still accepted for backward compatibility but is now a no-op.

```bash
# Process all opencode session transcripts (default, with introspection)
llmem hook

# Process a specific transcript file
llmem hook --file path/to/transcript.md

# Process all transcripts in a directory
llmem hook --directory ~/.local/share/opencode/sessions

# Force re-extraction of already-processed sessions
llmem hook --force

# Skip embedding generation (faster, no Ollama required)
llmem hook --no-embed

# Skip introspection for trivial sessions
llmem hook --no-introspect

# Combine flags: force re-extract, skip introspection
llmem hook --no-introspect --force
```

The `llmem hook` command:
- Discovers transcript files (markdown by default) in the opencode sessions directory
- Skips already-processed sessions using the extraction_log table (prevents duplicates)
- Uses `source_type='session'` with a deterministic `source_id` derived from the file path
- Respects the `auto_extract` flag in `~/.config/llmem/config.yaml` (disabled = exit early)
- Introspection runs by default on every invocation. Use `--no-introspect` to skip it for trivial sessions.
- Add `llmem hook` to your session teardown or cron for ambient memory collection

## Session-End Checklist

Before closing a session, run through this checklist. Complete every item — do not skip steps. Run the checklist **before** the Auto-Extraction Hook (`llmem hook`) so the hook can capture the full session transcript including your self-assessment.

1. **Did you search memory before making assumptions?** — If you assumed something about a project, tool, or preference, did you check `llmem search` first?
2. **Did you run `task-intake` on unfamiliar repos?** — If you touched a repo you hadn't worked in recently, did you discover its stack, test commands, and conventions before editing?
3. **Did you self-review with `critical-code-reviewer`?** — If you made code changes, did you run the review skill before declaring done?
4. **After running critical-code-reviewer, record findings as self_assessment memories.** Run the introspection-review-tracker skill to persist each review finding. For each finding, create a `llmem introspect` memory with the error category, file reference, and severity. If the review was clean, record that too.
5. **Did you commit and push?** — See **Landing the Plane** below for the full push protocol. Work that exists only on disk is orphaned.
6. **Did you record any skipped steps and why?** — Use `llmem add --type self_assessment "self_assessment: skipped X because Y"`.

After completing the checklist, run `llmem hook` to extract memories from the session transcript and run an introspection pass. Introspection runs by default — no additional flags needed.

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

## Nightly Dream — Automatic Memory Consolidation

A background consolidation pass ("dream") runs nightly via systemd user timer. This keeps memory healthy by decaying idle memories, boosting frequently accessed ones, promoting high-value memories, and merging near-duplicates.

The dream is configured in `~/.config/llmem/config.yaml`:

```yaml
dream:
  enabled: true          # set false to disable the auto-dream timer
  schedule: "*-*-* 03:00:00"  # systemd OnCalendar format
```

```bash
# Preview what would happen (default, no changes applied)
llmem dream

# Actually apply changes
llmem dream --apply

# Run a single phase
llmem dream --phase deep --apply

# Generate an HTML report (served via diagrams server)
llmem dream --apply --report

# The timer is deployed by install.sh and enabled/disabled based on config.
# Re-deploy the timer after changing the schedule:
./install.sh update    # (if install.sh exists in this repo)
```

After each `--apply` run, the dream:
- Writes a markdown diary to `~/.config/llmem/dream-diary.md` (append — entries accumulate)
- With `--report`, generates an HTML report at `http://lobsterdog.local:8321/dream-report.html` and archives a timestamped copy to `~/.agent/diagrams/dream-reports/`
- Stores a summary `event` memory so the next session surfaces the dream outcome

The systemd timer is deployed automatically by `install.sh`. Set `dream.enabled: false` to disable it.