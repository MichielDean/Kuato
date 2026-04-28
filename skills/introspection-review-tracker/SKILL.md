---
name: introspection-review-tracker
description: >
  Reference documentation for the automated ReviewOutcomeTracker hook that persists
  critical-code-reviewer findings as self_assessment memories. The hook fires
  automatically after every review — no manual steps required. This skill documents
  the mapping from severity tiers to error categories and the content format.
  Triggers on: "review outcome", "track findings", "record review",
  "critical-code-reviewer", "self_assessment".
license: MIT
---

# Introspection Review Tracker — Reference Documentation

This skill is **reference documentation** for the automated review outcome persistence hook. The `llmem track-review` CLI command fires automatically after every `critical-code-reviewer` run, creating `self_assessment` memories without requiring manual agent intervention. The Python API (`ReviewOutcomeTracker` in `memory/review_tracker.py`) provides the same functionality for programmatic use. This skill documents how the hook works, the severity-to-category mapping, and the content format.

## How Persistence Works

Persistence happens **mechanically** via `llmem track-review` (CLI) or `ReviewOutcomeTracker` (Python API). The agent must run this command after every `critical-code-reviewer` review — it is not optional.

### CLI: `llmem track-review`

The primary mechanism. Run after every review:

```bash
# For a review with findings (batch mode):
llmem track-review --finding-file /tmp/review-findings.json --context "handler.py"

# For a single finding:
llmem track-review --category NULL_SAFETY --what-happened "missing null check" --context "handler.py:42" --severity Required --caught-by self-review

# For a clean review (no findings):
llmem track-review --context "handler.py"

# See categories for a severity tier:
llmem suggest-categories Required
```

The `--finding-file` expects a JSON array of finding objects, each with `category`, `what_happened`, and optional `severity` keys.

### Python API: `ReviewOutcomeTracker`

```python
from memory.review_tracker import ReviewOutcomeTracker
from memory.store import MemoryStore

store = MemoryStore(db_path=db_path)
tracker = ReviewOutcomeTracker(store=store)

# For each finding:
tracker.track_finding(
    category="NULL_SAFETY",
    what_happened="missing null check before .field access",
    context="handler.py:42",
    severity="Required",
    caught_by="self-review",
)

# For an entire review (one memory per finding):
tracker.track_review(
    findings=[
        {"category": "NULL_SAFETY", "what_happened": "missing null check"},
        {"category": "ERROR_HANDLING", "what_happened": "swallowed exception"},
    ],
    context="handler.py",
)

# For a clean review (no findings):
tracker.track_review(findings=[], context="handler.py")
# → creates a single REVIEW_PASSED memory with "all clear"

# Suggest categories for a severity tier:
tracker.suggest_categories("Required")
# → ["NULL_SAFETY", "ERROR_HANDLING", "MISSING_VERIFICATION", "EDGE_CASE"]
```

## Verification

After a `critical-code-reviewer` review completes, verify that the post-review command was run:

1. Check that at least one `self_assessment` memory was created:
   ```bash
   llmem search "review_tracker" --type self_assessment
   ```

2. For each finding, confirm the category matches the severity tier mapping below.

3. For clean reviews, confirm one `REVIEW_PASSED` memory with outcomes "all clear".

## When This Skill Is Used

- **After every `critical-code-reviewer` completion** — the hook must be run mechanically.
- **To verify** that the hook ran correctly (see Verification above).
- **As reference** for understanding the severity-to-category mapping and content format.

## Severity-to-Category Mapping

The `REVIEW_SEVERITY_TAXONOMY` constant in `memory/taxonomy.py` maps each severity tier to applicable error taxonomy categories. The `ReviewOutcomeTracker.suggest_categories()` method and `llmem suggest-categories` CLI command use this mapping directly.

| Severity Tier | Applicable Categories | Guidance |
|---|---|---|
| Blocking | AUTH_BYPASS, RACE_CONDITION, DATA_INTEGRITY | Security holes, data corruption risks, logic errors |
| Required | NULL_SAFETY, ERROR_HANDLING, MISSING_VERIFICATION, EDGE_CASE | Quality gaps — slop, missing safety checks, unhandled cases |
| Strong Suggestions | PERFORMANCE, DESIGN | Suboptimal approaches, missing tests, unclear intent |
| Noted | OFF_BY_ONE | Minor style issues, small boundary errors |
| Passed | REVIEW_PASSED | Clean review with no findings — positive outcome |

These mappings are advisory — the agent should pick the most meaningful category for the actual finding, not follow them mechanically. A Required-tier finding about performance might still map to PERFORMANCE rather than NULL_SAFETY. Note that the reviewer's tier name is "Strong Suggestions" (not just "Suggestions"); the taxonomy key matches this exactly.

## Content Format

Memories created by `ReviewOutcomeTracker` use the `SELF_ASSESSMENT_FIELDS` format from `memory/taxonomy.py`, ensuring format parity with the `llmem introspect` manual mode:

```
Category: <ERROR_TAXONOMY category>
Context: <file:line or task identifier>
What_happened: <behavioral description>
Outcomes: <results or "all clear" for clean reviews>
What_caught_it: <how discovered, e.g. "self-review">
Estimates_vs_actual: <optional>
Recurring: <"yes" or "no">
Iteration_count: <optional integer>
Proposed_update: <optional>
```

## Key References

- **CLI command**: `llmem track-review` — the mechanical post-review hook
- **CLI command**: `llmem suggest-categories` — list categories for a severity tier
- **ReviewOutcomeTracker**: `memory/review_tracker.py` — the Python API implementation
- **Error taxonomy categories**: `memory/taxonomy.py` → `ERROR_TAXONOMY`
- **Severity mapping**: `memory/taxonomy.py` → `REVIEW_SEVERITY_TAXONOMY`
- **Self-assessment fields**: `memory/taxonomy.py` → `SELF_ASSESSMENT_FIELDS`
- **Reviewer severity tiers**: `skills/critical-code-reviewer/SKILL.md` (see Severity Tiers section)
- **llmem introspect command**: `skills/llmem/SKILL.md:134-137`