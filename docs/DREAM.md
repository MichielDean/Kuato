# LLMem Dream Cycle & Extraction

Automated memory maintenance and session extraction pipelines. [Back to README](../README.md)

## Dream Cycle

The dream cycle performs automated memory maintenance during idle periods. It can be invoked manually via `llmem dream` (see [CLI Reference](CLI.md#llmem-dream)) or run automatically by a systemd timer.

- **Light phase:** Sort and deduplicate near-duplicate memories (cosine similarity ≥ threshold).
- **Deep phase:** Score, promote, decay, and merge memories. Decays confidence on idle memories. Boosts frequently accessed memories. Auto-links memories with high cosine similarity (≥ `dream.auto_link_threshold`, default 0.85) by creating `related_to` relations between them. Procedure memories older than `dream.stale_procedure_days` (default 30 days) with no recent access decay at double the normal rate — proposed-but-never-adopted procedures fade faster than confirmed ones.
- **REM phase:** Extract themes from memory clusters and write a dream diary (read-only reflection). Also extracts behavioral insights (patterns exceeding `dream.behavioral_threshold` occurrences within `dream.behavioral_lookback_days` days). When Ollama is available, uses an LLM call to generate specific, actionable procedural rules with "Do" directives and "Verify" steps; also generates `[SKILL PATCH]` sections (Detection Rule, Checklist, Pitfall, Verification). Falls back to count-based summaries when Ollama is unavailable. When a `SkillPatcher` is configured, the REM phase validates previously applied skill patches by comparing error counts before and after each patch — patches where errors decreased are marked effective, patches where errors stayed the same or increased are flagged for review. Skill patches are applied directly to SKILL.md files immediately after introspection (no proposed-changes.md or human approval gate). The dream validates whether patches reduced errors; if not, they are flagged for review.

Configuration is under the `dream:` key in `config.yaml`. See [Configuration](CONFIGURATION.md) for all dream settings.

### Systemd Timer

The Go implementation includes systemd unit generation for running the dream cycle on a schedule:

```bash
# Generate service and timer unit files
llmem dream --apply  # manual run
```

The `internal/systemd` package provides `GenerateServiceUnit()` and `GenerateTimerUnit()` to produce systemd `.service` and `.timer` unit files from embedded templates. Timer schedules are validated via `ValidateSchedule()` to prevent shell metacharacter injection.

### Go API — Dream Package

The `internal/dream` package provides the Go dream cycle implementation:

```go
import "github.com/MichielDean/LLMem/internal/dream"

dreamer, err := dream.NewDreamer(dream.DreamerConfig{
    Store:                 ms,
    SimilarityThreshold:   0.92,   // default
    DecayRate:             0.05,    // default
    DecayIntervalDays:     30,     // default
    DecayFloor:            0.3,    // default
    ConfidenceFloor:       0.3,    // default
    BoostThreshold:        5,      // default
    BoostAmount:           0.05,   // default
    AutoLinkThreshold:     0.85,   // default
    BehavioralThreshold:   3,      // default
    BehavioralLookbackDays: 30,    // default
    StaleProcedureDays:    30,     // default — procedure memories older than this decay at 2x
    BaseURL:               "",     // defaults to "http://localhost:11434"
    Model:                 "",     // defaults to "glm-5.1:cloud"
    OllamaClient:          nil,    // nil → created from BaseURL; takes precedence if provided
    DiaryPath:             "",     // defaults from paths.GetDreamDiaryPath()
    ReportPath:            "",     // defaults from paths.GetDreamReportPath()
    ProposedChangesPath:   "",     // defaults from paths.GetProposedChangesPath()
    SkillPatcher:          nil,    // nil → skip patch validation in REM phase
})
if err != nil {
    log.Fatal(err)
}

// Run all phases (dry run)
result, err := dreamer.Run(ctx, false, "")

// Run a specific phase
result, err := dreamer.Run(ctx, true, "deep")

// Write dream diary (markdown with sync.Mutex for in-process concurrency)
err = dreamer.WriteDiary(result)

// Write proposed-changes.md (behavioral insights + skill patches, append-only)
err = dreamer.WriteProposedChanges(ctx, result)

// Skill patch validation happens automatically during REM phase when SkillPatcher is set.
// The REM phase compares error counts before and after each patch, marking effective
// or flagged-for-review patches. Patches are applied immediately after introspection
// (not via proposed-changes.md) — the dream validates whether patches reduced errors.

// Generate HTML dream report
err = dreamer.GenerateDreamReport(result, "/path/to/report.html")
```

#### DreamerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| Store | *store.MemoryStore | (required) | Memory store instance |
| SimilarityThreshold | float64 | 0.92 | Cosine similarity threshold for duplicate detection |
| DecayRate | float64 | 0.05 | Confidence decay rate for idle memories |
| DecayIntervalDays | int | 30 | Grace period before decay starts |
| DecayFloor | float64 | 0.3 | Minimum confidence after decay |
| ConfidenceFloor | float64 | 0.3 | Memories below this are invalidated |
| BoostThreshold | int | 5 | Access count threshold for boosting |
| BoostAmount | float64 | 0.05 | Confidence boost per event |
| AutoLinkThreshold | float64 | 0.85 | Cosine similarity threshold for auto-linking |
| BehavioralThreshold | int | 3 | Minimum occurrences for behavioral insight |
| BehavioralLookbackDays | int | 30 | Lookback window for behavioral insights |
| StaleProcedureDays | int | 30 | Age threshold (days) for double-decay of procedure memories |
| OllamaClient | *ollama.OllamaClient | nil | Pre-configured Ollama client. Takes precedence over BaseURL. When nil, the constructor creates one from BaseURL. |
| BaseURL | string | `"http://localhost:11434"` | Ollama API base URL for behavioral insight generation. Validated for SSRF. |
| HTTPClient | *http.Client | nil | Optional pre-configured HTTP client (for testing with httptest.NewServer). Only used when OllamaClient is nil. |
| Model | string | `"glm-5.1:cloud"` | Ollama model name for behavioral insight generation |
| DiaryPath | string | paths.GetDreamDiaryPath() | Path for dream diary markdown |
| ReportPath | string | paths.GetDreamReportPath() | Path for HTML dream report |
| ProposedChangesPath | string | paths.GetProposedChangesPath() | Path for proposed-changes.md (behavioral insights and skill patches) |
| SkillPatcher | *skillpatch.SkillPatcher | nil | Skill patcher for validating applied patches during REM phase. When nil, patch validation is skipped. |

#### DreamResult

```go
type DreamResult struct {
    Light *LightPhaseResult
    Deep  *DeepPhaseResult
    Rem   *RemPhaseResult
}

type LightPhaseResult struct {
    DuplicatePairs  int
    MergeCandidates []*store.DuplicatePair
}

type DeepPhaseResult struct {
    DecayedCount               int
    StaleProcedureDecayedCount int
    BoostedCount               int
    InvalidatedCount           int
    MergedCount                int
    AutoLinkedCount            int
}

type BehavioralInsight struct {
    Category       string
    Count          int
    InsightID      string
    ContentSnippet string
    Samples        []string
}

type RemPhaseResult struct {
    TotalMemories      int
    ActiveMemories     int
    Themes             []string
    BehavioralInsights []BehavioralInsight
}
```

## Extraction and Hooks

### Python

The `hooks` module provides automatic extraction from session transcripts:

- `process_file()`: Extract memories from a transcript file.
- `process_session()`: Extract from an OpenCode session ID.
- `process_all_session_sources()`: Process all session sources (delegates to `session_hooks.process_opencode_sessions`).
- Self-assessment extraction with structured error taxonomy.
- Correction detection for identifying mistakes.

The `session_hooks` module provides `process_opencode_sessions()` — the full pipeline that reads OpenCode sessions from the SQLite database, chunks them by message boundaries, and runs extraction and embedding.

The `extract` module uses Ollama (default: `qwen2.5:1.5b`) to extract structured memories from text. The `embed` module generates embeddings using Ollama (default: `nomic-embed-text`).

### Go

The `internal/extract` package provides LLM-based memory extraction via Ollama:

```go
import "github.com/MichielDean/LLMem/internal/extract"

engine, err := extract.NewExtractionEngine(extract.ExtractionConfig{
    Model:   "glm-5.1:cloud",     // defaults to "glm-5.1:cloud"
    BaseURL: "http://localhost:11434",  // defaults to localhost
})
if err != nil {
    log.Fatal(err)
}

// Extract memories from text. Returns empty slice if Ollama is unavailable (graceful degradation).
memories := engine.Extract(ctx, text)

// Check if the extraction model is available
available := engine.CheckAvailable(ctx)
```

The `internal/session` package provides session lifecycle hooks:

```go
import "github.com/MichielDean/LLMem/internal/session"

coord, err := session.NewSessionHookCoordinator(session.SessionHookConfig{
    Store: ms,
})

result, err := coord.OnCreated(ctx, "session-id")       // ("success"|"already_processed"|"error", ...)
result, err := coord.OnIdle(ctx, "session-id")          // ("success"|"debounced"|"no_transcript", count)
resultType, contextPath, err := coord.OnCompacting(ctx, "session-id")
result, err := coord.OnEnding(ctx, "session-id")

// OnEndingWithIntrospect: session.ending with automatic introspection
resultType, memoryID, err := coord.OnEndingWithIntrospect(ctx, "session-id")
```

The `internal/introspect` package provides failure analysis, lesson learning, and session transcript introspection:

```go
import "github.com/MichielDean/LLMem/internal/introspect"

// Analyze a failure and store self_assessment. Returns IntrospectResult with MemoryID, ProposedUpdate, and Category.
result, err := introspect.IntrospectFailure(ctx, ms, introspect.IntrospectFailureParams{
    WhatHappened: "null pointer dereference in handler",
    Category:     "NULL_SAFETY",
    Context:      "handler.go:42",
    CaughtBy:     "self-review",
    ProposedFix:  "add nil check before access",
    Model:        "glm-5.1:cloud",
    BaseURL:      "http://localhost:11434",
})
// result.MemoryID, result.ProposedUpdate, result.Category

// Learn a lesson from a wrong→right correction. Returns IntrospectResult with MemoryID, ProposedUpdate, and Category.
result, err := introspect.LearnLesson(ctx, ms, introspect.LearnLessonParams{
    WhatWasWrong: "used global state",
    WhatIsCorrect: "inject dependency via constructor",
    Context:       "service.go:15",
})

// Automatic introspection from text (e.g., a session transcript)
result, err := introspect.IntrospectAuto(ctx, ms, "Session transcript text...", "glm-5.1:cloud", "http://localhost:11434")
// result.MemoryID, result.ProposedUpdate, result.Category
```

All three functions use LLM expansion via Ollama when available, with graceful degradation to storage-only mode when Ollama is unavailable. `IntrospectFailure` and `LearnLesson` return `IntrospectResult{MemoryID, ProposedUpdate, Category}`. `IntrospectAuto` returns `IntrospectAutoResult{MemoryID, ProposedUpdate, Category}`. `ProposedUpdate` and `Category` are populated when LLM enrichment succeeds; empty on graceful degradation.

When `ProposedUpdate` and `Category` are both non-empty, callers should patch the relevant skill file using a `SkillPatcher` (see [Skill Patching](#skill-patching-internalskillpatch)). The CLI commands `introspect`, `learn`, and `hook --type ending` all perform this patching automatically after introspection.

```go
// Introspect a session transcript (called by OnEnding)
id, err := introspect.IntrospectTranscript(ctx, ms, transcript, "session-id", ollamaClient, "glm-5.1:cloud")
// When ollamaClient is nil, falls back to degraded storage (plain-text summary, no LLM call)
```

Both `IntrospectFailure` and `LearnLesson` use LLM expansion via Ollama when available, with graceful degradation to storage-only mode when Ollama is unavailable.

`IntrospectTranscript` analyzes a session transcript and stores a `self_assessment` memory. It accepts a pre-configured `*ollama.OllamaClient` (reusing the session's connection). When `ollamaClient` is nil, it produces a degraded memory with a plain-text summary. On LLM availability, the model generates a structured self-assessment from the transcript content. Note: `IntrospectTranscript` uses `context.Background()` for the final store operation (not the caller's `ctx`) to ensure persistence even if the calling context has expired during the LLM call.

### Skill Patching (internal/skillpatch)

The `internal/skillpatch` package provides direct skill file patching after introspection. When introspection produces a `ProposedUpdate` and `Category`, the relevant SKILL.md file is patched immediately — no proposed-changes.md or human approval gate. The dream cycle later validates whether the patch reduced errors in that category.

```go
import "github.com/MichielDean/LLMem/internal/skillpatch"

sp, err := skillpatch.NewSkillPatcher(skillpatch.SkillPatchConfig{
    SkillDir: "",  // empty → paths.GetSkillDir() (~/.config/llmem/skills/)
})

// Patch a skill file with a procedural update from introspection
err = sp.Patch(ctx, "NULL_SAFETY", "Always guard nil pointers in Go", "Missing null checks")

// Find the skill file for a category (returns "" if not found)
path, err := sp.FindSkillFile(ctx, "ERROR_HANDLING")

// Validate whether a patch was effective (pure function, no I/O)
validation := skillpatch.ValidatePatch("NULL_SAFETY", 10, 3)
// validation.Effective → true (errors decreased)
// validation.Flagged   → false
```

**Patch behavior:**
- If the category maps to a known skill directory (all 10 error categories map to `introspection`), the existing SKILL.md is patched in-place.
- If no SKILL.md exists, a new one is created with YAML frontmatter.
- Patches are additive — new `## Patch: CATEGORY (YYYY-MM-DD)` sections are appended, never overwriting existing content.
- Duplicate patches (same `proposedUpdate` text) are skipped (idempotent).
- Category names are validated against `^[A-Za-z0-9_]+$` to prevent path traversal.
- YAML frontmatter values are sanitized (newlines replaced) to prevent injection.

**Dream validation:** When a `SkillPatcher` is provided in `DreamerConfig`, the REM phase calls `ValidatePatch` for each category with behavioral insights, comparing error counts before and after the patch was applied. Effective patches (errors decreased) are noted; ineffective patches (errors stayed the same or increased) are flagged for review via `{flagged_for_review: true}` metadata on the insight memory.
