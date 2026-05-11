// Package introspect provides failure analysis and lesson learning for LLMem self-assessment.
// It uses an OllamaClient for LLM-assisted introspection with graceful degradation.
package introspect

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/MichielDean/LLMem/internal/ollama"
	"github.com/MichielDean/LLMem/internal/store"
	"github.com/MichielDean/LLMem/internal/taxonomy"
)

const (
	defaultModel             = "glm-5.1:cloud"
	defaultBaseURL           = "http://localhost:11434"
	introspectSource         = "introspect"
	introspectAutoSource     = "introspect-auto"
	learnSource              = "learn"
	introspectConfidence     = 0.9
	introspectAutoConfidence = 0.9
	learnConfidence          = 0.85

	// callModelTimeout is the default timeout for LLM calls in IntrospectFailure and LearnLesson.
	// CallModelTimeout in params takes precedence; this is the fallback when zero.
	callModelTimeout = 5 * time.Minute

	// callModelMinTimeout is the minimum allowed timeout for LLM calls.
	// Timeouts below this value are rejected to prevent accidental instant-failures.
	callModelMinTimeout = 10 * time.Second
)

// LLMEnrichment indicates whether LLM enrichment was used when storing a memory.
// It is a generic status type that could be reused by other LLM callers (learn, dream REM).
type LLMEnrichment string

const (
	// Enriched means the LLM successfully generated structured content.
	Enriched LLMEnrichment = "enriched"
	// Skipped means the LLM was unavailable or timed out, so raw fields were stored.
	Skipped LLMEnrichment = "skipped"
	// Disabled means the caller explicitly set NoLLM=true, skipping LLM entirely.
	Disabled LLMEnrichment = "disabled"
)

// IntrospectResult holds the result of an IntrospectFailure call.
// MemoryID is always non-empty on success (never empty string).
// Content is the stored memory content.
// LLMStatus indicates whether LLM enrichment was used.
type IntrospectResult struct {
	MemoryID   string
	Content    string
	LLMStatus  LLMEnrichment
}

// LearnResult holds the result of a LearnLesson call.
// MemoryID is always non-empty on success (never empty string).
// Content is the stored memory content.
// LLMStatus indicates whether LLM enrichment was used.
type LearnResult struct {
	MemoryID   string
	Content    string
	LLMStatus  LLMEnrichment
}

// IntrospectFailureParams contains the parameters for introspecting a failure.
type IntrospectFailureParams struct {
	WhatHappened string
	Category     string
	Context      string
	CaughtBy     string
	ProposedFix  string
	Model        string
	BaseURL      string

	// NoLLM skips all Ollama calls: no IsAvailable check, no Generate call.
	// When true, raw fields are stored immediately and LLMStatus is Disabled.
	NoLLM bool

	// Timeout for the LLM call. When zero, defaults to callModelTimeout (5 minutes).
	// Must be >= 10 seconds; values below are rejected with an error.
	Timeout time.Duration

	// HTTPClient is an optional pre-configured HTTP client (for testing with httptest.NewServer).
	// When provided, it is passed to OllamaClient constructor and bypasses URL validation.
	HTTPClient *http.Client
}

// LearnLessonParams contains the parameters for learning a lesson from a wrong→right correction.
type LearnLessonParams struct {
	WhatWasWrong string
	WhatIsCorrect string
	Context      string
	Model        string
	BaseURL      string

	// NoLLM skips all Ollama calls: no IsAvailable check, no Generate call.
	// When true, raw fields are stored immediately and LLMStatus is Disabled.
	NoLLM bool

	// Timeout for the LLM call. When zero, defaults to callModelTimeout (5 minutes).
	// Must be >= 10 seconds; values below are rejected with an error.
	Timeout time.Duration

	// HTTPClient is an optional pre-configured HTTP client (for testing with httptest.NewServer).
	// When provided, it is passed to OllamaClient constructor and bypasses URL validation.
	HTTPClient *http.Client
}

// fmtErr wraps an error with the "llmem: introspect:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: introspect: "+format, args...)
}

// IntrospectFailure analyzes a failure and stores a self_assessment memory.
// If the LLM is available, it uses the model to expand the bare description into
// a structured self-assessment. If unavailable, it stores a structured
// memory directly from the provided fields (graceful degradation).
//
// Contract: NEVER returns (IntrospectResult{}, nil) — either creates a memory or returns an error.
// Even on LLM failure, a storage-only memory is created with LLMStatus Skipped or Disabled.
func IntrospectFailure(ctx context.Context, ms *store.MemoryStore, params IntrospectFailureParams) (IntrospectResult, error) {
	if params.WhatHappened == "" {
		return IntrospectResult{}, fmtErr("what_happened is required")
	}
	if params.Model == "" {
		params.Model = defaultModel
	}
	if params.Timeout != 0 && params.Timeout < callModelMinTimeout {
		return IntrospectResult{}, fmtErr("timeout must be at least %v, got %v", callModelMinTimeout, params.Timeout)
	}

	// Warn about unknown categories but proceed anyway
	if params.Category != "" {
		if _, ok := taxonomy.ErrorTaxonomy[params.Category]; !ok {
			slog.Warn("llmem: introspect: unknown category, proceeding anyway", "category", params.Category)
		}
	}

	if params.NoLLM {
		// Explicit raw-only mode: skip LLM entirely
		content := buildRawFailureContent(params)
		id, err := ms.Add(ctx, store.AddParams{
			Type:       "self_assessment",
			Content:    content,
			Source:     introspectSource,
			Confidence: introspectConfidence,
		})
		if err != nil {
			return IntrospectResult{}, fmtErr("store self_assessment: %w", err)
		}
		slog.Info("llmem: introspect: stored self_assessment (LLM disabled)", "id", id)
		return IntrospectResult{MemoryID: id, Content: content, LLMStatus: Disabled}, nil
	}

	var content string
	llmResponse, llmStatus := callModel(ctx, params.Model, params.BaseURL, buildFailurePrompt(params), params.Timeout, params.HTTPClient)

	if llmStatus == Enriched && llmResponse != "" {
		content = llmResponse
	} else {
		// Graceful degradation: build from provided fields
		content = buildRawFailureContent(params)
	}

	id, err := ms.Add(ctx, store.AddParams{
		Type:       "self_assessment",
		Content:    content,
		Source:     introspectSource,
		Confidence: introspectConfidence,
	})
	if err != nil {
		return IntrospectResult{}, fmtErr("store self_assessment: %w", err)
	}

	slog.Info("llmem: introspect: stored self_assessment", "id", id, "llm_status", llmStatus)
	return IntrospectResult{MemoryID: id, Content: content, LLMStatus: llmStatus}, nil
}

// buildRawFailureContent constructs the fallback content string from provided fields
// when LLM enrichment is not available or was skipped.
func buildRawFailureContent(params IntrospectFailureParams) string {
	var lines []string
	if params.Category != "" {
		lines = append(lines, "Category: "+params.Category)
	}
	if params.Context != "" {
		lines = append(lines, "Context: "+params.Context)
	}
	lines = append(lines, "What_happened: "+params.WhatHappened)
	lines = append(lines, "What_caught_it: "+orDefault(params.CaughtBy, "mid-session introspection"))
	if params.ProposedFix != "" {
		lines = append(lines, "Proposed_update: "+params.ProposedFix)
	}
	lines = append(lines, "Recurring: unknown")
	return strings.Join(lines, "\n")
}

// IntrospectAuto performs automatic introspection from a text string or session transcript.
// It calls the LLM to produce a structured self_assessment, or falls back to raw fields
// when the LLM is unavailable (graceful degradation).
//
// Contract: NEVER returns ("", nil) — either creates a memory or returns an error.
func IntrospectAuto(ctx context.Context, ms *store.MemoryStore, text, model, baseURL string) (string, error) {
	if text == "" {
		return "", fmtErr("text is required")
	}
	if model == "" {
		model = defaultModel
	}

	var content string
	llmResponse, _ := callModel(ctx, model, baseURL, buildAutoPrompt(text), callModelTimeout, nil)

	if llmResponse != "" {
		content = llmResponse
	} else {
		// Graceful degradation: build from provided text directly
		content = "What_happened: " + text + "\nRecurring: unknown"
	}

	id, err := ms.Add(ctx, store.AddParams{
		Type:       "self_assessment",
		Content:    content,
		Source:     introspectAutoSource,
		Confidence: introspectAutoConfidence,
	})
	if err != nil {
		return "", fmtErr("store self_assessment: %w", err)
	}

	slog.Info("llmem: introspect: stored auto self_assessment", "id", id)
	return id, nil
}

// buildAutoPrompt builds the prompt for automatic introspection of a session
// transcript or arbitrary text. It instructs the LLM to analyze the text for
// errors, failures, and learning moments, producing structured self_assessment fields.
func buildAutoPrompt(text string) string {
	fieldLines := taxonomy.IntrospectFieldLines()
	prompt := "Analyze the following session transcript for errors, failures, and learning moments. " +
		"Identify what went wrong, why it happened, whether it's a recurring pattern, " +
		"and what procedural change would prevent it in the future.\n\n"
	prompt += "Format each field on its own line as \"Field: value\":\n\n"
	prompt += fieldLines + "\n\n"
	prompt += "Session transcript:\n" + text
	prompt += "\n\nProduce a structured self-assessment. Be specific about what went wrong and what should change."
	return prompt
}

// LearnLesson analyzes a wrong→right correction and stores a procedure memory.
// If the LLM is available, it distills the correction into a generalizable procedure.
// If unavailable, it stores the lesson directly (graceful degradation).
//
// Contract: NEVER returns (LearnResult{}, nil) — either creates a memory or returns an error.
func LearnLesson(ctx context.Context, ms *store.MemoryStore, params LearnLessonParams) (LearnResult, error) {
	if params.WhatWasWrong == "" || params.WhatIsCorrect == "" {
		return LearnResult{}, fmtErr("what_was_wrong and what_is_correct are required")
	}
	if params.Model == "" {
		params.Model = defaultModel
	}
	if params.Timeout != 0 && params.Timeout < callModelMinTimeout {
		return LearnResult{}, fmtErr("timeout must be at least %v, got %v", callModelMinTimeout, params.Timeout)
	}

	if params.NoLLM {
		// Explicit raw-only mode: skip LLM entirely
		content := buildRawLessonContent(params)
		id, err := ms.Add(ctx, store.AddParams{
			Type:       "procedure",
			Content:    content,
			Source:     learnSource,
			Confidence: learnConfidence,
		})
		if err != nil {
			return LearnResult{}, fmtErr("store procedure: %w", err)
		}
		slog.Info("llmem: learn: stored procedure (LLM disabled)", "id", id)
		return LearnResult{MemoryID: id, Content: content, LLMStatus: Disabled}, nil
	}

	var content string
	llmResponse, llmStatus := callModel(ctx, params.Model, params.BaseURL, buildLessonPrompt(params), params.Timeout, params.HTTPClient)

	if llmStatus == Enriched && llmResponse != "" {
		content = llmResponse
	} else {
		// Graceful degradation: build from provided fields
		content = buildRawLessonContent(params)
	}

	id, err := ms.Add(ctx, store.AddParams{
		Type:       "procedure",
		Content:    content,
		Source:     learnSource,
		Confidence: learnConfidence,
	})
	if err != nil {
		return LearnResult{}, fmtErr("store procedure: %w", err)
	}

	slog.Info("llmem: learn: stored procedure", "id", id, "llm_status", llmStatus)
	return LearnResult{MemoryID: id, Content: content, LLMStatus: llmStatus}, nil
}

// buildRawLessonContent constructs the fallback content string from provided fields
// when LLM enrichment is not available or was skipped.
func buildRawLessonContent(params LearnLessonParams) string {
	var lines []string
	lines = append(lines, "WRONG: "+params.WhatWasWrong)
	lines = append(lines, "RIGHT: "+params.WhatIsCorrect)
	if params.Context != "" {
		lines = append(lines, "Context: "+params.Context)
	}
	return strings.Join(lines, "\n")
}

// buildFailurePrompt builds the prompt for failure introspection.
func buildFailurePrompt(params IntrospectFailureParams) string {
	fieldLines := taxonomy.IntrospectFieldLines()
	prompt := "Analyze this failure from a coding agent's session and produce a structured self-assessment.\n\n"
	prompt += "The agent encountered a problem mid-session. Based on the context below, identify what went wrong, " +
		"why it happened, whether it's a recurring pattern, and what procedural change would prevent it in the future.\n\n"
	prompt += "Format each field on its own line as \"Field: value\":\n\n"
	prompt += fieldLines + "\n\n"
	prompt += "Failure context:\n  What happened: " + params.WhatHappened
	if params.Category != "" {
		prompt += "\n  Category: " + params.Category
	}
	if params.Context != "" {
		prompt += "\n  Context: " + params.Context
	}
	if params.CaughtBy != "" {
		prompt += "\n  How caught: " + params.CaughtBy
	}
	if params.ProposedFix != "" {
		prompt += "\n  Proposed fix: " + params.ProposedFix
	}
	prompt += "\n\nProduce a structured self-assessment. Be specific about what went wrong and what should change."
	return prompt
}

// buildLessonPrompt builds the prompt for lesson learning.
func buildLessonPrompt(params LearnLessonParams) string {
	prompt := "A coding agent made a mistake and then corrected it. Distill the lesson into an actionable, " +
		"generalizable procedure that will prevent this mistake in future sessions.\n\n"
	prompt += "Be specific and practical. The procedure should be a rule the agent can follow — not vague advice.\n\n"
	prompt += "What was WRONG:\n" + params.WhatWasWrong + "\n\n"
	prompt += "What is CORRECT:\n" + params.WhatIsCorrect
	if params.Context != "" {
		prompt += "\n\nContext: " + params.Context
	}
	prompt += "\n\nWrite the lesson as a clear, actionable procedure. Start with the correct behavior, " +
		"then explain what to avoid. Keep it under 200 words."
	return prompt
}

// callModel attempts to call the Ollama model. Returns the LLM response and enrichment status.
// When timeout is zero, defaults to callModelTimeout (5 minutes).
// Returns ("", Skipped) when Ollama is unavailable or the call times out.
// Returns (response, Enriched) when the model call succeeds with non-empty response.
// Never panics.
func callModel(ctx context.Context, model, baseURL, prompt string, timeout time.Duration, httpClient *http.Client) (string, LLMEnrichment) {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	if timeout == 0 {
		timeout = callModelTimeout
	}

	timeoutCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ollamaCfg := ollama.OllamaClientConfig{
		BaseURL: baseURL,
		HTTPClient: httpClient,
	}
	client, err := ollama.NewOllamaClient(ollamaCfg)
	if err != nil {
		slog.Error("llmem: introspect: create Ollama client failed", "error", err)
		return "", Skipped
	}

	if !client.IsAvailable(timeoutCtx) {
		slog.Debug("llmem: introspect: Ollama not available, using storage-only fallback")
		return "", Skipped
	}

	response, err := client.Generate(timeoutCtx, prompt, model)
	if err != nil {
		slog.Error("llmem: introspect: model call failed", "error", err)
		return "", Skipped
	}

	if response == "" {
		slog.Error("llmem: introspect: model returned empty response")
		return "", Skipped
	}

	return response, Enriched
}

// callModelWithClient calls the Ollama model using a pre-configured OllamaClient.
// Returns empty string on failure (never panics).
// If ollamaClient is nil, returns "" immediately (graceful no-op, no network call).
// Uses a bounded timeout so callers never block indefinitely.
func callModelWithClient(ctx context.Context, ollamaClient *ollama.OllamaClient, model, prompt string) string {
	if ollamaClient == nil {
		slog.Debug("llmem: introspect: no OllamaClient provided, using storage-only fallback")
		return ""
	}

	timeoutCtx, cancel := context.WithTimeout(ctx, callModelTimeout)
	defer cancel()

	if !ollamaClient.IsAvailable(timeoutCtx) {
		slog.Debug("llmem: introspect: Ollama not available, using storage-only fallback")
		return ""
	}

	response, err := ollamaClient.Generate(timeoutCtx, prompt, model)
	if err != nil {
		slog.Error("llmem: introspect: model call failed", "error", err)
		return ""
	}

	return response
}

// IntrospectTranscript analyzes a session transcript and stores a self_assessment memory.
// On LLM availability, it uses the model to produce a structured self-assessment from the
// transcript content. On LLM failure, it falls back to a plain-text summary of the session.
//
// The ollamaClient parameter provides the configured Ollama connection. If nil,
// IntrospectTranscript falls back to degraded storage immediately (no LLM call attempted).
//
// Contract: NEVER returns ("", nil) — either creates a memory or returns an error.
// Even on LLM failure, a degraded memory is created.
// Returns ("", error) only if the transcript is empty (validation error).
func IntrospectTranscript(ctx context.Context, ms *store.MemoryStore, transcript string, sessionID string, ollamaClient *ollama.OllamaClient, model string) (string, error) {
	if transcript == "" {
		return "", fmtErr("transcript is required for introspection")
	}
	if model == "" {
		model = defaultModel
	}

	var content string
	prompt := buildTranscriptPrompt(transcript, sessionID)
	llmResponse := callModelWithClient(ctx, ollamaClient, model, prompt)

	if llmResponse != "" {
		content = llmResponse
	} else {
		// Graceful degradation: build from provided fields
		var lines []string
		lines = append(lines, "Session_id: "+sessionID)
		lines = append(lines, "Summary: session completed")
		// Truncate transcript for storage if very long
		transcriptExcerpt := transcript
		if len(transcriptExcerpt) > 500 {
			transcriptExcerpt = transcriptExcerpt[:500] + "..."
		}
		lines = append(lines, "Transcript_excerpt: "+transcriptExcerpt)
		content = strings.Join(lines, "\n")
	}

	id, err := ms.Add(context.Background(), store.AddParams{
		Type:       "self_assessment",
		Content:    content,
		Source:     introspectSource,
		Confidence: 0.8,
	})
	if err != nil {
		return "", fmtErr("store self_assessment: %w", err)
	}

	slog.Info("llmem: introspect: stored transcript self_assessment", "id", id, "session_id", sessionID)
	return id, nil
}

// buildTranscriptPrompt builds the prompt for transcript introspection.
func buildTranscriptPrompt(transcript, sessionID string) string {
	prompt := "Analyze the following session transcript and produce a structured self-assessment.\n\n"
	prompt += "The session has ended. Identify key decisions, preferences, project state updates, " +
		"and any lessons learned during this session.\n\n"
	prompt += "Format each insight as a separate point:\n" +
		"- What was decided and why\n" +
		"- What preferences were expressed\n" +
		"- What project state changed\n" +
		"- What patterns or recurring issues appeared\n\n"
	prompt += "Session ID: " + sessionID + "\n\n"

	// Truncate very long transcripts to avoid overwhelming the model
	transcriptExcerpt := transcript
	if len(transcriptExcerpt) > 3000 {
		transcriptExcerpt = transcriptExcerpt[:3000] + "\n...(truncated)"
	}
	prompt += "Transcript:\n" + transcriptExcerpt
	prompt += "\n\nProduce a structured self-assessment of this session."
	return prompt
}

// orDefault returns val if non-empty, otherwise returns defaultVal.
func orDefault(val, defaultVal string) string {
	if val != "" {
		return val
	}
	return defaultVal
}