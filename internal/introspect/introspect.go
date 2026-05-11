// Package introspect provides failure analysis and lesson learning for LLMem self-assessment.
// It uses an OllamaClient for LLM-assisted introspection with graceful degradation.
package introspect

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/MichielDean/LLMem/internal/ollama"
	"github.com/MichielDean/LLMem/internal/store"
	"github.com/MichielDean/LLMem/internal/taxonomy"
)

const (
	defaultModel          = "glm-5.1:cloud"
	defaultBaseURL        = "http://localhost:11434"
	introspectSource      = "introspect"
	learnSource           = "learn"
	introspectConfidence  = 0.9
	learnConfidence       = 0.85
)

// IntrospectFailureParams contains the parameters for introspecting a failure.
type IntrospectFailureParams struct {
	WhatHappened string
	Category     string
	Context      string
	CaughtBy     string
	ProposedFix  string
	Model        string
	BaseURL      string
}

// LearnLessonParams contains the parameters for learning a lesson from a wrong→right correction.
type LearnLessonParams struct {
	WhatWasWrong string
	WhatIsCorrect string
	Context      string
	Model        string
	BaseURL      string
}

// fmtErr wraps an error with the "llmem: introspect:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: introspect: "+format, args...)
}

// IntrospectFailure analyzes a failure and stores a self_assessment memory.
// If the LLM is available, it uses the model to expand the bare description into
// a structured self-assessment. If the model is unavailable, it stores a structured
// memory directly from the provided fields (graceful degradation).
//
// Contract: NEVER returns ("", nil) — either creates a memory or returns an error.
// Even on LLM failure, a storage-only memory is created.
func IntrospectFailure(ctx context.Context, ms *store.MemoryStore, params IntrospectFailureParams) (string, error) {
	if params.WhatHappened == "" {
		return "", fmtErr("what_happened is required")
	}
	if params.Model == "" {
		params.Model = defaultModel
	}

	// Warn about unknown categories but proceed anyway
	if params.Category != "" {
		if _, ok := taxonomy.ErrorTaxonomy[params.Category]; !ok {
			slog.Warn("llmem: introspect: unknown category, proceeding anyway", "category", params.Category)
		}
	}

	var content string
	llmResponse := callModel(ctx, params.Model, params.BaseURL, buildFailurePrompt(params))

	if llmResponse != "" {
		content = llmResponse
	} else {
		// Graceful degradation: build from provided fields
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
		content = strings.Join(lines, "\n")
	}

	id, err := ms.Add(ctx, store.AddParams{
		Type:       "self_assessment",
		Content:    content,
		Source:     introspectSource,
		Confidence: introspectConfidence,
	})
	if err != nil {
		return "", fmtErr("store self_assessment: %w", err)
	}

	slog.Info("llmem: introspect: stored self_assessment", "id", id)
	return id, nil
}

// LearnLesson analyzes a wrong→right correction and stores a procedure memory.
// If the LLM is available, it distills the correction into a generalizable procedure.
// If unavailable, it stores the lesson directly (graceful degradation).
//
// Contract: NEVER returns ("", nil) — either creates a memory or returns an error.
func LearnLesson(ctx context.Context, ms *store.MemoryStore, params LearnLessonParams) (string, error) {
	if params.WhatWasWrong == "" || params.WhatIsCorrect == "" {
		return "", fmtErr("what_was_wrong and what_is_correct are required")
	}
	if params.Model == "" {
		params.Model = defaultModel
	}

	var content string
	llmResponse := callModel(ctx, params.Model, params.BaseURL, buildLessonPrompt(params))

	if llmResponse != "" {
		content = llmResponse
	} else {
		// Graceful degradation: build from provided fields
		var lines []string
		lines = append(lines, "WRONG: "+params.WhatWasWrong)
		lines = append(lines, "RIGHT: "+params.WhatIsCorrect)
		if params.Context != "" {
			lines = append(lines, "Context: "+params.Context)
		}
		content = strings.Join(lines, "\n")
	}

	id, err := ms.Add(ctx, store.AddParams{
		Type:       "procedure",
		Content:    content,
		Source:     learnSource,
		Confidence: learnConfidence,
	})
	if err != nil {
		return "", fmtErr("store procedure: %w", err)
	}

	slog.Info("llmem: learn: stored procedure", "id", id)
	return id, nil
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

const callModelTimeout = 5 * time.Minute

// callModel attempts to call the Ollama model. Returns empty string on failure (never panics).
// Uses a bounded timeout so callers never block indefinitely.
func callModel(ctx context.Context, model, baseURL, prompt string) string {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	timeoutCtx, cancel := context.WithTimeout(ctx, callModelTimeout)
	defer cancel()

	client, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL: baseURL,
	})
	if err != nil {
		slog.Error("llmem: introspect: create Ollama client failed", "error", err)
		return ""
	}

	if !client.IsAvailable(timeoutCtx) {
		slog.Debug("llmem: introspect: Ollama not available, using storage-only fallback")
		return ""
	}

	response, err := client.Generate(timeoutCtx, prompt, model)
	if err != nil {
		slog.Error("llmem: introspect: model call failed", "error", err)
		return ""
	}

	return response
}

// orDefault returns val if non-empty, otherwise returns defaultVal.
func orDefault(val, defaultVal string) string {
	if val != "" {
		return val
	}
	return defaultVal
}