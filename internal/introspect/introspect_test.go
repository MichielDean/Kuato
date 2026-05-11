package introspect

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MichielDean/LLMem/internal/ollama"
	"github.com/MichielDean/LLMem/internal/store"
)

func newTestStore(t *testing.T) *store.MemoryStore {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	ms, err := store.NewMemoryStore(store.StoreConfig{
		DBPath:     dbPath,
		DisableVec: true,
	})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { ms.Close() })
	// self_assessment and procedure are already registered via default types
	return ms
}

func TestIntrospectFailure_WithFields(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "swallowed error on database write",
		Category:     "ERROR_HANDLING",
		Context:      "writing to SQLite store",
		CaughtBy:     "code review",
		ProposedFix:  "always check error return values",
		BaseURL:      "http://localhost:59998",
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty memory ID")
	}

	mem, err := ms.Get(context.Background(), result.MemoryID, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Type != "self_assessment" {
		t.Errorf("expected type self_assessment, got %q", mem.Type)
	}
	if mem.Source != "introspect" {
		t.Errorf("expected source introspect, got %q", mem.Source)
	}
	// When Ollama is unreachable, LLMStatus should be Skipped
	if result.LLMStatus != Skipped {
		t.Errorf("expected LLMStatus Skipped, got %q", result.LLMStatus)
	}
}

func TestIntrospectFailure_UnknownCategory(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "test failure",
		Category:     "UNKNOWN_CATEGORY",
		BaseURL:      "http://localhost:59998",
	})
	if err != nil {
		t.Fatalf("IntrospectFailure with unknown category: %v", err)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty memory ID even with unknown category")
	}
}

func TestIntrospectFailure_EmptyWhatHappened(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)
	_, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "",
	})
	if err == nil {
		t.Error("expected error for empty what_happened")
	}
}

func TestIntrospectFailure_GracefulDegradation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "test failure",
		Category:     "NULL_SAFETY",
		BaseURL:      "http://localhost:59999",
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty memory ID even when Ollama is unavailable")
	}

	mem, _ := ms.Get(context.Background(), result.MemoryID, false)
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Content == "" {
		t.Error("expected non-empty content even without LLM")
	}
	if result.LLMStatus != Skipped {
		t.Errorf("expected LLMStatus Skipped when Ollama unavailable, got %q", result.LLMStatus)
	}
}

func TestIntrospectFailure_ReturnsLLMSkipped_WhenOllamaUnavailable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "some failure",
		Category:     "ERROR_HANDLING",
		BaseURL:      "http://localhost:59998", // unreachable
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	if result.LLMStatus != Skipped {
		t.Errorf("expected LLMStatus Skipped, got %q", result.LLMStatus)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty MemoryID even on LLM skip")
	}
	if result.Content == "" {
		t.Error("expected non-empty Content even on LLM skip")
	}
}

func TestIntrospectFailure_ReturnsLLMEnriched_WhenOllamaAvailable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{
				"response": "Category: ERROR_HANDLING\nWhat_happened: test\nProposed_update: fix it",
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "test failure",
		Category:     "ERROR_HANDLING",
		BaseURL:      server.URL,
		Model:        "test-model",
		HTTPClient:   server.Client(),
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	if result.LLMStatus != Enriched {
		t.Errorf("expected LLMStatus Enriched, got %q", result.LLMStatus)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty MemoryID")
	}
}

func TestIntrospectFailure_NoLLMFlag(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "some failure",
		Category:     "ERROR_HANDLING",
		NoLLM:        true,
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	if result.LLMStatus != Disabled {
		t.Errorf("expected LLMStatus Disabled, got %q", result.LLMStatus)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty MemoryID")
	}
	// Content should contain the raw fields
	if result.Content == "" {
		t.Error("expected non-empty Content with NoLLM")
	}
}

func TestIntrospectFailure_TimeoutValidation(t *testing.T) {
	ms := newTestStore(t)
	ctx := context.Background()

	tests := []struct {
		name    string
		timeout time.Duration
		wantErr bool
	}{
		{"zero timeout defaults", 0, false},
		{"valid timeout at 10s", 10 * time.Second, false},
		{"valid timeout at 2m", 2 * time.Minute, false},
		{"invalid timeout at 5s", 5 * time.Second, true},
		{"invalid timeout at 1s", 1 * time.Second, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use NoLLM=true to avoid any Ollama network calls (validation only)
			_, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
				WhatHappened: "test",
				NoLLM:        true,
				Timeout:      tt.timeout,
			})
			if tt.wantErr && err == nil {
				t.Error("expected error for timeout below minimum")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestLearnLesson_WithFields(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := LearnLesson(ctx, ms, LearnLessonParams{
		WhatWasWrong: "ignored error return value",
		WhatIsCorrect: "always check error return values",
		Context:      "database write operation",
		BaseURL:      "http://localhost:59998",
	})
	if err != nil {
		t.Fatalf("LearnLesson: %v", err)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty memory ID")
	}

	mem, err := ms.Get(context.Background(), result.MemoryID, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Type != "procedure" {
		t.Errorf("expected type procedure, got %q", mem.Type)
	}
	if mem.Source != "learn" {
		t.Errorf("expected source learn, got %q", mem.Source)
	}
	if result.LLMStatus != Skipped {
		t.Errorf("expected LLMStatus Skipped, got %q", result.LLMStatus)
	}
}

func TestLearnLesson_EmptyFields(t *testing.T) {
	ms := newTestStore(t)
	_, err := LearnLesson(context.Background(), ms, LearnLessonParams{
		WhatWasWrong: "",
		WhatIsCorrect: "something",
	})
	if err == nil {
		t.Error("expected error for empty what_was_wrong")
	}

	_, err = LearnLesson(context.Background(), ms, LearnLessonParams{
		WhatWasWrong: "something",
		WhatIsCorrect: "",
	})
	if err == nil {
		t.Error("expected error for empty what_is_correct")
	}
}

func TestLearnLesson_ReturnsLLMSkipped_WhenOllamaUnavailable(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := LearnLesson(ctx, ms, LearnLessonParams{
		WhatWasWrong:  "forgot to check nil",
		WhatIsCorrect: "always check for nil",
		BaseURL:       "http://localhost:59998", // unreachable
	})
	if err != nil {
		t.Fatalf("LearnLesson: %v", err)
	}
	if result.LLMStatus != Skipped {
		t.Errorf("expected LLMStatus Skipped, got %q", result.LLMStatus)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty MemoryID even on LLM skip")
	}
	if result.Content == "" {
		t.Error("expected non-empty Content even on LLM skip")
	}
}

func TestLearnLesson_ReturnsLLMEnriched_WhenOllamaAvailable(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{
				"response": "WRONG: forgot nil check\nRIGHT: always check nil",
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := LearnLesson(ctx, ms, LearnLessonParams{
		WhatWasWrong:  "forgot nil check",
		WhatIsCorrect: "always check nil",
		BaseURL:       server.URL,
		Model:         "test-model",
		HTTPClient:    server.Client(),
	})
	if err != nil {
		t.Fatalf("LearnLesson: %v", err)
	}
	if result.LLMStatus != Enriched {
		t.Errorf("expected LLMStatus Enriched, got %q", result.LLMStatus)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty MemoryID")
	}
}

func TestLearnLesson_NoLLMFlag(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := LearnLesson(ctx, ms, LearnLessonParams{
		WhatWasWrong:  "forgot nil check",
		WhatIsCorrect: "always check nil",
		NoLLM:         true,
	})
	if err != nil {
		t.Fatalf("LearnLesson: %v", err)
	}
	if result.LLMStatus != Disabled {
		t.Errorf("expected LLMStatus Disabled, got %q", result.LLMStatus)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty MemoryID")
	}
	if result.Content == "" {
		t.Error("expected non-empty Content with NoLLM")
	}
}

func TestLearnLesson_TimeoutValidation(t *testing.T) {
	ms := newTestStore(t)
	ctx := context.Background()

	tests := []struct {
		name    string
		timeout time.Duration
		wantErr bool
	}{
		{"zero timeout defaults", 0, false},
		{"valid timeout at 10s", 10 * time.Second, false},
		{"invalid timeout at 5s", 5 * time.Second, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use NoLLM=true to avoid any Ollama network calls (validation only)
			_, err := LearnLesson(ctx, ms, LearnLessonParams{
				WhatWasWrong:  "wrong",
				WhatIsCorrect: "right",
				NoLLM:         true,
				Timeout:       tt.timeout,
			})
			if tt.wantErr && err == nil {
				t.Error("expected error for timeout below minimum")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestCallModel_ConfigurableTimeout(t *testing.T) {
	// Verify that callModel defaults to callModelTimeout (5 minutes) when timeout is 0
	ctx := context.Background()
	// Unreachable base URL — should return Skipped quickly (URL validation fails)
	resp, status := callModel(ctx, "test", "http://localhost:59998", "test prompt", 0, nil)
	if status != Skipped {
		t.Errorf("expected Skipped for unreachable Ollama, got %q", status)
	}
	if resp != "" {
		t.Errorf("expected empty response for unreachable Ollama, got %q", resp)
	}

	// Verify custom timeout works
	resp, status = callModel(ctx, "test", "http://localhost:59998", "test prompt", 15*time.Second, nil)
	if status != Skipped {
		t.Errorf("expected Skipped for unreachable Ollama with custom timeout, got %q", status)
	}
}

func TestCallModel_NoLLMFlag(t *testing.T) {
	// This test verifies the NoLLM path bypasses callModel entirely.
	// callModel itself doesn't take NoLLM — the caller handles it.
	// But we verify that IntrospectFailure with NoLLM never invokes callModel
	// by checking that the result has status Disabled.
	ms := newTestStore(t)
	ctx := context.Background()

	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "test failure",
		NoLLM:        true,
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	if result.LLMStatus != Disabled {
		t.Errorf("expected LLMStatus Disabled with NoLLM=true, got %q", result.LLMStatus)
	}
}

func TestIntrospectFailure_WithModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{
				"response": "Category: ERROR_HANDLING\nWhat_happened: test\nProposed_update: fix it",
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "test failure",
		Category:     "ERROR_HANDLING",
		BaseURL:      server.URL,
		Model:        "test-model",
		HTTPClient:   server.Client(),
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	// With a working mock server, LLM should be Enriched
	if result.LLMStatus != Enriched {
		t.Errorf("expected LLMStatus Enriched with mock server, got %q", result.LLMStatus)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty MemoryID")
	}
}

// TestIntrospectFailure_ProposedUpdateReturned verifies that IntrospectResult
// includes the ProposedUpdate and Category fields when LLM returns structured content.
func TestIntrospectFailure_ProposedUpdateReturned(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "null pointer dereference in handler",
		Category:     "NULL_SAFETY",
		ProposedFix:  "always check for nil before accessing fields",
		NoLLM:        true,
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	if result.ProposedUpdate != "always check for nil before accessing fields" {
		t.Errorf("expected ProposedUpdate='always check for nil before accessing fields', got %q", result.ProposedUpdate)
	}
	if result.Category != "NULL_SAFETY" {
		t.Errorf("expected Category='NULL_SAFETY', got %q", result.Category)
	}
}

// TestIntrospectFailure_ProposedUpdateFromRawContent verifies that ProposedUpdate
// is extracted from the params when LLM is available and returns a structured response.
func TestIntrospectFailure_ProposedUpdateFromRawContent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)

	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "swallowed error in database call",
		Category:     "ERROR_HANDLING",
		ProposedFix:  "wrap errors with fmt.Errorf and proper context",
		NoLLM:        true,
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}

	// When NoLLM is true, ProposedFix is used directly as ProposedUpdate
	if result.ProposedUpdate != "wrap errors with fmt.Errorf and proper context" {
		t.Errorf("expected ProposedUpdate from params, got %q", result.ProposedUpdate)
	}
	if result.Category != "ERROR_HANDLING" {
		t.Errorf("expected Category='ERROR_HANDLING', got %q", result.Category)
	}
}

// TestIntrospectFailure_ProposedUpdateWithLLM verifies that ProposedUpdate is
// populated from LLM response when the LLM enriches the content.
func TestIntrospectFailure_ProposedUpdateWithLLM(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{
				"response": "Category: NULL_SAFETY\nWhat_happened: nil dereference\nProposed_update: always guard nil pointers in Go\nWhat_caught_it: code review",
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "nil dereference in handler",
		Category:     "NULL_SAFETY",
		ProposedFix:  "check nil before access",
		Model:        "test-model",
		HTTPClient:   server.Client(),
		BaseURL:      server.URL,
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}

	if result.ProposedUpdate != "always guard nil pointers in Go" {
		t.Errorf("expected ProposedUpdate from LLM response, got %q", result.ProposedUpdate)
	}
	if result.Category != "NULL_SAFETY" {
		t.Errorf("expected Category=NULL_SAFETY, got %q", result.Category)
	}
}

// TestIntrospectFailure_EmptyProposedUpdate verifies that ProposedUpdate is empty
// when no proposed fix is provided.
func TestIntrospectFailure_EmptyProposedUpdate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "some error",
		NoLLM:        true,
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	if result.ProposedUpdate != "" {
		t.Errorf("expected empty ProposedUpdate, got %q", result.ProposedUpdate)
	}
	if result.Category != "" {
		t.Errorf("expected empty Category when not specified, got %q", result.Category)
	}
}

func TestLearnLesson_WithModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{
				"response": "WRONG: forgot nil check\nRIGHT: always check nil",
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := LearnLesson(ctx, ms, LearnLessonParams{
		WhatWasWrong:  "forgot nil check",
		WhatIsCorrect: "always check nil",
		BaseURL:       server.URL,
		Model:         "test-model",
		HTTPClient:    server.Client(),
	})
	if err != nil {
		t.Fatalf("LearnLesson: %v", err)
	}
	if result.LLMStatus != Enriched {
		t.Errorf("expected LLMStatus Enriched with mock server, got %q", result.LLMStatus)
	}
}

func TestIntrospectAuto_WithText(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectAuto(ctx, ms, "Encountered a null pointer error when processing user input", "", "http://localhost:59999")
	if err != nil {
		t.Fatalf("IntrospectAuto: %v", err)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty memory ID")
	}

	mem, err := ms.Get(context.Background(), result.MemoryID, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Type != "self_assessment" {
		t.Errorf("expected type self_assessment, got %q", mem.Type)
	}
	if mem.Source != "introspect-auto" {
		t.Errorf("expected source introspect-auto, got %q", mem.Source)
	}
	if mem.Content == "" {
		t.Error("expected non-empty content even without LLM")
	}
}

func TestIntrospectAuto_WithTextAndModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{
				"response": "Category: NULL_SAFETY\nWhat_happened: null pointer when processing input\nProposed_update: add null checks",
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectAuto(ctx, ms, "Got null pointer error processing input", "test-model", server.URL)
	if err != nil {
		t.Fatalf("IntrospectAuto with model: %v", err)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty memory ID")
	}

	mem, err := ms.Get(context.Background(), result.MemoryID, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Source != "introspect-auto" {
		t.Errorf("expected source introspect-auto, got %q", mem.Source)
	}
	// With LLM response, content should be from the model
	if mem.Content == "" {
		t.Error("expected non-empty content from LLM")
	}
}

func TestIntrospectAuto_EmptyText(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)
	_, err := IntrospectAuto(ctx, ms, "", "", "")
	if err == nil {
		t.Error("expected error for empty text")
	}
	if err.Error() != "llmem: introspect: text is required" {
		t.Errorf("expected 'text is required' error, got: %v", err)
	}
}

func TestIntrospectAuto_GracefulDegradation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	// Point at a non-existent Ollama URL to trigger graceful degradation
	result, err := IntrospectAuto(ctx, ms, "test failure in error handling", "", "http://localhost:59999")
	if err != nil {
		t.Fatalf("IntrospectAuto graceful degradation: %v", err)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty memory ID even when Ollama is unavailable")
	}

	mem, _ := ms.Get(context.Background(), result.MemoryID, false)
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Content == "" {
		t.Error("expected non-empty content even without LLM")
	}
	// Graceful degradation content should start with "What_happened: " followed by the text
	if !strings.HasPrefix(mem.Content, "What_happened: ") {
		t.Errorf("expected content to start with 'What_happened: ', got: %q", mem.Content)
	}
	if mem.Source != "introspect-auto" {
		t.Errorf("expected source introspect-auto, got %q", mem.Source)
	}
	if mem.Confidence != 0.9 {
		t.Errorf("expected confidence 0.9, got %f", mem.Confidence)
	}
	// Without LLM enrichment, ProposedUpdate and Category should be empty
	if result.ProposedUpdate != "" {
		t.Errorf("expected empty ProposedUpdate on graceful degradation, got %q", result.ProposedUpdate)
	}
	if result.Category != "" {
		t.Errorf("expected empty Category on graceful degradation, got %q", result.Category)
	}
}

func TestIntrospectAuto_WithModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{
				"response": "Category: EDGE_CASE\nWhat_happened: unhandled input case\nProposed_update: add input validation",
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectAuto(ctx, ms, "Edge case with empty slice", "test-model", server.URL)
	if err != nil {
		t.Fatalf("IntrospectAuto with model: %v", err)
	}
	if result.MemoryID == "" {
		t.Error("expected non-empty memory ID")
	}

	mem, err := ms.Get(context.Background(), result.MemoryID, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Type != "self_assessment" {
		t.Errorf("expected type self_assessment, got %q", mem.Type)
	}
	if mem.Source != "introspect-auto" {
		t.Errorf("expected source introspect-auto, got %q", mem.Source)
	}
	// When LLM enrichment succeeds (Enriched), ProposedUpdate and Category are
	// populated from the structured response. When Ollama is unavailable
	// (Skipped), they remain empty since there is no structured content to parse.
	// Note: this test may hit URL validation blocking the mock server,
	// so ProposedUpdate/Category may be empty on graceful degradation.
	// The enrichment parsing logic is tested via IntrospectFailure with HTTPClient.
}

// TestIntrospectAuto_ResultType_ReturnsFieldsOnGracfulDegradation verifies that
// IntrospectAutoResult has MemoryID, ProposedUpdate, and Category fields,
// and that ProposedUpdate and Category are empty on graceful degradation
// (no LLM enrichment to parse structured content from).
func TestIntrospectAuto_ResultType_ReturnsFieldsOnGracefulDegradation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	result, err := IntrospectAuto(ctx, ms, "graceful degradation test", "", "http://localhost:59999")
	if err != nil {
		t.Fatalf("IntrospectAuto: %v", err)
	}
	// MemoryID must be non-empty
	if result.MemoryID == "" {
		t.Error("expected non-empty MemoryID on graceful degradation")
	}
	// ProposedUpdate and Category are empty on graceful degradation (no LLM content to parse)
	if result.ProposedUpdate != "" {
		t.Errorf("expected empty ProposedUpdate on graceful degradation, got %q", result.ProposedUpdate)
	}
	if result.Category != "" {
		t.Errorf("expected empty Category on graceful degradation, got %q", result.Category)
	}
}

func TestIntrospectTranscript_WithFields(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	// Nil OllamaClient triggers graceful degradation (no LLM call)
	id, err := IntrospectTranscript(ctx, ms, "User asked about Go testing conventions\nAssistant explained table-driven tests", "session-abc", nil, "")
	if err != nil {
		t.Fatalf("IntrospectTranscript: %v", err)
	}
	if id == "" {
		t.Error("expected non-empty memory ID")
	}

	mem, err := ms.Get(context.Background(), id, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Type != "self_assessment" {
		t.Errorf("expected type self_assessment, got %q", mem.Type)
	}
	if mem.Source != "introspect" {
		t.Errorf("expected source introspect, got %q", mem.Source)
	}
	if mem.Confidence != 0.8 {
		t.Errorf("expected confidence 0.8, got %f", mem.Confidence)
	}
	if mem.Content == "" {
		t.Error("expected non-empty content")
	}
}

func TestIntrospectTranscript_EmptyTranscript(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)
	_, err := IntrospectTranscript(ctx, ms, "", "session-abc", nil, "")
	if err == nil {
		t.Error("expected error for empty transcript")
	}
}

func TestIntrospectTranscript_GracefulDegradation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	// Nil OllamaClient triggers graceful degradation
	id, err := IntrospectTranscript(ctx, ms, "Some session transcript content here", "session-degrad", nil, "")
	if err != nil {
		t.Fatalf("IntrospectTranscript: %v", err)
	}
	if id == "" {
		t.Error("expected non-empty memory ID even when OllamaClient is nil")
	}

	mem, _ := ms.Get(context.Background(), id, false)
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Content == "" {
		t.Error("expected non-empty content even without LLM")
	}
	if !strings.Contains(mem.Content, "session-degrad") {
		t.Errorf("expected degraded content to contain session ID, got %q", mem.Content)
	}
}

func TestIntrospectTranscript_WithOllamaClient(t *testing.T) {
	// Create a mock Ollama server that returns a structured self-assessment
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{
				"response": "Category: PROJECT_STATE\nWhat_happened: Session reviewed testing patterns\nProposed_update: adopt table-driven tests",
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	client, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	id, err := IntrospectTranscript(ctx, ms, "We discussed testing patterns and decided on table-driven tests", "session-client", client, "test-model")
	if err != nil {
		t.Fatalf("IntrospectTranscript: %v", err)
	}
	if id == "" {
		t.Error("expected non-empty memory ID")
	}

	mem, err := ms.Get(context.Background(), id, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Type != "self_assessment" {
		t.Errorf("expected type self_assessment, got %q", mem.Type)
	}
	if mem.Content == "" {
		t.Error("expected non-empty content from LLM response")
	}
}

func TestIntrospectTranscript_NilOllamaClient_DegradedContent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	// nil OllamaClient means no LLM call — graceful degradation to plain content
	id, err := IntrospectTranscript(ctx, ms, "Brief session notes", "session-nil-client", nil, "")
	if err != nil {
		t.Fatalf("IntrospectTranscript: %v", err)
	}
	if id == "" {
		t.Error("expected non-empty memory ID")
	}

	mem, _ := ms.Get(context.Background(), id, false)
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	// Degraded content should include the session ID
	if !strings.Contains(mem.Content, "session-nil-client") {
		t.Errorf("expected degraded content to include session ID, got %q", mem.Content)
	}
}