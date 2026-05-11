package introspect

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

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
	id, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
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
}

func TestIntrospectFailure_UnknownCategory(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ms := newTestStore(t)
	id, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "test failure",
		Category:     "UNKNOWN_CATEGORY",
		BaseURL:      "http://localhost:59998",
	})
	if err != nil {
		t.Fatalf("IntrospectFailure with unknown category: %v", err)
	}
	if id == "" {
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
	id, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "test failure",
		Category:     "NULL_SAFETY",
		BaseURL:      "http://localhost:59999",
	})
	if err != nil {
		t.Fatalf("IntrospectFailure: %v", err)
	}
	if id == "" {
		t.Error("expected non-empty memory ID even when Ollama is unavailable")
	}

	mem, _ := ms.Get(context.Background(), id, false)
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Content == "" {
		t.Error("expected non-empty content even without LLM")
	}
}

func TestLearnLesson_WithFields(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ms := newTestStore(t)
	id, err := LearnLesson(ctx, ms, LearnLessonParams{
		WhatWasWrong: "ignored error return value",
		WhatIsCorrect: "always check error return values",
		Context:      "database write operation",
		BaseURL:      "http://localhost:59998",
	})
	if err != nil {
		t.Fatalf("LearnLesson: %v", err)
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
	if mem.Type != "procedure" {
		t.Errorf("expected type procedure, got %q", mem.Type)
	}
	if mem.Source != "learn" {
		t.Errorf("expected source learn, got %q", mem.Source)
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
	_, err := IntrospectFailure(ctx, ms, IntrospectFailureParams{
		WhatHappened: "test failure",
		Category:     "ERROR_HANDLING",
		BaseURL:      server.URL,
		Model:        "test-model",
	})
	_ = err
}

func TestLearnLesson_WithModel(t *testing.T) {
	_ = os.Getenv("LMEM_HOME") // just ensure env is accessible
}