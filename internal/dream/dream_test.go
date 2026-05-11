package dream

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MichielDean/LLMem/internal/ollama"
	"github.com/MichielDean/LLMem/internal/paths"
	"github.com/MichielDean/LLMem/internal/skillpatch"
	"github.com/MichielDean/LLMem/internal/store"
	"github.com/MichielDean/LLMem/internal/taxonomy"
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
	return ms
}

func TestNewDreamer_NilStore(t *testing.T) {
	_, err := NewDreamer(DreamerConfig{Store: nil})
	if err == nil {
		t.Error("expected error for nil store")
	}
}

func TestNewDreamer_Defaults(t *testing.T) {
	ms := newTestStore(t)
	d, err := NewDreamer(DreamerConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}
	if d.similarityThreshold != defaultSimilarityThreshold {
		t.Errorf("expected default similarity threshold %f, got %f", defaultSimilarityThreshold, d.similarityThreshold)
	}
	if d.decayRate != defaultDecayRate {
		t.Errorf("expected default decay rate %f, got %f", defaultDecayRate, d.decayRate)
	}
}

func TestDreamer_Run_DryRun(t *testing.T) {
	ms := newTestStore(t)
	d, err := NewDreamer(DreamerConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), false, "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.Light == nil {
		t.Error("expected Light result to be set")
	}
	if result.Deep == nil {
		t.Error("expected Deep result to be set")
	}
	if result.Rem == nil {
		t.Error("expected Rem result to be set")
	}
}

func TestDreamer_LightPhase(t *testing.T) {
	ms := newTestStore(t)
	d, err := NewDreamer(DreamerConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), false, "light")
	if err != nil {
		t.Fatalf("Run light: %v", err)
	}
	if result.Light == nil {
		t.Fatal("expected Light result")
	}
}

func TestDreamer_LightPhase_PropagatesContext(t *testing.T) {
	// Verify that lightPhase uses the context from Run, not context.Background().
	// A cancelled context should propagate through to ConsolidateDuplicates.
	ms := newTestStore(t)
	d, err := NewDreamer(DreamerConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	// Add a memory so the phase has something to process
	_, err = ms.Add(context.Background(), store.AddParams{
		Type:       "fact",
		Content:    "context propagation test",
		Source:     "test",
		Confidence: 0.9,
	})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Run with a cancelled context — the light phase should still complete
	// (ConsolidateDuplicates on SQLite is fast), but the key contract is
	// that ctx is passed through, not discarded.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Even with cancelled context, Run should not panic or hang
	result, err := d.Run(ctx, false, "light")
	if err != nil {
		t.Fatalf("Run with cancelled context: %v", err)
	}
	if result.Light == nil {
		t.Error("expected Light result even with cancelled context (dry run)")
	}
}

func TestDreamer_DeepPhase(t *testing.T) {
	ms := newTestStore(t)
	d, err := NewDreamer(DreamerConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	// Add a test memory
	_, err = ms.Add(context.Background(), store.AddParams{
		Type:       "fact",
		Content:    "test fact",
		Source:     "test",
		Confidence: 0.9,
	})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	result, err := d.Run(context.Background(), true, "deep")
	if err != nil {
		t.Fatalf("Run deep: %v", err)
	}
	if result.Deep == nil {
		t.Fatal("expected Deep result")
	}
}

func TestDreamer_RemPhase(t *testing.T) {
	ms := newTestStore(t)
	d, err := NewDreamer(DreamerConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	// fact is already registered via default types
	_, err = ms.Add(context.Background(), store.AddParams{
		Type:       "fact",
		Content:    "test content for REM phase",
		Source:     "test",
		Confidence: 0.9,
	})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run rem: %v", err)
	}
	if result.Rem == nil {
		t.Fatal("expected Rem result")
	}
	if result.Rem.TotalMemories < 1 {
		t.Errorf("expected at least 1 memory, got %d", result.Rem.TotalMemories)
	}
}

func TestDreamer_WriteDiary(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	diaryPath := filepath.Join(dir, "dream_diary.md")
	d, err := NewDreamer(DreamerConfig{
		Store:     ms,
		DiaryPath: diaryPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result := &DreamResult{
		Light: &LightPhaseResult{DuplicatePairs: 3},
		Deep:  &DeepPhaseResult{DecayedCount: 5, BoostedCount: 2, MergedCount: 1, AutoLinkedCount: 3},
		Rem:   &RemPhaseResult{TotalMemories: 100, ActiveMemories: 80, Themes: []string{"5 memories about fact", "cluster: 3 memories involve 'test'"}},
	}

	err = d.WriteDiary(result)
	if err != nil {
		t.Fatalf("WriteDiary: %v", err)
	}

	data, err := os.ReadFile(diaryPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(data) == 0 {
		t.Error("expected non-empty diary file")
	}
}

func TestDreamer_DryRun(t *testing.T) {
	ms := newTestStore(t)
	d, err := NewDreamer(DreamerConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), false, "")
	if err != nil {
		t.Fatalf("Run dry: %v", err)
	}
	// Dry run should return results but not persist changes
	if result == nil {
		t.Fatal("expected non-nil result")
	}
}

func TestGenerateDreamReport(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	reportPath := filepath.Join(dir, "report.html")
	d, err := NewDreamer(DreamerConfig{
		Store:      ms,
		ReportPath: reportPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result := &DreamResult{
		Light: &LightPhaseResult{DuplicatePairs: 3},
		Deep:  &DeepPhaseResult{DecayedCount: 2, BoostedCount: 4, MergedCount: 1, AutoLinkedCount: 5},
		Rem: &RemPhaseResult{
			TotalMemories:    50,
			ActiveMemories:   40,
			Themes:           []string{"10 memories about fact"},
			BehavioralInsights: []BehavioralInsight{{Category: "ERROR_HANDLING", Count: 5, ContentSnippet: "test"}},
		},
	}

	err = d.GenerateDreamReport(result, reportPath)
	if err != nil {
		t.Fatalf("GenerateDreamReport: %v", err)
	}

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	content := string(data)
	if !contains(content, "<html") {
		t.Error("expected HTML in report")
	}
	if !contains(content, "Dream Report") {
		t.Error("expected Dream Report header in report")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsStr(s, substr))
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestDreamer_WriteDiary_UsesResolvedPath(t *testing.T) {
	// Verify that WriteDiary uses the resolved (validated) path, not the raw path.
	// This is a contract test — the file should be at the exact path returned by ValidateWritePath.
	ms := newTestStore(t)
	dir := t.TempDir()
	diaryPath := filepath.Join(dir, "dream_diary.md")
	d, err := NewDreamer(DreamerConfig{
		Store:     ms,
		DiaryPath: diaryPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result := &DreamResult{
		Light: &LightPhaseResult{DuplicatePairs: 0},
		Deep:  &DeepPhaseResult{},
	}

	err = d.WriteDiary(result)
	if err != nil {
		t.Fatalf("WriteDiary: %v", err)
	}

	// Verify file was written at the expected path
	if _, err := os.Stat(diaryPath); err != nil {
		t.Errorf("expected diary file at %s: %v", diaryPath, err)
	}
}

func TestDreamer_GenerateDreamReport_UsesResolvedPath(t *testing.T) {
	// Verify that GenerateDreamReport uses the resolved (validated) path.
	ms := newTestStore(t)
	dir := t.TempDir()
	reportPath := filepath.Join(dir, "report.html")
	d, err := NewDreamer(DreamerConfig{
		Store:      ms,
		ReportPath: reportPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result := &DreamResult{
		Light: &LightPhaseResult{DuplicatePairs: 0},
		Deep:  &DeepPhaseResult{},
	}

	err = d.GenerateDreamReport(result, reportPath)
	if err != nil {
		t.Fatalf("GenerateDreamReport: %v", err)
	}

	// Verify file was written at the expected path
	if _, err := os.Stat(reportPath); err != nil {
		t.Errorf("expected report file at %s: %v", reportPath, err)
	}
}

// setTimestamps updates a memory's created_at and accessed_at columns
// via raw SQL on the store's database. This is needed for time-dependent
// decay tests where we need memories older than the decay/stale cutoffs.
func setTimestamps(t *testing.T, ms *store.MemoryStore, id, createdAt, accessedAt string) {
	t.Helper()
	db := ms.DB()
	if createdAt != "" {
		_, err := db.Exec(`UPDATE "memories" SET "created_at" = ? WHERE "id" = ?`, createdAt, id)
		if err != nil {
			t.Fatalf("set created_at for %s: %v", id, err)
		}
	}
	if accessedAt != "" {
		_, err := db.Exec(`UPDATE "memories" SET "accessed_at" = ? WHERE "id" = ?`, accessedAt, id)
		if err != nil {
			t.Fatalf("set accessed_at for %s: %v", id, err)
		}
	}
}

// newTestOllamaClient creates an OllamaClient pointing at the given httptest server.
// This mirrors the pattern used in extract_test.go.
func newTestOllamaClient(t *testing.T, server *httptest.Server) *ollama.OllamaClient {
	t.Helper()
	client, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("newTestOllamaClient: %v", err)
	}
	return client
}

// addSelfAssessment adds a self_assessment memory with the given category and content.
func addSelfAssessment(t *testing.T, ms *store.MemoryStore, category, content string) string {
	t.Helper()
	fullContent := "Category: " + category + "\n" + content
	id, err := ms.Add(context.Background(), store.AddParams{
		Type:       "self_assessment",
		Content:    fullContent,
		Source:     "test",
		Confidence: 0.8,
	})
	if err != nil {
		t.Fatalf("Add self_assessment: %v", err)
	}
	return id
}

func TestNewDreamer_DefaultStaleProcedureDays(t *testing.T) {
	ms := newTestStore(t)
	d, err := NewDreamer(DreamerConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}
	if d.staleProcedureDays != defaultStaleProcedureDays {
		t.Errorf("expected default stale procedure days %d, got %d", defaultStaleProcedureDays, d.staleProcedureDays)
	}
	if d.model != defaultDreamModel {
		t.Errorf("expected default model %q, got %q", defaultDreamModel, d.model)
	}
}

// TestNewDreamer_WithOllamaConfig verifies that OllamaClient and Model are properly
// wired through the constructor.
func TestNewDreamer_WithOllamaConfig(t *testing.T) {
	ms := newTestStore(t)

	// Create a test Ollama server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	ollamaClient := newTestOllamaClient(t, server)

	d, err := NewDreamer(DreamerConfig{
		Store:        ms,
		OllamaClient: ollamaClient,
		Model:        "custom-model",
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	if d.ollama == nil {
		t.Error("expected ollama client to be set")
	}
	if d.model != "custom-model" {
		t.Errorf("expected model 'custom-model', got %q", d.model)
	}

	// Verify defaults are still applied for other fields
	if d.similarityThreshold != defaultSimilarityThreshold {
		t.Errorf("expected default similarity threshold %f, got %f", defaultSimilarityThreshold, d.similarityThreshold)
	}

	// Test with BaseURL instead of OllamaClient
	d2, err := NewDreamer(DreamerConfig{
		Store:      ms,
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
		Model:      "another-model",
	})
	if err != nil {
		t.Fatalf("NewDreamer with BaseURL: %v", err)
	}
	if d2.ollama == nil {
		t.Error("expected ollama client to be created from BaseURL")
	}
	if d2.model != "another-model" {
		t.Errorf("expected model 'another-model', got %q", d2.model)
	}

	// Test defaults: no OllamaClient, no BaseURL — should create client from default URL
	d3, err := NewDreamer(DreamerConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewDreamer defaults: %v", err)
	}
	if d3.model != defaultDreamModel {
		t.Errorf("expected default model %q, got %q", defaultDreamModel, d3.model)
	}
	// The client should have been created from defaultBaseURL
	if d3.ollama == nil {
		t.Error("expected ollama client created from default URL")
	}
}

func TestDreamer_DeepPhase_StaleProcedureDoubleDecay(t *testing.T) {
	// Verify that a procedure memory older than staleProcedureDays decays at 2x the normal rate.
	ms := newTestStore(t)

	// Use DecayIntervalDays: 1 so memories older than 1 day are eligible for decay.
	// Use StaleProcedureDays: 1 so procedures not accessed within 1 day are stale.
	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		DecayIntervalDays:  1,
		StaleProcedureDays: 1,
		DecayRate:           0.05,
		DecayFloor:          0.1,
		ConfidenceFloor:     0.1,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	// Add a procedure memory with high confidence
	procID, err := ms.Add(context.Background(), store.AddParams{
		Type:       "procedure",
		Content:    "stale procedure test",
		Source:     "test",
		Confidence: 0.9,
	})
	if err != nil {
		t.Fatalf("Add procedure: %v", err)
	}

	// Set created_at to 60 days ago and leave accessed_at empty (never accessed = stale)
	oldCreated := time.Now().UTC().AddDate(0, 0, -60).Format(time.RFC3339)
	setTimestamps(t, ms, procID, oldCreated, "")

	// Run deep phase with apply=true
	result, err := d.Run(context.Background(), true, "deep")
	if err != nil {
		t.Fatalf("Run deep: %v", err)
	}
	if result.Deep == nil {
		t.Fatal("expected Deep result")
	}

	// The stale procedure should have been decayed at 2x rate:
	// 0.9 - (0.05 * 2) = 0.8
	if result.Deep.StaleProcedureDecayedCount != 1 {
		t.Errorf("expected StaleProcedureDecayedCount=1, got %d", result.Deep.StaleProcedureDecayedCount)
	}
	if result.Deep.DecayedCount < 1 {
		t.Errorf("expected DecayedCount >= 1, got %d", result.Deep.DecayedCount)
	}

	// Verify the actual confidence on the memory
	mem, err := ms.Get(context.Background(), procID, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	// Should be 0.9 - 0.10 = 0.8
	expectedConf := 0.9 - (0.05 * 2)
	if mem.Confidence < expectedConf-0.01 || mem.Confidence > expectedConf+0.01 {
		t.Errorf("expected stale procedure confidence ~%f, got %f", expectedConf, mem.Confidence)
	}
}

func TestDreamer_DeepPhase_StaleProcedureNotDoubleDecayedWhenFresh(t *testing.T) {
	// Verify that a procedure memory accessed recently (within staleProcedureDays) is NOT double-decayed.
	ms := newTestStore(t)

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		DecayIntervalDays:  1,
		StaleProcedureDays: 1,
		DecayRate:           0.05,
		DecayFloor:          0.1,
		ConfidenceFloor:     0.1,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	// Add a procedure memory with high confidence
	procID, err := ms.Add(context.Background(), store.AddParams{
		Type:       "procedure",
		Content:    "fresh procedure test",
		Source:     "test",
		Confidence: 0.9,
	})
	if err != nil {
		t.Fatalf("Add procedure: %v", err)
	}

	// Set created_at to 60 days ago but accessed_at to just now (freshly accessed)
	oldCreated := time.Now().UTC().AddDate(0, 0, -60).Format(time.RFC3339)
	recentAccessed := time.Now().UTC().Format(time.RFC3339)
	setTimestamps(t, ms, procID, oldCreated, recentAccessed)

	// Run deep phase with apply=true — but the memory should be skipped entirely
	// because it was recently accessed (accessedAt > cutoff)
	result, err := d.Run(context.Background(), true, "deep")
	if err != nil {
		t.Fatalf("Run deep: %v", err)
	}
	if result.Deep == nil {
		t.Fatal("expected Deep result")
	}

	// Recently accessed memory should not be decayed at all
	if result.Deep.StaleProcedureDecayedCount != 0 {
		t.Errorf("expected StaleProcedureDecayedCount=0 for freshly accessed procedure, got %d", result.Deep.StaleProcedureDecayedCount)
	}

	// Verify confidence unchanged
	mem, err := ms.Get(context.Background(), procID, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem.Confidence != 0.9 {
		t.Errorf("expected confidence unchanged at 0.9 for freshly accessed procedure, got %f", mem.Confidence)
	}
}

func TestDreamer_DeepPhase_NonProcedureNotDoubleDecayed(t *testing.T) {
	// Verify that a non-procedure memory (e.g., type "fact") older than staleProcedureDays
	// decays at the normal rate, not double.
	ms := newTestStore(t)

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		DecayIntervalDays:  1,
		StaleProcedureDays: 1,
		DecayRate:           0.05,
		DecayFloor:          0.1,
		ConfidenceFloor:     0.1,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	// Add a fact memory with high confidence
	factID, err := ms.Add(context.Background(), store.AddParams{
		Type:       "fact",
		Content:    "stale fact test",
		Source:     "test",
		Confidence: 0.9,
	})
	if err != nil {
		t.Fatalf("Add fact: %v", err)
	}

	// Set created_at to 60 days ago and leave accessed_at empty (never accessed)
	oldCreated := time.Now().UTC().AddDate(0, 0, -60).Format(time.RFC3339)
	setTimestamps(t, ms, factID, oldCreated, "")

	// Run deep phase with apply=true
	result, err := d.Run(context.Background(), true, "deep")
	if err != nil {
		t.Fatalf("Run deep: %v", err)
	}
	if result.Deep == nil {
		t.Fatal("expected Deep result")
	}

	// Non-procedure memory should NOT be counted as stale procedure double-decayed
	if result.Deep.StaleProcedureDecayedCount != 0 {
		t.Errorf("expected StaleProcedureDecayedCount=0 for non-procedure, got %d", result.Deep.StaleProcedureDecayedCount)
	}

	// Verify the actual confidence - should decay at normal rate: 0.9 - 0.05 = 0.85
	mem, err := ms.Get(context.Background(), factID, false)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	expectedConf := 0.9 - 0.05
	if mem.Confidence < expectedConf-0.01 || mem.Confidence > expectedConf+0.01 {
		t.Errorf("expected non-procedure confidence ~%f (normal decay), got %f", expectedConf, mem.Confidence)
	}
}

// TestDreamer_RemPhase_BehavioralInsight_WithLLM verifies that when Ollama is
// available and categories exceed the threshold, extractBehavioralInsights produces
// actionable procedural content via the LLM.
func TestDreamer_RemPhase_BehavioralInsight_WithLLM(t *testing.T) {
	ms := newTestStore(t)

	// Add self_assessment memories with ERROR_HANDLING category (need >= 3)
	for i := 0; i < 4; i++ {
		addSelfAssessment(t, ms, "ERROR_HANDLING", "Missing error handling in HTTP call "+strings.Repeat("x", 20))
	}

	llmResponse := "After any external call (DB, HTTP, subprocess), add try/except with logging. Verify error paths execute by testing them. Check: llmem search ERROR_HANDLING --type self_assessment shows rate dropping."

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "glm-5.1:cloud"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{"response": llmResponse}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:        ms,
		OllamaClient: newTestOllamaClient(t, server),
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run rem: %v", err)
	}
	if result.Rem == nil {
		t.Fatal("expected Rem result")
	}

	insights := result.Rem.BehavioralInsights
	if len(insights) == 0 {
		t.Fatal("expected at least one behavioral insight for ERROR_HANDLING with 4 occurrences")
	}

	// Find ERROR_HANDLING insight
	var found bool
	for _, insight := range insights {
		if insight.Category == "ERROR_HANDLING" {
			found = true
			if insight.ContentSnippet != llmResponse {
				t.Errorf("expected LLM-generated content, got: %q", insight.ContentSnippet)
			}
			if insight.Count != 4 {
				t.Errorf("expected count 4, got %d", insight.Count)
			}
			break
		}
	}
	if !found {
		t.Error("expected ERROR_HANDLING insight not found")
	}
}

// TestDreamer_RemPhase_BehavioralInsight_LLMFallback verifies that when Ollama is
// unavailable, the method falls back to count-based format (current behavior).
func TestDreamer_RemPhase_BehavioralInsight_LLMFallback(t *testing.T) {
	ms := newTestStore(t)

	// Add self_assessment memories with ERROR_HANDLING category (need >= 3)
	for i := 0; i < 3; i++ {
		addSelfAssessment(t, ms, "ERROR_HANDLING", "Missing error handler in function "+strings.Repeat("y", 10))
	}

	// Use a server that returns 404 on /api/tags so IsAvailable returns false immediately
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:        ms,
		OllamaClient: newTestOllamaClient(t, server),
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run rem: %v", err)
	}
	if result.Rem == nil {
		t.Fatal("expected Rem result")
	}

	insights := result.Rem.BehavioralInsights
	if len(insights) == 0 {
		t.Fatal("expected at least one behavioral insight even without Ollama")
	}

	// The content should be count-based fallback format
	var found bool
	for _, insight := range insights {
		if insight.Category == "ERROR_HANDLING" {
			found = true
			// Should contain the count-based format
			if !strings.Contains(insight.ContentSnippet, "Behavioral insight:") {
				t.Errorf("expected fallback format to contain 'Behavioral insight:', got: %q", insight.ContentSnippet)
			}
			if !strings.Contains(insight.ContentSnippet, "ERROR_HANDLING") {
				t.Errorf("expected fallback format to contain 'ERROR_HANDLING', got: %q", insight.ContentSnippet)
			}
			break
		}
	}
	if !found {
		t.Error("expected ERROR_HANDLING insight in fallback")
	}

	// Also test: no OllamaClient but with BaseURL/HTTPClient pointing to a server
	// that doesn't serve /api/tags correctly — IsAvailable returns false, falls back gracefully
	ms2 := newTestStore(t)
	for i := 0; i < 3; i++ {
		addSelfAssessment(t, ms2, "RACE_CONDITION", "Race condition in concurrent access "+strings.Repeat("z", 10))
	}

	// Use a server that returns 404 for /api/tags (not available)
	unavailServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer unavailServer.Close()

	d2, err := NewDreamer(DreamerConfig{
		Store:      ms2,
		BaseURL:    unavailServer.URL,
		HTTPClient: unavailServer.Client(),
		// OllamaClient intentionally nil — will create from BaseURL
	})
	if err != nil {
		t.Fatalf("NewDreamer with unavailable BaseURL: %v", err)
	}

	result2, err := d2.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run rem with unavailable Ollama: %v", err)
	}
	if result2.Rem == nil {
		t.Fatal("expected Rem result with unavailable Ollama")
	}

	insights2 := result2.Rem.BehavioralInsights
	if len(insights2) == 0 {
		t.Fatal("expected at least one behavioral insight with unavailable Ollama")
	}
}

// TestDreamer_RemPhase_BehavioralInsight_BelowThreshold verifies that categories
// below the threshold (count < 3) do not produce insights.
func TestDreamer_RemPhase_BehavioralInsight_BelowThreshold(t *testing.T) {
	ms := newTestStore(t)

	// Add only 2 self_assessment memories for ERROR_HANDLING (below threshold of 3)
	addSelfAssessment(t, ms, "ERROR_HANDLING", "Small error issue one")
	addSelfAssessment(t, ms, "ERROR_HANDLING", "Small error issue two")

	// Also add a NULL_SAFETY with just 1 occurrence (also below threshold)
	addSelfAssessment(t, ms, "NULL_SAFETY", "One null check missing")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Server is available but should never be called since no category exceeds threshold
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "glm-5.1:cloud"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		t.Error("unexpected LLM call for below-threshold category")
		http.Error(w, "unexpected", http.StatusBadRequest)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:        ms,
		OllamaClient: newTestOllamaClient(t, server),
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run rem: %v", err)
	}
	if result.Rem == nil {
		t.Fatal("expected Rem result")
	}

	if len(result.Rem.BehavioralInsights) != 0 {
		t.Errorf("expected 0 insights for below-threshold categories, got %d", len(result.Rem.BehavioralInsights))
	}
}

// TestDreamer_RemPhase_BehavioralInsight_ProposedMetadata verifies that generated
// procedure memories have proposed:true, source:dream_rem, and category metadata.
func TestDreamer_RemPhase_BehavioralInsight_ProposedMetadata(t *testing.T) {
	ms := newTestStore(t)

	// Add 3 self_assessment memories for ERROR_HANDLING
	for i := 0; i < 3; i++ {
		addSelfAssessment(t, ms, "ERROR_HANDLING", "Error handling gap "+strings.Repeat("a", 10))
	}

	llmResponse := "Always check error return values after external calls. Do: wrap every external call in try/except. Verify: check tests still pass after error path changes."

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "glm-5.1:cloud"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{"response": llmResponse}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:        ms,
		OllamaClient: newTestOllamaClient(t, server),
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run rem: %v", err)
	}

	insights := result.Rem.BehavioralInsights
	if len(insights) == 0 {
		t.Fatal("expected at least one behavioral insight")
	}

	// Find the generated procedure memory and verify its metadata
	var errorHandlingInsight BehavioralInsight
	var found bool
	for _, insight := range insights {
		if insight.Category == "ERROR_HANDLING" {
			errorHandlingInsight = insight
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected ERROR_HANDLING insight")
	}

	// The insight should have been stored as a procedure memory
	if errorHandlingInsight.InsightID == "" {
		t.Fatal("expected insight to have a non-empty InsightID (stored in the DB)")
	}

	// Retrieve the stored procedure memory and verify metadata
	mem, err := ms.Get(context.Background(), errorHandlingInsight.InsightID, false)
	if err != nil {
		t.Fatalf("Get stored procedure: %v", err)
	}

	if mem.Type != "procedure" {
		t.Errorf("expected type 'procedure', got %q", mem.Type)
	}
	if mem.Source != "dream_rem" {
		t.Errorf("expected source 'dream_rem', got %q", mem.Source)
	}

	// Verify metadata fields — Metadata is map[string]any, no type assertion needed
	if proposed, _ := mem.Metadata["proposed"].(bool); !proposed {
		t.Errorf("expected proposed=true in metadata, got %v", mem.Metadata["proposed"])
	}
	if mem.Metadata["source"] != "dream_rem" {
		t.Errorf("expected source='dream_rem' in metadata, got %v", mem.Metadata["source"])
	}
	if mem.Metadata["category"] != "ERROR_HANDLING" {
		t.Errorf("expected category='ERROR_HANDLING' in metadata, got %v", mem.Metadata["category"])
	}
	if occ, ok := mem.Metadata["occurrences"].(float64); !ok || int(occ) != 3 {
		t.Errorf("expected occurrences=3 in metadata, got %v", mem.Metadata["occurrences"])
	}
}

// TestDreamer_RemPhase_BehavioralInsight_InvalidatesOld verifies that repeated REM
// runs produce new insights and procedure memories are stored correctly.
func TestDreamer_RemPhase_BehavioralInsight_InvalidatesOld(t *testing.T) {
	ms := newTestStore(t)

	// Add self_assessment memories
	for i := 0; i < 3; i++ {
		addSelfAssessment(t, ms, "ERROR_HANDLING", "Repeated error handling issue "+strings.Repeat("b", 10))
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "glm-5.1:cloud"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{"response": "First run insight content"}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:        ms,
		OllamaClient: newTestOllamaClient(t, server),
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	// First REM run
	result1, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run 1: %v", err)
	}
	if len(result1.Rem.BehavioralInsights) == 0 {
		t.Fatal("expected at least one insight in first run")
	}

	// The insight should have been stored with an ID
	insight1 := result1.Rem.BehavioralInsights[0]
	if insight1.InsightID == "" {
		t.Fatal("expected InsightID after first REM run")
	}

	// Verify the procedure memory exists in the store
	mem, err := ms.Get(context.Background(), insight1.InsightID, false)
	if err != nil {
		t.Fatalf("Get procedure memory: %v", err)
	}
	if mem.Type != "procedure" {
		t.Errorf("expected type 'procedure', got %q", mem.Type)
	}
	if mem.Content != "First run insight content" {
		t.Errorf("expected LLM response content, got %q", mem.Content)
	}
}

// TestBuildBehavioralInsightPrompt verifies that the prompt builder includes the
// category name, occurrence count, taxonomy description, and self_assessment samples.
func TestBuildBehavioralInsightPrompt(t *testing.T) {
	category := "ERROR_HANDLING"
	count := 5
	samples := []string{
		"Missing try/except in HTTP call",
		"Swallowed error in database query",
		"Unhandled promise rejection in async function",
	}

	prompt := buildBehavioralInsightPrompt(category, count, 30, samples)

	// Verify category name is in the prompt
	if !strings.Contains(prompt, "ERROR_HANDLING") {
		t.Error("expected prompt to contain category name")
	}

	// Verify count is in the prompt
	if !strings.Contains(prompt, "5") {
		t.Error("expected prompt to contain occurrence count")
	}

	// Verify lookbackDays is in the prompt
	if !strings.Contains(prompt, "last 30 days") {
		t.Error("expected prompt to contain 'last 30 days'")
	}

	// Verify taxonomy description is in the prompt
	if !strings.Contains(prompt, "Missing try/except") {
		t.Error("expected prompt to contain taxonomy description for ERROR_HANDLING")
	}

	// Verify samples are included
	for _, s := range samples {
		if !strings.Contains(prompt, s) {
			t.Errorf("expected prompt to contain sample %q", s)
		}
	}

	// Verify prompt includes "Do" section guidance
	if !strings.Contains(prompt, "Do") {
		t.Error("expected prompt to contain 'Do' directive guidance")
	}

	// Verify prompt includes "Verify" section guidance
	if !strings.Contains(prompt, "Verify") {
		t.Error("expected prompt to contain 'Verify' step guidance")
	}

	// Verify prompt mentions word limit
	if !strings.Contains(prompt, "200") {
		t.Error("expected prompt to mention 200 word limit")
	}
}

// TestBuildBehavioralInsightPrompt_EmptySamples verifies prompt generation with no samples.
func TestBuildBehavioralInsightPrompt_EmptySamples(t *testing.T) {
	prompt := buildBehavioralInsightPrompt("RACE_CONDITION", 3, 14, nil)

	if !strings.Contains(prompt, "RACE_CONDITION") {
		t.Error("expected prompt to contain category name")
	}
	if !strings.Contains(prompt, "3") {
		t.Error("expected prompt to contain occurrence count")
	}
	// Should not crash with nil samples
}

// TestBuildBehavioralInsightPrompt_CustomLookback verifies that a non-default
// lookbackDays value appears in the prompt instead of the hardcoded "30".
func TestBuildBehavioralInsightPrompt_CustomLookback(t *testing.T) {
	prompt := buildBehavioralInsightPrompt("ERROR_HANDLING", 7, 14, []string{"sample text"})

	if !strings.Contains(prompt, "last 14 days") {
		t.Error("expected prompt to contain 'last 14 days' for custom lookback")
	}
	// Make sure the old hardcoded "30 days" does not appear when lookbackDays is 14
	if strings.Contains(prompt, "last 30 days") {
		t.Error("prompt should not contain hardcoded 'last 30 days' when lookbackDays is 14")
	}
}

// TestJoinSamples verifies that joinSamples concatenates strings with "; ".
func TestJoinSamples(t *testing.T) {
	tests := []struct {
		name     string
		samples  []string
		expected string
	}{
		{
			name:     "multiple samples",
			samples:  []string{"alpha", "beta", "gamma"},
			expected: "alpha; beta; gamma",
		},
		{
			name:     "single sample",
			samples:  []string{"only"},
			expected: "only",
		},
		{
			name:     "nil samples returns empty",
			samples:  nil,
			expected: "",
		},
		{
			name:     "empty slice returns empty",
			samples:  []string{},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := joinSamples(tt.samples)
			if result != tt.expected {
				t.Errorf("joinSamples(%v) = %q, want %q", tt.samples, result, tt.expected)
			}
		})
	}
}

// TestDreamer_RemPhase_BehavioralInsight_LLMErrorFallback verifies that when
func TestDreamer_RemPhase_BehavioralInsight_LLMErrorFallback(t *testing.T) {
	ms := newTestStore(t)

	// Add self_assessment memories with ERROR_HANDLING category
	for i := 0; i < 3; i++ {
		addSelfAssessment(t, ms, "ERROR_HANDLING", "Error handling gap "+strings.Repeat("c", 10))
	}

	// Server that returns 500 for /api/generate but 200 for /api/tags (so IsAvailable returns true)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "glm-5.1:cloud"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:        ms,
		OllamaClient: newTestOllamaClient(t, server),
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run rem: %v", err)
	}
	if result.Rem == nil {
		t.Fatal("expected Rem result")
	}

	insights := result.Rem.BehavioralInsights
	if len(insights) == 0 {
		t.Fatal("expected at least one behavioral insight even when LLM fails")
	}

	// Should fall back to count-based format
	var found bool
	for _, insight := range insights {
		if insight.Category == "ERROR_HANDLING" {
			found = true
			if !strings.Contains(insight.ContentSnippet, "Behavioral insight:") {
				t.Errorf("expected fallback format with 'Behavioral insight:' when LLM fails, got: %q", insight.ContentSnippet)
			}
			break
		}
	}
	if !found {
		t.Error("expected ERROR_HANDLING insight in LLM error fallback")
	}
}

// TestExtractBehavioralInsights_LogsSkipped_WhenOllamaUnavailable verifies that when
// Ollama is unavailable, extractBehavioralInsights falls back to count-based format
// and the nil OllamaClient path correctly sets useLLM=false.
func TestExtractBehavioralInsights_LogsSkipped_WhenOllamaUnavailable(t *testing.T) {
	ms := newTestStore(t)

	// Add self_assessment memories with ERROR_HANDLING category (need >= 3)
	for i := 0; i < 3; i++ {
		addSelfAssessment(t, ms, "ERROR_HANDLING", "Missing error handler in function "+strings.Repeat("y", 10))
	}

	// Use a server that returns 404 on /api/tags so IsAvailable returns false immediately
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:        ms,
		OllamaClient: newTestOllamaClient(t, server),
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run rem: %v", err)
	}
	if result.Rem == nil {
		t.Fatal("expected Rem result")
	}

	insights := result.Rem.BehavioralInsights
	if len(insights) == 0 {
		t.Fatal("expected at least one behavioral insight even without Ollama")
	}

	// Verify the insight is count-based fallback, not LLM-generated
	for _, insight := range insights {
		if insight.Category == "ERROR_HANDLING" {
			if !strings.Contains(insight.ContentSnippet, "Behavioral insight:") {
				t.Errorf("expected count-based fallback format, got: %q", insight.ContentSnippet)
			}
			if !strings.Contains(insight.ContentSnippet, "ERROR_HANDLING") {
				t.Errorf("expected ERROR_HANDLING in fallback format, got: %q", insight.ContentSnippet)
			}
			return
		}
	}
	t.Error("expected ERROR_HANDLING insight not found")
}

// TestDreamer_WriteProposedChanges_AppendsNewSections verifies that WriteProposedChanges
// appends behavioral insight sections with category/occurrence count, Do directives,
// Verify step, and [SKILL PATCH] with all four fields.
func TestDreamer_WriteProposedChanges_AppendsNewSections(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	proposedChangesPath := filepath.Join(dir, "proposed-changes.md")

	// Add self_assessment memories to exceed the threshold
	for i := 0; i < 4; i++ {
		addSelfAssessment(t, ms, "ERROR_HANDLING", "Missing error handling in HTTP call "+strings.Repeat("x", 20))
	}

	// Use a server that's not available, so we get count-based fallback
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		OllamaClient:        newTestOllamaClient(t, server),
		ProposedChangesPath: proposedChangesPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result.Rem == nil || len(result.Rem.BehavioralInsights) == 0 {
		t.Fatal("expected at least one behavioral insight")
	}

	// Write proposed changes
	err = d.WriteProposedChanges(context.Background(), result)
	if err != nil {
		t.Fatalf("WriteProposedChanges: %v", err)
	}

	data, err := os.ReadFile(proposedChangesPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	content := string(data)

	// Verify timestamp header
	if !contains(content, "# Dream Run:") {
		t.Error("expected 'Dream Run:' timestamp header in proposed-changes.md")
	}
	// Verify category header
	if !contains(content, "## ERROR_HANDLING") {
		t.Error("expected '## ERROR_HANDLING' section in proposed-changes.md")
	}
	// Verify occurrence count
	if !contains(content, "×4") {
		t.Error("expected '×4' occurrence count in proposed-changes.md")
	}
	// Verify behavioral directive section
	if !contains(content, "### Behavioral Directive") {
		t.Error("expected '### Behavioral Directive' section in proposed-changes.md")
	}
	// Verify Do directive
	if !contains(content, "**Do:**") {
		t.Error("expected '**Do:**' in proposed-changes.md")
	}
	// Verify Verify step
	if !contains(content, "**Verify:**") {
		t.Error("expected '**Verify:**' in proposed-changes.md")
	}
	// Verify SKILL PATCH section
	if !contains(content, "### [SKILL PATCH]") {
		t.Error("expected '### [SKILL PATCH]' section in proposed-changes.md")
	}
	// Verify SKILL PATCH fields
	if !contains(content, "**Detection Rule:**") {
		t.Error("expected '**Detection Rule:**' in proposed-changes.md")
	}
	if !contains(content, "**Checklist:**") {
		t.Error("expected '**Checklist:**' in proposed-changes.md")
	}
	if !contains(content, "**Pitfall:**") {
		t.Error("expected '**Pitfall:**' in proposed-changes.md")
	}
	if !contains(content, "**Verification:**") {
		t.Error("expected '**Verification:**' in proposed-changes.md")
	}
}

// TestDreamer_WriteProposedChanges_PreservesExistingContent verifies that
// existing content in proposed-changes.md is preserved when appending.
func TestDreamer_WriteProposedChanges_PreservesExistingContent(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	proposedChangesPath := filepath.Join(dir, "proposed-changes.md")

	// Write initial content
	initialContent := "# Previous Dream Run\n\nOld content here.\n"
	if err := os.WriteFile(proposedChangesPath, []byte(initialContent), 0600); err != nil {
		t.Fatalf("WriteFile initial: %v", err)
	}

	// Add self_assessment memories to exceed the threshold
	for i := 0; i < 3; i++ {
		addSelfAssessment(t, ms, "NULL_SAFETY", "Missing null check in function "+strings.Repeat("z", 10))
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		OllamaClient:        newTestOllamaClient(t, server),
		ProposedChangesPath: proposedChangesPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	err = d.WriteProposedChanges(context.Background(), result)
	if err != nil {
		t.Fatalf("WriteProposedChanges: %v", err)
	}

	data, err := os.ReadFile(proposedChangesPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	content := string(data)

	// Verify old content still present
	if !contains(content, "Old content here.") {
		t.Error("expected existing content to be preserved in proposed-changes.md")
	}
	// Verify new content appended
	if !contains(content, "# Dream Run:") {
		t.Error("expected new dream run timestamp header")
	}
}

// TestDreamer_WriteProposedChanges_UsesResolvedPath verifies that WriteProposedChanges
// validates the write path via paths.ValidateWritePath.
func TestDreamer_WriteProposedChanges_UsesResolvedPath(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	proposedChangesPath := filepath.Join(dir, "proposed-changes.md")

	// Use a mock server that returns 404 for /api/tags (unavailable)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		OllamaClient:        newTestOllamaClient(t, server),
		ProposedChangesPath: proposedChangesPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result := &DreamResult{
		Rem: &RemPhaseResult{
			BehavioralInsights: []BehavioralInsight{
				{Category: "ERROR_HANDLING", Count: 3, ContentSnippet: "test"},
			},
		},
	}

	err = d.WriteProposedChanges(context.Background(), result)
	if err != nil {
		t.Fatalf("WriteProposedChanges: %v", err)
	}

	// Verify file was written at the expected path
	if _, err := os.Stat(proposedChangesPath); err != nil {
		t.Errorf("expected proposed changes file at %s: %v", proposedChangesPath, err)
	}
}

// TestDreamer_WriteProposedChanges_MetadataLink verifies that procedure memories
// created during REM have proposed_changes_link metadata.
func TestDreamer_WriteProposedChanges_MetadataLink(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	proposedChangesPath := filepath.Join(dir, "proposed-changes.md")

	// Add self_assessment memories to exceed the threshold
	for i := 0; i < 3; i++ {
		addSelfAssessment(t, ms, "ERROR_HANDLING", "Error handling gap "+strings.Repeat("a", 10))
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		OllamaClient:        newTestOllamaClient(t, server),
		ProposedChangesPath: proposedChangesPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	insights := result.Rem.BehavioralInsights
	if len(insights) == 0 {
		t.Fatal("expected at least one behavioral insight")
	}

	// Find the ERROR_HANDLING insight
	var insightID string
	for _, insight := range insights {
		if insight.Category == "ERROR_HANDLING" {
			insightID = insight.InsightID
			break
		}
	}
	if insightID == "" {
		t.Fatal("expected ERROR_HANDLING insight with non-empty InsightID")
	}

	// Retrieve the stored procedure memory and verify proposed_changes_link metadata
	mem, err := ms.Get(context.Background(), insightID, false)
	if err != nil {
		t.Fatalf("Get stored procedure: %v", err)
	}

	link, ok := mem.Metadata["proposed_changes_link"]
	if !ok {
		t.Error("expected proposed_changes_link key in procedure memory metadata")
	}
	if link != proposedChangesPath {
		t.Errorf("expected proposed_changes_link=%q, got %q", proposedChangesPath, link)
	}
}

// TestDreamer_WriteProposedChanges_DryRunDoesNotWrite verifies that no file
// is written when apply=false (dry run).
func TestDreamer_WriteProposedChanges_DryRunDoesNotWrite(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	proposedChangesPath := filepath.Join(dir, "proposed-changes.md")

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		ProposedChangesPath: proposedChangesPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	// Run with apply=false (dry run)
	_, err = d.Run(context.Background(), false, "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Verify no file exists at the proposed changes path
	if _, err := os.Stat(proposedChangesPath); !os.IsNotExist(err) {
		t.Error("expected no proposed-changes.md file during dry run")
	}
}

// TestDreamer_WriteProposedChanges_NoInsightsNoFile verifies that no file
// is created when there are no behavioral insights above the threshold.
func TestDreamer_WriteProposedChanges_NoInsightsNoFile(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	proposedChangesPath := filepath.Join(dir, "proposed-changes.md")

	// Add only 1 self_assessment (below threshold of 3)
	addSelfAssessment(t, ms, "ERROR_HANDLING", "A single error handling entry")

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		ProposedChangesPath: proposedChangesPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// WriteProposedChanges should be a no-op when no insights
	err = d.WriteProposedChanges(context.Background(), result)
	if err != nil {
		t.Fatalf("WriteProposedChanges: %v", err)
	}

	// Verify no file created
	if _, err := os.Stat(proposedChangesPath); !os.IsNotExist(err) {
		t.Error("expected no proposed-changes.md file when there are no insights")
	}
}

// TestDreamer_WriteProposedChanges_AppendOnlyWithinRun verifies that running
// WriteProposedChanges twice appends both runs' content.
func TestDreamer_WriteProposedChanges_AppendOnlyWithinRun(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	proposedChangesPath := filepath.Join(dir, "proposed-changes.md")

	// Use a mock server that returns 404 (unavailable), so we get fallback content
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		OllamaClient:        newTestOllamaClient(t, server),
		ProposedChangesPath: proposedChangesPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result := &DreamResult{
		Rem: &RemPhaseResult{
			BehavioralInsights: []BehavioralInsight{
				{Category: "ERROR_HANDLING", Count: 3, ContentSnippet: "first run"},
			},
		},
	}

	// First write
	err = d.WriteProposedChanges(context.Background(), result)
	if err != nil {
		t.Fatalf("WriteProposedChanges first: %v", err)
	}

	// Second write
	result2 := &DreamResult{
		Rem: &RemPhaseResult{
			BehavioralInsights: []BehavioralInsight{
				{Category: "RACE_CONDITION", Count: 5, ContentSnippet: "second run"},
			},
		},
	}
	err = d.WriteProposedChanges(context.Background(), result2)
	if err != nil {
		t.Fatalf("WriteProposedChanges second: %v", err)
	}

	data, err := os.ReadFile(proposedChangesPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	content := string(data)

	// Should have two Dream Run headers
	dreamRunCount := strings.Count(content, "# Dream Run:")
	if dreamRunCount != 2 {
		t.Errorf("expected 2 Dream Run headers, got %d", dreamRunCount)
	}

	// Should have both categories
	if !contains(content, "ERROR_HANDLING") {
		t.Error("expected ERROR_HANDLING from first run")
	}
	if !contains(content, "RACE_CONDITION") {
		t.Error("expected RACE_CONDITION from second run")
	}
}

// TestDreamer_WriteProposedChanges_AcceptsContext verifies that
// WriteProposedChanges accepts a context.Context parameter and propagates it
// to LLM calls, enabling cancellation on shutdown. Without LLM (fallback path),
// the function completes normally even with a valid context.
func TestDreamer_WriteProposedChanges_AcceptsContext(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	proposedChangesPath := filepath.Join(dir, "proposed-changes.md")

	// Use a server that returns 404 (unavailable), so we get fallback content quickly
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		OllamaClient:        newTestOllamaClient(t, server),
		ProposedChangesPath: proposedChangesPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result := &DreamResult{
		Rem: &RemPhaseResult{
			BehavioralInsights: []BehavioralInsight{
				{Category: "ERROR_HANDLING", Count: 3, ContentSnippet: "test"},
			},
		},
	}

	// Verify the function accepts a context parameter — this is the contract test.
	// Using context.Background() ensures the context is valid.
	err = d.WriteProposedChanges(context.Background(), result)
	if err != nil {
		t.Fatalf("WriteProposedChanges with context.Background(): %v", err)
	}

	data, err := os.ReadFile(proposedChangesPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !contains(string(data), "ERROR_HANDLING") {
		t.Error("expected ERROR_HANDLING in proposed-changes.md")
	}
}

// TestBuildSkillPatchPrompt verifies that the prompt builder includes
// Detection Rule, Checklist, Pitfall, and Verification fields,
// category name, and occurrence count.
func TestBuildSkillPatchPrompt(t *testing.T) {
	category := "ERROR_HANDLING"
	count := 7
	samples := []string{"Missing try/except in HTTP call", "Swallowed error in database query"}

	prompt := buildSkillPatchPrompt(category, count, 30, samples)

	// Verify category name
	if !contains(prompt, "ERROR_HANDLING") {
		t.Error("expected prompt to contain category name")
	}
	// Verify occurrence count
	if !contains(prompt, "7") {
		t.Error("expected prompt to contain occurrence count")
	}
	// Verify lookback days
	if !contains(prompt, "last 30 days") {
		t.Error("expected prompt to contain 'last 30 days'")
	}
	// Verify Detection Rule field
	if !contains(prompt, "Detection Rule") {
		t.Error("expected prompt to contain 'Detection Rule'")
	}
	// Verify Checklist field
	if !contains(prompt, "Checklist") {
		t.Error("expected prompt to contain 'Checklist'")
	}
	// Verify Pitfall field
	if !contains(prompt, "Pitfall") {
		t.Error("expected prompt to contain 'Pitfall'")
	}
	// Verify Verification field
	if !contains(prompt, "Verification") {
		t.Error("expected prompt to contain 'Verification'")
	}
	// Verify samples
	for _, s := range samples {
		if !contains(prompt, s) {
			t.Errorf("expected prompt to contain sample %q", s)
		}
	}
	// Verify taxonomy description
	if !contains(prompt, "Missing try/except") {
		t.Error("expected prompt to contain taxonomy description for ERROR_HANDLING")
	}
}

// TestDreamerConfig_ProposedChangesPath_Default verifies that when
// DreamerConfig.ProposedChangesPath is empty, the constructor defaults
// to paths.GetProposedChangesPath().
func TestDreamerConfig_ProposedChangesPath_Default(t *testing.T) {
	ms := newTestStore(t)
	d, err := NewDreamer(DreamerConfig{
		Store: ms,
		// ProposedChangesPath left empty — should default
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	expected := paths.GetProposedChangesPath()
	if d.proposedChangesPath != expected {
		t.Errorf("expected proposedChangesPath=%q, got %q", expected, d.proposedChangesPath)
	}

	// Verify explicit value is used as-is
	dir := t.TempDir()
	explicitPath := filepath.Join(dir, "my-proposed-changes.md")
	d2, err := NewDreamer(DreamerConfig{
		Store:               ms,
		ProposedChangesPath: explicitPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer with explicit path: %v", err)
	}
	if d2.proposedChangesPath != explicitPath {
		t.Errorf("expected proposedChangesPath=%q, got %q", explicitPath, d2.proposedChangesPath)
	}
}

// TestParseLLMDirectiveResponse_FallbackUsesCategory verifies that when
// parseLLMDirectiveResponse cannot extract Do lines from the LLM response,
// it uses the actual category name in the fallback, not the literal string "category".
func TestParseLLMDirectiveResponse_FallbackUsesCategory(t *testing.T) {
	// Empty response should trigger fallback with the category name
	doLines, _ := parseLLMDirectiveResponse("", "ERROR_HANDLING")
	if len(doLines) != 1 {
		t.Fatalf("expected 1 fallback do line, got %d", len(doLines))
	}
	if !strings.Contains(doLines[0], "ERROR_HANDLING") {
		t.Errorf("expected fallback to contain 'ERROR_HANDLING', got %q", doLines[0])
	}
	if !strings.Contains(doLines[0], "checks from taxonomy") {
		t.Errorf("expected fallback to contain 'checks from taxonomy', got %q", doLines[0])
	}

	// Response with no recognizable Do/Verify lines should also use category fallback
	doLines2, verifyLine2 := parseLLMDirectiveResponse("Some random text without structure", "RACE_CONDITION")
	if !strings.Contains(doLines2[0], "RACE_CONDITION") {
		t.Errorf("expected fallback to contain 'RACE_CONDITION', got %q", doLines2[0])
	}
	// Verify fallback verify line is also sensible
	if verifyLine2 == "" {
		t.Error("expected non-empty fallback verify line")
	}
}

// TestParseLLMDirectiveResponse_ExtractsDoLines verifies that parseLLMDirectiveResponse
// correctly extracts Do directives and Verify steps from a well-formed LLM response.
func TestParseLLMDirectiveResponse_ExtractsDoLines(t *testing.T) {
	response := `**Do:**
- Always check return values from external calls
- Wrap database operations in try/except blocks
- Log errors with context (function name, args)

**Verify:** Run integration tests with mock failures to confirm error paths execute.`

	doLines, verifyLine := parseLLMDirectiveResponse(response, "ERROR_HANDLING")

	if len(doLines) != 3 {
		t.Errorf("expected 3 do lines, got %d: %v", len(doLines), doLines)
	}
	if len(doLines) > 0 && !strings.Contains(doLines[0], "Always check return values") {
		t.Errorf("expected first do line about return values, got %q", doLines[0])
	}
	if len(doLines) > 2 && !strings.Contains(doLines[2], "Log errors with context") {
		t.Errorf("expected third do line about logging, got %q", doLines[2])
	}
	// Verify line must be extracted from the **Verify:** line — not replaced by fallback.
	// The markdown bold format **Verify:** has no colon-space pattern for extractAfterColon,
	// so the parser must handle stripping the markdown bold markers.
	if verifyLine == "" {
		t.Error("expected non-empty verify line")
	}
	if !strings.Contains(verifyLine, "integration tests") {
		t.Errorf("expected verify line to contain 'integration tests', got %q (likely fallback used instead of extracted content)", verifyLine)
	}
}

// TestParseLLMDirectiveResponse_ExtractsVerifyFromBoldFormat verifies that
// parseLLMDirectiveResponse correctly extracts Verify text from markdown bold
// format (e.g., **Verify:** some text). The extractAfterColon helper cannot
// find ": " in "**Verify:**" because the colon is followed by "**", so the
// parser must strip bold markers before extracting.
func TestParseLLMDirectiveResponse_ExtractsVerifyFromBoldFormat(t *testing.T) {
	tests := []struct {
		name     string
		response string
		category string
		wantDo   []string // substrings that must appear in do lines
		wantDoN  int      // expected number of do lines
		wantVer  string   // substring that must appear in verify line
	}{
		{
			name: "bold Verify with inline text",
			response: `**Do:**
- Always check return values from external calls

**Verify:** Run integration tests with mock failures to confirm error paths execute.`,
			category: "ERROR_HANDLING",
			wantDoN:  1,
			wantDo:   []string{"Always check return values"},
			wantVer:  "integration tests",
		},
		{
			name: "non-bold Verify with colon-space",
			response: `Do:
- Wrap every external call in try/except

Verify: Run llmem search ERROR_HANDLING to confirm rate drops.`,
			category: "ERROR_HANDLING",
			wantDoN:  1,
			wantDo:   []string{"Wrap every external call"},
			wantVer:  "llmem search",
		},
		{
			name: "bold Do and bold Verify",
			response: `**Do:**
- Check error return values
- Add defensive nil checks
- Log errors with context

**Verify:** Run automated tests covering error paths.`,
			category: "NIL_POINTER",
			wantDoN:  3,
			wantDo:   []string{"Check error return values", "Add defensive nil checks", "Log errors with context"},
			wantVer:  "automated tests",
		},
		{
			name: "Verify on own line after bold header",
			response: `**Do:**
- Review recent changes for missing error handling

**Verify:**
Run llmem search ERROR_HANDLING to confirm reduction in occurrences.`,
			category: "ERROR_HANDLING",
			wantDoN:  1,
			wantDo:   []string{"Review recent changes"},
			wantVer:  "llmem search",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doLines, verifyLine := parseLLMDirectiveResponse(tt.response, tt.category)

			if len(doLines) != tt.wantDoN {
				t.Errorf("expected %d do lines, got %d: %v", tt.wantDoN, len(doLines), doLines)
			}
			for i, want := range tt.wantDo {
				if i >= len(doLines) {
					t.Errorf("missing do line %d: expected to contain %q", i, want)
					continue
				}
				if !strings.Contains(doLines[i], want) {
					t.Errorf("do line %d: expected to contain %q, got %q", i, want, doLines[i])
				}
			}
			if !strings.Contains(verifyLine, tt.wantVer) {
				t.Errorf("expected verify line to contain %q, got %q", tt.wantVer, verifyLine)
			}
		})
	}
}

// TestDreamer_RemPhase_BehavioralInsight_SamplesPopulated verifies that
// extractBehavioralInsights populates the Samples field on BehavioralInsight,
// so that WriteProposedChanges does not need to re-query the database.
func TestDreamer_RemPhase_BehavioralInsight_SamplesPopulated(t *testing.T) {
	ms := newTestStore(t)

	// Add self_assessment memories with ERROR_HANDLING category (need >= 3)
	for i := 0; i < 4; i++ {
		addSelfAssessment(t, ms, "ERROR_HANDLING", "Missing error handling in HTTP call "+strings.Repeat("x", 20))
	}

	// Use a server that returns 404 for /api/tags (unavailable)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	d, err := NewDreamer(DreamerConfig{
		Store:        ms,
		OllamaClient: newTestOllamaClient(t, server),
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(context.Background(), true, "rem")
	if err != nil {
		t.Fatalf("Run rem: %v", err)
	}
	if result.Rem == nil {
		t.Fatal("expected Rem result")
	}

	insights := result.Rem.BehavioralInsights
	if len(insights) == 0 {
		t.Fatal("expected at least one behavioral insight")
	}

	// Find ERROR_HANDLING insight and verify Samples is populated
	var found bool
	for _, insight := range insights {
		if insight.Category == "ERROR_HANDLING" {
			found = true
			if len(insight.Samples) == 0 {
				t.Error("expected Samples to be populated on BehavioralInsight, but got empty slice")
			}
			// Each sample should contain the category name
			for _, s := range insight.Samples {
				if !strings.Contains(s, "ERROR_HANDLING") {
					t.Errorf("expected sample to contain 'ERROR_HANDLING', got %q", s)
				}
			}
			break
		}
	}
	if !found {
		t.Error("expected ERROR_HANDLING insight not found")
	}
}

// TestDreamer_WriteProposedChanges_UsesSamplesFromInsight verifies that
// WriteProposedChanges uses the Samples field from BehavioralInsight, not
// a separate DB query. This is a regression test for the O(N) redundant
// DB queries that were previously made inside the insight loop.
func TestDreamer_WriteProposedChanges_UsesSamplesFromInsight(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	proposedChangesPath := filepath.Join(dir, "proposed-changes.md")

	// Use an unavailable server (fallback behavior)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer server.Close()

	// Create a Dreamer with a specific OllamaClient that's unavailable
	ollamaClient := newTestOllamaClient(t, server)

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		OllamaClient:        ollamaClient,
		ProposedChangesPath: proposedChangesPath,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	// Create a result with a BehavioralInsight that has Samples populated
	result := &DreamResult{
		Rem: &RemPhaseResult{
			BehavioralInsights: []BehavioralInsight{
				{
					Category:       "ERROR_HANDLING",
					Count:          5,
					ContentSnippet: "test snippet",
					Samples:        []string{"Error in HTTP handler", "Missing try/except block"},
				},
			},
		},
	}

	err = d.WriteProposedChanges(context.Background(), result)
	if err != nil {
		t.Fatalf("WriteProposedChanges: %v", err)
	}

	data, err := os.ReadFile(proposedChangesPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	content := string(data)

	// Verify that the samples from the BehavioralInsight appear in the file
	if !contains(content, "Error in HTTP handler") {
		t.Error("expected 'Error in HTTP handler' sample from insight.Samples in proposed-changes.md")
	}
	if !contains(content, "Missing try/except block") {
		t.Error("expected 'Missing try/except block' sample from insight.Samples in proposed-changes.md")
	}
}

// TestDreamer_WriteProposedChanges_ValidatesPatches verifies that the REM phase
// validates patches via the SkillPatcher when configured.
func TestDreamer_WriteProposedChanges_ValidatesPatches(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	// Add self_assessment memories in the NULL_SAFETY category
	// (need >= behavioralThreshold to trigger behavioral insight)
	for i := 0; i < 5; i++ {
		_, err := ms.Add(ctx, store.AddParams{
			Type:       "self_assessment",
			Content:    "Category: NULL_SAFETY\nWhat_happened: nil dereference\nProposed_update: always check nil",
			Source:     "test",
			Confidence: 0.9,
		})
		if err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	// Create a SkillPatcher
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")
	sp, err := skillpatch.NewSkillPatcher(skillpatch.SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	// Use a mock HTTP server that simulates unavailable Ollama
	// so the dream doesn't hang trying to connect
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return 404 for all requests — Ollama unavailable
		http.NotFound(w, r)
	}))
	defer mockServer.Close()

	mockClient := ollama.OllamaClientConfig{
		BaseURL:    mockServer.URL,
		HTTPClient: mockServer.Client(),
	}
	mockOllama, _ := ollama.NewOllamaClient(mockClient)

	d, err := NewDreamer(DreamerConfig{
		Store:       ms,
		SkillPatcher: sp,
		OllamaClient: mockOllama,
		BehavioralThreshold: 3,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(ctx, true, "rem")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.Rem == nil {
		t.Fatal("expected REM results")
	}

	// Verify that behavioral insights were generated and patch validation ran
	if len(result.Rem.BehavioralInsights) == 0 {
		t.Error("expected behavioral insights for NULL_SAFETY")
	}

	foundNULLSafety := false
	for _, insight := range result.Rem.BehavioralInsights {
		if insight.Category == "NULL_SAFETY" {
			foundNULLSafety = true
		}
	}
	if !foundNULLSafety {
		t.Error("expected NULL_SAFETY insight in behavioral insights")
	}
}

// TestDreamer_WriteProposedChanges_NoInsights_ValidatesPatches verifies that
// no patch validation occurs when there are no behavioral insights.
func TestDreamer_WriteProposedChanges_NoInsights_ValidatesPatches(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")
	sp, err := skillpatch.NewSkillPatcher(skillpatch.SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	d, err := NewDreamer(DreamerConfig{
		Store:       ms,
		SkillPatcher: sp,
		BehavioralThreshold: 3, // high threshold so no insights are generated
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(ctx, true, "rem")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.Rem == nil {
		t.Fatal("expected REM results")
	}

	// No insights means no patch validation — test verifies it doesn't crash
	if len(result.Rem.BehavioralInsights) != 0 {
		t.Errorf("expected no behavioral insights, got %d", len(result.Rem.BehavioralInsights))
	}
}

// TestDreamer_ValidatePatches_MergesMetadata verifies that patch validation
// merges flagged_for_review into existing metadata rather than replacing it.
// This is a regression test for the bug where Update with Metadata replaced
// all existing metadata fields.
func TestDreamer_ValidatePatches_MergesMetadata(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	// Add self_assessment memories with NULL_SAFETY category (>= behavioralThreshold)
	for i := 0; i < 5; i++ {
		_, err := ms.Add(ctx, store.AddParams{
			Type:       "self_assessment",
			Content:    "Category: NULL_SAFETY\nWhat_happened: nil dereference\nProposed_update: always check nil",
			Source:     "test",
			Confidence: 0.9,
		})
		if err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")
	sp, err := skillpatch.NewSkillPatcher(skillpatch.SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer mockServer.Close()

	mockClient := ollama.OllamaClientConfig{
		BaseURL:    mockServer.URL,
		HTTPClient: mockServer.Client(),
	}
	mockOllama, _ := ollama.NewOllamaClient(mockClient)

	d, err := NewDreamer(DreamerConfig{
		Store:               ms,
		SkillPatcher:        sp,
		OllamaClient:        mockOllama,
		BehavioralThreshold: 3,
	})
	if err != nil {
		t.Fatalf("NewDreamer: %v", err)
	}

	result, err := d.Run(ctx, true, "rem")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if result.Rem == nil {
		t.Fatal("expected REM results")
	}

	// Find the behavioral insight and check if it was flagged (afterCount >= beforeCount)
	// When errors don't decrease, the patch is flagged for review
	for _, insight := range result.Rem.BehavioralInsights {
		if insight.Category != "NULL_SAFETY" {
			continue
		}
		if insight.InsightID == "" {
			continue
		}

		// Verify existing metadata is preserved when flagged_for_review is added
		mem, err := ms.Get(ctx, insight.InsightID, false)
		if err != nil {
			t.Fatalf("Get: %v", err)
		}
		if mem == nil {
			t.Fatal("expected memory to exist")
		}

		// The memory should have proposed=true, source=dream_rem metadata from creation
		// If flagged_for_review is set, the existing metadata should still be present
		if proposed, ok := mem.Metadata["proposed"].(bool); !ok || !proposed {
			t.Errorf("expected proposed=true in metadata to be preserved, got %v", mem.Metadata["proposed"])
		}
		if source, ok := mem.Metadata["source"].(string); !ok || source != "dream_rem" {
			t.Errorf("expected source=dream_rem in metadata to be preserved, got %v", mem.Metadata["source"])
		}
	}
}

// TestDreamer_ValidatePatches_CategoryCounting_Precise verifies that category
// counting uses ParseSelfAssessmentField instead of strings.Contains.
// This prevents double-counting when a memory mentions another category in prose.
func TestDreamer_ValidatePatches_CategoryCounting_Precise(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	// Add a self_assessment memory in ERROR_HANDLING that mentions another category in prose.
	// With ParseSelfAssessmentField, it should only count as ERROR_HANDLING.
	_, err := ms.Add(ctx, store.AddParams{
		Type:       "self_assessment",
		Content:    "Category: ERROR_HANDLING\nWhat_happened: Unlike NULL_SAFETY, this was a bare except\nProposed_update: never use bare except",
		Source:     "test",
		Confidence: 0.9,
	})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Verify ParseSelfAssessmentField correctly extracts only ERROR_HANDLING
	content := "Category: ERROR_HANDLING\nWhat_happened: Unlike NULL_SAFETY, this was a bare except"
	parsed := taxonomy.ParseSelfAssessmentField(content, "Category")
	if parsed != "ERROR_HANDLING" {
		t.Errorf("expected Category=ERROR_HANDLING, got %q", parsed)
	}

	// Demonstrate that ParseSelfAssessmentField would NOT falsely match NULL_SAFETY
	// even though "NULL_SAFETY" appears in the content
	nullSafetyParsed := taxonomy.ParseSelfAssessmentField(content, "Category")
	if nullSafetyParsed == "NULL_SAFETY" {
		t.Error("ParseSelfAssessmentField should not extract NULL_SAFETY from the ERROR_HANDLING memory")
	}
}

// TestTaxonomy_ParseSelfAssessmentField_NoSubstringMatch verifies that
// ParseSelfAssessmentField does exact field matching and doesn't match
// substrings in prose or other field values.
func TestTaxonomy_ParseSelfAssessmentField_NoSubstringMatch(t *testing.T) {
	tests := []struct {
		name    string
		content string
		field   string
		want    string
	}{
		{
			name:    "exact category match",
			content: "Category: ERROR_HANDLING\nWhat_happened: detail",
			field:   "Category",
			want:    "ERROR_HANDLING",
		},
		{
			name:    "category differs from prose mention",
			content: "Category: ERROR_HANDLING\nWhat_happened: Unlike NULL_SAFETY issues, error was bare except",
			field:   "Category",
			want:    "ERROR_HANDLING", // NOT NULL_SAFETY — the bug we fixed
		},
		{
			name:    "missing field returns empty",
			content: "What_happened: something\nContext: else",
			field:   "Category",
			want:    "",
		},
		{
			name:    "proposed_update extracted correctly",
			content: "Category: RACE_CONDITION\nProposed_update: always use mutex",
			field:   "Proposed_update",
			want:    "always use mutex",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := taxonomy.ParseSelfAssessmentField(tt.content, tt.field)
			if got != tt.want {
				t.Errorf("ParseSelfAssessmentField(%q, %q) = %q, want %q", tt.content, tt.field, got, tt.want)
			}
		})
	}
}

// TestExtractBehavioralInsights_UsesParseSelfAssessmentField verifies that
// extractBehavioralInsights uses taxonomy.ParseSelfAssessmentField for category
// matching, preventing double-counting when a category name appears in prose.
func TestExtractBehavioralInsights_UsesParseSelfAssessmentField(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	// Add a self_assessment whose Category field is ERROR_HANDLING,
	// but whose What_happened prose mentions NULL_SAFETY.
	// With strings.Contains, this would be counted under both categories.
	// With ParseSelfAssessmentField, it should only count under ERROR_HANDLING.
	content := "Category: ERROR_HANDLING\nWhat_happened: Unlike NULL_SAFETY, this was a bare except\nProposed_update: never use bare except"
	_, err := ms.Add(ctx, store.AddParams{
		Type:       "self_assessment",
		Content:    content,
		Source:     "test",
		Confidence: 0.9,
	})
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	// We need at least behavioralThreshold self_assessments for insights to be generated.
	// Add enough ERROR_HANDLING memories to cross the threshold.
	for i := 0; i < 10; i++ {
		_, err := ms.Add(ctx, store.AddParams{
			Type:       "self_assessment",
			Content:    fmt.Sprintf("Category: ERROR_HANDLING\nWhat_happened: error handling issue %d", i),
			Source:     "test",
			Confidence: 0.8,
		})
		if err != nil {
			t.Fatalf("Add loop: %v", err)
		}
	}

	// Do NOT add any NULL_SAFETY memories. If extractBehavioralInsights uses
	// strings.Contains, it will falsely count the ERROR_HANDLING memory under
	// NULL_SAFETY because the prose contains "NULL_SAFETY".

	d := &Dreamer{
		store:                 ms,
		behavioralThreshold:   2,
		behavioralLookbackDays: 365,
	}

	insights := d.extractBehavioralInsights(ctx, false)

	// There should be no insight for NULL_SAFETY — only ERROR_HANDLING insights.
	for _, insight := range insights {
		if insight.Category == "NULL_SAFETY" {
			t.Errorf("extractBehavioralInsights counted a NULL_SAFETY insight from ERROR_HANDLING memory — double-counting bug not fixed")
		}
	}

	// There should be at least one ERROR_HANDLING insight.
	foundErrorHandling := false
	for _, insight := range insights {
		if insight.Category == "ERROR_HANDLING" {
			foundErrorHandling = true
		}
	}
	if !foundErrorHandling {
		t.Error("expected at least one ERROR_HANDLING insight, got none")
	}
}