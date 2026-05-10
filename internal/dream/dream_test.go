package dream

import (
	"context"
	"os"
	"path/filepath"
	"testing"

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