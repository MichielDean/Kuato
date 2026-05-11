package skillpatch

import (
	"context"
	"os"
	"path/filepath"
	"strings"
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
	return ms
}

func TestSkillPatcher_PatchExistingSkill(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")
	introDir := filepath.Join(skillDir, "introspection")
	if err := os.MkdirAll(introDir, 0700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	// Create an existing SKILL.md
	existingContent := "---\nname: introspection\ndescription: >\n  Test skill\nlicense: MIT\n---\n\n# Introspection\n\nSome existing content.\n"
	skillFile := filepath.Join(introDir, "SKILL.md")
	if err := os.WriteFile(skillFile, []byte(existingContent), 0600); err != nil {
		t.Fatalf("write skill: %v", err)
	}

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	err = sp.Patch(ctx, "NULL_SAFETY", "Always check for nil before dereferencing pointers in Go", "Missing null checks")
	if err != nil {
		t.Fatalf("Patch: %v", err)
	}

	// Verify the content was appended
	data, err := os.ReadFile(skillFile)
	if err != nil {
		t.Fatalf("read skill: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "Patch: NULL_SAFETY") {
		t.Error("expected patch section header in skill file")
	}
	if !strings.Contains(content, "Always check for nil") {
		t.Error("expected proposed update content in skill file")
	}
	if !strings.Contains(content, "Some existing content") {
		t.Error("expected existing content to be preserved")
	}
}

func TestSkillPatcher_CreateNewSkill(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	err = sp.Patch(ctx, "ERROR_HANDLING", "Always check error return values in Go", "Missing error handling")
	if err != nil {
		t.Fatalf("Patch: %v", err)
	}

	// Verify the file was created in the introspection directory
	skillFile := filepath.Join(skillDir, "introspection", "SKILL.md")
	data, err := os.ReadFile(skillFile)
	if err != nil {
		t.Fatalf("read skill: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "Patch: ERROR_HANDLING") {
		t.Error("expected patch section in new skill file")
	}
	if !strings.Contains(content, "---") {
		t.Error("expected YAML frontmatter in new skill file")
	}
}

func TestSkillPatcher_PatchWithEmptyCategory(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: filepath.Join(dir, "skills"),
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	err = sp.Patch(context.Background(), "", "some content", "")
	if err == nil {
		t.Error("expected error for empty category")
	}
	if !strings.Contains(err.Error(), "category is required") {
		t.Errorf("expected 'category is required' error, got: %v", err)
	}
}

func TestSkillPatcher_PatchWithEmptyProposedUpdate(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: filepath.Join(dir, "skills"),
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	err = sp.Patch(context.Background(), "NULL_SAFETY", "", "")
	if err != nil {
		t.Errorf("expected nil for empty proposed update, got: %v", err)
	}
}

func TestSkillPatcher_IdempotentPatch(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")
	introDir := filepath.Join(skillDir, "introspection")
	if err := os.MkdirAll(introDir, 0700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	existingContent := "---\nname: introspection\ndescription: >\n  Test\nlicense: MIT\n---\n\n# Introspection\n"
	skillFile := filepath.Join(introDir, "SKILL.md")
	if err := os.WriteFile(skillFile, []byte(existingContent), 0600); err != nil {
		t.Fatalf("write skill: %v", err)
	}

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	proposedUpdate := "Always check nil before dereferencing"
	err = sp.Patch(ctx, "NULL_SAFETY", proposedUpdate, "")
	if err != nil {
		t.Fatalf("First Patch: %v", err)
	}

	data1, _ := os.ReadFile(skillFile)

	// Patch again with the same content — should be idempotent
	err = sp.Patch(ctx, "NULL_SAFETY", proposedUpdate, "")
	if err != nil {
		t.Fatalf("Second Patch: %v", err)
	}

	data2, _ := os.ReadFile(skillFile)

	// Content should not have grown
	if len(data2) != len(data1) {
		t.Errorf("expected idempotent patch (same length), got %d bytes then %d bytes", len(data1), len(data2))
	}
}

func TestSkillPatcher_InvalidSkillDir(t *testing.T) {
	ms := newTestStore(t)

	// Use a path that can't be created (will fail when trying to create a new file)
	invalidDir := "/proc/no-skills-here"

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: invalidDir,
		Store:    ms,
	})
	if err != nil {
		// NewSkillPatcher itself should succeed — the error happens on Patch
		t.Fatalf("NewSkillPatcher should not fail: %v", err)
	}

	// Use a category not in categorySkillMap so a new directory needs to be created
	err = sp.Patch(context.Background(), "UNKNOWN_CATEGORY", "test content", "")
	// This should fail because the directory can't be created under /proc
	if err == nil {
		t.Error("expected error for invalid skill directory")
	}
}

func TestSkillPatcher_ParsesFrontmatterCorrectly(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")
	introDir := filepath.Join(skillDir, "introspection")
	if err := os.MkdirAll(introDir, 0700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	// Create a skill file with frontmatter
	existingContent := "---\nname: introspection\ndescription: >\n  Introspection skill\nlicense: MIT\n---\n\n# Introspection\n\nExisting content.\n"
	skillFile := filepath.Join(introDir, "SKILL.md")
	if err := os.WriteFile(skillFile, []byte(existingContent), 0600); err != nil {
		t.Fatalf("write skill: %v", err)
	}

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	err = sp.Patch(ctx, "ERROR_HANDLING", "Always wrap errors with fmt.Errorf", "")
	if err != nil {
		t.Fatalf("Patch: %v", err)
	}

	data, err := os.ReadFile(skillFile)
	if err != nil {
		t.Fatalf("read skill: %v", err)
	}
	content := string(data)

	// Frontmatter should still be present
	if !strings.Contains(content, "---\nname: introspection") {
		t.Error("expected frontmatter to be preserved after patch")
	}
	if !strings.Contains(content, "license: MIT") {
		t.Error("expected license field to be preserved after patch")
	}
	if !strings.Contains(content, "Existing content.") {
		t.Error("expected existing content to be preserved after patch")
	}
}

func TestSkillPatcher_MalformedFile_NoFrontmatter(t *testing.T) {
	ctx := context.Background()
	ms := newTestStore(t)

	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")
	introDir := filepath.Join(skillDir, "introspection")
	if err := os.MkdirAll(introDir, 0700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	// Create a skill file WITHOUT frontmatter
	existingContent := "# Introspection\n\nSome content without frontmatter.\n"
	skillFile := filepath.Join(introDir, "SKILL.md")
	if err := os.WriteFile(skillFile, []byte(existingContent), 0600); err != nil {
		t.Fatalf("write skill: %v", err)
	}

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	err = sp.Patch(ctx, "NULL_SAFETY", "Check nil before deref", "")
	if err != nil {
		t.Fatalf("Patch on malformed file: %v", err)
	}

	data, err := os.ReadFile(skillFile)
	if err != nil {
		t.Fatalf("read skill: %v", err)
	}
	content := string(data)

	// Should have the fallback comment
	if !strings.Contains(content, "patch appended without frontmatter") {
		t.Error("expected fallback comment for malformed file")
	}
	if !strings.Contains(content, "Patch: NULL_SAFETY") {
		t.Error("expected patch section after fallback comment")
	}
}

func TestNewSkillPatcher_NilStore(t *testing.T) {
	_, err := NewSkillPatcher(SkillPatchConfig{
		Store: nil,
	})
	if err == nil {
		t.Error("expected error for nil store")
	}
}

func TestNewSkillPatcher_DefaultSkillDir(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("LMEM_HOME", dir)

	ms := newTestStore(t)
	sp, err := NewSkillPatcher(SkillPatchConfig{
		Store: ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}
	expected := filepath.Join(dir, "skills")
	if sp.skillDir != expected {
		t.Errorf("expected skill dir %q, got %q", expected, sp.skillDir)
	}
}

func TestSkillPatcher_FindSkillFile_Existing(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")
	introDir := filepath.Join(skillDir, "introspection")
	if err := os.MkdirAll(introDir, 0700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	skillFile := filepath.Join(introDir, "SKILL.md")
	if err := os.WriteFile(skillFile, []byte("---\nname: introspection\n---\n"), 0600); err != nil {
		t.Fatalf("write: %v", err)
	}

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	path, err := sp.FindSkillFile(context.Background(), "NULL_SAFETY")
	if err != nil {
		t.Fatalf("FindSkillFile: %v", err)
	}
	if path != skillFile {
		t.Errorf("expected %q, got %q", skillFile, path)
	}
}

func TestSkillPatcher_FindSkillFile_NotFound(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	path, err := sp.FindSkillFile(context.Background(), "NULL_SAFETY")
	if err != nil {
		t.Fatalf("FindSkillFile: %v", err)
	}
	if path != "" {
		t.Errorf("expected empty string for not found, got %q", path)
	}
}

func TestSkillPatcher_FindSkillFile_UnknownCategory(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: filepath.Join(dir, "skills"),
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	path, err := sp.FindSkillFile(context.Background(), "UNKNOWN_CATEGORY")
	if err != nil {
		t.Fatalf("FindSkillFile: %v", err)
	}
	if path != "" {
		t.Errorf("expected empty string for unknown category, got %q", path)
	}
}

func TestSkillPatcher_ValidatePatch_Effective(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: filepath.Join(dir, "skills"),
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	result, err := sp.ValidatePatch(context.Background(), "NULL_SAFETY", 10, 3)
	if err != nil {
		t.Fatalf("ValidatePatch: %v", err)
	}
	if !result.Effective {
		t.Error("expected Effective=true when after < before")
	}
	if result.Flagged {
		t.Error("expected Flagged=false when after < before")
	}
}

func TestSkillPatcher_ValidatePatch_Flagged(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: filepath.Join(dir, "skills"),
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	result, err := sp.ValidatePatch(context.Background(), "NULL_SAFETY", 5, 8)
	if err != nil {
		t.Fatalf("ValidatePatch: %v", err)
	}
	if result.Effective {
		t.Error("expected Effective=false when after >= before")
	}
	if !result.Flagged {
		t.Error("expected Flagged=true when after >= before")
	}
}

func TestSkillPatcher_ValidatePatch_ZeroBeforeCount(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: filepath.Join(dir, "skills"),
		Store:    ms,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	result, err := sp.ValidatePatch(context.Background(), "NULL_SAFETY", 0, 0)
	if err != nil {
		t.Fatalf("ValidatePatch: %v", err)
	}
	if result.Effective {
		t.Error("expected Effective=false when before=0")
	}
	if result.Flagged {
		t.Error("expected Flagged=false when before=0")
	}
}

func TestBuildPatchContent(t *testing.T) {
	content := buildPatchContent("NULL_SAFETY", "Always guard nil pointers", time.Date(2025, 6, 15, 0, 0, 0, 0, time.UTC))
	if !strings.Contains(content, "## Patch: NULL_SAFETY (2025-06-15)") {
		t.Errorf("expected patch header, got: %s", content)
	}
	if !strings.Contains(content, "**Detection Rule:**") {
		t.Error("expected Detection Rule field")
	}
	if !strings.Contains(content, "**Checklist:**") {
		t.Error("expected Checklist field")
	}
	if !strings.Contains(content, "**Pitfall:**") {
		t.Error("expected Pitfall field")
	}
	if !strings.Contains(content, "**Verification:**") {
		t.Error("expected Verification field")
	}
}

func TestIsDuplicatePatch(t *testing.T) {
	content := "Some existing content\nAlways check for nil before dereferencing\nMore content"
	if !isDuplicatePatch(content, "Always check for nil before dereferencing") {
		t.Error("expected duplicate to be detected")
	}
	if isDuplicatePatch(content, "Unique text not in file") {
		t.Error("expected non-duplicate to not be detected")
	}
}

func TestHasYAMLFrontmatter(t *testing.T) {
	if !hasYAMLFrontmatter("---\nname: test\n---\nContent") {
		t.Error("expected YAML frontmatter to be detected")
	}
	if hasYAMLFrontmatter("# Just a heading\nNo frontmatter") {
		t.Error("expected no frontmatter detection for plain markdown")
	}
}