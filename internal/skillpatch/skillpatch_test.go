package skillpatch

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSkillPatcher_PatchExistingSkill(t *testing.T) {
	ctx := context.Background()

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

	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
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
	dir := t.TempDir()

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: filepath.Join(dir, "skills"),
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
	dir := t.TempDir()

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: filepath.Join(dir, "skills"),
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
	// Use a path that can't be created (will fail when trying to create a new file)
	invalidDir := "/proc/no-skills-here"

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: invalidDir,
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

func TestNewSkillPatcher_DefaultSkillDir(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("LMEM_HOME", dir)

	sp, err := NewSkillPatcher(SkillPatchConfig{})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}
	expected := filepath.Join(dir, "skills")
	if sp.skillDir != expected {
		t.Errorf("expected skill dir %q, got %q", expected, sp.skillDir)
	}
}

func TestSkillPatcher_FindSkillFile_Existing(t *testing.T) {
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
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
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
	dir := t.TempDir()

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: filepath.Join(dir, "skills"),
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

func TestValidatePatch_Effective(t *testing.T) {
	result := ValidatePatch("NULL_SAFETY", 10, 3)
	if !result.Effective {
		t.Error("expected Effective=true when after < before")
	}
	if result.Flagged {
		t.Error("expected Flagged=false when after < before")
	}
}

func TestValidatePatch_Flagged(t *testing.T) {
	result := ValidatePatch("NULL_SAFETY", 5, 8)
	if result.Effective {
		t.Error("expected Effective=false when after >= before")
	}
	if !result.Flagged {
		t.Error("expected Flagged=true when after >= before")
	}
}

func TestValidatePatch_ZeroBeforeCount(t *testing.T) {
	result := ValidatePatch("NULL_SAFETY", 0, 0)
	if result.Effective {
		t.Error("expected Effective=false when before=0")
	}
	if result.Flagged {
		t.Error("expected Flagged=false when before=0")
	}
}

func TestValidatePatch_EqualCounts_Flagged(t *testing.T) {
	result := ValidatePatch("ERROR_HANDLING", 5, 5)
	if result.Effective {
		t.Error("expected Effective=false when after == before")
	}
	if !result.Flagged {
		t.Error("expected Flagged=true when after >= before")
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

func TestSkillPatcher_PathTraversal_Rejected(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	skillDir := filepath.Join(dir, "skills")

	sp, err := NewSkillPatcher(SkillPatchConfig{
		SkillDir: skillDir,
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	// Attempt path traversal via category not in categorySkillMap
	// Categories with path separators or dots should be rejected
	traversalCategories := []string{
		"../../etc",
		"../config",
		"foo/bar",
		"foo\\bar",
		".hidden",
		"attacker..evil",
	}
	for _, tc := range traversalCategories {
		t.Run(tc, func(t *testing.T) {
			err := sp.Patch(ctx, tc, "malicious update", "malicious description")
			if err == nil {
				t.Errorf("expected error for path-traversal category %q, got nil", tc)
			}
			if err != nil && !strings.Contains(err.Error(), "invalid category") {
				t.Errorf("expected 'invalid category' error for %q, got: %v", tc, err)
			}
		})
	}
}

func TestSkillPatcher_YAMLInjection_Prevented(t *testing.T) {
	ctx := context.Background()
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
	})
	if err != nil {
		t.Fatalf("NewSkillPatcher: %v", err)
	}

	// Injected newlines in description should be sanitized
	maliciousDesc := "Safe description\nmalicious: injected"
	err = sp.Patch(ctx, "ERROR_HANDLING", "Check error returns", maliciousDesc)
	if err != nil {
		t.Fatalf("Patch: %v", err)
	}

	data, err := os.ReadFile(skillFile)
	if err != nil {
		t.Fatalf("read skill: %v", err)
	}
	content := string(data)

	// The YAML frontmatter should not contain unescaped newlines in the description
	// Newlines should be replaced with spaces
	if strings.Contains(content, "malicious: injected") {
		t.Error("YAML injection: newline in description was not sanitized")
	}
}

func TestSanitizeCategory(t *testing.T) {
	tests := []struct {
		input    string
		wantErr bool
	}{
		{"NULL_SAFETY", false},
		{"ERROR_HANDLING", false},
		{"simple", false},
		{"../../etc", true},
		{"../config", true},
		{"foo/bar", true},
		{".hidden", true},
		{"a..b", true},
		{"", true},
		{"valid_name123", false},
		{"has space", true},
		{"has\nnewline", true},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			err := sanitizeCategory(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("sanitizeCategory(%q) = %v, want error=%v", tt.input, err, tt.wantErr)
			}
		})
	}
}

func TestSanitizeYAMLValue(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"plain text", "hello world", "hello world"},
		{"newline replaced", "line1\nline2", "line1 line2"},
		{"carriage return replaced", "line1\rline2", "line1 line2"},
		{"crlf replaced", "line1\r\nline2", "line1  line2"},
		{"tab preserved", "tab\there", "tab\there"},
		{"empty string", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sanitizeYAMLValue(tt.input)
			if got != tt.want {
				t.Errorf("sanitizeYAMLValue(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}