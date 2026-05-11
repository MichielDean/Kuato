// Package skillpatch provides direct skill file patching after introspection.
// The SkillPatcher is specific to LLMem SKILL.md files and will not be reused outside this package.
package skillpatch

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/MichielDean/LLMem/internal/paths"
)

// categorySkillMap maps error taxonomy categories to their skill directory.
// All 10 error categories currently map to the introspection skill.
// If no matching skill file exists, a new skill file is created in the
// lowercase category directory.
var categorySkillMap = map[string]string{
	"NULL_SAFETY":        "introspection",
	"ERROR_HANDLING":     "introspection",
	"OFF_BY_ONE":         "introspection",
	"RACE_CONDITION":     "introspection",
	"AUTH_BYPASS":        "introspection",
	"DATA_INTEGRITY":     "introspection",
	"MISSING_VERIFICATION": "introspection",
	"EDGE_CASE":          "introspection",
	"PERFORMANCE":        "introspection",
	"DESIGN":             "introspection",
}

// PatchValidation holds the result of validating whether a skill patch was effective.
type PatchValidation struct {
	Category   string
	BeforeCount int
	AfterCount  int
	Effective  bool
	Flagged    bool
}

// SkillPatcher patches SKILL.md files with procedural updates from introspection.
// The SkillPatcher is specific to LLMem SKILL.md files and will not be reused outside this package.
type SkillPatcher struct {
	skillDir string
}

// SkillPatchConfig contains configuration for creating a SkillPatcher.
// SkillDir defaults to paths.GetSkillDir() if empty.
type SkillPatchConfig struct {
	// SkillDir is the root directory containing skill files.
	// Defaults to paths.GetSkillDir() if empty.
	SkillDir string
}

// fmtErr wraps an error with the "llmem: skillpatch:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: skillpatch: "+format, args...)
}

// validCategoryRe matches category names that are safe to use as directory names.
// Only alphanumeric characters and underscores are allowed — no path separators,
// dots, or whitespace that could enable traversal or injection.
var validCategoryRe = regexp.MustCompile(`^[A-Za-z0-9_]+$`)

// sanitizeCategory validates that a category name is safe for use as a directory
// name. It rejects categories containing path separators, dots, whitespace, or
// other characters that could enable path traversal.
func sanitizeCategory(category string) error {
	if category == "" {
		return fmtErr("invalid category: empty")
	}
	if !validCategoryRe.MatchString(category) {
		return fmtErr("invalid category %q: must contain only alphanumeric characters and underscores", category)
	}
	return nil
}

// sanitizeYAMLValue replaces newlines and carriage returns in a string to prevent
// YAML frontmatter injection. Newlines in YAML values can break the frontmatter
// structure and inject arbitrary YAML keys.
func sanitizeYAMLValue(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "  ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	return s
}

// NewSkillPatcher creates and initializes a SkillPatcher.
// All config fields default to sensible values if zero.
// The constructor leaves the SkillPatcher in a fully usable state.
func NewSkillPatcher(cfg SkillPatchConfig) (*SkillPatcher, error) {
	skillDir := cfg.SkillDir
	if skillDir == "" {
		skillDir = paths.GetSkillDir()
	}

	return &SkillPatcher{
		skillDir: skillDir,
	}, nil
}

// Patch appends a structured section to the matching skill file for the given category.
// If no matching skill file exists, creates one in the appropriate directory.
// If category is empty, returns fmtErr("category is required").
// If proposedUpdate is empty, returns nil (no-op, not an error).
// Creates parent directories with 0700 permissions.
// Writes with 0600 permissions (following paths.go pattern).
func (sp *SkillPatcher) Patch(ctx context.Context, category, proposedUpdate, categoryDescription string) error {
	if category == "" {
		return fmtErr("category is required")
	}
	if err := sanitizeCategory(category); err != nil {
		return err
	}
	if proposedUpdate == "" {
		// No-op: nothing to patch
		slog.Debug("llmem: skillpatch: empty proposed update, skipping patch", "category", category)
		return nil
	}

	skillFile, err := sp.FindSkillFile(ctx, category)
	if err != nil {
		return fmtErr("find skill file: %w", err)
	}

	if skillFile == "" {
		// No matching skill file found; create a new one
		skillFile, err = sp.createSkillFile(category, categoryDescription)
		if err != nil {
			return fmtErr("create skill file: %w", err)
		}
		slog.Info("llmem: skillpatch: created new skill file", "path", skillFile, "category", category)
	}

	// Check for duplicate patch content
	existingContent, readErr := os.ReadFile(skillFile)
	if readErr != nil && !os.IsNotExist(readErr) {
		return fmtErr("read skill file %s: %w", skillFile, readErr)
	}

	// Build the patch content
	patchContent := buildPatchContent(category, proposedUpdate, time.Now().UTC())

	// Check idempotency: if the proposed_update text already exists in the file
	if string(existingContent) != "" && isDuplicatePatch(string(existingContent), proposedUpdate) {
		slog.Debug("llmem: skillpatch: duplicate patch detected, skipping", "category", category)
		return nil
	}

	// Handle malformed files (no YAML frontmatter)
	if string(existingContent) != "" && !hasYAMLFrontmatter(string(existingContent)) {
		slog.Warn("llmem: skillpatch: skill file has no YAML frontmatter, appending with comment", "path", skillFile)
		patchContent = "\n<!-- patch appended without frontmatter -->\n" + patchContent
	}

	// Append patch to skill file
	var newContent string
	if string(existingContent) == "" {
		newContent = patchContent
	} else {
		existing := string(existingContent)
		if !strings.HasSuffix(existing, "\n") {
			existing += "\n"
		}
		newContent = existing + patchContent
	}

	if err := os.WriteFile(skillFile, []byte(newContent), 0600); err != nil {
		return fmtErr("write skill file %s: %w", skillFile, err)
	}

	slog.Info("llmem: skillpatch: patched skill file", "path", skillFile, "category", category)
	return nil
}

// FindSkillFile locates the SKILL.md file matching the given category.
// Performs a categorySkillMap lookup to find the skill directory name, then checks
// whether SKILL.md exists as a regular file in that directory.
// Returns the file path or empty string if not found (unknown category or missing file).
// Returns error only on I/O failures, not for "not found" (empty string is a valid result).
func (sp *SkillPatcher) FindSkillFile(ctx context.Context, category string) (string, error) {
	skillDirName, ok := categorySkillMap[category]
	if !ok {
		// Unknown category: no matching skill directory
		slog.Debug("llmem: skillpatch: no skill mapping for category", "category", category)
		return "", nil
	}

	// Look for SKILL.md in the mapped skill directory
	candidatePath := filepath.Join(sp.skillDir, skillDirName, "SKILL.md")

	info, err := os.Stat(candidatePath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmtErr("stat skill file %s: %w", candidatePath, err)
	}

	// Verify it's a regular file
	if !info.Mode().IsRegular() {
		return "", nil
	}

	return candidatePath, nil
}

// ValidatePatch checks whether the error rate in the given category decreased
// after a skill patch was applied.
// This is a pure function: it compares two integer counts and returns a PatchValidation.
// Effective is true when AfterCount < BeforeCount.
// Flagged is true when AfterCount >= BeforeCount.
// Never returns an error for zero-count categories — returns PatchValidation{Effective: false, Flagged: false}.
func ValidatePatch(category string, beforeCount, afterCount int) PatchValidation {
	result := PatchValidation{
		Category:    category,
		BeforeCount: beforeCount,
		AfterCount:  afterCount,
	}

	if beforeCount == 0 {
		// Zero before-count means no baseline — cannot determine effectiveness
		result.Effective = false
		result.Flagged = false
		return result
	}

	result.Effective = afterCount < beforeCount
	result.Flagged = afterCount >= beforeCount
	return result
}

// createSkillFile creates a new SKILL.md file in the appropriate category directory
// with proper YAML frontmatter.
// The category must already pass sanitizeCategory validation before reaching this method.
func (sp *SkillPatcher) createSkillFile(category, categoryDescription string) (string, error) {
	skillDirName, ok := categorySkillMap[category]
	if !ok {
		// No mapping: use sanitized lowercase category as directory name.
		// sanitizeCategory has already validated no path traversal characters.
		skillDirName = strings.ToLower(category)
	}

	dirPath := filepath.Join(sp.skillDir, skillDirName)

	// Security: verify the resolved path stays within skillDir (defense in depth
	// against path traversal, even though sanitizeCategory already rejects
	// separator characters).
	absDirPath, err := filepath.Abs(dirPath)
	if err != nil {
		return "", fmtErr("resolve skill directory path %s: %w", dirPath, err)
	}
	absSkillDir, err := filepath.Abs(sp.skillDir)
	if err != nil {
		return "", fmtErr("resolve skill dir root %s: %w", sp.skillDir, err)
	}
	if !strings.HasPrefix(absDirPath, absSkillDir+string(filepath.Separator)) && absDirPath != absSkillDir {
		return "", fmtErr("invalid category %q: resolved path escapes skill directory", category)
	}

	if err := os.MkdirAll(dirPath, 0700); err != nil {
		return "", fmtErr("create skill directory %s: %w", dirPath, err)
	}

	description := categoryDescription
	if description == "" {
		description = category
	}

	// Sanitize values before inserting into YAML frontmatter to prevent injection
	sanitizedCategory := sanitizeYAMLValue(strings.ToLower(category))
	sanitizedDescription := sanitizeYAMLValue(description)
	sanitizedHeading := sanitizeYAMLValue(category)

	// Build frontmatter
	var sb strings.Builder
	sb.WriteString("---\n")
	sb.WriteString("name: ")
	sb.WriteString(sanitizedCategory)
	sb.WriteString("\n")
	sb.WriteString("description: >\n  ")
	sb.WriteString(sanitizedDescription)
	sb.WriteString("\nlicense: MIT\n")
	sb.WriteString("---\n\n")
	sb.WriteString("# ")
	sb.WriteString(sanitizedHeading)
	sb.WriteString("\n\n")

	filePath := filepath.Join(dirPath, "SKILL.md")
	if err := os.WriteFile(filePath, []byte(sb.String()), 0600); err != nil {
		return "", fmtErr("write skill file %s: %w", filePath, err)
	}

	return filePath, nil
}

// buildPatchContent constructs the patch section text.
func buildPatchContent(category, proposedUpdate string, now time.Time) string {
	var sb strings.Builder
	date := now.Format("2006-01-02")
	sb.WriteString(fmt.Sprintf("\n## Patch: %s (%s)\n\n", category, date))
	sb.WriteString(fmt.Sprintf("**Detection Rule:** %s\n\n", extractDetectionRule(proposedUpdate)))
	sb.WriteString("**Checklist:**\n")
	items := extractChecklistItems(proposedUpdate)
	for _, item := range items {
		sb.WriteString(fmt.Sprintf("- [ ] %s\n", item))
	}
	sb.WriteString("\n")
	sb.WriteString(fmt.Sprintf("**Pitfall:** %s\n\n", extractPitfall(proposedUpdate)))
	sb.WriteString(fmt.Sprintf("**Verification:** %s\n", extractVerification(proposedUpdate)))
	return sb.String()
}

// isDuplicatePatch checks if the proposed_update text already exists in the file content.
func isDuplicatePatch(content, proposedUpdate string) bool {
	return strings.Contains(content, proposedUpdate)
}

// hasYAMLFrontmatter checks if the content starts with "---" YAML frontmatter.
func hasYAMLFrontmatter(content string) bool {
	return strings.HasPrefix(content, "---\n")
}

// extractDetectionRule extracts a detection rule from proposed update content.
// Falls back to a generic rule based on the content itself.
func extractDetectionRule(proposedUpdate string) string {
	// Try to find "Detection Rule:" in the content
	if idx := strings.Index(proposedUpdate, "Detection Rule:"); idx >= 0 {
		after := proposedUpdate[idx+len("Detection Rule:"):]
		line := strings.TrimSpace(strings.Split(after, "\n")[0])
		if line != "" {
			return line
		}
	}
	// Fallback: use first sentence or truncate
	first := strings.Split(proposedUpdate, "\n")[0]
	if len(first) > 100 {
		first = first[:100] + "..."
	}
	return first
}

// extractChecklistItems extracts checklist items from proposed update content.
// Falls back to a single item from the proposed update.
func extractChecklistItems(proposedUpdate string) []string {
	items := []string{}
	// Look for "Checklist:" section
	if idx := strings.Index(proposedUpdate, "Checklist:"); idx >= 0 {
		after := proposedUpdate[idx:]
		for _, line := range strings.Split(after, "\n") {
			trimmed := strings.TrimSpace(line)
			if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") {
				item := strings.TrimPrefix(trimmed, "- ")
				item = strings.TrimPrefix(item, "* ")
				item = strings.TrimSpace(item)
				if item != "" {
					items = append(items, item)
				}
			}
			// Stop at next section header
			if strings.HasPrefix(trimmed, "**") && !strings.HasPrefix(trimmed, "**Checklist") {
				break
			}
		}
	}
	if len(items) == 0 {
		items = append(items, "Review "+strings.Split(proposedUpdate, "\n")[0])
	}
	return items
}

// extractPitfall extracts a pitfall from proposed update content.
func extractPitfall(proposedUpdate string) string {
	if idx := strings.Index(proposedUpdate, "Pitfall:"); idx >= 0 {
		after := proposedUpdate[idx+len("Pitfall:"):]
		line := strings.TrimSpace(strings.Split(after, "\n")[0])
		// Strip markdown bold
		line = strings.TrimPrefix(line, "**")
		line = strings.TrimSuffix(line, "**")
		if line != "" {
			return line
		}
	}
	// Try **Pitfall:** format
	if idx := strings.Index(proposedUpdate, "**Pitfall:**"); idx >= 0 {
		after := proposedUpdate[idx+len("**Pitfall:**"):]
		line := strings.TrimSpace(strings.Split(after, "\n")[0])
		if line != "" {
			return line
		}
	}
	return "Incomplete handling may recur"
}

// extractVerification extracts a verification step from proposed update content.
func extractVerification(proposedUpdate string) string {
	if idx := strings.Index(proposedUpdate, "Verification:"); idx >= 0 {
		after := proposedUpdate[idx+len("Verification:"):]
		line := strings.TrimSpace(strings.Split(after, "\n")[0])
		line = strings.TrimPrefix(line, "**")
		line = strings.TrimSuffix(line, "**")
		if line != "" {
			return line
		}
	}
	return "Run llmem search to confirm reduction"
}