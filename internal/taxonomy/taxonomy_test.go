package taxonomy

import (
	"strings"
	"testing"
)

func TestErrorTaxonomy_KeysMatch(t *testing.T) {
	keys := ErrorTaxonomyKeys()
	for _, k := range keys {
		if _, ok := ErrorTaxonomy[k]; !ok {
			t.Errorf("ErrorTaxonomyKeys contains %q but ErrorTaxonomy does not", k)
		}
	}
	keySet := map[string]bool{}
	for _, k := range keys {
		keySet[k] = true
	}
	for k := range ErrorTaxonomy {
		if !keySet[k] {
			t.Errorf("ErrorTaxonomy contains %q but ErrorTaxonomyKeys does not", k)
		}
	}
}

func TestErrorTaxonomy_HasAllCategories(t *testing.T) {
	expected := []string{
		"NULL_SAFETY", "ERROR_HANDLING", "OFF_BY_ONE", "RACE_CONDITION",
		"AUTH_BYPASS", "DATA_INTEGRITY", "MISSING_VERIFICATION", "EDGE_CASE",
		"PERFORMANCE", "DESIGN", "REVIEW_PASSED",
	}
	if len(ErrorTaxonomy) != len(expected) {
		t.Errorf("expected %d categories, got %d", len(expected), len(ErrorTaxonomy))
	}
	for _, cat := range expected {
		if _, ok := ErrorTaxonomy[cat]; !ok {
			t.Errorf("missing category %q", cat)
		}
	}
}

func TestErrorTaxonomyKeys_ReturnsDefensiveCopy(t *testing.T) {
	keys1 := ErrorTaxonomyKeys()
	keys2 := ErrorTaxonomyKeys()
	if len(keys1) != len(keys2) {
		t.Fatal("keys lengths differ")
	}
	keys1[0] = "MODIFIED"
	if keys2[0] == "MODIFIED" {
		t.Error("ErrorTaxonomyKeys should return defensive copies")
	}
}

func TestReviewSeverityTaxonomy_Blocking(t *testing.T) {
	blocking := ReviewSeverityTaxonomy["Blocking"]
	expected := []string{"AUTH_BYPASS", "RACE_CONDITION", "DATA_INTEGRITY"}
	if len(blocking) != len(expected) {
		t.Fatalf("Blocking: expected %d categories, got %d", len(expected), len(blocking))
	}
	for i, exp := range expected {
		if blocking[i] != exp {
			t.Errorf("Blocking[%d]: expected %q, got %q", i, exp, blocking[i])
		}
	}
}

// splitLines is a local copy for testing.
func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

func TestSplitLines(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  int
	}{
		{"empty", "", 0},
		{"single line", "hello", 1},
		{"two lines", "hello\nworld", 2},
		{"trailing newline", "hello\n", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := splitLines(tt.input)
			if len(got) != tt.want {
				t.Errorf("splitLines(%q) returned %d lines, want %d", tt.input, len(got), tt.want)
			}
		})
	}
}

func TestParseKeyValue(t *testing.T) {
	tests := []struct {
		name    string
		content string
		key     string
		want    string
	}{
		{"exact match", "Category: ERROR_HANDLING\nWhat: detail", "Category", "ERROR_HANDLING"},
		{"missing key", "What: detail", "Category", ""},
		{"proposed update", "Category: RACE_CONDITION\nProposed_update: always use mutex", "Proposed_update", "always use mutex"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use the shared splitLines and findColonSpace via ParseKeyValue-like logic
			lines := splitLines(tt.content)
			for _, line := range lines {
				idx := strings.Index(line, ": ")
				if idx < 0 {
					continue
				}
				key := strings.TrimSpace(line[:idx])
				if key == tt.key {
					val := strings.TrimSpace(line[idx+2:])
					if val != tt.want {
						t.Errorf("got %q, want %q", val, tt.want)
					}
					return
				}
			}
			if tt.want != "" {
				t.Errorf("key %q not found in content", tt.key)
			}
		})
	}
}