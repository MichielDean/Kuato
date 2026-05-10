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
	// Verify all taxonomy entries are in keys
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
	// Modifying one should not affect the other
	keys1[0] = "MODIFIED"
	if keys2[0] == "MODIFIED" {
		t.Error("ErrorTaxonomyKeys should return defensive copies")
	}
}

func TestIntrospectCategoryChoices(t *testing.T) {
	choices := IntrospectCategoryChoices()
	if !strings.Contains(choices, "NULL_SAFETY") {
		t.Error("should contain NULL_SAFETY")
	}
	if !strings.Contains(choices, "REVIEW_PASSED") {
		t.Error("should contain REVIEW_PASSED")
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

func TestSelfAssessmentFields_OrderAndCount(t *testing.T) {
	fields := SelfAssessmentFields()
	if len(fields) != 9 {
		t.Errorf("expected 9 fields, got %d", len(fields))
	}
	if fields[0].Name != "Category" {
		t.Errorf("first field should be Category, got %q", fields[0].Name)
	}
	if fields[len(fields)-1].Name != "Iteration_count" {
		t.Errorf("last field should be Iteration_count, got %q", fields[len(fields)-1].Name)
	}
}

func TestSelfAssessmentFields_DefensiveCopy(t *testing.T) {
	f1 := SelfAssessmentFields()
	f2 := SelfAssessmentFields()
	f1[0].Name = "MODIFIED"
	if f2[0].Name == "MODIFIED" {
		t.Error("SelfAssessmentFields should return defensive copies")
	}
}

func TestIntrospectFieldLines(t *testing.T) {
	lines := IntrospectFieldLines()
	if !strings.Contains(lines, "Category:") {
		t.Error("should contain Category field")
	}
	if !strings.Contains(lines, "Iteration_count:") {
		t.Error("should contain Iteration_count field")
	}
}

func TestParseSelfAssessment_Basic(t *testing.T) {
	content := "Category: ERROR_HANDLING\nWhat_happened: swallowed error\nProposed_update: always check errors"
	result := ParseSelfAssessment(content)
	if result["Category"] != "ERROR_HANDLING" {
		t.Errorf("Category: expected ERROR_HANDLING, got %q", result["Category"])
	}
	if result["What_happened"] != "swallowed error" {
		t.Errorf("What_happened: expected 'swallowed error', got %q", result["What_happened"])
	}
	if result["Proposed_update"] != "always check errors" {
		t.Errorf("Proposed_update: expected 'always check errors', got %q", result["Proposed_update"])
	}
}

func TestParseSelfAssessment_SkipsLinesWithoutColonSpace(t *testing.T) {
	content := "Category: EDGE_CASE\njust a line without colon-space\nWhat_happened: detail"
	result := ParseSelfAssessment(content)
	if len(result) != 2 {
		t.Errorf("expected 2 entries, got %d", len(result))
	}
}

func TestParseSelfAssessment_Empty(t *testing.T) {
	result := ParseSelfAssessment("")
	if len(result) != 0 {
		t.Errorf("expected empty map, got %d entries", len(result))
	}
}