// Package taxonomy provides error taxonomy constants and structured format
// for self_assessment memories in the LLMem project.
package taxonomy

// ErrorTaxonomy maps error category keys to their descriptions.
// This is a constant map — it should never be modified at runtime.
var ErrorTaxonomy = map[string]string{
	"NULL_SAFETY":        "Missing null/None/undefined checks before property access or method calls",
	"ERROR_HANDLING":     "Missing try/except, bare except, swallowed errors, unhandled promise rejections",
	"OFF_BY_ONE":         "Boundary errors, wrong loop bounds, fencepost errors",
	"RACE_CONDITION":     "Concurrency issues, async/await problems, missing locks",
	"AUTH_BYPASS":        "Missing auth checks, SSRF, injection vulnerabilities, security oversights",
	"DATA_INTEGRITY":     "Stale derived fields, out-of-sync caches/embeddings/indexes, source-of-truth divergence",
	"MISSING_VERIFICATION": "Skipped test steps, unverified outputs, assumed-it-works",
	"EDGE_CASE":          "Unhandled empty input, unexpected types, boundary values",
	"PERFORMANCE":        "N+1 queries, unnecessary recomputation, memory leaks",
	"DESIGN":             "Architectural issues, wrong abstraction level, coupling problems",
	"REVIEW_PASSED":      "Clean review with no findings — positive outcome for tracking purposes",
}

// ErrorTaxonomyKeys returns the ordered list of error taxonomy category keys.
// Returns a new slice each time (defensive copy).
func ErrorTaxonomyKeys() []string {
	return []string{
		"NULL_SAFETY",
		"ERROR_HANDLING",
		"OFF_BY_ONE",
		"RACE_CONDITION",
		"AUTH_BYPASS",
		"DATA_INTEGRITY",
		"MISSING_VERIFICATION",
		"EDGE_CASE",
		"PERFORMANCE",
		"DESIGN",
		"REVIEW_PASSED",
	}
}

// IntrospectCategoryChoices returns a comma-separated string of error taxonomy keys.
func IntrospectCategoryChoices() string {
	keys := ErrorTaxonomyKeys()
	result := keys[0]
	for _, k := range keys[1:] {
		result += ", " + k
	}
	return result
}

// ReviewSeverityTaxonomy maps severity levels to their associated error categories.
// This maps human-facing severity labels to the taxonomy categories they encompass.
var ReviewSeverityTaxonomy = map[string][]string{
	"Blocking":          {"AUTH_BYPASS", "RACE_CONDITION", "DATA_INTEGRITY"},
	"Required":          {"NULL_SAFETY", "ERROR_HANDLING", "MISSING_VERIFICATION", "EDGE_CASE"},
	"Strong Suggestions": {"PERFORMANCE", "DESIGN"},
	"Noted":             {"OFF_BY_ONE"},
	"Passed":            {"REVIEW_PASSED"},
}

// Field represents a structured field with a name and description.
type Field struct {
	Name        string
	Description string
}

// SelfAssessmentFields returns the ordered list of field name+description pairs
// for self_assessment memories. Returns a new slice each time (defensive copy).
func SelfAssessmentFields() []Field {
	return []Field{
		{Name: "Category", Description: "Error category from the taxonomy above"},
		{Name: "Context", Description: "What you were doing when this happened"},
		{Name: "What_happened", Description: "Describe the error or issue"},
		{Name: "Outcomes", Description: "What happened as a result (broke tests, deployed bug, etc.)"},
		{Name: "What_caught_it", Description: "How was this caught? (self-review, CI, user report, etc.)"},
		{Name: "Estimates_vs_actual", Description: "How long did you think vs how long it took"},
		{Name: "Recurring", Description: "Is this a pattern? (yes/no, with reference to prior)"},
		{Name: "Proposed_update", Description: "What rule or procedure should change to prevent recurrence"},
		{Name: "Iteration_count", Description: "How many attempts before success (integer). 1 = first try"},
	}
}

// IntrospectFieldLines returns a formatted string of self-assessment field names
// and descriptions, one per line, suitable for inclusion in prompts.
func IntrospectFieldLines() string {
	fields := SelfAssessmentFields()
	result := ""
	for i, f := range fields {
		if i > 0 {
			result += "\n"
		}
		result += "  " + f.Name + ": " + f.Description
	}
	return result
}

// ParseSelfAssessment parses "Key: Value" lines from self-assessment content
// into a map. Lines without ": " are skipped.
func ParseSelfAssessment(content string) map[string]string {
	result := map[string]string{}
	for _, line := range splitLines(content) {
		idx := findColonSpace(line)
		if idx < 0 {
			continue
		}
		key := trimSpace(line[:idx])
		val := trimSpace(line[idx+2:])
		if key != "" {
			result[key] = val
		}
	}
	return result
}

// splitLines splits content into lines.
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

// findColonSpace finds the index of ": " in s, or -1 if not found.
func findColonSpace(s string) int {
	for i := 0; i < len(s)-1; i++ {
		if s[i] == ':' && s[i+1] == ' ' {
			return i
		}
	}
	return -1
}

// trimSpace trims leading and trailing whitespace.
func trimSpace(s string) string {
	// Fast path for common cases
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\t' || s[0] == '\r') {
		s = s[1:]
	}
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\t' || s[len(s)-1] == '\r') {
		s = s[:len(s)-1]
	}
	return s
}