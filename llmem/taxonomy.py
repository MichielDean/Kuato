"""Error taxonomy and structured format for self_assessment memories."""

ERROR_TAXONOMY: dict[str, str] = {
    "NULL_SAFETY": "Missing null/None/undefined checks before property access or method calls",
    "ERROR_HANDLING": "Missing try/except, bare except, swallowed errors, unhandled promise rejections",
    "OFF_BY_ONE": "Boundary errors, wrong loop bounds, fencepost errors",
    "RACE_CONDITION": "Concurrency issues, async/await problems, missing locks",
    "AUTH_BYPASS": "Missing auth checks, SSRF, injection vulnerabilities, security oversights",
    "DATA_INTEGRITY": "Stale derived fields, out-of-sync caches/embeddings/indexes, source-of-truth divergence",
    "MISSING_VERIFICATION": "Skipped test steps, unverified outputs, assumed-it-works",
    "EDGE_CASE": "Unhandled empty input, unexpected types, boundary values",
    "PERFORMANCE": "N+1 queries, unnecessary recomputation, memory leaks",
    "DESIGN": "Architectural issues, wrong abstraction level, coupling problems",
    "REVIEW_PASSED": "Clean review with no findings — positive outcome for tracking purposes",
}

ERROR_TAXONOMY_KEYS: list[str] = list(ERROR_TAXONOMY.keys())

INTROSPECT_CATEGORY_CHOICES = ", ".join(ERROR_TAXONOMY_KEYS)

REVIEW_SEVERITY_TAXONOMY: dict[str, list[str]] = {
    "Blocking": ["AUTH_BYPASS", "RACE_CONDITION", "DATA_INTEGRITY"],
    "Required": ["NULL_SAFETY", "ERROR_HANDLING", "MISSING_VERIFICATION", "EDGE_CASE"],
    "Strong Suggestions": ["PERFORMANCE", "DESIGN"],
    "Noted": ["OFF_BY_ONE"],
    "Passed": ["REVIEW_PASSED"],
}

SELF_ASSESSMENT_FIELDS: list[tuple[str, str]] = [
    ("Category", "Error category from the taxonomy above"),
    ("Context", "What you were doing when this happened"),
    ("What_happened", "Describe the error or issue"),
    ("Outcomes", "What happened as a result (broke tests, deployed bug, etc.)"),
    ("What_caught_it", "How was this caught? (self-review, CI, user report, etc.)"),
    ("Estimates_vs_actual", "How long did you think vs how long it took"),
    ("Recurring", "Is this a pattern? (yes/no, with reference to prior)"),
    ("Proposed_update", "What rule or procedure should change to prevent recurrence"),
    ("Iteration_count", "How many attempts before success (integer). 1 = first try"),
]

INTROSPECT_FIELD_LINES = "\n".join(
    f"  {name}: {desc}" for name, desc in SELF_ASSESSMENT_FIELDS
)


def _parse_self_assessment(content: str) -> dict:
    """Parse a self_assessment memory content into a dict.

    Fields are formatted as 'Key: value' lines.
    """
    result = {}
    for line in content.split("\n"):
        if ": " in line:
            key, _, val = line.partition(": ")
            result[key.strip()] = val.strip()
    return result
