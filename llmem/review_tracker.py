"""Review outcome tracker — persist review findings as self_assessment memories."""

import json
import logging
from datetime import datetime, timezone

from .store import MemoryStore
from .taxonomy import ERROR_TAXONOMY_KEYS

log = logging.getLogger(__name__)


class ReviewOutcomeTracker:
    """Track review findings as self_assessment memories."""

    def __init__(self, store: MemoryStore):
        self._store = store

    def track_finding(
        self,
        category: str,
        what_happened: str,
        context: str = "",
        severity: str = "",
        caught_by: str = "self-review",
        iteration_count: int | None = None,
        outcomes: str = "",
    ) -> str:
        """Track a single review finding as a self_assessment memory.

        Args:
            category: Error category from ERROR_TAXONOMY.
            what_happened: Description of the finding.
            context: Additional context.
            severity: Severity level.
            caught_by: How the finding was caught.
            iteration_count: Optional iteration count.
            outcomes: Result of the finding.

        Returns:
            The memory ID of the stored self_assessment.
        """
        fields = [
            f"Category: {category}",
            f"Context: {context}",
            f"What_happened: {what_happened}",
            f"Outcomes: {outcomes}",
            f"What_caught_it: {caught_by}",
            f"Estimates_vs_actual: ",
            f"Recurring: no",
            f"Proposed_update: ",
        ]
        content = "\n".join(fields)

        metadata: dict = {"category": category}
        if severity:
            metadata["severity"] = severity
        if iteration_count is not None:
            metadata["iteration_count"] = iteration_count

        return self._store.add(
            type="self_assessment",
            content=content,
            source="review_tracker",
            confidence=0.9,
            metadata=metadata,
        )

    def track_review(
        self,
        findings: list[dict],
        context: str = "",
        iteration_count: int | None = None,
    ) -> list[str]:
        """Track multiple review findings.

        Args:
            findings: List of finding dicts with 'category', 'what_happened', etc.
            context: Shared context for all findings.
            iteration_count: Optional iteration count.

        Returns:
            List of memory IDs for all stored findings.
        """
        if not findings:
            # Clean review — record a positive outcome
            content = "Category: REVIEW_PASSED\nContext: Clean review with no findings"
            metadata: dict = {"category": "REVIEW_PASSED"}
            if iteration_count is not None:
                metadata["iteration_count"] = iteration_count
            mid = self._store.add(
                type="self_assessment",
                content=content,
                source="review_tracker",
                confidence=0.9,
                metadata=metadata,
            )
            return [mid]

        mem_ids = []
        for finding in findings:
            mid = self.track_finding(
                category=finding.get("category", "UNKNOWN"),
                what_happened=finding.get("what_happened", ""),
                context=context or finding.get("context", ""),
                severity=finding.get("severity", ""),
                caught_by=finding.get("caught_by", "self-review"),
                iteration_count=iteration_count,
                outcomes=finding.get("outcomes", ""),
            )
            mem_ids.append(mid)
        return mem_ids
