"""Test outcome tracker — persist test results as memories."""

import json
import logging
from datetime import datetime, timezone

from .store import MemoryStore

log = logging.getLogger(__name__)

# Command categories for test result classification
_COMMAND_CATEGORY_MAP: dict[str, str] = {
    "pytest": "python",
    "python -m pytest": "python",
    "go test": "go",
    "npm test": "javascript",
    "npm run test": "javascript",
    "make test": "make",
}


class TestOutcomeTracker:
    """Track test outcomes as event memories."""

    def __init__(self, store: MemoryStore):
        self._store = store

    def track_result(
        self,
        command: str,
        passed: bool,
        output: str = "",
        duration: float | None = None,
    ) -> str:
        """Track a single test result as an event memory.

        Args:
            command: The test command that was run.
            passed: Whether the tests passed.
            output: Test output (truncated to 500 chars).
            duration: Test duration in seconds.

        Returns:
            The memory ID.
        """
        status = "PASSED" if passed else "FAILED"
        content = f"Test {status}: {command}"
        if duration is not None:
            content += f" ({duration:.1f}s)"

        metadata: dict = {
            "test_command": command,
            "test_passed": passed,
            "category": _COMMAND_CATEGORY_MAP.get(command, "unknown"),
        }
        if duration is not None:
            metadata["duration_seconds"] = duration
        if output:
            metadata["output_preview"] = output[:500]

        return self._store.add(
            type="event",
            content=content,
            source="test_tracker",
            confidence=1.0 if passed else 0.7,
            metadata=metadata,
        )

    def track_run(
        self,
        results: list[dict],
    ) -> list[str]:
        """Track multiple test results.

        Args:
            results: List of dicts with 'command', 'passed', 'output', 'duration'.

        Returns:
            List of memory IDs.
        """
        mem_ids = []
        for r in results:
            mid = self.track_result(
                command=r.get("command", "unknown"),
                passed=r.get("passed", False),
                output=r.get("output", ""),
                duration=r.get("duration"),
            )
            mem_ids.append(mid)
        return mem_ids
