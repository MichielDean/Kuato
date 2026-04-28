"""Tests for forbidden references in Python source code.

Ensures no lobsterdog, cistern, or Michiel references appear in
the llmem Python package or test code, EXCEPT for the explicit
backward-compatibility migration function (migrate_from_lobsterdog)
and its test references.
"""

import os
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parent.parent
_FORBIDDEN_WORDS = {
    "lobsterdog": re.compile(r"\blobsterdog\b", re.IGNORECASE),
    "cistern": re.compile(r"\bcistern\b", re.IGNORECASE),
    "Michiel": re.compile(r"\bMichiel\b"),
}
# Lines that are allowed to contain forbidden words (backward compat)
_ALLOWED_PATTERNS = [
    re.compile(r"migrate_from_lobsterdog"),  # Required function name
    re.compile(r"\.lobsterdog"),  # Path references for backward compat
    re.compile(r"~/.lobsterdog"),  # Path references in comments
    re.compile(r"# test.*lobsterdog"),  # Test references
    re.compile(r"_FORBIDDEN_WORDS"),  # Test fixture definitions
    re.compile(r'["\'].*lobsterdog.*["\']'),  # String literals in test fixtures
    re.compile(r'["\'].*cistern.*["\']'),  # String literals in test fixtures
    re.compile(r'["\'].*Michiel.*["\']'),  # String literals in test fixtures
    re.compile(r"_ALLOWED_PATTERNS"),  # Test fixture definitions
    re.compile(r"_line_is_allowed"),  # Test fixture definitions
    re.compile(r"test_cli_forbidden_refs"),  # Self-reference in test file
    re.compile(r"Forbidden.*lobsterdog"),  # Test descriptions
    re.compile(r"No lobsterdog"),  # Test descriptions
    re.compile(r"ensures no lobsterdog"),  # Module docstring
    re.compile(r"backward-compatibility"),  # Comments about compat
]
_EXCLUDED_DIRS = {".git", "__pycache__", "node_modules", ".cistern"}
_EXCLUDED_FILES = {"test_cli_forbidden_refs.py"}  # This test file itself


def _find_python_files(root: Path) -> list[Path]:
    """Walk the repo and find all .py files."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
        for fname in filenames:
            if fname.endswith(".py") and fname not in _EXCLUDED_FILES:
                files.append(Path(dirpath) / fname)
    return files


def _line_is_allowed(line: str) -> bool:
    """Check if a line that contains a forbidden word is allowed anyway."""
    for pattern in _ALLOWED_PATTERNS:
        if pattern.search(line):
            return True
    return False


class TestForbiddenRefs_NoLobsterdogInCode:
    """No lobsterdog/cistern/Michiel references in Python source.

    Exception: the migrate_from_lobsterdog function and ~/.lobsterdog
    path references are required for backward compatibility.
    """

    @pytest.fixture(autouse=True)
    def _scan_files(self):
        self._py_files = _find_python_files(_REPO_ROOT / "llmem") + _find_python_files(
            _REPO_ROOT / "tests"
        )

    def test_no_branding_references(self):
        """No Lobsterdog branding (docstrings, descriptions, etc.)."""
        violations = []
        for fpath in self._py_files:
            content = fpath.read_text(errors="ignore")
            lines = content.split("\n")
            for lineno, line in enumerate(lines, 1):
                if _line_is_allowed(line):
                    continue
                for word, pattern in _FORBIDDEN_WORDS.items():
                    if pattern.search(line):
                        violations.append(
                            f"{fpath.relative_to(_REPO_ROOT)}:{lineno}: found '{word}'"
                        )
        assert violations == [], "Forbidden branding references found:\n" + "\n".join(
            violations
        )
