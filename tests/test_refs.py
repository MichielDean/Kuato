"""Tests for llmem.refs module — resolve_code_ref()."""

from pathlib import Path

import pytest

from llmem.refs import resolve_code_ref


class TestRefs_ResolveCodeRef:
    """Test resolve_code_ref() — parses code reference strings and reads file content."""

    def test_resolve_code_ref_valid_format(self, tmp_path):
        """Given target_id='path:start:end', returns dict with content from that range."""
        f = tmp_path / "src" / "lib.rs"
        f.parent.mkdir(parents=True)
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        target_id = f"{f}:2:4"
        result = resolve_code_ref(target_id)
        assert result is not None
        assert result["file_path"] == str(f)
        assert result["start_line"] == 2
        assert result["end_line"] == 4
        assert result["content"] == "line2\nline3\nline4"
        assert result["target_type"] == "code"

    def test_resolve_code_ref_missing_file(self, tmp_path):
        """Given a non-existent file path, returns None."""
        target_id = f"{tmp_path / 'nonexistent.py'}:1:5"
        result = resolve_code_ref(target_id)
        assert result is None

    def test_resolve_code_ref_invalid_format(self):
        """Given target_id='not-a-valid-ref', returns None."""
        result = resolve_code_ref("not-a-valid-ref")
        assert result is None

    def test_resolve_code_ref_partial_format(self, tmp_path):
        """Given target_id='path:42' (missing end_line), returns None."""
        f = tmp_path / "src.rs"
        f.write_text("some content\n")
        target_id = f"{f}:42"
        result = resolve_code_ref(target_id)
        assert result is None

    def test_resolve_code_ref_line_range(self, tmp_path):
        """Given a 10-line file and start_line=2:end_line=5, returns only lines 2-5."""
        f = tmp_path / "ten_lines.txt"
        lines = [f"line{i}" for i in range(1, 11)]
        f.write_text("\n".join(lines) + "\n")
        target_id = f"{f}:2:5"
        result = resolve_code_ref(target_id)
        assert result is not None
        assert result["content"] == "line2\nline3\nline4\nline5"
        assert result["start_line"] == 2
        assert result["end_line"] == 5

    def test_resolve_code_ref_blocked_path(self):
        """A code ref targeting a system directory (/etc/...) returns None."""
        result = resolve_code_ref("/etc/passwd:1:5")
        assert result is None

    def test_resolve_code_ref_start_line_exceeds_file(self, tmp_path):
        """When start_line exceeds file length, returns None."""
        f = tmp_path / "short.txt"
        f.write_text("only one line\n")
        target_id = f"{f}:10:15"
        result = resolve_code_ref(target_id)
        assert result is None

    def test_resolve_code_ref_end_line_clamped(self, tmp_path):
        """When end_line exceeds file length, clamps to available lines."""
        f = tmp_path / "clamp.txt"
        f.write_text("line1\nline2\nline3\n")
        target_id = f"{f}:1:100"
        result = resolve_code_ref(target_id)
        assert result is not None
        assert result["end_line"] == 3
        assert result["content"] == "line1\nline2\nline3"
