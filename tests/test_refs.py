"""Tests for llmem.refs module — resolve_code_ref() and validate_code_ref_path()."""

from pathlib import Path

import pytest

from llmem.refs import resolve_code_ref, validate_code_ref_path


class TestRefs_NoDefaultMaxRefDepth:
    """Regression test: DEFAULT_MAX_REF_DEPTH was dead code in refs.py and
    has been removed. Verify it no longer exists."""

    def test_default_max_ref_depth_not_in_module(self):
        """refs.py must not export DEFAULT_MAX_REF_DEPTH — it was dead code."""
        import llmem.refs as refs_module

        assert not hasattr(refs_module, "DEFAULT_MAX_REF_DEPTH")


class TestRefs_ValidateCodeRefPath:
    """Test validate_code_ref_path() — validates that code ref paths are safe."""

    def test_relative_path_accepted(self):
        """A simple relative path is accepted."""
        assert validate_code_ref_path("src/lib.rs") == "src/lib.rs"

    def test_relative_path_with_subdir(self):
        """A relative path with subdirectories is accepted."""
        assert validate_code_ref_path("src/utils/helpers.py") == "src/utils/helpers.py"

    def test_absolute_path_rejected(self):
        """An absolute path is rejected (prevents reading /etc/passwd etc)."""
        assert validate_code_ref_path("/etc/passwd") is None

    def test_absolute_home_path_rejected(self):
        """An absolute path under /home is rejected (prevents SSH key reads etc)."""
        assert validate_code_ref_path("/home/user/.ssh/id_rsa") is None

    def test_traversal_rejected(self):
        """A path with '..' traversal is rejected."""
        assert validate_code_ref_path("../../etc/passwd") is None

    def test_traversal_in_middle_rejected(self):
        """A path with '..' in the middle is rejected."""
        assert validate_code_ref_path("src/../etc/passwd") is None

    def test_empty_path_rejected(self):
        """An empty path is rejected."""
        assert validate_code_ref_path("") is None

    def test_whitespace_path_rejected(self):
        """A whitespace-only path is rejected."""
        assert validate_code_ref_path("   ") is None

    def test_current_dir_relative_accepted(self):
        """A path starting with './' is accepted (it resolves to relative)."""
        # './src/lib.rs' does not start with '/' and does not contain '..'
        assert validate_code_ref_path("./src/lib.rs") == "./src/lib.rs"


class TestRefs_ResolveCodeRef:
    """Test resolve_code_ref() — parses code reference strings and reads file content."""

    def test_resolve_code_ref_valid_format(self, tmp_path):
        """Given a relative target_id='path:start:end' with allowed_paths,
        returns dict with content from that range."""
        f = tmp_path / "src" / "lib.rs"
        f.parent.mkdir(parents=True)
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        # Use relative path within allowed directory
        result = resolve_code_ref("src/lib.rs:2:4", allowed_paths=[tmp_path])
        assert result is not None
        assert result["start_line"] == 2
        assert result["end_line"] == 4
        assert result["content"] == "line2\nline3\nline4"
        assert result["target_type"] == "code"

    def test_resolve_code_ref_missing_file(self, tmp_path):
        """Given a non-existent code ref path, returns None."""
        result = resolve_code_ref("nonexistent.py:1:5", allowed_paths=[tmp_path])
        assert result is None

    def test_resolve_code_ref_invalid_format(self):
        """Given target_id='not-a-valid-ref', returns None."""
        result = resolve_code_ref("not-a-valid-ref")
        assert result is None

    def test_resolve_code_ref_partial_format(self, tmp_path):
        """Given target_id='path:42' (missing end_line), returns None."""
        # Create a file so the path is valid, but the format is wrong
        f = tmp_path / "src.rs"
        f.write_text("some content\n")
        result = resolve_code_ref("src.rs:42", allowed_paths=[tmp_path])
        assert result is None

    def test_resolve_code_ref_line_range(self, tmp_path):
        """Given a 10-line file and start_line=2:end_line=5, returns only lines 2-5."""
        f = tmp_path / "ten_lines.txt"
        lines = [f"line{i}" for i in range(1, 11)]
        f.write_text("\n".join(lines) + "\n")
        result = resolve_code_ref("ten_lines.txt:2:5", allowed_paths=[tmp_path])
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
        result = resolve_code_ref("short.txt:10:15", allowed_paths=[tmp_path])
        assert result is None

    def test_resolve_code_ref_end_line_clamped(self, tmp_path):
        """When end_line exceeds file length, clamps to available lines."""
        f = tmp_path / "clamp.txt"
        f.write_text("line1\nline2\nline3\n")
        result = resolve_code_ref("clamp.txt:1:100", allowed_paths=[tmp_path])
        assert result is not None
        assert result["end_line"] == 3
        assert result["content"] == "line1\nline2\nline3"


class TestRefs_SecurityPathRestrictions:
    """Security tests for resolve_code_ref() — prevents arbitrary file reads.

    These tests verify that code refs cannot read files outside allowed
    directories, which prevents attacks like reading /home/user/.ssh/id_rsa
    or other sensitive files.
    """

    def test_absolute_path_rejected(self):
        """Absolute paths are rejected — prevents reading /etc/passwd etc."""
        result = resolve_code_ref("/etc/passwd:1:5")
        assert result is None

    def test_absolute_home_path_rejected(self):
        """Absolute path under /home is rejected — prevents reading SSH keys etc."""
        result = resolve_code_ref("/home/user/.ssh/id_rsa:1:5")
        assert result is None

    def test_traversal_path_rejected(self):
        """Paths with '..' traversal are rejected."""
        result = resolve_code_ref("../../etc/passwd:1:5")
        assert result is None

    def test_path_outside_allowed_dirs_rejected(self, tmp_path):
        """A relative path resolving outside allowed_paths is rejected.

        Create a file in tmp_path, but set allowed_paths to a different
        directory. Even though the path is relative and valid, it resolves
        to a directory not in the allowlist.
        """
        subdir = tmp_path / "project"
        subdir.mkdir()
        f = subdir / "test.py"
        f.write_text("secret = True\n")

        # allowed_paths set to a different subdirectory — resolve should fail
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        result = resolve_code_ref("project/test.py:1:1", allowed_paths=[other_dir])
        assert result is None

    def test_path_inside_allowed_dirs_accepted(self, tmp_path):
        """A relative path resolving inside allowed_paths is accepted."""
        f = tmp_path / "test.py"
        f.write_text("safe = True\n")

        result = resolve_code_ref("test.py:1:1", allowed_paths=[tmp_path])
        assert result is not None
        assert result["content"] == "safe = True"

    def test_default_allowed_paths_is_cwd(self, tmp_path, monkeypatch):
        """When allowed_paths is None, defaults to [Path.cwd()]."""
        f = tmp_path / "local.txt"
        f.write_text("local content\n")

        # Change cwd to tmp_path so the relative path resolves inside cwd
        monkeypatch.chdir(tmp_path)
        result = resolve_code_ref("local.txt:1:1")
        assert result is not None
        assert result["content"] == "local content"

    def test_default_allowed_paths_blocks_outside_cwd(self):
        """When allowed_paths is None (defaults to cwd), absolute paths
        pointing outside cwd are blocked."""
        result = resolve_code_ref("/etc/passwd:1:5")
        assert result is None

    def test_multiple_allowed_paths(self, tmp_path):
        """Files under any of the allowed directories can be read."""
        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"
        dir_a.mkdir()
        dir_b.mkdir()

        f_a = dir_a / "a.py"
        f_a.write_text("project A code\n")
        f_b = dir_b / "b.py"
        f_b.write_text("project B code\n")

        # Both files should be accessible with both dirs in allowed_paths
        result_a = resolve_code_ref("a.py:1:1", allowed_paths=[dir_a, dir_b])
        assert result_a is not None
        assert result_a["content"] == "project A code"

        result_b = resolve_code_ref("b.py:1:1", allowed_paths=[dir_a, dir_b])
        assert result_b is not None
        assert result_b["content"] == "project B code"

    def test_symlink_outside_allowed_rejected(self, tmp_path):
        """A symlink that resolves outside allowed_paths is rejected."""
        # Create outside directory and file
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "outside.txt"
        outside_file.write_text("outside content\n")

        # Create allowed directory with symlink pointing outside
        link_dir = tmp_path / "links"
        link_dir.mkdir()
        link = link_dir / "link.txt"
        try:
            link.symlink_to(outside_file)
        except OSError:
            pytest.skip("symlinks not supported on this platform")

        # Even though the link is inside link_dir, it resolves to outside_dir
        # which is NOT in allowed_paths — should be rejected
        result = resolve_code_ref("link.txt:1:1", allowed_paths=[link_dir])
        # The resolved path is outside link_dir, so it should be rejected
        assert result is None

    def test_relative_path_with_subdirectory_in_allowed(self, tmp_path):
        """A relative path with subdirectories resolved under allowed_paths works."""
        subdir = tmp_path / "src" / "utils"
        subdir.mkdir(parents=True)
        f = subdir / "helpers.py"
        f.write_text("def helper(): pass\n")

        result = resolve_code_ref("src/utils/helpers.py:1:1", allowed_paths=[tmp_path])
        assert result is not None
        assert "helper" in result["content"]
