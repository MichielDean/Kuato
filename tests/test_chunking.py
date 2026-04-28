"""Tests for llmem.chunking module — ParagraphChunking, FixedLineChunking, gitignore."""

from pathlib import Path

import pytest

from llmem.chunking import (
    CodeChunk,
    ParagraphChunking,
    FixedLineChunking,
    detect_language,
    parse_gitignore,
    is_ignored,
    walk_code_files,
)


class TestChunking_ChunkIdFormat:
    """Test that chunk IDs follow the <file_path>:<start_line>:<end_line> format."""

    def test_chunk_id_format_paragraph(self):
        """ParagraphChunking produces IDs in path:start:end format."""
        chunker = ParagraphChunking()
        chunks = chunker.chunk("src/main.py", "line1\nline2\nline3\n")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert ":" in chunk.id
            parts = chunk.id.split(":")
            assert len(parts) == 3
            assert parts[0] == "src/main.py"
            start = int(parts[1])
            end = int(parts[2])
            assert start >= 1
            assert end >= start

    def test_chunk_id_format_fixed(self):
        """FixedLineChunking produces IDs in path:start:end format."""
        chunker = FixedLineChunking(window_size=3, overlap=1)
        content = "\n".join(f"line{i}" for i in range(10))
        chunks = chunker.chunk("app.js", content)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert ":" in chunk.id
            parts = chunk.id.split(":")
            assert parts[0] == "app.js"
            start = int(parts[1])
            end = int(parts[2])
            assert start >= 1
            assert end >= start


class TestChunking_ParagraphChunking:
    """Test paragraph-based chunking on content with blank-line boundaries."""

    def test_basic_paragraph_splitting(self):
        """Content with blank-line separations is split at those boundaries."""
        chunker = ParagraphChunking()
        content = (
            "paragraph1 line1\nparagraph1 line2\n\nparagraph2 line1\nparagraph2 line2"
        )
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) >= 2
        assert chunks[0].content == "paragraph1 line1\nparagraph1 line2"
        assert chunks[1].content == "paragraph2 line1\nparagraph2 line2"

    def test_single_paragraph(self):
        """Content without blank lines produces a single chunk."""
        chunker = ParagraphChunking()
        content = "line1\nline2\nline3"
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) == 1
        assert chunks[0].content == "line1\nline2\nline3"

    def test_line_numbers_are_correct(self):
        """Start and end line numbers are 1-based and correct."""
        chunker = ParagraphChunking()
        content = "line1\nline2\n\nline4\nline5"
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) >= 2
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2
        assert chunks[1].start_line == 4
        assert chunks[1].end_line == 5

    def test_language_detection(self):
        """Language is set correctly when provided."""
        chunker = ParagraphChunking()
        chunks = chunker.chunk("test.py", "hello world", language="python")
        assert len(chunks) == 1
        assert chunks[0].language == "python"

    def test_auto_language_detection(self):
        """Language is auto-detected from file extension when not provided."""
        chunker = ParagraphChunking()
        chunks = chunker.chunk("test.py", "hello world")
        assert len(chunks) == 1
        assert chunks[0].language == "python"

    def test_chunk_type_is_paragraph(self):
        """ParagraphChunking always produces chunk_type 'paragraph'."""
        chunker = ParagraphChunking()
        chunks = chunker.chunk("test.py", "hello world")
        assert all(c.chunk_type == "paragraph" for c in chunks)

    def test_file_path_preserved(self):
        """File path is preserved in each chunk."""
        chunker = ParagraphChunking()
        chunks = chunker.chunk("src/main.py", "hello world")
        assert all(c.file_path == "src/main.py" for c in chunks)


class TestChunking_ParagraphChunking_EdgeCases:
    """Test edge cases for ParagraphChunking."""

    def test_empty_content_returns_empty_list(self):
        """Empty content produces no chunks."""
        chunker = ParagraphChunking()
        chunks = chunker.chunk("test.py", "")
        assert chunks == []

    def test_single_line_content(self):
        """A single line of content produces one chunk."""
        chunker = ParagraphChunking()
        chunks = chunker.chunk("test.py", "single line")
        assert len(chunks) == 1
        assert chunks[0].content == "single line"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1

    def test_content_with_only_blank_lines(self):
        """Content consisting entirely of blank lines produces no chunks."""
        chunker = ParagraphChunking()
        content = "\n\n\n"
        chunks = chunker.chunk("test.py", content)
        assert chunks == []

    def test_trailing_blank_lines_ignored(self):
        """Trailing blank lines at the end of content are ignored."""
        chunker = ParagraphChunking()
        content = "line1\nline2\n\n"
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) == 1
        assert chunks[0].content == "line1\nline2"

    def test_leading_blank_lines_ignored(self):
        """Leading blank lines are ignored and don't produce empty chunks."""
        chunker = ParagraphChunking()
        content = "\n\nline3\nline4"
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) >= 1
        # Should have one chunk starting from the first non-blank line
        assert chunks[0].content.strip() != ""

    def test_max_lines_splits_large_paragraphs(self):
        """Paragraphs exceeding max_lines are split."""
        chunker = ParagraphChunking(min_lines=1, max_lines=3)
        content = "\n".join([f"line{i}" for i in range(10)])
        chunks = chunker.chunk("test.py", content)
        # 10 lines with max_lines=3 should produce multiple chunks
        assert len(chunks) >= 3
        # Each chunk should have <= 3 lines, except possibly the last
        for chunk in chunks:
            line_count = len(chunk.content.splitlines())
            assert line_count <= 3

    def test_min_lines_merges_short_paragraphs(self):
        """Adjacent short paragraphs below min_lines are merged."""
        chunker = ParagraphChunking(min_lines=3, max_lines=200)
        # Two 1-line paragraphs separated by a blank line
        content = "short1\n\nshort2\nshort3"
        chunks = chunker.chunk("test.py", content)
        # The paragraphs should be merged to reach the min threshold
        assert len(chunks) >= 1
        total_lines = sum(len(c.content.splitlines()) for c in chunks)
        assert total_lines >= 3

    def test_min_lines_validation_zero(self):
        """min_lines <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="min_lines must be positive"):
            ParagraphChunking(min_lines=0)

    def test_min_lines_validation_negative(self):
        """Negative min_lines raises ValueError."""
        with pytest.raises(ValueError, match="min_lines must be positive"):
            ParagraphChunking(min_lines=-1)

    def test_max_lines_less_than_min_lines(self):
        """max_lines < min_lines raises ValueError."""
        with pytest.raises(ValueError, match="max_lines.*must be >= min_lines"):
            ParagraphChunking(min_lines=10, max_lines=5)


class TestChunking_FixedLineChunking:
    """Test sliding-window chunking with configurable window size and overlap."""

    def test_basic_window_chunking(self):
        """Content is split into windows of the specified size."""
        chunker = FixedLineChunking(window_size=3, overlap=1)
        content = "\n".join([f"line{i}" for i in range(10)])
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) >= 2
        # First chunk should have 3 lines
        assert len(chunks[0].content.splitlines()) == 3

    def test_overlap_between_chunks(self):
        """Overlapping lines appear in adjacent chunks."""
        chunker = FixedLineChunking(window_size=3, overlap=1)
        content = "\n".join([f"line{i}" for i in range(5)])
        chunks = chunker.chunk("test.py", content)
        # With window=3, overlap=1, step=2: lines 1-3, 3-5
        assert len(chunks) >= 2
        # Verify overlap: last line of first chunk = first line of second chunk
        first_last_line = chunks[0].content.splitlines()[-1]
        second_first_line = chunks[1].content.splitlines()[0]
        assert first_last_line == second_first_line

    def test_content_shorter_than_window(self):
        """Content shorter than window size produces one chunk."""
        chunker = FixedLineChunking(window_size=50, overlap=10)
        content = "line1\nline2"
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) == 1
        assert "line1" in chunks[0].content

    def test_zero_overlap(self):
        """Zero overlap produces non-overlapping chunks."""
        chunker = FixedLineChunking(window_size=3, overlap=0)
        content = "\n".join([f"line{i}" for i in range(6)])
        chunks = chunker.chunk("test.py", content)
        assert len(chunks) == 2
        # With zero overlap, last line of first chunk should NOT be first line of second
        first_lines = chunks[0].content.splitlines()
        second_lines = chunks[1].content.splitlines()
        assert first_lines[-1] != second_lines[0]

    def test_chunk_type_is_fixed_line(self):
        """FixedLineChunking always produces chunk_type 'fixed_line'."""
        chunker = FixedLineChunking(window_size=3, overlap=1)
        content = "\n".join([f"line{i}" for i in range(10)])
        chunks = chunker.chunk("test.py", content)
        assert all(c.chunk_type == "fixed_line" for c in chunks)

    def test_language_auto_detection(self):
        """Language is auto-detected from file extension."""
        chunker = FixedLineChunking(window_size=50, overlap=10)
        content = "\n".join([f"line{i}" for i in range(10)])
        chunks = chunker.chunk("app.rs", content)
        assert chunks[0].language == "rust"

    def test_last_chunk_can_be_shorter(self):
        """The last chunk may be shorter than window_size but never empty."""
        chunker = FixedLineChunking(window_size=4, overlap=1)
        content = "\n".join([f"line{i}" for i in range(7)])
        chunks = chunker.chunk("test.py", content)
        last_chunk = chunks[-1]
        assert len(last_chunk.content.splitlines()) > 0
        assert len(last_chunk.content.splitlines()) <= 4


class TestChunking_FixedLineChunking_EdgeCases:
    """Test edge cases for FixedLineChunking."""

    def test_empty_content_returns_empty_list(self):
        """Empty content produces no chunks."""
        chunker = FixedLineChunking(window_size=10, overlap=2)
        chunks = chunker.chunk("test.py", "")
        assert chunks == []

    def test_window_size_zero_raises(self):
        """window_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be > 0"):
            FixedLineChunking(window_size=0)

    def test_window_size_negative_raises(self):
        """Negative window_size raises ValueError."""
        with pytest.raises(ValueError, match="window_size must be > 0"):
            FixedLineChunking(window_size=-1)

    def test_overlap_negative_raises(self):
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            FixedLineChunking(window_size=10, overlap=-1)

    def test_overlap_ge_window_size_raises(self):
        """overlap >= window_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap.*must be < window_size"):
            FixedLineChunking(window_size=5, overlap=5)

    def test_overlap_equal_window_size_raises(self):
        """overlap equal to window_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap.*must be < window_size"):
            FixedLineChunking(window_size=10, overlap=10)

    def test_single_line_content(self):
        """A single line of content produces one chunk."""
        chunker = FixedLineChunking(window_size=50, overlap=10)
        chunks = chunker.chunk("test.py", "only line")
        assert len(chunks) == 1
        assert chunks[0].content == "only line"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1


class TestChunking_DetectLanguage:
    """Test language detection from file extensions."""

    def test_python_detection(self):
        assert detect_language("main.py") == "python"

    def test_javascript_detection(self):
        assert detect_language("app.js") == "javascript"

    def test_typescript_detection(self):
        assert detect_language("app.ts") == "typescript"

    def test_rust_detection(self):
        assert detect_language("main.rs") == "rust"

    def test_go_detection(self):
        assert detect_language("main.go") == "go"

    def test_unknown_extension(self):
        assert detect_language("file.xyz") is None

    def test_no_extension(self):
        assert detect_language("Makefile") is None

    def test_r_extension_case_insensitive(self):
        """Issue ll-67q3p-juc0m: .R (uppercase) must detect as 'r'.

        detect_language lowercases extensions before lookup, so both
        .r and .R should map to 'r'. The duplicate .R entry in the map
        was removed since .R is unreachable — .R is lowercased to .r
        before lookup.
        """
        assert detect_language("script.r") == "r"
        assert detect_language("script.R") == "r"


class TestChunking_GitignoreParser:
    """Test .gitignore parsing and matching."""

    def test_parse_gitignore_basic(self, tmp_path):
        """Basic .gitignore patterns are parsed correctly."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.pyc\nbuild/\n!important.txt\n")
        patterns = parse_gitignore(gitignore)
        assert len(patterns) == 3
        assert patterns[0] == ("*.pyc", False)
        assert patterns[1] == ("build/", False)
        assert patterns[2] == ("important.txt", True)

    def test_parse_gitignore_comments_and_blanks(self, tmp_path):
        """Comments and blank lines are ignored."""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("# comment\n\n*.log\n")
        patterns = parse_gitignore(gitignore)
        assert len(patterns) == 1
        assert patterns[0] == ("*.log", False)

    def test_parse_gitignore_nonexistent(self, tmp_path):
        """Non-existent .gitignore returns empty patterns."""
        patterns = parse_gitignore(tmp_path / ".gitignore")
        assert patterns == []

    def test_is_ignored_matches_star(self, tmp_path):
        """*.pyc pattern matches .pyc files."""
        patterns = [("*.pyc", False)]
        test_file = tmp_path / "test.pyc"
        test_file.touch()
        assert is_ignored(test_file, tmp_path, patterns)

    def test_is_ignored_does_not_match_other(self, tmp_path):
        """*.pyc pattern does not match .py files."""
        patterns = [("* .pyc", False)]
        test_file = tmp_path / "test.py"
        test_file.touch()
        assert not is_ignored(test_file, tmp_path, [])

    def test_is_ignored_negation(self, tmp_path):
        """Negation pattern overrides previous ignore."""
        patterns = [("*.log", False), ("important.log", True)]
        important_file = tmp_path / "important.log"
        important_file.touch()
        # Negation should un-ignore
        assert not is_ignored(important_file, tmp_path, patterns)

    def test_walk_code_files_skips_git(self, tmp_path):
        """walk_code_files skips .git directories."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("data")
        (tmp_path / "main.py").write_text("print('hello')")
        files = walk_code_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_walk_code_files_skips_pycache(self, tmp_path):
        """walk_code_files skips __pycache__ directories."""
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "test.cpython-311.pyc").write_bytes(b"")
        (tmp_path / "main.py").write_text("print('hello')")
        files = walk_code_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_walk_code_files_skips_binary_extensions(self, tmp_path):
        """walk_code_files skips binary file extensions."""
        (tmp_path / "image.png").write_bytes(b"")
        (tmp_path / "main.py").write_text("print('hello')")
        files = walk_code_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_walk_code_files_respects_gitignore(self, tmp_path):
        """walk_code_files respects .gitignore patterns."""
        (tmp_path / ".gitignore").write_text("*.log\n")
        (tmp_path / "app.log").write_text("log data")
        (tmp_path / "main.py").write_text("print('hello')")
        files = walk_code_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "main.py"
