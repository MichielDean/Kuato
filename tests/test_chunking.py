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
    _matches_pattern,
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


class TestChunking_WalkCodeFiles_Security:
    """Security-related tests for walk_code_files: symlinks, size limits, depth limits."""

    def test_walk_skips_file_symlink(self, tmp_path):
        """walk_code_files skips symlinks to files (path traversal prevention).

        A symlink pointing to a file outside the root must not be followed,
        preventing indexing of arbitrary files on the filesystem.
        """
        # Create a real file outside the project root
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.txt"
        secret_file.write_text("secret data")

        # Create a symlink inside the project root pointing to the outside file
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text("print('hello')")
        symlink_path = project_dir / "linked_secret.txt"
        symlink_path.symlink_to(secret_file)

        files = walk_code_files(project_dir)
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert "linked_secret.txt" not in file_names

    def test_walk_skips_directory_symlink(self, tmp_path):
        """walk_code_files skips symlinks to directories (path traversal prevention).

        A symlink pointing to a directory outside the root must not be
        traversed, preventing indexing of entire directory trees outside
        the project root.
        """
        # Create a directory outside the project root with a code file
        outside_dir = tmp_path / "outside_code"
        outside_dir.mkdir()
        (outside_dir / "leaked.py").write_text("sensitive = True")

        # Create a symlink inside the project root pointing outside
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text("print('hello')")
        symlink_dir = project_dir / "evil_link"
        symlink_dir.symlink_to(outside_dir)

        files = walk_code_files(project_dir)
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert "leaked.py" not in file_names

    def test_walk_skips_circular_symlink(self, tmp_path):
        """walk_code_files skips circular directory symlinks.

        A symlink pointing back to an ancestor directory would create an
        infinite loop. walk_code_files must not follow it.
        """
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text("print('hello')")
        # Create a circular symlink: project/sub -> project
        sub_dir = project_dir / "sub"
        sub_dir.symlink_to(project_dir)

        files = walk_code_files(project_dir)
        # Should find main.py exactly once, not infinitely
        assert len(files) == 1
        assert files[0].name == "main.py"

    def test_walk_skips_large_files(self, tmp_path):
        """walk_code_files skips files exceeding max_file_size.

        A file larger than the configured size limit is skipped entirely,
        preventing memory exhaustion from indexing large files.
        """
        (tmp_path / "small.py").write_text("x = 1")
        # Create a file that exceeds the default 1 MiB limit
        large_content = "x = 1\n" * 200000  # ~1.4 MB
        (tmp_path / "large.py").write_text(large_content)

        files = walk_code_files(tmp_path)
        file_names = {f.name for f in files}
        assert "small.py" in file_names
        assert "large.py" not in file_names

    def test_walk_custom_max_file_size(self, tmp_path):
        """walk_code_files respects a custom max_file_size parameter.

        When max_file_size is set to a small value, files exceeding that
        threshold are skipped.
        """
        (tmp_path / "small.py").write_text("x = 1")
        (tmp_path / "medium.py").write_text("x = 1\n" * 100)  # ~700 bytes

        files = walk_code_files(tmp_path, max_file_size=500)
        file_names = {f.name for f in files}
        assert "small.py" in file_names
        assert "medium.py" not in file_names

    def test_walk_respects_max_depth(self, tmp_path):
        """walk_code_files respects the max_depth parameter.

        Directories deeper than max_depth are not traversed, preventing
        stack overflow from excessively deep directory nesting.
        """
        # Create a 5-level deep directory structure
        current = tmp_path
        for i in range(5):
            current = current / f"level{i}"
            current.mkdir()
        (current / "deep.py").write_text("deep = True")

        # Also add a shallow file
        (tmp_path / "shallow.py").write_text("shallow = True")

        # With max_depth=2, only files within 2 levels should be found
        files = walk_code_files(tmp_path, max_depth=2)
        file_names = {f.name for f in files}
        assert "shallow.py" in file_names
        assert "deep.py" not in file_names

    def test_walk_default_depth_allows_reasonable_nesting(self, tmp_path):
        """walk_code_files with default max_depth=50 allows reasonable nesting.

        A 10-level deep directory structure is well within the default
        limit of 50 and should be fully traversed.
        """
        current = tmp_path
        for i in range(10):
            current = current / f"dir{i}"
            current.mkdir()
        (current / "deep.py").write_text("deep = True")

        files = walk_code_files(tmp_path)
        file_names = {f.name for f in files}
        assert "deep.py" in file_names

    def test_walk_symlink_file_not_in_results(self, tmp_path):
        """Issue ll-67q3p-16s63: symlinked files are excluded from results.

        Even when a symlink appears to be a regular file with a valid
        extension, it must not appear in the results to prevent path
        traversal and data exposure attacks.
        """
        # Create an external file and symlink it into the project
        outside = tmp_path / "external"
        outside.mkdir()
        (outside / "targets.py").write_text("secret = 'data'")

        project = tmp_path / "project"
        project.mkdir()
        (project / "real.py").write_text("ok = True")
        (project / "attack.py").symlink_to(outside / "targets.py")

        files = walk_code_files(project)
        file_names = {f.name for f in files}
        assert "real.py" in file_names
        assert "attack.py" not in file_names

    def test_walk_zero_max_file_size_skips_everything(self, tmp_path):
        """walk_code_files with max_file_size=0 skips all files.

        Setting max_file_size to zero means no file can be indexed,
        which is a valid (if unusual) configuration.
        """
        (tmp_path / "tiny.py").write_text("x = 1")

        files = walk_code_files(tmp_path, max_file_size=0)
        assert len(files) == 0

    def test_walk_skips_env_files(self, tmp_path):
        """walk_code_files skips .env files to prevent indexing secrets.

        .env files and their variants (.env.local, .env.production, etc.)
        typically contain API keys, passwords, and database credentials.
        Indexing them would expose secrets via search.
        """
        (tmp_path / ".env").write_text("API_KEY=sk-12345\nDB_PASSWORD=secret")
        (tmp_path / ".env.local").write_text("DEBUG=true")
        (tmp_path / ".env.production").write_text("DATABASE_URL=postgres://...")
        (tmp_path / "main.py").write_text("print('hello')")

        files = walk_code_files(tmp_path)
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert ".env" not in file_names
        assert ".env.local" not in file_names
        assert ".env.production" not in file_names

    def test_walk_skips_pem_key_files(self, tmp_path):
        """walk_code_files skips .pem and .key files to prevent indexing private keys.

        .pem and .key files contain private cryptographic keys that must
        never be indexed into searchable code chunks.
        """
        (tmp_path / "server.pem").write_text("-----BEGIN PRIVATE KEY-----\n...")
        (tmp_path / "private.key").write_text("-----BEGIN RSA PRIVATE KEY-----\n...")
        (tmp_path / "id_rsa.key").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n...")
        (tmp_path / "cert.pem").write_text("-----BEGIN CERTIFICATE-----\n...")
        (tmp_path / "main.py").write_text("print('hello')")

        files = walk_code_files(tmp_path)
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert "server.pem" not in file_names
        assert "private.key" not in file_names
        assert "id_rsa.key" not in file_names
        assert "cert.pem" not in file_names

    def test_walk_skips_credential_filenames(self, tmp_path):
        """walk_code_files skips SSH private keys and credential files.

        Files like id_rsa, id_ed25519, .netrc, .htpasswd, .npmrc, and
        .pypirc contain secrets (SSH keys, network passwords, package
        manager tokens) and must never be indexed.
        """
        (tmp_path / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n...")
        (tmp_path / "id_dsa").write_text("-----BEGIN DSA PRIVATE KEY-----\n...")
        (tmp_path / "id_ed25519").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n...")
        (tmp_path / "id_ecdsa").write_text("-----BEGIN EC PRIVATE KEY-----\n...")
        (tmp_path / ".netrc").write_text("machine example.com login user password pass")
        (tmp_path / ".htpasswd").write_text("user:$apr1$hash$salt\n")
        (tmp_path / ".npmrc").write_text("//registry.npmjs.org/:_authToken=secret\n")
        (tmp_path / ".pypirc").write_text(
            "[pypi]\n  username = user\n  password = pass\n"
        )
        (tmp_path / "main.py").write_text("print('hello')")

        files = walk_code_files(tmp_path)
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        for secret_name in [
            "id_rsa",
            "id_dsa",
            "id_ed25519",
            "id_ecdsa",
            ".netrc",
            ".htpasswd",
            ".npmrc",
            ".pypirc",
        ]:
            assert secret_name not in file_names, f"{secret_name} should be skipped"


class TestChunking_DirOnlyGitignorePattern:
    """Test that gitignore patterns ending with / only match directories, not files.

    Issue ll-67q3p-cokil: _matches_pattern computes a dir_only flag from
    patterns ending in '/' but previously never used it, causing patterns
    like 'build/' to incorrectly match a file named 'build'.
    """

    def test_matches_pattern_dir_only_skips_file(self):
        """Issue ll-67q3p-cokil: patterns ending in / must not match files.

        A gitignore pattern like 'build/' should only match directories.
        A regular file named 'build' must NOT be ignored by this pattern.
        """
        # 'build/' should match directories only
        assert not _matches_pattern("build", "build/", is_dir=False)

    def test_matches_pattern_dir_only_matches_directory(self):
        """Patterns ending in / must match directories.

        A gitignore pattern like 'build/' should match a directory named
        'build'.
        """
        assert _matches_pattern("build", "build/", is_dir=True)

    def test_matches_pattern_without_slash_matches_both(self):
        """Patterns without trailing / match both files and directories.

        A pattern like 'build' (no trailing /) matches both a file and
        a directory named 'build'.
        """
        assert _matches_pattern("build", "build", is_dir=False)
        assert _matches_pattern("build", "build", is_dir=True)

    def test_is_ignored_dir_pattern_does_not_ignore_file(self, tmp_path):
        """is_ignored with a dir-only pattern does not ignore a file.

        A .gitignore pattern 'build/' should not cause a file named
        'build' to be ignored.
        """
        patterns = [("build/", False)]
        build_file = tmp_path / "build"
        build_file.touch()
        assert not is_ignored(build_file, tmp_path, patterns, is_dir=False)

    def test_is_ignored_dir_pattern_ignores_directory(self, tmp_path):
        """is_ignored with a dir-only pattern ignores a directory.

        A .gitignore pattern 'build/' should cause a directory named
        'build' to be ignored.
        """
        patterns = [("build/", False)]
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        assert is_ignored(build_dir, tmp_path, patterns, is_dir=True)

    def test_walk_code_files_dir_pattern_does_not_skip_file(self, tmp_path):
        """walk_code_files does not skip a file matching a dir-only pattern.

        A .gitignore with 'dist/' should skip a directory named 'dist'
        but NOT skip a file named 'dist' (if it were a code file).
        """
        (tmp_path / ".gitignore").write_text("dist/\n")
        # dist/ pattern should skip the dist directory
        (tmp_path / "dist" / "output.py").parent.mkdir(parents=True)
        (tmp_path / "dist" / "output.py").write_text("print('dist')")
        # main.py should still be indexed
        (tmp_path / "main.py").write_text("print('hello')")
        files = walk_code_files(tmp_path)
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert "output.py" not in file_names

    def test_walk_code_files_dir_pattern_preserves_file(self, tmp_path):
        """A file named like a dir-only gitignore pattern is not skipped.

        When .gitignore contains 'logs/', a file named 'logs' should
        still be indexed if it has a recognized extension, while a
        'logs/' directory should be skipped.
        """
        (tmp_path / ".gitignore").write_text("logs/\n")
        # Create a logs directory (should be skipped)
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "app.py").write_text("app code")
        # Create a main.py (should not be skipped)
        (tmp_path / "main.py").write_text("print('hello')")

        files = walk_code_files(tmp_path)
        file_names = {f.name for f in files}
        assert "main.py" in file_names
        assert "app.py" not in file_names  # Inside skipped logs/ directory

    def test_is_ignored_negation_with_dir_only(self, tmp_path):
        """Negation patterns can override dir-only patterns in is_ignored.

        A pattern '!build/' combined with 'build/' should un-ignore
        the build directory.
        """
        patterns = [("build/", False), ("build/", True)]
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        # First pattern ignores build/, second negation un-ignores it
        assert not is_ignored(build_dir, tmp_path, patterns, is_dir=True)
