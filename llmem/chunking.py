"""Code chunking strategies for llmem learn command.

Provides ParagraphChunking (blank-line boundaries) and FixedLineChunking
(sliding window with overlap) for splitting source files into searchable
code chunks. Also includes a minimal .gitignore parser for directory walking.
"""

import logging
import re
from collections import namedtuple
from pathlib import Path

log = logging.getLogger(__name__)

# Named tuple representing a single code chunk.
# id format: "<file_path>:<start_line>:<end_line>"
CodeChunk = namedtuple(
    "CodeChunk",
    ["id", "file_path", "start_line", "end_line", "content", "language", "chunk_type"],
)

# Common language mapping from file extensions
_EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".txt": "text",
    ".r": "r",
    ".lua": "lua",
    ".php": "php",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".vim": "vim",
}


def detect_language(file_path: str) -> str | None:
    """Detect programming language from a file extension.

    Args:
        file_path: Path to the source file.

    Returns:
        Language name string or None if extension is unknown.
    """
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_LANGUAGE_MAP.get(ext)


class ParagraphChunking:
    """Split file content into chunks at blank-line boundaries.

    Adjacent paragraphs are merged when their combined line count is
    below ``min_lines``. Paragraphs exceeding ``max_lines`` are split
    at the ``max_lines`` boundary.

    Args:
        min_lines: Minimum lines per chunk. Adjacent short paragraphs
            merge to reach this threshold. Defaults to 1.
        max_lines: Maximum lines per chunk. Paragraphs exceeding this
            are split. Defaults to 200.

    Raises:
        ValueError: If min_lines <= 0 or max_lines < min_lines.
    """

    def __init__(self, min_lines: int = 1, max_lines: int = 200):
        if min_lines <= 0:
            raise ValueError(
                f"llmem: chunking: min_lines must be positive, got {min_lines}"
            )
        if max_lines < min_lines:
            raise ValueError(
                f"llmem: chunking: max_lines ({max_lines}) must be >= min_lines ({min_lines})"
            )
        self._min_lines = min_lines
        self._max_lines = max_lines

    def chunk(
        self, file_path: str, content: str, language: str | None = None
    ) -> list[CodeChunk]:
        """Split content into chunks at blank-line boundaries.

        Args:
            file_path: Path of the source file (used in chunk IDs).
            content: The file content to chunk.
            language: Detected or explicit language. If None, auto-detected
                from file_path.

        Returns:
            List of CodeChunk named tuples. Empty list for empty content.
        """
        if not content:
            return []

        if language is None:
            language = detect_language(file_path)

        lines = content.splitlines()
        if not lines:
            return []

        # Split into paragraphs (groups of non-blank lines separated by blanks)
        # Each entry tracks (original_line_number, text) for correct line tracking.
        # Line numbers are 1-based.
        paragraphs: list[list[tuple[int, str]]] = []
        current_paragraph: list[tuple[int, str]] = []

        for i, line in enumerate(lines, start=1):
            if line.strip() == "":
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
            else:
                current_paragraph.append((i, line))

        if current_paragraph:
            paragraphs.append(current_paragraph)

        if not paragraphs:
            # All lines were blank
            return []

        # Merge adjacent short paragraphs
        merged: list[list[tuple[int, str]]] = []
        buffer: list[tuple[int, str]] = []

        for para in paragraphs:
            buffer.extend(para)
            if len(buffer) >= self._min_lines:
                merged.append(buffer)
                buffer = []

        # Flush remaining buffer
        if buffer:
            if merged and len(buffer) < self._min_lines:
                # Merge with previous chunk if too short
                merged[-1].extend(buffer)
            else:
                merged.append(buffer)

        # Split chunks exceeding max_lines
        result_chunks: list[CodeChunk] = []

        for chunk_entries in merged:
            # Split into max_lines-sized pieces
            sub_start = 0
            while sub_start < len(chunk_entries):
                sub_end = min(sub_start + self._max_lines, len(chunk_entries))
                sub_entries = chunk_entries[sub_start:sub_end]

                chunk_start = sub_entries[0][0]
                chunk_end = sub_entries[-1][0]
                chunk_content = "\n".join(text for _, text in sub_entries)
                chunk_id = f"{file_path}:{chunk_start}:{chunk_end}"

                result_chunks.append(
                    CodeChunk(
                        id=chunk_id,
                        file_path=file_path,
                        start_line=chunk_start,
                        end_line=chunk_end,
                        content=chunk_content,
                        language=language,
                        chunk_type="paragraph",
                    )
                )
                sub_start = sub_end

        return result_chunks


class FixedLineChunking:
    """Split file content into overlapping sliding-window chunks.

    Args:
        window_size: Number of lines per chunk window. Must be > 0.
            Defaults to 50.
        overlap: Number of overlap lines between consecutive chunks.
            Must be >= 0 and < window_size. Defaults to 10.

    Raises:
        ValueError: If window_size <= 0, overlap < 0, or overlap >= window_size.
    """

    def __init__(self, window_size: int = 50, overlap: int = 10):
        if window_size <= 0:
            raise ValueError(
                f"llmem: chunking: window_size must be > 0, got {window_size}"
            )
        if overlap < 0:
            raise ValueError(f"llmem: chunking: overlap must be >= 0, got {overlap}")
        if overlap >= window_size:
            raise ValueError(
                f"llmem: chunking: overlap ({overlap}) must be < window_size ({window_size})"
            )
        self._window_size = window_size
        self._overlap = overlap

    def chunk(
        self, file_path: str, content: str, language: str | None = None
    ) -> list[CodeChunk]:
        """Split content into overlapping sliding-window chunks.

        Args:
            file_path: Path of the source file (used in chunk IDs).
            content: The file content to chunk.
            language: Detected or explicit language. If None, auto-detected
                from file_path.

        Returns:
            List of CodeChunk named tuples. Empty list for empty content.
            The last chunk may be shorter than window_size but is never empty.
        """
        if not content:
            return []

        if language is None:
            language = detect_language(file_path)

        lines = content.splitlines()
        if not lines:
            return []

        step = self._window_size - self._overlap
        chunks: list[CodeChunk] = []

        start = 0
        while start < len(lines):
            end = min(start + self._window_size, len(lines))
            chunk_lines = lines[start:end]

            # Never produce empty chunks
            if not chunk_lines:
                break

            chunk_start = start + 1  # 1-based
            chunk_end = start + len(chunk_lines)  # 1-based, inclusive
            chunk_content = "\n".join(chunk_lines)
            chunk_id = f"{file_path}:{chunk_start}:{chunk_end}"

            chunks.append(
                CodeChunk(
                    id=chunk_id,
                    file_path=file_path,
                    start_line=chunk_start,
                    end_line=chunk_end,
                    content=chunk_content,
                    language=language,
                    chunk_type="fixed_line",
                )
            )

            # If we've consumed all lines, we're done
            if end >= len(lines):
                break

            start += step

        return chunks


def parse_gitignore(gitignore_path: Path) -> list[tuple[str, bool]]:
    """Parse a .gitignore file into a list of pattern/negation tuples.

    Each tuple is (pattern: str, is_negation: bool).

    Handles:
    - Blank lines (ignored)
    - Comments starting with # (ignored)
    - Negation patterns starting with !
    - Trailing whitespace is stripped
    - Leading/trailing backslash-escaped spaces are preserved

    Args:
        gitignore_path: Path to the .gitignore file.

    Returns:
        List of (pattern, is_negation) tuples.
    """
    if not gitignore_path.exists():
        return []

    patterns: list[tuple[str, bool]] = []
    try:
        text = gitignore_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.debug(
            "llmem: chunking: could not read .gitignore at %s: %s", gitignore_path, e
        )
        return []

    for line in text.splitlines():
        # Strip trailing whitespace but preserve leading spaces (significant in gitignore)
        line = line.rstrip()
        # Skip blank lines and comments
        if not line or line.startswith("#"):
            continue
        # Check for negation
        is_negation = line.startswith("!")
        if is_negation:
            line = line[1:]
        # Strip leading slash if present (anchors to root)
        # but keep the pattern for matching
        patterns.append((line, is_negation))

    return patterns


def _matches_pattern(path: str, pattern: str, is_dir: bool = False) -> bool:
    """Check if a relative path matches a single gitignore pattern.

    Supports:
    - ``*`` matches any sequence of characters (except /)
    - ``?`` matches any single character (except /)
    - ``**`` matches any sequence of characters including /
    - Patterns ending with ``/`` match directories only
    - Patterns without ``/`` match at any depth

    Args:
        path: Relative path from the repo root (using / as separator).
        pattern: A single gitignore pattern (not a negation).
        is_dir: Whether the path is a directory. When False, patterns
            that end with ``/`` (directory-only patterns) will not match.

    Returns:
        True if the path matches the pattern.
    """
    # Handle directory-only patterns (ending with /)
    dir_only = pattern.endswith("/")
    if dir_only:
        if not is_dir:
            return False
        pattern = pattern[:-1]

    # Determine if pattern is anchored (contains /)
    anchored = "/" in pattern

    # Build regex from the gitignore pattern
    # Convert glob-style pattern to regex
    regex_parts: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*" and i + 1 < len(pattern) and pattern[i + 1] == "*":
            # ** matches any path including /
            if i + 2 < len(pattern) and pattern[i + 2] == "/":
                regex_parts.append("(?:.*/)?")
                i += 3
            else:
                regex_parts.append(".*")
                i += 2
        elif c == "*":
            # * matches any sequence except /
            regex_parts.append("[^/]*")
            i += 1
        elif c == "?":
            # ? matches any single character except /
            regex_parts.append("[^/]")
            i += 1
        elif c in ".+()[]{}^$|":
            regex_parts.append("\\" + c)
            i += 1
        else:
            regex_parts.append(c)
            i += 1

    regex = "".join(regex_parts)

    if anchored:
        # Pattern contains /, so it's anchored to gitignore location
        full_regex = "^" + regex + "$"
    else:
        # Pattern without / matches at any depth
        full_regex = "^(?:.*/)?" + regex + "$"

    try:
        if re.match(full_regex, path):
            return True
    except re.error:
        log.debug("llmem: chunking: invalid gitignore regex pattern: %s", pattern)
        return False

    return False


def is_ignored(
    path: Path, root: Path, patterns: list[tuple[str, bool]], is_dir: bool = False
) -> bool:
    """Check if a file path should be ignored based on gitignore patterns.

    Args:
        path: Absolute path to the file or directory.
        root: Absolute path to the repository root (where .gitignore lives).
        patterns: List of (pattern, is_negation) tuples from parse_gitignore.
        is_dir: Whether the path is a directory. When True, directory-only
            patterns (those ending in ``/``) can match. When False (the
            default for files), directory-only patterns are skipped.

    Returns:
        True if the path should be ignored.
    """
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        return False

    rel_str = str(rel_path).replace("\\", "/")
    name = path.name

    ignored = False
    for pattern, is_negation in patterns:
        # Check both the full relative path and just the filename
        # for unanchored patterns. Pass is_dir so that directory-only
        # patterns (ending in /) match directories but not files.
        if _matches_pattern(rel_str, pattern, is_dir=is_dir) or (
            "/" not in pattern and _matches_pattern(name, pattern, is_dir=is_dir)
        ):
            if is_negation:
                ignored = False
            else:
                ignored = True

    return ignored


# Default maximum file size for indexing: 1 MiB prevents memory exhaustion
# from accidentally indexing large binary or generated files.
_DEFAULT_MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MiB

# Default maximum recursion depth for directory walking.
_DEFAULT_MAX_DEPTH = 50


def walk_code_files(
    root_path: Path,
    patterns: list[tuple[str, bool]] | None = None,
    max_file_size: int = _DEFAULT_MAX_FILE_SIZE,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> list[Path]:
    """Walk a directory tree and return paths of code files to index.

    Respects .gitignore files found at any level. Skips common
    non-code directories (``.git``, ``__pycache__``, ``node_modules``,
    ``.venv``, ``venv``).

    Symlinks to both files and directories are skipped to prevent
    path traversal and data exposure. Use ``-L`` / ``follow_symlinks``
    in the CLI if you explicitly want to follow trusted symlinks.

    Args:
        root_path: Root directory to walk.
        patterns: Optional pre-parsed gitignore patterns. If None,
            reads ``.gitignore`` from root_path.
        max_file_size: Maximum file size in bytes to index. Files
            exceeding this size are skipped. Defaults to 1 MiB.
        max_depth: Maximum directory recursion depth. Prevents
            stack overflow from deeply nested directory trees.
            Defaults to 50.

    Returns:
        List of absolute Path objects for files to index.
    """
    # Always-skip directories
    _SKIP_DIRS: frozenset[str] = frozenset(
        {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            ".mypy_cache",
            ".pytest_cache",
        }
    )

    # File names to always skip (project metadata, build artifacts, and secrets).
    # Includes credential files that must never be indexed: SSH private keys
    # (id_rsa, id_dsa, id_ed25519, id_ecdsa), network credentials (.netrc),
    # web secrets (.htpasswd), and package-manager tokens (.npmrc, .pypirc).
    _SKIP_FILENAMES: frozenset[str] = frozenset(
        {
            ".gitignore",
            ".gitattributes",
            ".gitmodules",
            ".env",
            "id_rsa",
            "id_dsa",
            "id_ed25519",
            "id_ecdsa",
            ".netrc",
            ".htpasswd",
            ".npmrc",
            ".pypirc",
        }
    )

    # Filename prefixes that indicate secret-containing files (e.g. .env.local,
    # .env.production).  Any file whose name starts with one of these prefixes
    # is excluded from indexing to prevent credential leakage.
    _SECRET_FILENAME_PREFIXES: tuple[str, ...] = (".env",)

    # Extensions to skip (binary/cached/generated files and secrets)
    _SKIP_EXTENSIONS: frozenset[str] = frozenset(
        {
            ".pyc",
            ".pyo",
            ".so",
            ".dylib",
            ".dll",
            ".exe",
            ".o",
            ".a",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".ico",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tiff",
            ".svg",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".xz",
            ".7z",
            ".jar",
            ".war",
            ".egg",
            ".whl",
            ".db",
            ".sqlite",
            ".pyd",
            ".pem",
            ".key",
        }
    )

    if patterns is None:
        patterns = parse_gitignore(root_path / ".gitignore")

    result: list[Path] = []

    def _walk(
        directory: Path,
        parent_patterns: list[tuple[str, bool]],
        depth: int,
    ) -> None:
        if depth > max_depth:
            log.debug(
                "llmem: chunking: max depth %d exceeded at %s", max_depth, directory
            )
            return

        try:
            entries = sorted(directory.iterdir())
        except PermissionError as e:
            log.debug("llmem: chunking: permission denied walking %s: %s", directory, e)
            return

        # Check for .gitignore in this directory
        local_gitignore = directory / ".gitignore"
        current_patterns = parent_patterns
        if local_gitignore.exists():
            sub_patterns = parse_gitignore(local_gitignore)
            current_patterns = parent_patterns + sub_patterns

        for entry in entries:
            # Skip symlinks to prevent path traversal and data exposure
            if entry.is_symlink():
                log.debug("llmem: chunking: skipping symlink %s", entry)
                continue

            if entry.is_dir():
                if entry.name in _SKIP_DIRS:
                    continue
                if is_ignored(entry, root_path, current_patterns, is_dir=True):
                    continue
                _walk(entry, current_patterns, depth + 1)
            elif entry.is_file():
                if entry.name in _SKIP_FILENAMES:
                    continue
                if any(entry.name.startswith(p) for p in _SECRET_FILENAME_PREFIXES):
                    continue
                if entry.suffix.lower() in _SKIP_EXTENSIONS:
                    continue
                if is_ignored(entry, root_path, current_patterns, is_dir=False):
                    continue
                # Skip files exceeding the size limit to prevent memory exhaustion
                try:
                    file_size = entry.stat().st_size
                except OSError as e:
                    log.debug("llmem: chunking: cannot stat %s: %s", entry, e)
                    continue
                if file_size > max_file_size:
                    log.debug(
                        "llmem: chunking: skipping file %s "
                        "(%d bytes exceeds max_file_size %d)",
                        entry,
                        file_size,
                        max_file_size,
                    )
                    continue
                result.append(entry)

    _walk(root_path, patterns, depth=0)
    return result
