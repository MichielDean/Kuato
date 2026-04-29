# Contributing to LLMem

Thank you for your interest in contributing to LLMem! This guide covers how to set up your development environment, coding conventions, and how to submit pull requests.

## Development Setup

### Prerequisites

- Python 3.11 or later
- Node.js 20 or later
- Git

### Install for Development

```bash
git clone https://github.com/MichielDean/LLMem.git
cd LLMem
pip install ".[dev]"
```

This installs LLMem along with pytest for running tests.

### Optional Extras

For vector similarity search support:

```bash
pip install ".[vec]"
```

For local embedding without any server (uses sentence-transformers):

```bash
pip install ".[local]"
```

You can combine extras:

```bash
pip install ".[vec,local,dev]"
```

## Running Tests

### Python Tests

```bash
python -m pytest
```

The test suite uses `pytest` with test paths configured in `pyproject.toml`. Tests create temporary databases via the `tmp_path` fixture — no external services are required.

### Node.js Tests (Root Package)

```bash
npm install
npm test
```

This validates skill directory structure, frontmatter, forbidden references, and the install script.

### opencode-llmem Package Tests

```bash
cd opencode-llmem
npm install
npm test
```

## Code Style

### Python

- Use the standard library first. Use `urllib.request` for HTTP — do not add `requests` or `httpx` as dependencies.
- Use `argparse` for CLI argument parsing — no `click` or `typer`.
- Use `logging.getLogger(__name__)` for logging — never `print()` for operational output.
- Follow the logging prefix convention: `"llmem: <module>: <message>"`.
- CLI user-facing errors use `print(f"Error: ...", file=sys.stderr)` then `sys.exit(1)`.
- Library errors raise `ValueError` with an `"llmem: <module>: "` prefix.

### Node.js

- Use Node.js built-in modules only (`fs`, `path`, `os`, `child_process`) — no third-party dependencies.
- Test runner: custom `test.js` with `assert()` function.

### General

- All identifiers in SQL must be double-quoted (SQLite convention).
- Migration files are numbered: `001_xxx.sql`, `002_xxx.sql`, etc.
- DDL and DML must be in separate migration files. DML files are wrapped in `BEGIN TRANSACTION; ... COMMIT;`.

## No Internal References

All files in this repository are checked against a list of forbidden patterns (internal names, personal references, project-internal terminology). See `test.js` for the full list. Before submitting a PR, verify your changes pass `npm test` at the repository root, which includes the forbidden reference check.

## Submitting Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, including tests for any new functionality.
3. Ensure all tests pass: `python -m pytest`, `npm test`, and `cd opencode-llmem && npm test`.
4. Ensure no forbidden patterns are introduced (covered by `npm test`).
5. Open a pull request against the `main` branch with a clear description of the change.

## License

By contributing to LLMem, you agree that your contributions will be licensed under the [MIT License](LICENSE).
