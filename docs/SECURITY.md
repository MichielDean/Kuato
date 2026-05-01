# LLMem Security

Security measures, validation, and hardening in LLMem. [Back to README](../README.md)

## Security

- `LMEM_HOME` is validated against path traversal, system directories, and symlink attacks.
- Write paths are validated against system directories and symbolic links. `is_symlink()` checks are wrapped in `try/except OSError` to handle inaccessible paths gracefully.
- System directory blocking uses a shared `_BLOCKED_PATH_PREFIXES` tuple (with prefix + `/` matching to avoid false positives) across `_validate_home_path`, `_validate_write_path`, `OpenCodeAdapter.__init__`, and `CopilotAdapter.__init__` to prevent DRY violations.
- `validate_session_id()` rejects session IDs containing `/`, `\`, or `..` to prevent path traversal when constructing context file paths.
- **CLI path validation**: `llmem add --file`, `llmem import`, and `llmem export --output` block access to protected system directories (e.g. `/etc`, `/bin`, `/usr/bin`, `/sbin`, `/usr/sbin`, `/dev`, `/proc`, `/sys`, `/var`, `/boot`, `/root`). These checks use prefix + `/` matching to avoid false positives (e.g. `/binary_search` is not blocked as `/bin`).
- **Import file size limit**: `llmem import` rejects files larger than 10 MiB.
- URL validation (`is_safe_url`) blocks private/reserved IPs and SSRF vectors, including percent-encoded IP hostnames (e.g. `%31%32%37%2e%30%2e%30%2e%31` is decoded before IP checks). When `allow_remote=False` (the default), only loopback addresses on the Ollama default port are permitted — all other IPs including public addresses are rejected. `safe_urlopen` enforces URL validation, blocks all redirects (via `_NoRedirectHandler`), mitigates DNS rebinding by re-resolving the hostname immediately before the request, strips credentials from error messages, and requires an explicit `allow_remote` parameter (defaults to `False`) for non-loopback addresses. It accepts both string URLs and `urllib.request.Request` objects, and applies a default 30-second timeout to prevent indefinite hangs.
- OpenCode session extraction validates the database path: rejects path traversal (`..`), system directories, symlinks, and URI injection (`?` and `#` characters). `get_opencode_db_path()` validates via `_validate_home_path` before returning.
- CopilotAdapter validates `state_dir` via `_validate_home_path` to reject path traversal and system directory attacks. `share_dir` is resolved but not validated (it points to user-controlled output).
- API keys are masked in `__repr__` on provider instances (`***masked***`).
- API keys are refused over plain HTTP to non-loopback hosts. `OpenAIProvider` and `AnthropicProvider` raise `ValueError` if `base_url` is `http://` and the hostname is not a loopback address (checked via exact string match and `ipaddress` for IPv6-mapped addresses like `::ffff:127.0.0.1`). Substring matches like `localhost.evil.com` are blocked.
- A warning is logged when API keys are sent to a non-default base URL to alert the user of potential credential exfiltration risk.
- Validation error messages use `_strip_credentials()` to remove userinfo from URLs — never embed user-supplied URL credentials in error messages or logs.
- `_strip_credentials()` is used consistently across `is_safe_url()`, `safe_urlopen()`, provider error messages, and config URL validation to prevent credential leaking.
- Embedding and generation inputs are validated against size limits: `MAX_TEXT_LENGTH` (100,000 characters per text) and `MAX_BATCH_SIZE` (2,048 texts per batch) to prevent OOM and resource exhaustion.
- Embedding dimension validation in `MemoryStore.add()` rejects vectors whose dimension doesn't match `vec_dimensions`, preventing dimension mismatch bugs from silently corrupting the vector index.
- All SQL queries use parameterized statements (no injection risk).
- **Code reference path validation**: `validate_code_ref_path()` rejects absolute paths (leading `/`) and directory traversal (`..`) in code ref target_ids. `resolve_code_ref()` enforces an `allowed_paths` allowlist (defaulting to `[Path.cwd()]`) and blocks resolved paths targeting system directories (`/etc`, `/var`, etc.). Code refs must use the relative format `path:start_line:end_line`.
- `add_relation()` validates code ref `target_id` paths at insertion time — unsafe paths are rejected with `ValueError`.
- SQLite extension loading is disabled immediately after `sqlite-vec` loads, preventing runtime loading of arbitrary shared libraries.
- Database files are created with `umask(0o177)` before creation, then `chmod(0o600)` applied to the DB file and its WAL/SHM sidecars (prevents a race window where sensitive memory content is world-readable on multi-user systems). Parent directories use `0o700`.
- `config.yaml` is written with `0o600` file permissions (owner-only read/write) to protect API keys and secrets from other users on shared systems.
- **Server auth token strength**: `server.auth_token` in `config.yaml` must be at least 16 characters. Short tokens are rejected with a hint to generate a strong token.
- `import_memories()` validates entry IDs (string, max 256 chars), embeddings (bytes, max 1 MB), and confidence (numeric) before insertion. Invalid entries are skipped with warnings rather than crashing.
- `export_all()` defaults to a limit of 10,000 memories to prevent unbounded memory consumption; pass `limit=None` to export all.
- `_search_by_embedding_brute()` uses a `LIMIT` clause (10,000 rows max) to prevent OOM on large databases.
- **Embedding metrics computation capping**: `get_embeddings_with_types(limit=)` applies a SQL `LIMIT` clause (default: 10,000) and `compute_metrics(max_embeddings=10000)` truncates input vectors. These caps prevent O(n²) pairwise metrics computations from causing OOM or CPU hangs on large stores.
- `process_transcript()` enforces the same size limit as `process_file()` to prevent OOM from large session transcripts.
- **Dream diary locking**: On platforms with `fcntl` (Linux/macOS), dream diary writes use an exclusive file lock to prevent corruption from concurrent dream cycles.
- OpenCode tool invocations (`_llmem.ts`) prepend `--` before user arguments to prevent argparse flag injection.
- JavaScript hooks use `execFileSync` (not shell-based `execSync`) and `validateSessionId()` for path traversal protection, with `canSpawnProcess()` rate limiting and `MAX_CONCURRENT=3` process cap to prevent resource exhaustion.
- Prototype pollution protection in `_parseSimpleYaml`: keys `__proto__`, `constructor`, and `prototype` are filtered from parsed YAML to prevent Object prototype mutation.
- ProviderDetector.detect() only returns `provider` and `ollama_url` — no API key presence is exposed.
- Migration from `~/.lobsterdog/` skips symlinks (using `follow_symlinks=False`).

**Code indexing security:**

- `walk_code_files()` skips all symlinks (both file and directory) to prevent path traversal and data exposure.
- Default file size limit of 1 MiB (`--max-file-size`) prevents memory exhaustion from large files.
- Default directory depth limit of 50 (`--max-depth`) prevents stack overflow from deeply nested trees.
- Credential files are excluded from indexing: `.env`, `.env.*` variants, `.pem`, `.key`, SSH private keys (`id_rsa`, `id_dsa`, `id_ed25519`, `id_ecdsa`), `.netrc`, `.htpasswd`, `.npmrc`, `.pypirc`.
- `.gitignore` patterns are respected at every directory level, with correct handling of anchored patterns (leading `/`), negation patterns (`!`), and directory-only patterns (trailing `/`).