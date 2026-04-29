"""llmem — LLMem memory management CLI."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .store import MemoryStore, register_memory_type, get_registered_types
from .metrics import (
    compute_metrics,
    bytes_to_vec,
    ANISOTROPY_WARNING_THRESHOLD,
    SIMILARITY_RANGE_WARNING_THRESHOLD,
    METRICS_MAX_EMBEDDINGS,
)
from .paths import get_db_path, get_config_path, get_llmem_home
from .paths import _validate_write_path, _is_blocked_path
from .registry import get_registered_cli_plugins
from .config import write_config_yaml
from .ollama import ProviderDetector
from .paths import migrate_from_lobsterdog
from .url_validate import is_safe_url
from .chunking import (
    ParagraphChunking,
    FixedLineChunking,
    detect_language,
    walk_code_files,
    _DEFAULT_MAX_FILE_SIZE,
    _DEFAULT_MAX_DEPTH,
)
from .code_index import CodeIndex

log = logging.getLogger(__name__)


def _report_embedding_metrics(store: MemoryStore) -> None:
    """Compute and print embedding quality metrics from the store.

    Fetches embeddings via store.get_embeddings_with_types(), capped at
    ``METRICS_MAX_EMBEDDINGS``, computes anisotropy, similarity range,
    and discrimination gap, and prints the results. Emits warnings to
    stderr when metrics exceed warning thresholds or when the result
    set is capped.

    Args:
        store: An open MemoryStore instance.
    """
    total_count = store.count_embeddings()
    limit = METRICS_MAX_EMBEDDINGS
    rows = store.get_embeddings_with_types(limit=limit)

    if not rows:
        print("No embedded memories found.")
        return

    embeddings = []
    labels = []
    for emb_bytes, mem_type in rows:
        vec = bytes_to_vec(emb_bytes)
        if vec:
            embeddings.append(vec)
            labels.append(mem_type)

    if not embeddings:
        print("No valid embeddings found.")
        return

    capped = total_count > limit
    m = compute_metrics(embeddings, labels=labels)

    if capped:
        print(
            f"Embedding metrics ({len(embeddings)} of {total_count} vectors, "
            f"capped at {limit}):"
        )
    else:
        print(f"Embedding metrics ({len(embeddings)} vectors):")
    print(f"  Anisotropy:        {m.anisotropy:.4f}")
    print(f"  Similarity range:  {m.similarity_range:.4f}")
    if m.discrimination_gap is not None:
        print(f"  Discrimination gap: {m.discrimination_gap:.4f}")

    warnings = []
    if m.anisotropy > ANISOTROPY_WARNING_THRESHOLD:
        warnings.append(
            f"Anisotropy ({m.anisotropy:.4f}) exceeds threshold "
            f"({ANISOTROPY_WARNING_THRESHOLD})."
        )
    if m.similarity_range < SIMILARITY_RANGE_WARNING_THRESHOLD:
        warnings.append(
            f"Similarity range ({m.similarity_range:.4f}) is below threshold "
            f"({SIMILARITY_RANGE_WARNING_THRESHOLD})."
        )
    if warnings:
        print()
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)
        print(
            "Embeddings may be poor quality, consider using a different model.",
            file=sys.stderr,
        )


VALID_SOURCES = ["manual", "session", "heartbeat", "extraction", "import"]

# Maximum size for imported files (10 MiB, matching config max_file_size default)
MAX_IMPORT_FILE_SIZE = 10 * 1024 * 1024


def cmd_add(args):
    store = MemoryStore(args.db)
    content = args.content
    if not content and args.file:
        file_path = Path(args.file)
        # Validate the file path to prevent arbitrary file read
        # Uses shared _is_blocked_path from paths.py for consistency
        resolved = file_path.resolve()
        if _is_blocked_path(resolved):
            print(
                f"Error: --file path targets a protected directory: {resolved}",
                file=sys.stderr,
            )
            sys.exit(1)
        content = file_path.read_text().strip()
    if not content:
        print("Error: provide --content or --file", file=sys.stderr)
        sys.exit(1)

    # Validate type at runtime against the global registry — argparse choices
    # cannot be used because register-type can add types after import time.
    registered = get_registered_types()
    if args.type not in registered:
        print(
            f"Error: unregistered type '{args.type}'. "
            f"Register it with: llmem register-type {args.type}",
            file=sys.stderr,
        )
        sys.exit(1)

    mid = store.add(
        type=args.type,
        content=content,
        summary=args.summary,
        source=args.source,
        confidence=args.confidence,
        valid_until=args.valid_until,
        metadata=json.loads(args.metadata) if args.metadata else None,
    )
    print(f"Added memory {mid} [{args.type}]")

    if args.relation and args.relation_to:
        store.add_relation(mid, args.relation_to, args.relation)
        print(f"  -> {args.relation} {args.relation_to}")

    store.close()


def cmd_get(args):
    store = MemoryStore(args.db)
    m = store.get(args.id)
    if not m:
        print(f"Memory {args.id} not found", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(m, indent=2, default=str))
    store.close()


def cmd_search(args):
    from .store import MemoryStore
    from .retrieve import Retriever
    from .embed import EmbeddingEngine

    store = MemoryStore(args.db)

    # Determine search mode from mutually exclusive flags
    if args.fts_only:
        search_mode = "fts"
    elif args.semantic_only:
        search_mode = "semantic"
    else:
        search_mode = "hybrid"

    # Create embedder based on --provider flag or legacy Ollama logic
    embedder = None
    provider_name = getattr(args, "provider", None)

    if provider_name and search_mode != "fts":
        try:
            from memory.providers import resolve_provider

            embed_provider, _ = resolve_provider(
                {"provider": {"default": provider_name}}
            )
            embedder = embed_provider
        except Exception:
            log.warning(
                "llmem: cli: --provider %s failed, falling back to FTS5-only",
                provider_name,
            )
            embedder = None
    elif search_mode != "fts":
        try:
            ollama_url = getattr(args, "ollama_url", None) or "http://localhost:11434"
            embedder = EmbeddingEngine(base_url=ollama_url)
        except Exception:
            embedder = None

    retriever = Retriever(store=store, embedder=embedder)

    try:
        results = retriever.hybrid_search(
            query=args.query,
            limit=args.limit,
            type_filter=args.type,
            search_mode=search_mode,
        )
    except ValueError:
        # semantic-only without embedder
        print("Error: semantic search requires an embedder", file=sys.stderr)
        store.close()
        sys.exit(1)

    # Interleave code chunk results if --include-code is set
    code_results: list[dict] = []
    if getattr(args, "include_code", False):
        from .code_index import CodeIndex
        from .retrieve import DEFAULT_RRF_K

        code_index = CodeIndex(db_path=args.db)
        try:
            code_results = code_index.search_content(query=args.query, limit=args.limit)
        except Exception as e:
            log.warning("llmem: cli: search: code search failed: %s", e)
        finally:
            code_index.close()

        # Merge with code results, assigning RRF scores based on FTS rank
        for i, cr in enumerate(code_results):
            cr["_source"] = "code"
            # Assign RRF score based on FTS rank position (1-based).
            # Using alpha=0.0 (pure FTS) and default k to match the FTS-only
            # scoring pattern used by the Retriever for consistency.
            fts_rank = i + 1
            cr["_rrf_score"] = (1 - 0.0) * (1 / (DEFAULT_RRF_K + fts_rank))

    # Traverse code refs if --traverse-refs is set
    ref_results: list[dict] = []
    if getattr(args, "traverse_refs", False):
        from .refs import resolve_code_ref

        max_ref_depth = getattr(args, "max_ref_depth", 3)
        # Get code ref edges from result memory IDs
        mem_ids = [r["id"] for r in results if r.get("id")]
        if mem_ids:
            code_refs = store.traverse_relations(
                mem_ids, max_depth=max_ref_depth, target_type="code"
            )
            seen_refs = set()
            for ref in code_refs:
                ref_id = ref["target_id"]
                if ref_id in seen_refs:
                    continue
                seen_refs.add(ref_id)
                resolved = resolve_code_ref(ref_id)
                if resolved is not None:
                    resolved["_source"] = "code"
                    ref_results.append(resolved)

    # Mark memory results with source
    for m in results:
        m["_source"] = "memory"

    combined = results + code_results + ref_results
    # Sort by score descending, then by source for stability
    combined.sort(key=lambda x: (-x.get("_rrf_score", 0.0), x.get("_source", "")))

    if args.json:
        print(json.dumps(combined, indent=2, default=str))
    else:
        for m in combined:
            source = m.get("_source", "memory")
            prefix = "[code]" if source == "code" else f"[{m.get('type', '?')}]"
            score = m.get("_rrf_score", 0.0)
            if source == "code":
                print(
                    f"  {m.get('id', '?')}  {prefix}  file={m.get('file_path', '?')}  "
                    f"lines={m.get('start_line', '?')}-{m.get('end_line', '?')}  rrf={score:.4f}"
                )
                print(f"    {m.get('content', '')[:120]}")
            else:
                print(
                    f"  {m['id']}  {prefix}  conf={m.get('confidence', 0):.2f}  rrf={score:.4f}"
                )
                print(f"    {m['content'][:120]}")

    store.close()


def cmd_list(args):
    store = MemoryStore(args.db)
    memories = store.list_all(type=args.type, valid_only=not args.all, limit=args.limit)
    if not memories:
        print("No memories found.")
        store.close()
        return
    for m in memories:
        valid = (
            ""
            if m.get("valid_until") is None
            else f" [EXPIRED:{m['valid_until'][:10]}]"
        )
        print(f"  {m['id']}  [{m['type']}]  conf={m.get('confidence', 0):.2f}{valid}")
        print(f"    {m['content'][:120]}")
    store.close()


def cmd_stats(args):
    store = MemoryStore(args.db)
    total = store.count()
    active = store.count(valid_only=True)
    by_type = store.count_by_type()
    print(f"Total memories: {total}")
    print(f"Active (not expired): {active}")
    print(f"Expired: {total - active}")
    print()
    print("By type:")
    for t, c in by_type.items():
        print(f"  {t}: {c}")
    store.close()


def cmd_update(args):
    store = MemoryStore(args.db)
    metadata = json.loads(args.metadata) if args.metadata else None
    ok = store.update(
        args.id,
        content=args.content,
        summary=args.summary,
        confidence=args.confidence,
        valid_until=args.valid_until,
        metadata=metadata,
    )
    if ok:
        print(f"Updated {args.id}")
    else:
        print(f"Memory {args.id} not found", file=sys.stderr)
        sys.exit(1)
    store.close()


def cmd_invalidate(args):
    store = MemoryStore(args.db)
    ok = store.invalidate(args.id, reason=args.reason)
    if ok:
        print(f"Invalidated {args.id}")
    else:
        print(f"Memory {args.id} not found", file=sys.stderr)
        sys.exit(1)
    store.close()


def cmd_delete(args):
    store = MemoryStore(args.db)
    ok = store.delete(args.id)
    if ok:
        print(f"Deleted {args.id}")
    else:
        print(f"Memory {args.id} not found", file=sys.stderr)
        sys.exit(1)
    store.close()


def cmd_export(args):
    store = MemoryStore(args.db)
    data = store.export_all(limit=None)
    out = json.dumps(data, indent=2, default=str)
    if args.output:
        output_path = _validate_write_path(Path(args.output), "export")
        output_path.write_text(out)
        print(f"Exported {len(data)} memories to {args.output}")
    else:
        print(out)
    store.close()


def cmd_import(args):
    store = MemoryStore(args.db)
    import_path = Path(args.file)
    # Validate import file path to prevent arbitrary file read
    # Uses shared _is_blocked_path from paths.py for consistency
    resolved_import = import_path.resolve()
    if _is_blocked_path(resolved_import):
        print(
            f"Error: import file targets a protected directory: {resolved_import}",
            file=sys.stderr,
        )
        sys.exit(1)
    # Enforce file size limit
    try:
        file_size = import_path.stat().st_size
    except OSError as e:
        print(f"Error: cannot stat {args.file}: {e}", file=sys.stderr)
        sys.exit(1)
    if file_size > MAX_IMPORT_FILE_SIZE:
        print(
            f"Error: import file too large ({file_size} bytes, max {MAX_IMPORT_FILE_SIZE} bytes)",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        raw = import_path.read_text()
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {args.file}: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: cannot read {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(
            "Error: import file must contain a JSON array of memory objects",
            file=sys.stderr,
        )
        sys.exit(1)

    # Schema validation: each entry must have at least 'type' and 'content'
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"Error: entry {i} is not a JSON object", file=sys.stderr)
            sys.exit(1)
        if "type" not in entry or "content" not in entry:
            print(
                f"Error: entry {i} missing required 'type' or 'content' field",
                file=sys.stderr,
            )
            sys.exit(1)
        if not isinstance(entry["type"], str) or not isinstance(entry["content"], str):
            print(
                f"Error: entry {i} 'type' and 'content' must be strings",
                file=sys.stderr,
            )
            sys.exit(1)

    count = store.import_memories(data)
    print(f"Imported {count} memories.")
    store.close()


def cmd_learn(args):
    """Ingest a codebase directory into the code index.

    Walks the directory at <path> respecting .gitignore, chunks files
    using the selected strategy, embeds each chunk, and stores them
    in the code_chunks table.

    Args:
        args: An argparse Namespace with attributes:
            - path (str): Root directory to ingest.
            - db (Path): Database path.
            - strategy (str): "paragraph" or "fixed".
            - window_size (int): Window size for fixed strategy.
            - overlap (int): Overlap for fixed strategy.
            - no_embed (bool): If True, skip embedding step.
            - ollama_url (str): Ollama base URL for embedding.
    """
    root_path = Path(args.path).resolve()
    if not root_path.is_dir():
        print(f"Error: {args.path} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Select chunking strategy
    if args.strategy == "fixed":
        chunker = FixedLineChunking(window_size=args.window_size, overlap=args.overlap)
    else:
        chunker = ParagraphChunking()

    # Walk directory for files to index (symlinks are always skipped to
    # prevent path traversal; size and depth limits prevent resource exhaustion)
    max_file_size = getattr(args, "max_file_size", None)
    if max_file_size is None:
        max_file_size = _DEFAULT_MAX_FILE_SIZE
    max_depth = getattr(args, "max_depth", None)
    if max_depth is None:
        max_depth = _DEFAULT_MAX_DEPTH

    code_files = walk_code_files(
        root_path, max_file_size=max_file_size, max_depth=max_depth
    )
    if not code_files:
        print("No code files found to index.")
        return

    # Initialize code index (shares the same database as MemoryStore)
    # NOTE: disable_vec controls sqlite-vec extension loading only.
    # --no-embed skips embedding generation but should NOT disable vec,
    # since vec triggers are needed for existing embedding searches.
    code_index = CodeIndex(db_path=args.db)
    total_chunks = 0
    total_files = 0
    embedder = None

    # Create embedder for embedding chunks
    if not hasattr(args, "no_embed") or not args.no_embed:
        try:
            from .embed import EmbeddingEngine

            ollama_url = getattr(args, "ollama_url", None) or "http://localhost:11434"
            embedder = EmbeddingEngine(base_url=ollama_url)
        except Exception as e:
            log.warning("llmem: cli: learn: embedding engine unavailable: %s", e)
            embedder = None

    try:
        for file_path in code_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                log.debug("llmem: cli: learn: cannot read %s: %s", file_path, e)
                continue

            if not content.strip():
                continue

            # Chunk the file
            rel_path = str(file_path.relative_to(root_path))
            language = detect_language(rel_path)
            chunks = chunker.chunk(rel_path, content, language=language)

            if not chunks:
                continue

            # Remove stale chunks for this file before re-inserting,
            # making cmd_learn idempotent for re-indexing.
            code_index.remove_by_path(rel_path)

            # Add chunks to the index (without embeddings first)
            chunk_ids = code_index.add_chunks(chunks)

            # Embed and update each chunk if embedder is available
            if embedder is not None:
                conn = code_index._connect()
                for chunk, chunk_id in zip(chunks, chunk_ids):
                    try:
                        vec = embedder.embed(chunk.content)

                        embedding_bytes = EmbeddingEngine.vec_to_bytes(vec)
                        conn.execute(
                            'UPDATE "code_chunks" SET "embedding" = ? WHERE "id" = ?',
                            (embedding_bytes, chunk_id),
                        )
                    except Exception as e:
                        log.warning(
                            "llmem: cli: learn: failed to embed chunk %s: %s",
                            chunk_id,
                            e,
                        )
                conn.commit()

            total_chunks += len(chunks)
            total_files += 1

        print(f"Ingested {total_chunks} chunks from {total_files} files")
    finally:
        code_index.close()


def cmd_register_type(args):
    try:
        register_memory_type(args.type_name)
        print(f"Registered type '{args.type_name}'")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_types(args):
    """List all registered memory types."""
    types = sorted(get_registered_types())
    print("Registered memory types:")
    for t in types:
        print(f"  {t}")


def cmd_note(args):
    """Add content to the working memory inbox.

    Args:
        args: argparse Namespace with attributes:
            - content (str): The note content text.
            - source (str): Source of the note. Default: 'note'.
            - attention_score (float): Attention score 0-1. Default: 0.5.
            - metadata (str or None): JSON metadata string.

    Prints the new inbox item ID and exits with 0 on success.
    Prints to stderr and exits with 1 on validation error.
    """
    store = MemoryStore(args.db)
    src = args.source
    score = args.attention_score
    metadata = json.loads(args.metadata) if args.metadata else None

    if src not in ("note", "learn", "extract", "consolidation"):
        print(
            f"Error: invalid source '{src}'. Must be note, learn, extract, or consolidation",
            file=sys.stderr,
        )
        store.close()
        sys.exit(1)

    if score < 0.0 or score > 1.0:
        print(
            f"Error: attention-score must be between 0.0 and 1.0, got {score}",
            file=sys.stderr,
        )
        store.close()
        sys.exit(1)

    try:
        inbox_id = store.add_to_inbox(
            content=args.content,
            source=src,
            attention_score=score,
            metadata=metadata,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        store.close()
        sys.exit(1)

    print(f"Added to inbox {inbox_id} [source={src}] score={score}")
    store.close()


def cmd_inbox(args):
    """List items in the working memory inbox.

    Args:
        args: argparse Namespace with attributes:
            - limit (int): Maximum items to show. Default: 20.
            - json (bool): If True, output full JSON.

    Prints each inbox item's id, source, attention_score, content (truncated),
    and created_at. With --json, outputs full JSON.
    """
    store = MemoryStore(args.db)
    items = store.list_inbox(limit=args.limit)
    if args.json:
        print(json.dumps(items, indent=2, default=str))
    else:
        if not items:
            print("Inbox is empty.")
        else:
            for item in items:
                content_preview = item["content"][:120]
                print(
                    f"  {item['id']}  [source={item['source']}]  "
                    f"score={item['attention_score']:.2f}  "
                    f"{content_preview}  {item['created_at']}"
                )
    store.close()


def cmd_embed(args):
    """Report embedding quality metrics for existing embeddings.

    Computes anisotropy, similarity range, and discrimination gap
    from the embeddings already stored in the database. Does NOT
    generate new embeddings — only analyses existing ones.

    Since the sole purpose of the embed subcommand is to report
    metrics, they are always reported (no --metrics flag needed).

    Args:
        args: argparse Namespace with attributes:
            - db (Path): Database path.

    Prints metric values to stdout. Emits warnings if anisotropy
    exceeds the threshold or similarity range is below the threshold.
    """
    store = MemoryStore(args.db)
    _report_embedding_metrics(store)
    store.close()


def cmd_consolidate(args):
    """Promote inbox items to long-term memory.

    Args:
        args: argparse Namespace with attributes:
            - min_score (float): Minimum attention_score for promotion. Default: 0.0.
            - dry_run (bool): If True, show what would happen without making changes.
            - metrics (bool): If True, compute and report embedding metrics after
              consolidation.

    Prints the number of promoted and evicted items.
    """
    store = MemoryStore(args.db)
    result = store.consolidate(min_score=args.min_score, dry_run=args.dry_run)

    prefix = "[DRY RUN] " if args.dry_run else ""
    promoted = result["promoted"]
    evicted = result["evicted"]

    print(f"{prefix}Promoted: {len(promoted)} items")
    print(f"{prefix}Evicted: {len(evicted)} items")

    for item in promoted:
        content_preview = item["content"][:80]
        mem_id = item.get("memory_id", "?")
        print(f"{prefix}  promoted: {item['id']} -> {mem_id}  {content_preview}")

    for item in evicted:
        content_preview = item["content"][:80]
        print(f"{prefix}  evicted: {item['id']}  {content_preview}")

    # Report embedding metrics if requested
    if getattr(args, "metrics", False):
        print()
        _report_embedding_metrics(store)

    store.close()


def cmd_init(args):
    """Initialize the llmem memory system: config, database, and provider detection.

    Creates ``~/.config/llmem/`` (or ``LMEM_HOME``) with ``config.yaml``
    and initializes the SQLite database (``memory.db``). Detects available
    LLM providers (Ollama, OpenAI, Anthropic).

    In interactive mode (default), prompts the user for configuration
    values with sensible defaults shown in brackets. In non-interactive
    mode (``--non-interactive``), uses all defaults without prompting.

    Idempotent: if ``config.yaml`` already exists and ``--force`` is not
    given, prints a message and returns without error.

    Args:
        args: An argparse Namespace with attributes:
            - ollama_url (str or None): Override Ollama URL.
            - non_interactive (bool): Skip prompts, use defaults.
            - force (bool): Overwrite existing config.yaml.
            - db (Path or None): Not used by init (always creates default).

    Returns:
        None. Prints to stdout on success, or stderr + sys.exit(1) on error.
    """
    detector = ProviderDetector()
    ollama_url = args.ollama_url or "http://localhost:11434"

    # Detect providers
    try:
        detection = detector.detect(ollama_url=ollama_url)
    except ValueError as e:
        print(f"Error: invalid Ollama URL {ollama_url!r}: {e}", file=sys.stderr)
        sys.exit(1)

    # Build config dict
    config = {
        "memory": {
            "ollama_url": detection["ollama_url"],
            "embed_model": "nomic-embed-text",
            "extract_model": "qwen2.5:1.5b",
        },
        "dream": {
            "enabled": True,
        },
    }

    if detection["provider"] != "none":
        config["memory"]["provider"] = detection["provider"]

    # Interactive prompts (unless --non-interactive)
    if not args.non_interactive:
        try:
            url_input = input(f"Ollama URL [{detection['ollama_url']}]: ").strip()
            if url_input:
                if not url_input.startswith(("http://", "https://")) or not is_safe_url(
                    url_input, allow_remote=True
                ):
                    print(
                        f"Error: invalid or unsafe Ollama URL: {url_input!r} "
                        "(must be http:// or https:// with a valid hostname)",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                config["memory"]["ollama_url"] = url_input

            dream_default = "Y"
            dream_input = (
                input(f"Enable dream cycle? [{dream_default}]: ").strip().lower()
            )
            if dream_input not in ("", "y", "yes"):
                config["dream"]["enabled"] = False
        except KeyboardInterrupt:
            print("\nInit cancelled.")
            sys.exit(1)

    # Migrate from ~/.lobsterdog/ if it exists
    # Must run before resolving paths so get_llmem_home() returns the
    # new location (~/.config/llmem/) after migration rather than the
    # legacy path (~/.lobsterdog/).
    old_home = Path.home() / ".lobsterdog"
    migrated = False
    if old_home.exists():
        migrated = migrate_from_lobsterdog()

    # Resolve home AFTER migration so the path matches config/db paths
    home = get_llmem_home()

    if migrated:
        print(f"Migrated data from {old_home} to {home}")

    # Write config.yaml
    config_path = get_config_path()
    written = write_config_yaml(config_path, config, force=args.force)
    if not written and not args.force:
        print(f"Config already exists at {config_path}. Use --force to overwrite.")
        print("Skipping config write. Database will be verified.")

    # Initialize the database via MemoryStore (runs migrations)
    db_path = get_db_path()
    try:
        store = MemoryStore(db_path=db_path, disable_vec=True)
        store.close()
    except Exception as e:
        print(f"Error: failed to initialize database: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Initialized llmem in {home}")
    print(f"  Config: {config_path}")
    print(f"  Database: {db_path}")
    print(f"  Provider: {detection['provider']}")


def cmd_context(args):
    """Inject relevant memory context for a session.

    Used by session hooks (e.g., Copilot CLI, OpenCode plugins) to inject
    memories into a new or compacting session. Writes a context file and
    prints its content to stdout.

    Args:
        args: An argparse Namespace with attributes:
            - session_id (str): The session ID to inject context for.
            - compacting (bool): If True, inject key memories for compaction.
            - db (Path): Database path.

    Prints the context content to stdout on success.
    Prints to stderr and exits with 1 on validation or coordinator error.
    """
    from .session_hooks import create_session_hook_coordinator, SESSION_CREATED_SUCCESS
    from .session_hooks import (
        SESSION_COMPACTING_SUCCESS,
        SESSION_COMPACTING_NO_MEMORIES,
        SESSION_COMPACTING_ERROR,
        SESSION_CREATED_ALREADY_PROCESSED,
        SESSION_CREATED_ERROR,
    )
    from .paths import validate_session_id, get_context_dir

    try:
        validate_session_id(args.session_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        coordinator = create_session_hook_coordinator()
    except Exception as e:
        log.error("llmem: cli: context: failed to create coordinator: %s", e)
        print(f"Error: llmem: context: failed to initialize: {e}", file=sys.stderr)
        sys.exit(1)

    if args.compacting:
        result_type, context_file = coordinator.on_compacting(args.session_id)
        if result_type == SESSION_COMPACTING_SUCCESS and context_file:
            try:
                print(Path(context_file).read_text())
            except OSError as e:
                log.error("llmem: cli: context: failed to read compact context: %s", e)
                print(
                    f"Error: llmem: context: failed to read context file: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif result_type == SESSION_COMPACTING_NO_MEMORIES:
            log.debug(
                "llmem: cli: context: no key memories for session %s",
                args.session_id,
            )
            # Print nothing — no key memories to inject
        elif result_type == SESSION_COMPACTING_ERROR:
            print(
                f"Error: llmem: context: compacting hook failed for session "
                f"{args.session_id}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        result_type, context_file = coordinator.on_created(args.session_id)
        if result_type == SESSION_CREATED_SUCCESS and context_file:
            try:
                print(Path(context_file).read_text())
            except OSError as e:
                log.error("llmem: cli: context: failed to read context: %s", e)
                print(
                    f"Error: llmem: context: failed to read context file: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif result_type == SESSION_CREATED_ALREADY_PROCESSED:
            log.debug(
                "llmem: cli: context: session %s already processed",
                args.session_id,
            )
            # Re-read existing context file from disk if available
            context_dir = get_context_dir()
            existing_file = context_dir / f"{args.session_id}.md"
            if existing_file.exists():
                try:
                    print(existing_file.read_text())
                except OSError:
                    pass
        elif result_type == SESSION_CREATED_ERROR:
            print(
                f"Error: llmem: context: created hook failed for session "
                f"{args.session_id}",
                file=sys.stderr,
            )
            sys.exit(1)


def cmd_hook(args):
    """Handle session lifecycle hook events.

    Delegates to the appropriate session hook handler based on the
    subcommand (e.g., 'idle' triggers memory extraction and introspection).

    Args:
        args: An argparse Namespace with attributes:
            - hook_type (str): The hook type ('idle').
            - session_id (str): The session ID.
            - db (Path): Database path.

    Prints extraction summary on stdout for 'idle' hook.
    Prints to stderr and exits with 1 on validation error.
    """
    from .session_hooks import (
        create_session_hook_coordinator,
        SESSION_IDLE_DEBOUNCED,
        SESSION_IDLE_NO_TRANSCRIPT,
    )
    from .paths import validate_session_id

    try:
        validate_session_id(args.session_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.hook_type == "idle":
        try:
            coordinator = create_session_hook_coordinator()
        except Exception as e:
            log.error("llmem: cli: hook: idle: failed to create coordinator: %s", e)
            print(
                f"Error: llmem: hook: idle: failed to initialize: {e}", file=sys.stderr
            )
            sys.exit(1)

        result_type, count = coordinator.on_idle(args.session_id)
        if result_type == SESSION_IDLE_DEBOUNCED:
            log.debug(
                "llmem: cli: hook: idle: debounced for session %s",
                args.session_id,
            )
        elif result_type == SESSION_IDLE_NO_TRANSCRIPT:
            log.debug(
                "llmem: cli: hook: idle: no transcript for session %s",
                args.session_id,
            )
        else:
            log.info(
                "llmem: cli: hook: idle: %s (%d memories) for session %s",
                result_type,
                count,
                args.session_id,
            )
    else:
        print(
            f"Error: unknown hook type '{args.hook_type}'. Supported: idle",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_track_review(args):
    """Persist review findings as self_assessment memories.

    Three modes:
    - Single finding: --category + --what-happened (optionally --severity, --caught-by)
    - Batch from file: --finding-file (JSON array of finding objects)
    - Clean review: no flags → creates a REVIEW_PASSED memory

    --category and --finding-file are mutually exclusive.
    Every invocation MUST produce at least one memory.

    Args:
        args: An argparse Namespace with attributes:
            - context (str|None): File/task identifier.
            - category (str|None): Error taxonomy category for single finding.
            - what_happened (str|None): Behavioral description for single finding.
            - severity (str|None): Severity tier (Blocking, Required, etc.).
            - caught_by (str|None): How the finding was discovered.
            - finding_file (str|None): Path to JSON file with findings array.
            - db (Path): Database path.
    """
    from .taxonomy import ERROR_TAXONOMY

    store = MemoryStore(args.db)

    if args.category and args.finding_file:
        print(
            "Error: --category and --finding-file are mutually exclusive",
            file=sys.stderr,
        )
        store.close()
        sys.exit(1)

    if args.category:
        # Single finding mode
        category = args.category
        if category not in ERROR_TAXONOMY:
            print(
                f"Error: unknown category '{category}'. "
                f"Valid categories: {', '.join(ERROR_TAXONOMY.keys())}",
                file=sys.stderr,
            )
            store.close()
            sys.exit(1)

        if not args.what_happened:
            print("Error: --what-happened is required with --category", file=sys.stderr)
            store.close()
            sys.exit(1)

        # Build structured content
        content_lines = [f"Category: {category}"]
        if args.context:
            content_lines.append(f"Context: {args.context}")
        content_lines.append(f"What_happened: {args.what_happened}")
        content_lines.append(
            f"Outcomes: {args.severity} finding"
            if args.severity
            else "Outcomes: finding"
        )
        if args.caught_by:
            content_lines.append(f"What_caught_it: {args.caught_by}")
        else:
            content_lines.append("What_caught_it: self-review")
        content_lines.append("Recurring: no")
        content = "\n".join(content_lines)

        mid = store.add(
            type="self_assessment",
            content=content,
            source="review_tracker",
            confidence=0.9,
        )
        print(f"Added self_assessment memory {mid} [{category}]")

    elif args.finding_file:
        # Batch mode from file
        finding_path = Path(args.finding_file)
        if not finding_path.exists():
            print(
                f"Error: llmem: track-review: finding file not found: {finding_path}",
                file=sys.stderr,
            )
            store.close()
            sys.exit(1)

        try:
            findings = json.loads(finding_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"Error: llmem: track-review: failed to read finding file: {e}",
                file=sys.stderr,
            )
            store.close()
            sys.exit(1)

        if not isinstance(findings, list):
            print(
                "Error: llmem: track-review: finding file must contain a JSON array",
                file=sys.stderr,
            )
            store.close()
            sys.exit(1)

        for finding in findings:
            category = finding.get("category", "MISSING_VERIFICATION")
            if category not in ERROR_TAXONOMY:
                log.warning(
                    "llmem: track-review: unknown category '%s', using MISSING_VERIFICATION",
                    category,
                )
                category = "MISSING_VERIFICATION"

            what_happened = finding.get(
                "what_happened", finding.get("whatHappened", "review finding")
            )
            severity = finding.get("severity", "")

            content_lines = [f"Category: {category}"]
            if args.context:
                content_lines.append(f"Context: {args.context}")
            content_lines.append(f"What_happened: {what_happened}")
            if severity:
                content_lines.append(f"Outcomes: {severity} finding")
            content_lines.append("What_caught_it: self-review")
            content_lines.append("Recurring: no")
            content = "\n".join(content_lines)

            mid = store.add(
                type="self_assessment",
                content=content,
                source="review_tracker",
                confidence=0.9,
            )
            print(f"Added self_assessment memory {mid} [{category}]")

    else:
        # Clean review — create REVIEW_PASSED memory
        content_lines = ["Category: REVIEW_PASSED"]
        if args.context:
            content_lines.append(f"Context: {args.context}")
        content_lines.append("What_happened: clean review — no findings")
        content_lines.append("Outcomes: all clear")
        content_lines.append("What_caught_it: self-review")
        content_lines.append("Recurring: no")
        content = "\n".join(content_lines)

        mid = store.add(
            type="self_assessment",
            content=content,
            source="review_tracker",
            confidence=0.9,
        )
        print(f"Added self_assessment memory {mid} [REVIEW_PASSED]")

    store.close()


def cmd_suggest_categories(args):
    """List error taxonomy categories applicable to a severity tier.

    Args:
        args: An argparse Namespace with attributes:
            - tier (str): Severity tier name (Blocking, Required, etc.).

    Prints one category per line for the given tier.
    """
    from .taxonomy import REVIEW_SEVERITY_TAXONOMY

    categories = REVIEW_SEVERITY_TAXONOMY.get(args.tier, [])
    for cat in categories:
        print(cat)


def main():
    """Entry point for the llmem CLI."""
    # Backward-compat: warn when invoked as 'lobmem'
    invoked_as = os.path.basename(sys.argv[0])
    if invoked_as == "lobmem":
        print(
            "warning: 'lobmem' is deprecated, use 'llmem'",
            file=sys.stderr,
        )

    parser = argparse.ArgumentParser(
        prog="llmem",
        description="LLMem — structured memory with semantic search",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to memory database (default: ~/.config/llmem/memory.db)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # add
    p_add = subparsers.add_parser("add", help="Add a memory")
    p_add.add_argument(
        "--type",
        required=True,
        help="Memory type (use 'llmem types' to list, or register new with 'llmem register-type')",
    )
    p_add.add_argument("--content", help="Memory content text")
    p_add.add_argument("--file", help="Read content from file")
    p_add.add_argument("--summary", help="Short summary")
    p_add.add_argument("--source", default="manual", help="Source (default: manual)")
    p_add.add_argument("--confidence", type=float, default=0.8, help="Confidence 0-1")
    p_add.add_argument("--valid-until", help="Expiry timestamp")
    p_add.add_argument("--metadata", help="JSON metadata")
    p_add.add_argument("--relation", help="Relation type")
    p_add.add_argument("--relation-to", help="Target memory ID for relation")

    # get
    p_get = subparsers.add_parser("get", help="Get a memory by ID")
    p_get.add_argument("id", help="Memory ID")

    # search
    p_search = subparsers.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--type", help="Filter by type")
    p_search.add_argument("--limit", type=int, default=20, help="Max results")
    p_search.add_argument("--json", action="store_true", help="JSON output")
    p_search.add_argument(
        "--include-code",
        action="store_true",
        help="Include code chunks in search results",
    )
    p_search.add_argument(
        "--traverse-refs",
        action="store_true",
        help="Follow code reference edges from search results",
    )
    p_search.add_argument(
        "--max-ref-depth",
        type=int,
        default=3,
        help="Max depth for ref expansion (default: 3)",
    )
    p_search.add_argument(
        "--provider",
        choices=["ollama", "openai", "local", "none"],
        help="Embedding provider to use (overrides config-based selection)",
    )
    search_mode_group = p_search.add_mutually_exclusive_group()
    search_mode_group.add_argument(
        "--fts-only", action="store_true", help="FTS5 keyword search only"
    )
    search_mode_group.add_argument(
        "--semantic-only", action="store_true", help="Semantic (embedding) search only"
    )

    # list
    p_list = subparsers.add_parser("list", help="List memories")
    p_list.add_argument("--type", help="Filter by type")
    p_list.add_argument("--all", action="store_true", help="Include expired")
    p_list.add_argument("--limit", type=int, default=100, help="Max results")

    # stats
    subparsers.add_parser("stats", help="Show memory statistics")

    # update
    p_update = subparsers.add_parser("update", help="Update a memory")
    p_update.add_argument("id", help="Memory ID")
    p_update.add_argument("--content", help="New content")
    p_update.add_argument("--summary", help="New summary")
    p_update.add_argument("--confidence", type=float, help="New confidence")
    p_update.add_argument("--valid-until", help="New expiry")
    p_update.add_argument("--metadata", help="New JSON metadata")

    # invalidate
    p_inv = subparsers.add_parser("invalidate", help="Invalidate a memory")
    p_inv.add_argument("id", help="Memory ID")
    p_inv.add_argument("--reason", help="Invalidation reason")

    # delete
    p_del = subparsers.add_parser("delete", help="Delete a memory")
    p_del.add_argument("id", help="Memory ID")

    # export
    p_export = subparsers.add_parser("export", help="Export all memories")
    p_export.add_argument("--output", help="Output file path")

    # import
    p_import = subparsers.add_parser("import", help="Import memories from JSON")
    p_import.add_argument("file", help="JSON file to import")

    # register-type
    p_reg = subparsers.add_parser("register-type", help="Register a new memory type")
    p_reg.add_argument("type_name", help="Type name to register")

    # types
    subparsers.add_parser("types", help="List registered memory types")

    # init
    p_init = subparsers.add_parser("init", help="Initialize the llmem memory system")
    p_init.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL (default: http://localhost:11434)",
    )
    p_init.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip prompts and use defaults",
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config.yaml",
    )

    # note
    p_note = subparsers.add_parser(
        "note", help="Add a note to the working memory inbox"
    )
    p_note.add_argument("content", help="Note content text")
    p_note.add_argument(
        "--source",
        default="note",
        help="Source of the note (default: note)",
    )
    p_note.add_argument(
        "--attention-score",
        type=float,
        default=0.5,
        help="Attention score 0-1 (default: 0.5)",
    )
    p_note.add_argument("--metadata", help="JSON metadata")

    # inbox
    p_inbox = subparsers.add_parser(
        "inbox", help="List items in the working memory inbox"
    )
    p_inbox.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum items to show (default: 20)",
    )
    p_inbox.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # embed
    subparsers.add_parser("embed", help="Report embedding quality metrics")

    # consolidate
    p_consolidate = subparsers.add_parser(
        "consolidate", help="Promote inbox items to long-term memory"
    )
    p_consolidate.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum attention score for promotion (default: 0.0)",
    )
    p_consolidate.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )
    p_consolidate.add_argument(
        "--metrics",
        action="store_true",
        help="Compute and report embedding quality metrics after consolidation",
    )

    # learn
    p_learn = subparsers.add_parser(
        "learn", help="Ingest a codebase into the code index"
    )
    p_learn.add_argument("path", help="Root directory to ingest")
    p_learn.add_argument(
        "--strategy",
        choices=["paragraph", "fixed"],
        default="paragraph",
        help="Chunking strategy (default: paragraph)",
    )
    p_learn.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Window size for fixed-line chunking (default: 50)",
    )
    p_learn.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Overlap for fixed-line chunking (default: 10)",
    )
    p_learn.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding generation (store chunks only)",
    )
    p_learn.add_argument(
        "--max-file-size",
        type=int,
        default=None,
        help="Maximum file size in bytes to index (default: 1048576 = 1 MiB)",
    )
    p_learn.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum directory recursion depth (default: 50)",
    )
    p_learn.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL (default: http://localhost:11434)",
    )

    # track-review
    p_track_review = subparsers.add_parser(
        "track-review",
        help="Persist review findings as self_assessment memories",
    )
    p_track_review.add_argument(
        "--context",
        help="File or task identifier (e.g. 'handler.py:42')",
    )
    p_track_review.add_argument(
        "--category",
        help="Error taxonomy category for a single finding (e.g. NULL_SAFETY)",
    )
    p_track_review.add_argument(
        "--what-happened",
        help="Behavioral description of the finding",
    )
    p_track_review.add_argument(
        "--severity",
        help="Severity tier (Blocking, Required, Strong Suggestions, Noted)",
    )
    p_track_review.add_argument(
        "--caught-by",
        help="How the finding was discovered (e.g. self-review, CI)",
    )
    p_track_review.add_argument(
        "--finding-file",
        help="Path to a JSON file with an array of finding objects",
    )

    # suggest-categories
    p_suggest_categories = subparsers.add_parser(
        "suggest-categories",
        help="List error taxonomy categories for a severity tier",
    )
    p_suggest_categories.add_argument(
        "tier",
        choices=["Blocking", "Required", "Strong Suggestions", "Noted", "Passed"],
        help="Severity tier to list categories for",
    )

    # context
    p_context = subparsers.add_parser(
        "context",
        help="Inject relevant memory context for a session",
    )
    p_context.add_argument(
        "session_id",
        help="Session ID to inject context for",
    )
    p_context.add_argument(
        "--compacting",
        action="store_true",
        help="Inject key memories for compaction instead of session start context",
    )

    # hook
    p_hook = subparsers.add_parser(
        "hook",
        help="Handle session lifecycle hook events",
    )
    p_hook.add_argument(
        "hook_type",
        choices=["idle"],
        help="Hook type to dispatch",
    )
    p_hook.add_argument(
        "session_id",
        help="Session ID for the hook event",
    )

    # Register CLI plugins
    for plugin_name in sorted(get_registered_cli_plugins()):
        from .registry import get_cli_plugin_setup_fn

        setup_fn = get_cli_plugin_setup_fn(plugin_name)
        if setup_fn is not None:
            try:
                setup_fn(subparsers)
            except Exception as e:
                log.error("llmem: cli: plugin '%s' setup failed: %s", plugin_name, e)

    args = parser.parse_args()

    # Resolve db path using get_db_path() if not provided
    if not hasattr(args, "db") or args.db is None:
        args.db = get_db_path()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "add": cmd_add,
        "get": cmd_get,
        "search": cmd_search,
        "list": cmd_list,
        "stats": cmd_stats,
        "update": cmd_update,
        "invalidate": cmd_invalidate,
        "delete": cmd_delete,
        "export": cmd_export,
        "import": cmd_import,
        "register-type": cmd_register_type,
        "types": cmd_types,
        "init": cmd_init,
        "note": cmd_note,
        "inbox": cmd_inbox,
        "embed": cmd_embed,
        "consolidate": cmd_consolidate,
        "learn": cmd_learn,
        "context": cmd_context,
        "hook": cmd_hook,
        "track-review": cmd_track_review,
        "suggest-categories": cmd_suggest_categories,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
