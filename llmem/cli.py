"""llmem — LLMem memory management CLI."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .store import MemoryStore, get_registered_types
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
from .config import write_config_yaml, get_dream_config, get_ollama_url
from .ollama import ProviderDetector
from .paths import migrate_from_lobsterdog
from .url_validate import is_safe_url

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

    embedder = None
    if search_mode != "fts":
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

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for m in results:
            prefix = f"[{m.get('type', '?')}]"
            score = m.get("_rrf_score", 0.0)
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


def cmd_dream(args):
    """Run the dream consolidation cycle.

    Args:
        args: An argparse Namespace with attributes:
            - db (Path or None): Database path. Resolved via get_db_path() if None.
            - apply (bool): If True, apply changes. If False, dry run.
            - phase (str or None): Run a specific phase ('light', 'deep', 'rem'),
              or None for all phases.
            - report (str or None): Path to write an HTML dream report.

    Prints a summary of dream phase results to stdout.
    On --report path validation errors, prints to stderr and exits with 1.
    """
    from .dream import Dreamer
    from .dream_report import generate_dream_report

    # Resolve DB path if not provided
    if not hasattr(args, "db") or args.db is None:
        args.db = get_db_path()

    # Resolve dream config
    dream_config = get_dream_config()

    # Resolve ollama_url from the memory section (not dream section)
    try:
        ollama_url = get_ollama_url()
    except ValueError:
        ollama_url = "http://localhost:11434"

    store = MemoryStore(args.db, disable_vec=True)

    try:
        # Construct Dreamer from config
        dreamer = Dreamer(
            store=store,
            similarity_threshold=dream_config.get("similarity_threshold", 0.92),
            decay_rate=dream_config.get("decay_rate", 0.05),
            decay_interval_days=dream_config.get("decay_interval_days", 30),
            decay_floor=dream_config.get("decay_floor", 0.3),
            confidence_floor=dream_config.get("confidence_floor", 0.3),
            boost_threshold=dream_config.get("boost_threshold", 5),
            boost_amount=dream_config.get("boost_amount", 0.05),
            min_score=dream_config.get("min_score", 0.5),
            min_recall_count=dream_config.get("min_recall_count", 3),
            min_unique_queries=dream_config.get("min_unique_queries", 1),
            boost_on_promote=dream_config.get("boost_on_promote", 0.1),
            merge_model=dream_config.get("merge_model", "qwen2.5:1.5b"),
            ollama_url=ollama_url,
            behavioral_threshold=dream_config.get("behavioral_threshold", 3),
            behavioral_lookback_days=dream_config.get("behavioral_lookback_days", 30),
            auto_link_threshold=dream_config.get("auto_link_threshold", 0.85),
        )

        result = dreamer.run(apply=args.apply, phase=args.phase)

        prefix = "[DRY RUN] " if not args.apply else ""

        # Print phase-specific output
        if result.light is not None:
            print(f"{prefix}Light phase:")
            print(f"{prefix}  Duplicate pairs: {result.light.duplicate_pairs}")

        if result.deep is not None:
            print(f"{prefix}Deep phase:")
            print(
                f"{prefix}  Decayed: {result.deep.decayed_count}  "
                f"Boosted: {result.deep.boosted_count}  "
                f"Promoted: {result.deep.promoted_count}  "
                f"Invalidated: {result.deep.invalidated_count}  "
                f"Merged: {result.deep.merged_count}  "
                f"Auto-linked: {result.deep.auto_linked_count}"
            )

        if result.rem is not None:
            print(f"{prefix}REM phase:")
            print(
                f"{prefix}  Total memories: {result.rem.total_memories}  "
                f"Active: {result.rem.active_memories}"
            )
            if result.rem.behavioral_insights:
                print(f"{prefix}  Behavioral insights:")
                for insight in result.rem.behavioral_insights:
                    cat = insight.get("category", "?")
                    count = insight.get("count", 0)
                    iid = insight.get("insight_id", "not written")
                    print(
                        f"{prefix}    {cat}: {count} occurrences (id: {iid})"
                    )

        # Generate report if requested
        if args.report:
            try:
                report_path = Path(args.report)
                generate_dream_report(result, report_path)
                print(f"Report written to {report_path}")
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
    finally:
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

    # Detect session adapter automatically
    opencode_db = Path("~/.local/share/opencode/opencode.db").expanduser()
    if opencode_db.exists():
        config["session"] = {"adapter": "opencode"}
    else:
        config["session"] = {"adapter": "none"}

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

            # Detect session adapter
            adapter_default = "opencode"
            opencode_db = Path(
                "~/.local/share/opencode/opencode.db"
            ).expanduser()
            if not opencode_db.exists():
                adapter_default = "none"
            adapter_input = (
                input(
                    f"Session adapter (opencode/none) [{adapter_default}]: "
                )
                .strip()
                .lower()
            )
            if adapter_input in ("opencode", "none"):
                adapter_type = adapter_input
            elif adapter_input == "":
                adapter_type = adapter_default
            else:
                adapter_type = adapter_default
            config["session"] = {"adapter": adapter_type}
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
    adapter_type = config.get("session", {}).get("adapter", "opencode")
    print(f"  Session adapter: {adapter_type}")


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
                except OSError as e:
                    log.debug(
                        "llmem: cli: context: failed to read existing context file: %s",
                        e,
                    )
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
    elif args.hook_type == "ending":
        try:
            coordinator = create_session_hook_coordinator()
        except Exception as e:
            log.error("llmem: cli: hook: ending: failed to create coordinator: %s", e)
            print(
                f"Error: llmem: hook: ending: failed to initialize: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        from .session_hooks import SESSION_ENDING_NO_TRANSCRIPT

        result_type, count = coordinator.on_ending(args.session_id)
        if result_type == SESSION_ENDING_NO_TRANSCRIPT:
            log.debug(
                "llmem: cli: hook: ending: no transcript for session %s",
                args.session_id,
            )
        else:
            log.info(
                "llmem: cli: hook: ending: %s (%d total) for session %s",
                result_type,
                count,
                args.session_id,
            )
    else:
        print(
            f"Error: unknown hook type '{args.hook_type}'. Supported: idle, ending",
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

        if len(findings) == 0:
            # Empty array — semantically equivalent to a clean review.
            # Satisfies the docstring contract: every invocation MUST produce at
            # least one memory.
            content_lines = ["Category: REVIEW_PASSED"]
            if args.context:
                content_lines.append(f"Context: {args.context}")
            content_lines.append("What_happened: clean review — no findings")
            content_lines.append("Outcomes: all clear")
            caught_by = args.caught_by or "self-review"
            content_lines.append(f"What_caught_it: {caught_by}")
            content_lines.append("Recurring: no")
            content = "\n".join(content_lines)

            mid = store.add(
                type="self_assessment",
                content=content,
                source="review_tracker",
                confidence=0.9,
            )
            print(f"Added self_assessment memory {mid} [REVIEW_PASSED]")

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
            caught_by = args.caught_by or finding.get("caughtBy", "self-review")
            content_lines.append(f"What_caught_it: {caught_by}")
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
        caught_by = args.caught_by or "self-review"
        content_lines.append(f"What_caught_it: {caught_by}")
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

    # embed
    subparsers.add_parser("embed", help="Report embedding quality metrics")

    # dream
    p_dream = subparsers.add_parser("dream", help="Run dream consolidation cycle")
    p_dream.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry run)",
    )
    p_dream.add_argument(
        "--phase",
        choices=["light", "deep", "rem"],
        default=None,
        help="Run a specific dream phase (default: all phases)",
    )
    p_dream.add_argument(
        "--report",
        default=None,
        help="Path to write an HTML dream report",
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
        choices=["idle", "ending"],
        help="Hook type to dispatch (idle: extract memories, ending: extract + introspect)",
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
        "init": cmd_init,
        "embed": cmd_embed,
        "dream": cmd_dream,
        "context": cmd_context,
        "hook": cmd_hook,
        "track-review": cmd_track_review,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
