"""llmem — LLMem memory management CLI."""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

log = logging.getLogger(__name__)

from .store import MemoryStore, register_memory_type, get_registered_types
from .paths import get_db_path, get_config_path


VALID_SOURCES = ["manual", "session", "heartbeat", "extraction", "import"]


def cmd_add(args):
    store = MemoryStore(args.db)
    content = args.content
    if not content and args.file:
        content = Path(args.file).read_text().strip()
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

    store = MemoryStore(args.db)

    if args.json:
        results = store.search(args.query, limit=args.limit, type=args.type)
        print(json.dumps(results, indent=2, default=str))
    else:
        results = store.search(args.query, limit=args.limit, type=args.type)
        for m in results:
            print(f"  {m['id']}  [{m['type']}]  conf={m.get('confidence', 0):.2f}")
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
    data = store.export_all()
    out = json.dumps(data, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(out)
        print(f"Exported {len(data)} memories to {args.output}")
    else:
        print(out)
    store.close()


def cmd_import(args):
    store = MemoryStore(args.db)
    try:
        raw = Path(args.file).read_text()
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


def main():
    """Entry point for the llmem CLI."""
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
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
