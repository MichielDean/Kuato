"""OpenCode session database adapter.

Reads session transcripts from the opencode SQLite database
(~/.local/share/opencode/opencode.db) and reconstructs them as
markdown for the llmem hook/introspect pipeline.

This adapter does NOT contain any pipeline detection logic.
Pipeline-specific filtering is an extension concern that should
be implemented via a custom SessionAdapter subclass.
"""

import json
import logging
import sqlite3
from pathlib import Path

from .base import SessionAdapter
from ..paths import BLOCKED_SYSTEM_PREFIXES

log = logging.getLogger(__name__)

OPENCODE_SESSION_SOURCE_TYPE = "opencode_session"


def _create_opencode_schema(conn: sqlite3.Connection) -> None:
    """Create minimal opencode-compatible tables for testing.

    Creates the session, message, and part tables with the schema
    that the real opencode database uses, so tests can verify against
    realistic data.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS "session" (
            "id" TEXT PRIMARY KEY,
            "title" TEXT NOT NULL DEFAULT '',
            "slug" TEXT NOT NULL DEFAULT '',
            "directory" TEXT NOT NULL DEFAULT '',
            "time_created" INTEGER NOT NULL DEFAULT 0,
            "time_updated" INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS "message" (
            "id" TEXT PRIMARY KEY,
            "session_id" TEXT NOT NULL,
            "role" TEXT NOT NULL,
            "data" TEXT NOT NULL DEFAULT '{}',
            "time_created" INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS "part" (
            "id" TEXT PRIMARY KEY,
            "message_id" TEXT NOT NULL,
            "type" TEXT NOT NULL,
            "data" TEXT NOT NULL DEFAULT '{}',
            "time_created" INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()


def session_to_markdown(parts: list[dict]) -> str:
    """Convert a list of opencode part dicts to a markdown transcript.

    Each part dict is expected to have a 'type' key and a 'data' key
    (containing JSON). Supported part types:

    - text: rendered as **<role>**: <text> if role is available,
      or just the text content if not.
    - tool: rendered with tool name and truncated output.
    - reasoning: rendered with reasoning text content.
    - step-start, step-finish, compaction: skipped (no content contribution).

    Args:
        parts: List of dicts, each with at least 'type' and 'data' keys.
            The 'data' key should contain a JSON string or dict.

    Returns:
        A markdown transcript string. Returns an empty string if no
        renderable parts are found. Never raises on malformed JSON —
        invalid parts are skipped with log.debug.
    """
    lines: list[str] = []

    for part in parts:
        part_type = part.get("type", "")
        raw_data = part.get("data", "{}")

        # Parse data if it's a string
        if isinstance(raw_data, str):
            try:
                data = json.loads(raw_data)
            except (json.JSONDecodeError, TypeError):
                log.debug("opencode: skipping part with invalid JSON data")
                continue
        elif isinstance(raw_data, dict):
            data = raw_data
        else:
            log.debug(
                "opencode: skipping part with unexpected data type: %s", type(raw_data)
            )
            continue

        if part_type == "step-start" or part_type == "step-finish":
            continue

        if part_type == "compaction":
            continue

        if part_type == "text":
            text_content = data.get("text", "")
            role = part.get("role", data.get("role", ""))
            if text_content:
                if role:
                    display_role = role.capitalize()
                    lines.append(f"**{display_role}**: {text_content}")
                else:
                    lines.append(text_content)

        elif part_type == "tool":
            tool_name = data.get("name", data.get("tool_name", "unknown_tool"))
            output = data.get("output", data.get("result", ""))
            if isinstance(output, str) and len(output) > 500:
                output = output[:500] + "..."
            elif not isinstance(output, str):
                output = json.dumps(output)[:500]
            lines.append(f"**Tool ({tool_name})**: {output}")

        elif part_type == "reasoning":
            text_content = data.get("text", data.get("reasoning", ""))
            if text_content:
                lines.append(f"**Reasoning**: {text_content}")

        else:
            log.debug("opencode: skipping unknown part type: %s", part_type)

    return "\n\n".join(lines)


class OpenCodeAdapter(SessionAdapter):
    """Read opencode sessions from a SQLite database.

    Opens a read-only connection to the opencode SQLite database and
    provides methods to list sessions and retrieve session transcripts
    as markdown strings.

    Args:
        db_path: Path to the opencode SQLite database file.
            Must point to an existing file — raises FileNotFoundError
            if the file doesn't exist. Must not target system directories
            — raises ValueError if the path resolves to a blocked prefix.
            Symlink paths are also rejected.

    Raises:
        FileNotFoundError: If db_path doesn't point to an existing file.
        ValueError: If db_path targets a system directory, is a symlink,
            or contains '..' traversal.
        sqlite3.Error: If the database connection cannot be established.
    """

    def __init__(self, db_path: Path) -> None:
        # Validate the path before any filesystem access to prevent
        # URI injection attacks (e.g. file:path?mode=rw to escalate
        # from read-only to read-write).
        path_str = str(db_path)
        for char in ("?", "#"):
            if char in path_str:
                raise ValueError(
                    f"llmem: opencode: database path contains disallowed "
                    f"character {char!r}: {db_path}"
                )
        if ".." in path_str:
            raise ValueError(
                f"llmem: opencode: database path contains '..' traversal: {db_path}"
            )

        resolved = Path(db_path).resolve()

        # Must not target a system directory — checked before symlink
        # check because is_symlink() requires stat access which may
        # OSError on inaccessible paths under blocked prefixes (e.g. /root).
        for prefix in BLOCKED_SYSTEM_PREFIXES:
            if str(resolved).startswith(prefix + "/") or str(resolved) == prefix:
                raise ValueError(
                    f"llmem: opencode adapter: db_path targets a system directory: {resolved}"
                )

        # Must not be a symlink — OSError on inaccessible paths is treated
        # as unsafe (matching _validate_home_path's handling in paths.py).
        try:
            if Path(db_path).is_symlink():
                raise ValueError(
                    f"llmem: opencode adapter: db_path is a symlink (not allowed): {db_path}"
                )
        except OSError:
            raise ValueError(
                f"llmem: opencode adapter: db_path cannot be accessed (permission denied): {db_path}"
            )

        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"llmem: opencode adapter: database not found: {db_path}"
            )
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._has_role_column: bool | None = None
        self._has_type_column: bool | None = None
        self._connect()

    def _connect(self) -> None:
        """Open a read-only connection to the opencode database.

        Raises:
            sqlite3.Error: If the connection cannot be established.
        """
        uri = f"file:{self._db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True)
        self._conn.row_factory = sqlite3.Row

    def _check_role_column(self) -> bool:
        if self._has_role_column is not None:
            return self._has_role_column
        try:
            cursor = self._conn.execute("PRAGMA table_info(message)")
            columns = {row["name"] for row in cursor.fetchall()}
            self._has_role_column = "role" in columns
        except sqlite3.Error:
            self._has_role_column = False
        return self._has_role_column

    def _check_type_column(self) -> bool:
        if self._has_type_column is not None:
            return self._has_type_column
        try:
            cursor = self._conn.execute("PRAGMA table_info(part)")
            columns = {row["name"] for row in cursor.fetchall()}
            self._has_type_column = "type" in columns
        except sqlite3.Error:
            self._has_type_column = False
        return self._has_type_column

    def _role_query_fragment(self) -> str:
        if self._check_role_column():
            return 'COALESCE(json_extract("data", \'$.role\'), "role") as "role"'
        return 'json_extract("data", \'$.role\') as "role"'

    def _type_query_fragment(self) -> str:
        if self._check_type_column():
            return 'COALESCE(json_extract("data", \'$.type\'), "type") as "type"'
        return 'json_extract("data", \'$.type\') as "type"'

    def close(self) -> None:
        """Close the database connection. Safe to call multiple times."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """List opencode sessions ordered by time_updated descending.

        Args:
            limit: Maximum number of sessions to return. Defaults to 50.

        Returns:
            A list of dicts, each with keys: id, title, slug,
            time_created, time_updated. Returns an empty list after
            close() has been called or on query errors (logs a warning).
        """
        if self._conn is None:
            log.warning("opencode: database connection not available (adapter closed)")
            return []

        try:
            rows = self._conn.execute(
                'SELECT "id", "title", "slug", "time_created", "time_updated" '
                'FROM "session" ORDER BY "time_updated" DESC LIMIT ?',
                (limit,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "title": r["title"] or "",
                    "slug": r["slug"] or "",
                    "time_created": r["time_created"],
                    "time_updated": r["time_updated"],
                }
                for r in rows
            ]
        except sqlite3.Error as e:
            log.warning("opencode: query error listing sessions: %s", e)
            return []

    def get_session_transcript(self, session_id: str) -> str | None:
        """Reconstruct a session transcript as markdown.

        Args:
            session_id: The opencode session ID (e.g., ses_abc123).

        Returns:
            A markdown transcript string, or None if the session_id
            doesn't exist in the database.
        """
        if self._conn is None:
            log.warning("opencode: database connection not available (adapter closed)")
            return None

        try:
            row = self._conn.execute(
                'SELECT 1 FROM "session" WHERE "id" = ?', (session_id,)
            ).fetchone()
            if row is None:
                log.info("opencode: session not found: %s", session_id)
                return None
        except sqlite3.Error as e:
            log.warning("opencode: query error checking session %s: %s", session_id, e)
            return None

        try:
            messages = self._conn.execute(
                f'SELECT "id", {self._role_query_fragment()} FROM "message" '
                f'WHERE "session_id" = ? ORDER BY "time_created"',
                (session_id,),
            ).fetchall()
        except sqlite3.Error as e:
            log.warning(
                "opencode: query error fetching messages for %s: %s", session_id, e
            )
            return None

        if not messages:
            log.info("opencode: no messages for session %s", session_id)
            return ""

        all_parts: list[dict] = []
        message_ids = [m["id"] for m in messages]
        role_map = {m["id"]: m["role"] for m in messages}

        try:
            placeholders = ",".join("?" for _ in message_ids)
            parts = self._conn.execute(
                f'SELECT "id", "message_id", {self._type_query_fragment()}, "data" FROM "part" '
                f'WHERE "message_id" IN ({placeholders}) '
                f'ORDER BY "time_created"',
                message_ids,
            ).fetchall()
        except sqlite3.Error as e:
            log.warning(
                "opencode: query error fetching parts for %s: %s", session_id, e
            )
            return None

        for p in parts:
            part_dict = {
                "type": p["type"],
                "data": p["data"],
                "role": role_map.get(p["message_id"], ""),
            }
            all_parts.append(part_dict)

        return session_to_markdown(all_parts)

    def get_session_chunks(self, session_id: str) -> list[str] | None:
        """Chunk a session transcript by user-message boundaries.

        Groups messages by user-message boundaries: each chunk starts with a
        user message and includes all subsequent assistant/tool/reasoning
        messages until the next user message.

        Args:
            session_id: The opencode session ID.

        Returns:
            None if the session doesn't exist or the adapter is closed.
            An empty list if the session exists but has no messages.
            A list of 1+ non-empty markdown strings otherwise.
        """
        if self._conn is None:
            log.warning("opencode: database connection not available (adapter closed)")
            return None

        try:
            row = self._conn.execute(
                'SELECT 1 FROM "session" WHERE "id" = ?', (session_id,)
            ).fetchone()
            if row is None:
                log.info("opencode: session not found: %s", session_id)
                return None

            messages = self._conn.execute(
                f'SELECT "id", {self._role_query_fragment()} FROM "message" '
                f'WHERE "session_id" = ? ORDER BY "time_created"',
                (session_id,),
            ).fetchall()
        except sqlite3.Error as e:
            log.warning(
                "opencode: query error fetching chunks for %s: %s", session_id, e
            )
            transcript = self.get_session_transcript(session_id)
            if transcript is None:
                return None
            return [transcript] if transcript else []

        if not messages:
            log.info("opencode: no messages for session %s", session_id)
            return []

        user_message_count = sum(1 for m in messages if m["role"] == "user")
        if user_message_count < 2:
            transcript = self.get_session_transcript(session_id)
            if transcript is None:
                return None
            return [transcript] if transcript else []

        message_ids = [m["id"] for m in messages]
        role_map = {m["id"]: m["role"] for m in messages}

        try:
            placeholders = ",".join("?" for _ in message_ids)
            parts = self._conn.execute(
                f'SELECT "id", "message_id", {self._type_query_fragment()}, "data" FROM "part" '
                f'WHERE "message_id" IN ({placeholders}) '
                f'ORDER BY "time_created"',
                message_ids,
            ).fetchall()
        except sqlite3.Error as e:
            log.warning(
                "opencode: query error fetching parts for chunks in %s: %s",
                session_id,
                e,
            )
            transcript = self.get_session_transcript(session_id)
            if transcript is None:
                return None
            return [transcript] if transcript else []

        parts_by_message: dict[str, list[dict]] = {}
        for p in parts:
            msg_id = p["message_id"]
            if msg_id not in parts_by_message:
                parts_by_message[msg_id] = []
            parts_by_message[msg_id].append(
                {
                    "type": p["type"],
                    "data": p["data"],
                    "role": role_map.get(msg_id, ""),
                }
            )

        chunks: list[str] = []
        current_parts: list[dict] = []

        for m in messages:
            msg_parts = parts_by_message.get(m["id"], [])
            if m["role"] == "user":
                if current_parts:
                    rendered = session_to_markdown(current_parts)
                    if rendered.strip():
                        chunks.append(rendered)
                    current_parts = []
            current_parts.extend(msg_parts)

        if current_parts:
            rendered = session_to_markdown(current_parts)
            if rendered.strip():
                chunks.append(rendered)

        return chunks

    def session_exists(self, session_id: str) -> bool:
        """Check whether a session ID exists in the opencode database.

        Args:
            session_id: The opencode session ID to check.

        Returns:
            True if the session exists, False otherwise.
        """
        if self._conn is None:
            log.warning("opencode: database connection not available (adapter closed)")
            return False

        try:
            row = self._conn.execute(
                'SELECT 1 FROM "session" WHERE "id" = ?', (session_id,)
            ).fetchone()
            return row is not None
        except sqlite3.Error as e:
            log.warning("opencode: query error checking session %s: %s", session_id, e)
            return False
