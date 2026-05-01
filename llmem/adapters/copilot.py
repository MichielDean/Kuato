"""Copilot CLI session adapter.

Reads session transcripts from the Copilot CLI session state directory
(~/.copilot/session-state/) and optional --share markdown exports.

Copilot CLI stores session metadata in workspace.yaml files within
UUID-named directories. Full conversation transcripts are only available
when the --share flag is used during the session, which writes a
markdown file to a configurable location.

This adapter provides:
- Session listing from workspace.yaml metadata
- Full transcripts from --share markdown files (if available)
- Metadata-only fallback when no share file exists
"""

import logging
import re
from pathlib import Path

import yaml

from .base import SessionAdapter
from ..paths import _validate_home_path

log = logging.getLogger(__name__)

COPILOT_SESSION_SOURCE_TYPE = "copilot_session"

_DEFAULT_STATE_DIR = Path("~/.copilot/session-state").expanduser()
_DEFAULT_SHARE_DIR = Path(".")


def _parse_workspace_yaml(path: Path) -> dict | None:
    """Parse a workspace.yaml file safely.

    Args:
        path: Path to the workspace.yaml file.

    Returns:
        Parsed dict, or None on error.
    """
    try:
        text = path.read_text()
        return yaml.safe_load(text) or {}
    except Exception as e:
        log.debug("llmem: adapters.copilot: failed to parse %s: %s", path, e)
        return None


class CopilotAdapter(SessionAdapter):
    """Session adapter for GitHub Copilot CLI.

    Reads session metadata from workspace.yaml files in
    ~/.copilot/session-state/ and full transcripts from --share
    markdown files.

    Args:
        state_dir: Path to the Copilot session-state directory.
            Defaults to ~/.copilot/session-state/.
        share_dir: Directory where --share markdown files are written.
            Defaults to current directory. Can be overridden via config.
    """

    def __init__(
        self,
        state_dir: str | Path | None = None,
        share_dir: str | Path | None = None,
    ):
        if state_dir is not None:
            candidate = Path(state_dir).expanduser()
            validated = _validate_home_path(candidate, "copilot.state_dir")
            self._state_dir = Path(validated)
        else:
            self._state_dir = _DEFAULT_STATE_DIR

        if share_dir is not None:
            self._share_dir = Path(share_dir).expanduser()
        else:
            self._share_dir = _DEFAULT_SHARE_DIR

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """List Copilot sessions by scanning session-state directories.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            A list of dicts with 'id', 'title', 'slug', and 'directory' keys.
        """
        if not self._state_dir.is_dir():
            log.debug(
                "llmem: adapters.copilot: state directory not found: %s",
                self._state_dir,
            )
            return []

        sessions = []
        for entry in sorted(
            self._state_dir.iterdir(), key=lambda e: e.stat().st_mtime, reverse=True
        ):
            if not entry.is_dir():
                continue

            workspace_path = entry / "workspace.yaml"
            if not workspace_path.exists():
                continue

            ws = _parse_workspace_yaml(workspace_path)
            if ws is None:
                continue

            sessions.append(
                {
                    "id": ws.get("id", entry.name),
                    "title": ws.get("cwd", ""),
                    "slug": ws.get("branch", ""),
                    "directory": ws.get("cwd", ""),
                    "created_at": ws.get("created_at", ""),
                    "updated_at": ws.get("updated_at", ""),
                }
            )

            if len(sessions) >= limit:
                break

        return sessions

    def get_session_transcript(self, session_id: str) -> str | None:
        """Get session transcript from a --share markdown file.

        Looks for a file matching the pattern copilot-session-{session_id}.md
        in the share directory.

        Args:
            session_id: The Copilot session ID (UUID format).

        Returns:
            Markdown string of the transcript, or None if not found.
        """
        share_file = self._share_dir / f"copilot-session-{session_id}.md"
        if share_file.exists():
            try:
                return share_file.read_text()
            except Exception as e:
                log.debug(
                    "llmem: adapters.copilot: failed to read share file %s: %s",
                    share_file,
                    e,
                )
                return None

        # Try without the full UUID — some share files use shortened IDs
        for f in self._share_dir.glob(f"copilot-session-{session_id[:8]}*.md"):
            try:
                return f.read_text()
            except Exception as e:
                log.debug(
                    "llmem: adapters.copilot: failed to read share file %s: %s", f, e
                )
                return None

        log.debug(
            "llmem: adapters.copilot: no share file for session %s in %s",
            session_id,
            self._share_dir,
        )
        return None

    def get_session_chunks(self, session_id: str) -> list[str] | None:
        """Chunk a session transcript by message boundaries.

        Splits the markdown transcript on heading boundaries (##) to
        approximate per-message chunks.

        Args:
            session_id: The session identifier.

        Returns:
            None if no transcript exists. Empty list if transcript is empty.
            Otherwise a list of non-empty markdown strings.
        """
        transcript = self.get_session_transcript(session_id)
        if transcript is None:
            return None

        if not transcript.strip():
            return []

        chunks = re.split(r"(?=\n## )", transcript)
        result = [c.strip() for c in chunks if c.strip()]
        return result if result else []

    def session_exists(self, session_id: str) -> bool:
        """Check whether a Copilot session exists.

        Args:
            session_id: The session ID (UUID format).

        Returns:
            True if the session directory exists.
        """
        session_dir = self._state_dir / session_id
        return session_dir.is_dir()

    def close(self) -> None:
        """Release resources. No-op for filesystem-based adapter."""
        pass