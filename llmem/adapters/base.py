"""Abstract base class for session adapters.

A SessionAdapter provides a pluggable interface for reading session
transcripts from any source (e.g., opencode DB, Claude sessions, etc.).
Extensions implement this ABC to integrate new session sources.
"""

import abc


class SessionAdapter(abc.ABC):
    """Abstract base class for reading session transcripts.

    Subclasses must implement all abstract methods. Constructor
    parameters are defined by the subclass — there is no required
    base constructor signature beyond **kwargs.
    """

    @abc.abstractmethod
    def list_sessions(self, limit: int = 50) -> list[dict]:
        """List recent sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            A list of dicts, each with at least 'id' and 'title' keys.
        """
        ...

    @abc.abstractmethod
    def get_session_transcript(self, session_id: str) -> str | None:
        """Reconstruct a full session transcript as markdown.

        Args:
            session_id: The session identifier.

        Returns:
            A markdown string, or None if the session doesn't exist.
        """
        ...

    @abc.abstractmethod
    def get_session_chunks(self, session_id: str) -> list[str] | None:
        """Chunk a session transcript by message boundaries.

        Args:
            session_id: The session identifier.

        Returns:
            None if the session doesn't exist. An empty list if
            the session has no messages. A list of 1+ non-empty
            markdown strings otherwise.
        """
        ...

    @abc.abstractmethod
    def session_exists(self, session_id: str) -> bool:
        """Check whether a session exists.

        Args:
            session_id: The session identifier.

        Returns:
            True if the session exists, False otherwise.
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources (e.g., close database connections)."""
        ...
