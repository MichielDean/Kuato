"""Tests for the llmem CLI 'context' and 'hook' subcommands.

These subcommands are used by external plugins (Copilot CLI, OpenCode)
to inject memory context and extract memories at session lifecycle events.
"""

import argparse
import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmem.cli import cmd_context, cmd_hook, main
from llmem.session_hooks import (
    SESSION_CREATED_SUCCESS,
    SESSION_CREATED_ALREADY_PROCESSED,
    SESSION_CREATED_ERROR,
    SESSION_COMPACTING_SUCCESS,
    SESSION_COMPACTING_NO_MEMORIES,
    SESSION_COMPACTING_ERROR,
    SESSION_IDLE_DEBOUNCED,
    SESSION_IDLE_NO_TRANSCRIPT,
)
from llmem.hooks import PROCESS_RESULT_SUCCESS
from llmem.store import MemoryStore

# Patch target for create_session_hook_coordinator — imported inside
# cmd_context and cmd_hook function bodies from llmem.session_hooks.
_COORDINATOR_PATCH = "llmem.session_hooks.create_session_hook_coordinator"
# Patch target for get_context_dir — imported inside cmd_context from llmem.paths.
_CONTEXT_DIR_PATCH = "llmem.paths.get_context_dir"


# ── cmd_context tests ─────────────────────────────────────────────────────


class TestCmdContext_Validation:
    """Test that cmd_context validates session_id."""

    def test_rejects_path_traversal_session_id(self, tmp_path):
        """cmd_context rejects a session_id containing '..'."""
        args = argparse.Namespace(
            session_id="../etc/passwd",
            compacting=False,
            db=tmp_path / "test.db",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_context(args)
        assert exc_info.value.code == 1

    def test_rejects_slash_in_session_id(self, tmp_path):
        """cmd_context rejects a session_id containing '/'."""
        args = argparse.Namespace(
            session_id="foo/bar",
            compacting=False,
            db=tmp_path / "test.db",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_context(args)
        assert exc_info.value.code == 1

    def test_rejects_backslash_in_session_id(self, tmp_path):
        """cmd_context rejects a session_id containing '\\'."""
        args = argparse.Namespace(
            session_id="foo\\bar",
            compacting=False,
            db=tmp_path / "test.db",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_context(args)
        assert exc_info.value.code == 1

    def test_rejects_empty_session_id(self, tmp_path):
        """cmd_context rejects an empty session_id."""
        args = argparse.Namespace(
            session_id="",
            compacting=False,
            db=tmp_path / "test.db",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_context(args)
        assert exc_info.value.code == 1


class TestCmdContext_CreatedSuccess:
    """Test cmd_context happy path for session start (created)."""

    def test_prints_context_on_success(self, tmp_path, capsys):
        """cmd_context prints context content when on_created succeeds."""
        mock_coordinator = MagicMock()
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_abc123.md"
        context_file.write_text("- [fact] the project uses pytest")
        mock_coordinator.on_created.return_value = (
            SESSION_CREATED_SUCCESS,
            str(context_file),
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                session_id="ses_abc123",
                compacting=False,
                db=tmp_path / "test.db",
            )
            cmd_context(args)

        captured = capsys.readouterr()
        assert "the project uses pytest" in captured.out

    def test_exits_on_created_error(self, tmp_path):
        """cmd_context exits with 1 when on_created returns error."""
        mock_coordinator = MagicMock()
        mock_coordinator.on_created.return_value = (SESSION_CREATED_ERROR, None)

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                session_id="ses_abc123",
                compacting=False,
                db=tmp_path / "test.db",
            )
            with pytest.raises(SystemExit) as exc_info:
                cmd_context(args)
            assert exc_info.value.code == 1


class TestCmdContext_CreatedAlreadyProcessed:
    """Test cmd_context when session was already processed."""

    def test_reads_existing_file_when_already_processed(self, tmp_path, capsys):
        """cmd_context reads existing context file on already_processed."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_exists.md"
        context_file.write_text("- [fact] previously stored fact")

        mock_coordinator = MagicMock()
        mock_coordinator.on_created.return_value = (
            SESSION_CREATED_ALREADY_PROCESSED,
            None,
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            with patch(_CONTEXT_DIR_PATCH, return_value=context_dir):
                args = argparse.Namespace(
                    session_id="ses_exists",
                    compacting=False,
                    db=tmp_path / "test.db",
                )
                cmd_context(args)

        captured = capsys.readouterr()
        assert "previously stored fact" in captured.out

    def test_no_output_when_already_processed_and_no_file(self, tmp_path, capsys):
        """cmd_context prints nothing when already_processed and file missing."""
        # Use a temp dir that has no context files
        context_dir = tmp_path / "empty_context"
        context_dir.mkdir()

        mock_coordinator = MagicMock()
        mock_coordinator.on_created.return_value = (
            SESSION_CREATED_ALREADY_PROCESSED,
            None,
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            with patch(_CONTEXT_DIR_PATCH, return_value=context_dir):
                args = argparse.Namespace(
                    session_id="ses_missing",
                    compacting=False,
                    db=tmp_path / "test.db",
                )
                cmd_context(args)

        captured = capsys.readouterr()
        assert captured.out == ""


class TestCmdContext_Compacting:
    """Test cmd_context --compacting for compaction events."""

    def test_prints_compact_context_on_success(self, tmp_path, capsys):
        """cmd_context --compacting prints context when on_compacting succeeds."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_abc123-compact.md"
        context_file.write_text("- [decision] use PostgreSQL\n- [preference] dark mode")

        mock_coordinator = MagicMock()
        mock_coordinator.on_compacting.return_value = (
            SESSION_COMPACTING_SUCCESS,
            str(context_file),
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                session_id="ses_abc123",
                compacting=True,
                db=tmp_path / "test.db",
            )
            cmd_context(args)

        captured = capsys.readouterr()
        assert "use PostgreSQL" in captured.out
        assert "dark mode" in captured.out

    def test_no_output_when_no_key_memories(self, tmp_path, capsys):
        """cmd_context --compacting prints nothing when no key memories found."""
        mock_coordinator = MagicMock()
        mock_coordinator.on_compacting.return_value = (
            SESSION_COMPACTING_NO_MEMORIES,
            None,
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                session_id="ses_abc123",
                compacting=True,
                db=tmp_path / "test.db",
            )
            cmd_context(args)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_exits_on_compacting_error(self, tmp_path):
        """cmd_context --compacting exits with 1 on compacting error."""
        mock_coordinator = MagicMock()
        mock_coordinator.on_compacting.return_value = (
            SESSION_COMPACTING_ERROR,
            None,
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                session_id="ses_abc123",
                compacting=True,
                db=tmp_path / "test.db",
            )
            with pytest.raises(SystemExit) as exc_info:
                cmd_context(args)
            assert exc_info.value.code == 1


class TestCmdContext_CoordinatorFailure:
    """Test cmd_context when coordinator creation fails."""

    def test_exits_when_coordinator_creation_fails(self, tmp_path):
        """cmd_context exits with 1 when create_session_hook_coordinator raises."""
        with patch(
            _COORDINATOR_PATCH,
            side_effect=RuntimeError("no config"),
        ):
            args = argparse.Namespace(
                session_id="ses_abc123",
                compacting=False,
                db=tmp_path / "test.db",
            )
            with pytest.raises(SystemExit) as exc_info:
                cmd_context(args)
            assert exc_info.value.code == 1


# ── cmd_hook tests ────────────────────────────────────────────────────────


class TestCmdHook_Validation:
    """Test that cmd_hook validates session_id."""

    def test_rejects_path_traversal_session_id(self, tmp_path):
        """cmd_hook rejects a session_id containing '..'."""
        args = argparse.Namespace(
            hook_type="idle",
            session_id="../etc/passwd",
            db=tmp_path / "test.db",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_hook(args)
        assert exc_info.value.code == 1

    def test_rejects_slash_in_session_id(self, tmp_path):
        """cmd_hook rejects a session_id containing '/'."""
        args = argparse.Namespace(
            hook_type="idle",
            session_id="foo/bar",
            db=tmp_path / "test.db",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_hook(args)
        assert exc_info.value.code == 1

    def test_rejects_empty_session_id(self, tmp_path):
        """cmd_hook rejects an empty session_id."""
        args = argparse.Namespace(
            hook_type="idle",
            session_id="",
            db=tmp_path / "test.db",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_hook(args)
        assert exc_info.value.code == 1


class TestCmdHook_Idle:
    """Test cmd_hook idle for memory extraction."""

    def test_calls_on_idle(self, tmp_path):
        """cmd_hook idle calls coordinator.on_idle with correct session_id."""
        mock_coordinator = MagicMock()
        mock_coordinator.on_idle.return_value = (PROCESS_RESULT_SUCCESS, 2)

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                hook_type="idle",
                session_id="ses_abc123",
                db=tmp_path / "test.db",
            )
            cmd_hook(args)

        mock_coordinator.on_idle.assert_called_once_with("ses_abc123")

    def test_debounced_does_not_error(self, tmp_path):
        """cmd_hook idle does not error when debounced."""
        mock_coordinator = MagicMock()
        mock_coordinator.on_idle.return_value = (SESSION_IDLE_DEBOUNCED, 0)

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                hook_type="idle",
                session_id="ses_abc123",
                db=tmp_path / "test.db",
            )
            # Should not raise
            cmd_hook(args)

    def test_no_transcript_does_not_error(self, tmp_path):
        """cmd_hook idle does not error when no transcript available."""
        mock_coordinator = MagicMock()
        mock_coordinator.on_idle.return_value = (SESSION_IDLE_NO_TRANSCRIPT, 0)

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                hook_type="idle",
                session_id="ses_abc123",
                db=tmp_path / "test.db",
            )
            # Should not raise
            cmd_hook(args)


class TestCmdHook_UnknownType:
    """Test cmd_hook with unknown hook types."""

    def test_unknown_hook_type_exits(self, tmp_path):
        """cmd_hook exits with 1 for unknown hook type.

        Note: argparse choices=['idle'] should reject unknown types
        before cmd_hook is called, but cmd_hook also validates internally.
        """
        args = argparse.Namespace(
            hook_type="unknown_event",
            session_id="ses_abc123",
            db=tmp_path / "test.db",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_hook(args)
        assert exc_info.value.code == 1


class TestCmdHook_CoordinatorFailure:
    """Test cmd_hook when coordinator creation fails."""

    def test_exits_when_coordinator_creation_fails(self, tmp_path):
        """cmd_hook idle exits with 1 when coordinator creation raises."""
        with patch(
            _COORDINATOR_PATCH,
            side_effect=RuntimeError("no config"),
        ):
            args = argparse.Namespace(
                hook_type="idle",
                session_id="ses_abc123",
                db=tmp_path / "test.db",
            )
            with pytest.raises(SystemExit) as exc_info:
                cmd_hook(args)
            assert exc_info.value.code == 1


# ── CLI integration tests ──────────────────────────────────────────────────


class TestCli_ContextSubcommand:
    """Test that 'llmem context' is a recognized CLI subcommand."""

    def test_context_appears_in_help(self):
        """'llmem --help' lists the context subcommand."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sys.argv = ["llmem", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "context" in output

    def test_context_help_shows_compacting_flag(self):
        """'llmem context --help' shows --compacting flag."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sys.argv = ["llmem", "context", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "--compacting" in output

    def test_context_help_shows_session_id(self):
        """'llmem context --help' shows session_id argument."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sys.argv = ["llmem", "context", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "session_id" in output


class TestCli_HookSubcommand:
    """Test that 'llmem hook' is a recognized CLI subcommand."""

    def test_hook_appears_in_help(self):
        """'llmem --help' lists the hook subcommand."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sys.argv = ["llmem", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "hook" in output

    def test_hook_help_shows_idle_choice(self):
        """'llmem hook --help' shows 'idle' as a choice."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sys.argv = ["llmem", "hook", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "idle" in output

    def test_hook_help_shows_session_id(self):
        """'llmem hook --help' shows session_id argument."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sys.argv = ["llmem", "hook", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
        assert "session_id" in output


class TestCli_ContextAndHook_InvalidSessionId:
    """Test that invalid session IDs are rejected at the CLI level."""

    def test_context_path_traversal_rejected(self):
        """'llmem context ../etc/passwd' is rejected."""
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "context", "../etc/passwd"]
            main()
        except SystemExit:
            pass
        finally:
            stderr = sys.stderr.getvalue()
            sys.stderr = old_stderr
        assert "path traversal" in stderr.lower() or "error" in stderr.lower()

    def test_hook_path_traversal_rejected(self):
        """'llmem hook idle ../etc/passwd' is rejected."""
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "hook", "idle", "../etc/passwd"]
            main()
        except SystemExit:
            pass
        finally:
            stderr = sys.stderr.getvalue()
            sys.stderr = old_stderr
        assert "path traversal" in stderr.lower() or "error" in stderr.lower()


class TestCmdContext_SingleExecution:
    """Test that cmd_context calls the coordinator exactly once.

    Regression test: a duplicate code block caused cmd_context to create
    two coordinators and execute the hook twice, producing doubled output.
    """

    def test_on_created_called_once(self, tmp_path, capsys):
        """cmd_context calls coordinator.on_created exactly once."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_once.md"
        context_file.write_text("- [fact] single execution fact")

        mock_coordinator = MagicMock()
        mock_coordinator.on_created.return_value = (
            SESSION_CREATED_SUCCESS,
            str(context_file),
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                session_id="ses_once",
                compacting=False,
                db=tmp_path / "test.db",
            )
            cmd_context(args)

        mock_coordinator.on_created.assert_called_once_with("ses_once")

    def test_on_compacting_called_once(self, tmp_path, capsys):
        """cmd_context --compacting calls coordinator.on_compacting exactly once."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_once-compact.md"
        context_file.write_text("- [decision] compact once")

        mock_coordinator = MagicMock()
        mock_coordinator.on_compacting.return_value = (
            SESSION_COMPACTING_SUCCESS,
            str(context_file),
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                session_id="ses_once",
                compacting=True,
                db=tmp_path / "test.db",
            )
            cmd_context(args)

        mock_coordinator.on_compacting.assert_called_once_with("ses_once")

    def test_context_content_appears_once_in_output(self, tmp_path, capsys):
        """cmd_context prints context content exactly once (not doubled)."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_exact.md"
        content = "- [fact] unique marker xyzzy"
        context_file.write_text(content)

        mock_coordinator = MagicMock()
        mock_coordinator.on_created.return_value = (
            SESSION_CREATED_SUCCESS,
            str(context_file),
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                session_id="ses_exact",
                compacting=False,
                db=tmp_path / "test.db",
            )
            cmd_context(args)

        captured = capsys.readouterr()
        assert captured.out.count(content) == 1, (
            f"Context content should appear exactly once, but appeared "
            f"{captured.out.count(content)} times"
        )

    def test_compacting_content_appears_once_in_output(self, tmp_path, capsys):
        """cmd_context --compacting prints content exactly once (not doubled)."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_compact_exact.md"
        content = "- [decision] compact marker plugh"
        context_file.write_text(content)

        mock_coordinator = MagicMock()
        mock_coordinator.on_compacting.return_value = (
            SESSION_COMPACTING_SUCCESS,
            str(context_file),
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            args = argparse.Namespace(
                session_id="ses_compact_exact",
                compacting=True,
                db=tmp_path / "test.db",
            )
            cmd_context(args)

        captured = capsys.readouterr()
        assert captured.out.count(content) == 1, (
            f"Compacting content should appear exactly once, but appeared "
            f"{captured.out.count(content)} times"
        )

    def test_coordinator_created_once(self, tmp_path):
        """cmd_context creates the coordinator exactly once."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_coord.md"
        context_file.write_text("- [fact] coord test")

        mock_coordinator = MagicMock()
        mock_coordinator.on_created.return_value = (
            SESSION_CREATED_SUCCESS,
            str(context_file),
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator) as mock_factory:
            args = argparse.Namespace(
                session_id="ses_coord",
                compacting=False,
                db=tmp_path / "test.db",
            )
            cmd_context(args)

        assert mock_factory.call_count == 1, (
            f"create_session_hook_coordinator should be called exactly once, "
            f"but was called {mock_factory.call_count} times"
        )


class TestCmdContext_ReadFileFailure:
    """Test cmd_context when reading the context file fails."""

    def test_exits_when_context_file_unreadable(self, tmp_path):
        """cmd_context exits with 1 when the context file cannot be read."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_bad.md"
        context_file.write_text("some context")

        mock_coordinator = MagicMock()
        mock_coordinator.on_created.return_value = (
            SESSION_CREATED_SUCCESS,
            str(context_file),
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            # Patch Path.read_text to simulate unreadable file
            original_read_text = Path.read_text

            def _failing_read_text(self, *args, **kwargs):
                if str(self) == str(context_file):
                    raise OSError("permission denied")
                return original_read_text(self, *args, **kwargs)

            with patch.object(Path, "read_text", _failing_read_text):
                args = argparse.Namespace(
                    session_id="ses_bad",
                    compacting=False,
                    db=tmp_path / "test.db",
                )
                with pytest.raises(SystemExit) as exc_info:
                    cmd_context(args)
                assert exc_info.value.code == 1

    def test_compacting_exits_when_context_file_unreadable(self, tmp_path):
        """cmd_context --compacting exits with 1 when context file unreadable."""
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        context_file = context_dir / "ses_bad-compact.md"
        context_file.write_text("some compact context")

        mock_coordinator = MagicMock()
        mock_coordinator.on_compacting.return_value = (
            SESSION_COMPACTING_SUCCESS,
            str(context_file),
        )

        with patch(_COORDINATOR_PATCH, return_value=mock_coordinator):
            original_read_text = Path.read_text

            def _failing_read_text(self, *args, **kwargs):
                if str(self) == str(context_file):
                    raise OSError("permission denied")
                return original_read_text(self, *args, **kwargs)

            with patch.object(Path, "read_text", _failing_read_text):
                args = argparse.Namespace(
                    session_id="ses_bad",
                    compacting=True,
                    db=tmp_path / "test.db",
                )
                with pytest.raises(SystemExit) as exc_info:
                    cmd_context(args)
                assert exc_info.value.code == 1
