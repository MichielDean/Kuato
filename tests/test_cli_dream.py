"""Tests for the llmem dream CLI subcommand."""

import argparse
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from llmem.cli import cmd_dream, main
from llmem.dream import (
    DreamResult,
    LightPhaseResult,
)
from llmem.store import MemoryStore


class TestCliDream_DryRun:
    """Test cmd_dream with apply=False (dry run)."""

    def test_dry_run_prints_prefix(self, tmp_path, capsys):
        """cmd_dream with apply=False prints [DRY RUN] prefix on phase output."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=False,
            phase=None,
            report=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_dream(args)

        captured = capsys.readouterr()
        assert "[DRY RUN]" in captured.out

    def test_dry_run_does_not_modify_database(self, tmp_path, capsys):
        """cmd_dream with apply=False should not promote inbox items."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add_to_inbox(content="should not be promoted", attention_score=0.9)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=False,
            phase=None,
            report=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_dream(args)

        # Verify inbox still has the item (dry run)
        store2 = MemoryStore(db_path=db, disable_vec=True)
        assert store2.inbox_count() == 1
        store2.close()


class TestCliDream_Apply:
    """Test cmd_dream with apply=True."""

    def test_apply_prints_results_without_dry_run_prefix(self, tmp_path, capsys):
        """cmd_dream with apply=True prints results without [DRY RUN] prefix."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add_to_inbox(content="promote me", attention_score=0.9)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=True,
            phase=None,
            report=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_dream(args)

        captured = capsys.readouterr()
        # Should not have [DRY RUN] prefix when apply=True
        assert "[DRY RUN]" not in captured.out
        # Should have some output about phases
        assert (
            "Dream complete" in captured.out
            or "Phase" in captured.out.lower()
            or "promoted" in captured.out.lower()
            or "Duplicate pairs" in captured.out
        )


class TestCliDream_PhaseLight:
    """Test cmd_dream with --phase=light."""

    def test_phase_light_runs_only_light_phase(self, tmp_path, capsys):
        """cmd_dream with phase='light' only runs the light phase."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=True,
            phase="light",
            report=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_dream(args)

        captured = capsys.readouterr()
        assert "Duplicate pairs" in captured.out


class TestCliDream_PhaseDeep:
    """Test cmd_dream with --phase=deep."""

    def test_phase_deep_runs_only_deep_phase(self, tmp_path, capsys):
        """cmd_dream with phase='deep' only runs the deep phase."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add_to_inbox(content="promote me", attention_score=0.8)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=True,
            phase="deep",
            report=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_dream(args)

        captured = capsys.readouterr()
        assert "Promoted" in captured.out or "Decayed" in captured.out


class TestCliDream_PhaseRem:
    """Test cmd_dream with --phase=rem."""

    def test_phase_rem_runs_only_rem_phase(self, tmp_path, capsys):
        """cmd_dream with phase='rem' only runs the rem phase."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add(type="fact", content="test memory")
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=True,
            phase="rem",
            report=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_dream(args)

        captured = capsys.readouterr()
        assert "Total memories" in captured.out
        assert "Active" in captured.out


class TestCliDream_WithReport:
    """Test cmd_dream with --report flag."""

    def test_with_report_generates_html_file(self, tmp_path, capsys):
        """cmd_dream with report=<path> generates an HTML report file."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        report_path = tmp_path / "dream-report.html"
        args = argparse.Namespace(
            db=db,
            apply=True,
            phase=None,
            report=str(report_path),
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_dream(args)

        assert report_path.exists()
        html = report_path.read_text()
        assert "LLMem Dream Report" in html

    def test_with_report_and_dry_run(self, tmp_path, capsys):
        """cmd_dream with report path and apply=False still generates the report."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        report_path = tmp_path / "dream-report.html"
        args = argparse.Namespace(
            db=db,
            apply=False,
            phase=None,
            report=str(report_path),
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_dream(args)

        assert report_path.exists()


class TestCliDream_ReportValidatesPath:
    """Test cmd_dream report path validation."""

    def test_report_blocked_path_exits_with_error(self, tmp_path):
        """cmd_dream with a blocked report path (e.g., /etc/) exits with an error."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=True,
            phase=None,
            report="/etc/dream-report.html",
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            with pytest.raises(SystemExit):
                cmd_dream(args)


class TestCliDream_InvalidPhase:
    """Test that argparse rejects invalid --phase values."""

    def test_invalid_phase_produces_argparse_error(self):
        """An invalid --phase value (not light/deep/rem) produces an argparse error."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "dream", "--phase", "invalid"]
            with pytest.raises(SystemExit):
                main()
        finally:
            stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        assert "invalid choice" in stderr.lower()


class TestCli_DreamSubparser:
    """Verify the dream subcommand appears in --help and accepts flags."""

    def test_dream_in_help_output(self):
        """The 'dream' subcommand appears in --help output."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "--help"]
            with pytest.raises(SystemExit):
                main()
        finally:
            help_output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        assert "dream" in help_output

    def test_dream_help_shows_apply_flag(self):
        """The dream subparser shows --apply flag in help."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "dream", "--help"]
            with pytest.raises(SystemExit):
                main()
        finally:
            help_output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        assert "--apply" in help_output

    def test_dream_help_shows_phase_flag(self):
        """The dream subparser shows --phase flag in help."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "dream", "--help"]
            with pytest.raises(SystemExit):
                main()
        finally:
            help_output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        assert "--phase" in help_output

    def test_dream_help_shows_report_flag(self):
        """The dream subparser shows --report flag in help."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "dream", "--help"]
            with pytest.raises(SystemExit):
                main()
        finally:
            help_output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        assert "--report" in help_output


class TestCliDream_DefaultDbPath:
    """Test that cmd_dream resolves DB path via get_db_path() when args.db is None."""

    def test_default_db_path_resolved(self, tmp_path, capsys):
        """cmd_dream with db=None resolves the default DB path via get_db_path()."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        with patch("llmem.cli.get_db_path", return_value=db) as mock_get_db:
            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                args = argparse.Namespace(
                    db=None,
                    apply=True,
                    phase=None,
                    report=None,
                )
                cmd_dream(args)
                mock_get_db.assert_called_once()

        captured = capsys.readouterr()
        # Should have run successfully without error
        assert captured.err == "" or "Error" not in captured.err


class TestCliDream_OllamaUrlFromMemoryConfig:
    """Test that cmd_dream reads ollama_url from the memory config section,
    not from the dream section.

    Regression test for ll-kingr-jywet: cmd_dream previously read
    ollama_url from dream_config, which always fell back to the
    hardcoded default since ollama_url is a memory-section key.
    """

    def test_ollama_url_read_from_memory_section(self, tmp_path, capsys):
        """cmd_dream calls get_ollama_url() to read from the memory section."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=False,
            phase="light",
            report=None,
        )

        with patch(
            "llmem.cli.get_ollama_url", return_value="http://custom-ollama:11434"
        ) as mock_get_url:
            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                with patch("llmem.dream.Dreamer") as MockDreamer:
                    mock_result = DreamResult(light=LightPhaseResult())
                    MockDreamer.return_value.run.return_value = mock_result
                    cmd_dream(args)
                    mock_get_url.assert_called_once()
                    # Verify the Dreamer was constructed with the URL from get_ollama_url
                    call_kwargs = MockDreamer.call_args[1]
                    assert call_kwargs["ollama_url"] == "http://custom-ollama:11434"

    def test_ollama_url_falls_back_on_validation_error(self, tmp_path, capsys):
        """cmd_dream falls back to default when get_ollama_url() raises ValueError."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=False,
            phase="light",
            report=None,
        )

        with patch("llmem.cli.get_ollama_url", side_effect=ValueError("unsafe URL")):
            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                with patch("llmem.dream.Dreamer") as MockDreamer:
                    mock_result = DreamResult(light=LightPhaseResult())
                    MockDreamer.return_value.run.return_value = mock_result
                    cmd_dream(args)
                    # Verify the Dreamer was constructed with the fallback default
                    call_kwargs = MockDreamer.call_args[1]
                    assert call_kwargs["ollama_url"] == "http://localhost:11434"


class TestCliDream_AutoLinkThresholdInConfig:
    """Test that auto_link_threshold is picked up from user config via get_dream_config().

    Regression test for ll-kingr-87izo: auto_link_threshold was missing
    from DEFAULTS['dream'], so user config was silently ignored because
    get_dream_config() only iterated over keys present in the defaults.
    """

    def test_auto_link_threshold_in_dream_config(self, tmp_path, capsys):
        """cmd_dream passes auto_link_threshold from dream_config to Dreamer."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=False,
            phase="light",
            report=None,
        )

        custom_threshold = 0.75
        with patch(
            "llmem.cli.get_dream_config",
            return_value={
                "enabled": True,
                "schedule": "*-*-* 03:00:00",
                "similarity_threshold": 0.92,
                "decay_rate": 0.05,
                "decay_interval_days": 30,
                "decay_floor": 0.3,
                "confidence_floor": 0.3,
                "boost_threshold": 5,
                "boost_amount": 0.05,
                "min_score": 0.5,
                "min_recall_count": 3,
                "min_unique_queries": 1,
                "boost_on_promote": 0.1,
                "merge_model": "qwen2.5:1.5b",
                "diary_path": str(tmp_path / "diary.md"),
                "report_path": str(tmp_path / "report.html"),
                "behavioral_threshold": 3,
                "behavioral_lookback_days": 30,
                "proposed_changes_path": str(tmp_path / "proposed.json"),
                "calibration_enabled": True,
                "stale_procedure_days": 30,
                "calibration_lookback_days": 90,
                "auto_link_threshold": custom_threshold,
            },
        ):
            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                with patch("llmem.dream.Dreamer") as MockDreamer:
                    mock_result = DreamResult(light=LightPhaseResult())
                    MockDreamer.return_value.run.return_value = mock_result
                    cmd_dream(args)
                    call_kwargs = MockDreamer.call_args[1]
                    assert call_kwargs["auto_link_threshold"] == custom_threshold


class TestCliDream_DiaryAndProposedPathsFromConfig:
    """Test that cmd_dream passes diary_path and proposed_changes_path from
    dream_config to the Dreamer constructor.

    Regression test for ll-kingr-4vmi8: cmd_dream did not pass diary_path
    or proposed_changes_path from dream_config to Dreamer.__init__, so user
    config for these paths was silently ignored. Dreamer defaulted to None,
    which triggered its own hardcoded path resolution inside the constructor,
    writing diary entries to the wrong location.
    """

    def test_diary_path_passed_from_config(self, tmp_path, capsys):
        """cmd_dream passes diary_path from dream_config to Dreamer as a Path."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        custom_diary = str(tmp_path / "custom-diary.md")
        args = argparse.Namespace(
            db=db,
            apply=False,
            phase="light",
            report=None,
        )

        with patch(
            "llmem.cli.get_dream_config",
            return_value={
                "enabled": True,
                "schedule": "*-*-* 03:00:00",
                "similarity_threshold": 0.92,
                "decay_rate": 0.05,
                "decay_interval_days": 30,
                "decay_floor": 0.3,
                "confidence_floor": 0.3,
                "boost_threshold": 5,
                "boost_amount": 0.05,
                "min_score": 0.5,
                "min_recall_count": 3,
                "min_unique_queries": 1,
                "boost_on_promote": 0.1,
                "merge_model": "qwen2.5:1.5b",
                "diary_path": custom_diary,
                "report_path": str(tmp_path / "report.html"),
                "behavioral_threshold": 3,
                "behavioral_lookback_days": 30,
                "proposed_changes_path": str(tmp_path / "proposed.json"),
                "calibration_enabled": True,
                "stale_procedure_days": 30,
                "calibration_lookback_days": 90,
                "auto_link_threshold": 0.85,
            },
        ):
            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                with patch("llmem.dream.Dreamer") as MockDreamer:
                    mock_result = DreamResult(light=LightPhaseResult())
                    MockDreamer.return_value.run.return_value = mock_result
                    cmd_dream(args)
                    call_kwargs = MockDreamer.call_args[1]
                    assert call_kwargs["diary_path"] == Path(custom_diary)

    def test_proposed_changes_path_passed_from_config(self, tmp_path, capsys):
        """cmd_dream passes proposed_changes_path from dream_config to Dreamer as a Path."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        custom_proposed = str(tmp_path / "custom-proposed.json")
        args = argparse.Namespace(
            db=db,
            apply=False,
            phase="light",
            report=None,
        )

        with patch(
            "llmem.cli.get_dream_config",
            return_value={
                "enabled": True,
                "schedule": "*-*-* 03:00:00",
                "similarity_threshold": 0.92,
                "decay_rate": 0.05,
                "decay_interval_days": 30,
                "decay_floor": 0.3,
                "confidence_floor": 0.3,
                "boost_threshold": 5,
                "boost_amount": 0.05,
                "min_score": 0.5,
                "min_recall_count": 3,
                "min_unique_queries": 1,
                "boost_on_promote": 0.1,
                "merge_model": "qwen2.5:1.5b",
                "diary_path": str(tmp_path / "diary.md"),
                "report_path": str(tmp_path / "report.html"),
                "behavioral_threshold": 3,
                "behavioral_lookback_days": 30,
                "proposed_changes_path": custom_proposed,
                "calibration_enabled": True,
                "stale_procedure_days": 30,
                "calibration_lookback_days": 90,
                "auto_link_threshold": 0.85,
            },
        ):
            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                with patch("llmem.dream.Dreamer") as MockDreamer:
                    mock_result = DreamResult(light=LightPhaseResult())
                    MockDreamer.return_value.run.return_value = mock_result
                    cmd_dream(args)
                    call_kwargs = MockDreamer.call_args[1]
                    assert call_kwargs["proposed_changes_path"] == Path(custom_proposed)

    def test_diary_and_proposed_paths_none_when_not_in_config(self, tmp_path, capsys):
        """cmd_dream passes None for diary_path and proposed_changes_path when
        config values are None, letting Dreamer use its own default path resolution."""
        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.close()

        args = argparse.Namespace(
            db=db,
            apply=False,
            phase="light",
            report=None,
        )

        with patch(
            "llmem.cli.get_dream_config",
            return_value={
                "enabled": True,
                "schedule": "*-*-* 03:00:00",
                "similarity_threshold": 0.92,
                "decay_rate": 0.05,
                "decay_interval_days": 30,
                "decay_floor": 0.3,
                "confidence_floor": 0.3,
                "boost_threshold": 5,
                "boost_amount": 0.05,
                "min_score": 0.5,
                "min_recall_count": 3,
                "min_unique_queries": 1,
                "boost_on_promote": 0.1,
                "merge_model": "qwen2.5:1.5b",
                "diary_path": None,
                "report_path": str(tmp_path / "report.html"),
                "behavioral_threshold": 3,
                "behavioral_lookback_days": 30,
                "proposed_changes_path": None,
                "calibration_enabled": True,
                "stale_procedure_days": 30,
                "calibration_lookback_days": 90,
                "auto_link_threshold": 0.85,
            },
        ):
            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                with patch("llmem.dream.Dreamer") as MockDreamer:
                    mock_result = DreamResult(light=LightPhaseResult())
                    MockDreamer.return_value.run.return_value = mock_result
                    cmd_dream(args)
                    call_kwargs = MockDreamer.call_args[1]
                    assert call_kwargs["diary_path"] is None
                    assert call_kwargs["proposed_changes_path"] is None
