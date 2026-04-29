"""Tests for the llmem CLI 'track-review' and 'suggest-categories' subcommands.

These subcommands persist review findings and list error taxonomy categories,
supporting the introspection-review-tracker skill.
"""

import argparse
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from llmem.cli import cmd_track_review, cmd_suggest_categories, main
from llmem.taxonomy import ERROR_TAXONOMY, REVIEW_SEVERITY_TAXONOMY
from llmem.store import MemoryStore


# ── suggest-categories tests ────────────────────────────────────────────


class TestSuggestCategories_CLI:
    """Test that suggest-categories CLI lists correct categories for each tier."""

    def test_blocking_tier(self, capsys):
        """suggest-categories Blocking lists AUTH_BYPASS, RACE_CONDITION, DATA_INTEGRITY."""
        args = argparse.Namespace(tier="Blocking")
        cmd_suggest_categories(args)
        output = capsys.readouterr().out.strip()
        lines = output.split("\n")
        assert set(lines) == {"AUTH_BYPASS", "RACE_CONDITION", "DATA_INTEGRITY"}

    def test_required_tier(self, capsys):
        """suggest-categories Required lists the four Required categories."""
        args = argparse.Namespace(tier="Required")
        cmd_suggest_categories(args)
        output = capsys.readouterr().out.strip()
        lines = output.split("\n")
        assert set(lines) == {
            "NULL_SAFETY",
            "ERROR_HANDLING",
            "MISSING_VERIFICATION",
            "EDGE_CASE",
        }

    def test_strong_suggestions_tier(self, capsys):
        """suggest-categories Strong Suggestions lists PERFORMANCE and DESIGN."""
        args = argparse.Namespace(tier="Strong Suggestions")
        cmd_suggest_categories(args)
        output = capsys.readouterr().out.strip()
        lines = output.split("\n")
        assert set(lines) == {"PERFORMANCE", "DESIGN"}

    def test_noted_tier(self, capsys):
        """suggest-categories Noted lists OFF_BY_ONE."""
        args = argparse.Namespace(tier="Noted")
        cmd_suggest_categories(args)
        output = capsys.readouterr().out.strip()
        assert output.strip() == "OFF_BY_ONE"

    def test_passed_tier(self, capsys):
        """suggest-categories Passed lists REVIEW_PASSED."""
        args = argparse.Namespace(tier="Passed")
        cmd_suggest_categories(args)
        output = capsys.readouterr().out.strip()
        assert output.strip() == "REVIEW_PASSED"


class TestSuggestCategories_Integration:
    """Test suggest-categories via main() entry point."""

    def test_suggest_categories_in_help(self):
        """suggest-categories appears in llmem --help output."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        assert "suggest-categories" in output

    def test_suggest_categories_via_main(self, capsys):
        """Calling suggest-categories Required through main() produces correct output."""
        sys.argv = ["llmem", "suggest-categories", "Required"]
        try:
            main()
        except SystemExit:
            pass
        output = capsys.readouterr().out.strip()
        lines = output.split("\n")
        assert set(lines) == {
            "NULL_SAFETY",
            "ERROR_HANDLING",
            "MISSING_VERIFICATION",
            "EDGE_CASE",
        }

    def test_suggest_categories_invalid_tier(self):
        """Calling suggest-categories with an invalid tier exits with error."""
        sys.argv = ["llmem", "suggest-categories", "Invalid"]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


# ── track-review: clean review mode ─────────────────────────────────────


class TestTrackReview_CleanReview:
    """Test track-review clean review mode (no findings → REVIEW_PASSED)."""

    def test_clean_review_creates_review_passed_memory(self, tmp_path, capsys):
        """track-review with only --context creates a REVIEW_PASSED memory."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=None,
            db=db,
        )
        cmd_track_review(args)
        captured = capsys.readouterr()
        assert "REVIEW_PASSED" in captured.out

        store = MemoryStore(db_path=db)
        results = store.search("REVIEW_PASSED", limit=10)
        store.close()
        assert any("REVIEW_PASSED" in r.get("content", "") for r in results)

    def test_clean_review_includes_context(self, tmp_path, capsys):
        """track-review clean review includes the context in the memory content."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py:42",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=None,
            db=db,
        )
        cmd_track_review(args)
        store = MemoryStore(db_path=db)
        results = store.search("handler.py:42", limit=10)
        store.close()
        assert any("handler.py:42" in r.get("content", "") for r in results)


# ── track-review: single finding mode ───────────────────────────────────


class TestTrackReview_SingleFinding:
    """Test track-review single finding mode (--category + --what-happened)."""

    def test_single_finding_creates_self_assessment(self, tmp_path, capsys):
        """track-review --category + --what-happened creates a self_assessment memory."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py:42",
            category="NULL_SAFETY",
            what_happened="missing null check before property access",
            severity=None,
            caught_by="self-review",
            finding_file=None,
            db=db,
        )
        cmd_track_review(args)
        captured = capsys.readouterr()
        assert "NULL_SAFETY" in captured.out

        store = MemoryStore(db_path=db)
        results = store.search("null check", limit=10)
        store.close()
        assert any("NULL_SAFETY" in r.get("content", "") for r in results)

    def test_single_finding_requires_what_happened(self, tmp_path):
        """track-review --category without --what-happened exits with error."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py",
            category="NULL_SAFETY",
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=None,
            db=db,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_track_review(args)
        assert exc_info.value.code == 1

    def test_single_finding_invalid_category(self, tmp_path):
        """track-review --category with unknown category exits with error."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py",
            category="INVALID_CATEGORY",
            what_happened="something happened",
            severity=None,
            caught_by=None,
            finding_file=None,
            db=db,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_track_review(args)
        assert exc_info.value.code == 1

    def test_single_finding_includes_all_fields(self, tmp_path, capsys):
        """track-review single finding includes category, context, what_happened, in content."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py:42",
            category="ERROR_HANDLING",
            what_happened="swallowed exception",
            severity="Required",
            caught_by="CI",
            finding_file=None,
            db=db,
        )
        cmd_track_review(args)
        store = MemoryStore(db_path=db)
        results = store.search("swallowed", limit=10)
        store.close()
        content = results[0]["content"]
        assert "Category: ERROR_HANDLING" in content
        assert "Context: handler.py:42" in content
        assert "What_happened: swallowed exception" in content
        assert "What_caught_it: CI" in content

    def test_single_finding_default_caught_by(self, tmp_path, capsys):
        """track-review single finding defaults caught_by to 'self-review'."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py",
            category="DESIGN",
            what_happened="tight coupling",
            severity=None,
            caught_by=None,
            finding_file=None,
            db=db,
        )
        cmd_track_review(args)
        store = MemoryStore(db_path=db)
        results = store.search("tight coupling", limit=10)
        store.close()
        content = results[0]["content"]
        assert "What_caught_it: self-review" in content

    def test_single_finding_without_severity_outcomes_not_all_clear(
        self, tmp_path, capsys
    ):
        """track-review single finding without --severity must not produce 'Outcomes: all clear'.

        A finding is never 'all clear' — that phrase is reserved for zero-finding
        clean reviews. Without a severity tier, the outcome should be 'Outcomes: finding'.
        """
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py",
            category="NULL_SAFETY",
            what_happened="missing null check",
            severity=None,
            caught_by=None,
            finding_file=None,
            db=db,
        )
        cmd_track_review(args)
        store = MemoryStore(db_path=db)
        results = store.search("missing null check", limit=10)
        store.close()
        content = results[0]["content"]
        assert "Outcomes: finding" in content
        assert "Outcomes: all clear" not in content

    def test_clean_review_outcomes_all_clear(self, tmp_path, capsys):
        """track-review clean review produces 'Outcomes: all clear'."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=None,
            db=db,
        )
        cmd_track_review(args)
        store = MemoryStore(db_path=db)
        results = store.search("REVIEW_PASSED", limit=10)
        store.close()
        content = results[0]["content"]
        assert "Outcomes: all clear" in content

    def test_single_finding_with_severity_outcomes(self, tmp_path, capsys):
        """track-review single finding with --severity produces 'Outcomes: <tier> finding'."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="handler.py:10",
            category="ERROR_HANDLING",
            what_happened="swallowed exception",
            severity="Required",
            caught_by="CI",
            finding_file=None,
            db=db,
        )
        cmd_track_review(args)
        store = MemoryStore(db_path=db)
        results = store.search("swallowed exception", limit=10)
        store.close()
        content = results[0]["content"]
        assert "Outcomes: Required finding" in content


# ── track-review: batch mode ─────────────────────────────────────────────


class TestTrackReview_BatchMode:
    """Test track-review batch mode (--finding-file)."""

    def test_batch_mode_creates_one_memory_per_finding(self, tmp_path, capsys):
        """track-review --finding-file creates one memory per finding entry."""
        db = tmp_path / "test.db"
        findings = [
            {"category": "NULL_SAFETY", "what_happened": "null pointer dereference"},
            {"category": "ERROR_HANDLING", "what_happened": "missing error check"},
        ]
        finding_path = tmp_path / "findings.json"
        finding_path.write_text(json.dumps(findings))

        args = argparse.Namespace(
            context="app.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=str(finding_path),
            db=db,
        )
        cmd_track_review(args)
        captured = capsys.readouterr()
        assert "NULL_SAFETY" in captured.out
        assert "ERROR_HANDLING" in captured.out

        store = MemoryStore(db_path=db)
        results = store.search("null pointer", limit=10)
        store.close()
        assert len(results) >= 1

    def test_batch_mode_file_not_found(self, tmp_path):
        """track-review --finding-file with missing file exits with error."""
        db = tmp_path / "test.db"
        args = argparse.Namespace(
            context="app.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file="/nonexistent/path.json",
            db=db,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_track_review(args)
        assert exc_info.value.code == 1

    def test_batch_mode_invalid_json(self, tmp_path):
        """track-review --finding-file with invalid JSON exits with error."""
        db = tmp_path / "test.db"
        finding_path = tmp_path / "bad.json"
        finding_path.write_text("not json{")

        args = argparse.Namespace(
            context="app.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=str(finding_path),
            db=db,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_track_review(args)
        assert exc_info.value.code == 1

    def test_batch_mode_non_array_json(self, tmp_path):
        """track-review --finding-file with JSON object (not array) exits with error."""
        db = tmp_path / "test.db"
        finding_path = tmp_path / "obj.json"
        finding_path.write_text(json.dumps({"category": "DESIGN"}))

        args = argparse.Namespace(
            context="app.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=str(finding_path),
            db=db,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_track_review(args)
        assert exc_info.value.code == 1

    def test_batch_mode_unknown_category_defaults_to_missing_verification(
        self, tmp_path, capsys
    ):
        """track-review --finding-file with unknown category defaults to MISSING_VERIFICATION."""
        db = tmp_path / "test.db"
        findings = [
            {"category": "UNKNOWN_CAT", "what_happened": "something"},
        ]
        finding_path = tmp_path / "findings.json"
        finding_path.write_text(json.dumps(findings))

        args = argparse.Namespace(
            context="app.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=str(finding_path),
            db=db,
        )
        cmd_track_review(args)
        captured = capsys.readouterr()
        assert "MISSING_VERIFICATION" in captured.out

    def test_batch_mode_camel_case_fallback(self, tmp_path, capsys):
        """track-review --finding-file falls back to 'whatHappened' when 'what_happened' is absent."""
        db = tmp_path / "test.db"
        findings = [
            {"category": "NULL_SAFETY", "whatHappened": "camelCase null issue"},
        ]
        finding_path = tmp_path / "findings.json"
        finding_path.write_text(json.dumps(findings))

        args = argparse.Namespace(
            context="app.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=str(finding_path),
            db=db,
        )
        cmd_track_review(args)
        store = MemoryStore(db_path=db)
        results = store.search("camelCase null issue", limit=10)
        store.close()
        content = results[0]["content"]
        assert "What_happened: camelCase null issue" in content

    def test_batch_mode_snake_case_takes_precedence_over_camel(self, tmp_path, capsys):
        """track-review --finding-file prefers 'what_happened' over 'whatHappened'."""
        db = tmp_path / "test.db"
        findings = [
            {
                "category": "NULL_SAFETY",
                "what_happened": "snake_case value",
                "whatHappened": "camelCase value",
            },
        ]
        finding_path = tmp_path / "findings.json"
        finding_path.write_text(json.dumps(findings))

        args = argparse.Namespace(
            context="app.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=str(finding_path),
            db=db,
        )
        cmd_track_review(args)
        store = MemoryStore(db_path=db)
        results = store.search("snake_case value", limit=10)
        store.close()
        content = results[0]["content"]
        assert "What_happened: snake_case value" in content
        assert "What_happened: camelCase value" not in content

    def test_batch_mode_missing_what_happened_defaults(self, tmp_path, capsys):
        """track-review --finding-file defaults to 'review finding' when neither key is present."""
        db = tmp_path / "test.db"
        findings = [
            {"category": "DESIGN"},
        ]
        finding_path = tmp_path / "findings.json"
        finding_path.write_text(json.dumps(findings))

        args = argparse.Namespace(
            context="app.py",
            category=None,
            what_happened=None,
            severity=None,
            caught_by=None,
            finding_file=str(finding_path),
            db=db,
        )
        cmd_track_review(args)
        store = MemoryStore(db_path=db)
        results = store.search("DESIGN", limit=10)
        store.close()
        content = results[0]["content"]
        assert "What_happened: review finding" in content


# ── track-review: mutual exclusivity ─────────────────────────────────────


class TestTrackReview_MutualExclusivity:
    """Test that --category and --finding-file are mutually exclusive."""

    def test_category_and_finding_file_mutually_exclusive(self, tmp_path):
        """track-review rejects both --category and --finding-file."""
        db = tmp_path / "test.db"
        finding_path = tmp_path / "findings.json"
        finding_path.write_text("[]")

        args = argparse.Namespace(
            context="handler.py",
            category="NULL_SAFETY",
            what_happened="something",
            severity=None,
            caught_by=None,
            finding_file=str(finding_path),
            db=db,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_track_review(args)
        assert exc_info.value.code == 1


# ── track-review: CLI integration ────────────────────────────────────────


class TestTrackReview_CLIIntegration:
    """Test track-review through the main() CLI entry point."""

    def test_track_review_in_help(self):
        """track-review appears in llmem --help output."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        assert "track-review" in output

    def test_track_review_clean_via_main(self, tmp_path, capsys):
        """Calling track-review --context through main() creates REVIEW_PASSED."""
        db = tmp_path / "test.db"
        sys.argv = ["llmem", "--db", str(db), "track-review", "--context", "handler.py"]
        main()
        captured = capsys.readouterr()
        assert "REVIEW_PASSED" in captured.out

    def test_track_review_single_finding_via_main(self, tmp_path, capsys):
        """Calling track-review --category --what-happened through main() creates finding."""
        db = tmp_path / "test.db"
        sys.argv = [
            "llmem",
            "--db",
            str(db),
            "track-review",
            "--category",
            "NULL_SAFETY",
            "--what-happened",
            "missing null check",
            "--context",
            "handler.py",
        ]
        main()
        captured = capsys.readouterr()
        assert "NULL_SAFETY" in captured.out
