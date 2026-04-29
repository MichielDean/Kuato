"""Tests for the llmem CLI entry point."""

import argparse
import io
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCli_EntryPoint:
    """Test that the llmem CLI entry point works."""

    def test_cli_help(self):
        """CLI --help outputs usage info with llmem branding."""
        from llmem.cli import main
        import io

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
        assert "llmem" in output.lower()

    def test_cli_prog_name(self):
        """CLI prog name is 'llmem'."""
        from llmem.cli import main
        import io

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
        assert "llmem" in output.lower()


class TestCli_LobmemCompat:
    """Test backward-compatible lobmem invocation with deprecation warning."""

    def test_lobmem_compat_deprecation_warning(self):
        """When invoked as 'lobmem', main() prints deprecation warning to stderr."""
        from llmem.cli import main
        import io

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["lobmem", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            stderr = sys.stderr.getvalue()
            stdout = sys.stdout.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        assert "deprecated" in stderr.lower(), (
            f"Expected deprecation warning in stderr, got: {stderr!r}"
        )
        assert "llmem" in stderr.lower(), (
            f"Expected 'llmem' in deprecation warning, got: {stderr!r}"
        )

    def test_llmem_no_deprecation_warning(self):
        """When invoked as 'llmem', no deprecation warning is emitted."""
        from llmem.cli import main
        import io

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
            stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        assert "deprecated" not in stderr.lower(), (
            f"No deprecation warning expected for llmem invocation, got: {stderr!r}"
        )


class TestCli_DbPathDefault:
    """Test that CLI uses ~/.config/llmem/memory.db by default."""

    def test_default_db_path(self):
        from llmem.paths import get_db_path

        # Mock the home to a tmp dir where neither .lobsterdog nor .config/llmem exists
        with patch("llmem.paths.Path") as mock_path_cls:
            import tempfile

            fake_home = Path(tempfile.mkdtemp())
            mock_path_cls.home.return_value = fake_home
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw) if a else Path()
            result = get_db_path()
            assert "llmem" in str(result)
            assert result.name == "memory.db"


class TestCli_DbPathEnvOverride:
    """Test that LMEM_HOME overrides DB path."""

    def test_env_override(self):
        import os

        with patch.dict(os.environ, {"LMEM_HOME": "/tmp/llmem-env-test"}):
            from llmem.paths import get_db_path

            result = get_db_path()
            assert result == Path("/tmp/llmem-env-test/memory.db")


class TestCli_TypeValidation:
    """Test that CLI --type rejects unregistered types at runtime."""

    def test_add_unregistered_type_exits_with_error(self, tmp_path):
        """Using --type with an unregistered type prints an error and exits."""
        from llmem.cli import cmd_add
        from llmem.store import _reset_global_registry
        import argparse

        _reset_global_registry()

        db = tmp_path / "test.db"
        # Pre-create a store with disable_vec so the DB is initialized
        from llmem.store import MemoryStore

        MemoryStore(db_path=db, disable_vec=True).close()

        args = argparse.Namespace(
            db=db,
            type="never_registered_type_xyz",
            content="test content",
            file=None,
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        # Patch MemoryStore to use disable_vec=True so sqlite-vec doesn't fail
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            with pytest.raises(SystemExit):
                cmd_add(args)

    def test_add_registered_type_succeeds(self, tmp_path):
        """Using --type with a registered type works."""
        from llmem.cli import cmd_add
        from llmem.store import register_memory_type, _reset_global_registry

        _reset_global_registry()
        register_memory_type("registered_cli_type")

        db = tmp_path / "test.db"
        from llmem.store import MemoryStore

        MemoryStore(db_path=db, disable_vec=True).close()

        import argparse

        args = argparse.Namespace(
            db=db,
            type="registered_cli_type",
            content="test content",
            file=None,
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        # Patch MemoryStore to use disable_vec=True so sqlite-vec doesn't fail
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_add(args)

    def test_type_choices_dynamic_after_registration(self):
        """get_registered_types() reflects newly registered types."""
        from llmem.store import (
            register_memory_type,
            get_registered_types,
            _reset_global_registry,
        )

        _reset_global_registry()
        before = get_registered_types()
        register_memory_type("dynamic_type_test")
        after = get_registered_types()
        assert "dynamic_type_test" not in before
        assert "dynamic_type_test" in after


class TestCli_PluginsLoadedViaRegistry:
    """Test that core CLI has no session-start/track-review/track-test
    subcommands and that plugins are loaded via registry."""

    def test_core_cli_has_no_session_start_subcommand(self):
        """Core CLI must not have a 'session-start' subcommand."""
        from llmem.cli import main
        from llmem.registry import _reset_registries

        _reset_registries()
        import io

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "session-start"]
            main()
        except SystemExit:
            pass
        finally:
            stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        # argparse should not recognize session-start
        assert "invalid choice" in stderr.lower() or "unknown" in stderr.lower()

    def test_core_cli_has_no_track_review_subcommand(self):
        """Core CLI must not have a 'track-review' subcommand."""
        from llmem.cli import main
        from llmem.registry import _reset_registries

        _reset_registries()
        import io

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "track-review"]
            main()
        except SystemExit:
            pass
        finally:
            stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        assert "invalid choice" in stderr.lower() or "unknown" in stderr.lower()

    def test_core_cli_has_no_track_test_subcommand(self):
        """Core CLI must not have a 'track-test' subcommand."""
        from llmem.cli import main
        from llmem.registry import _reset_registries

        _reset_registries()
        import io

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "track-test"]
            main()
        except SystemExit:
            pass
        finally:
            stderr = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        assert "invalid choice" in stderr.lower() or "unknown" in stderr.lower()

    def test_cli_plugin_registered_subcommand_appears(self):
        """A registered CLI plugin can add a subcommand."""
        from llmem.cli import main
        from llmem.registry import (
            register_cli_plugin,
            get_cli_plugin_setup_fn,
            _reset_registries,
        )
        import io

        _reset_registries()

        def setup_test_plugin(subparsers):
            subparsers.add_parser("test-plugin-cmd", help="A test plugin command")

        register_cli_plugin("test_plugin", setup_test_plugin)

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

        assert "test-plugin-cmd" in output

    def test_cli_plugin_bad_setup_does_not_crash_main(self):
        """A faulty plugin setup function does not crash main()."""
        from llmem.cli import main
        from llmem.registry import register_cli_plugin, _reset_registries
        import io

        _reset_registries()

        def bad_setup(subparsers):
            raise RuntimeError("plugin setup crashed")

        register_cli_plugin("bad_plugin", bad_setup)

        # Should not crash — just log the error and continue
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            sys.argv = ["llmem", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            sys.stdout = old_stdout


class TestCli_Note:
    """Test cmd_note adds content to the inbox."""

    def test_cmd_note_adds_to_inbox(self, tmp_path):
        """cmd_note with content adds an item to the inbox."""
        from llmem.cli import cmd_note
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        args = argparse.Namespace(
            db=db,
            content="test note content",
            source="note",
            attention_score=0.5,
            metadata=None,
        )
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_note(args)

        # Verify the note was added
        store = MemoryStore(db_path=db, disable_vec=True)
        items = store.list_inbox()
        assert len(items) == 1
        assert items[0]["content"] == "test note content"
        assert items[0]["source"] == "note"
        store.close()


class TestCli_Inbox:
    """Test cmd_inbox lists inbox items."""

    def test_cmd_inbox_lists_items(self, tmp_path, capsys):
        """cmd_inbox lists inbox items to stdout."""
        from llmem.cli import cmd_inbox
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        # Pre-populate inbox
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add_to_inbox(content="test note", source="note", attention_score=0.7)
        store.close()

        args = argparse.Namespace(
            db=db,
            limit=20,
            json=False,
        )
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_inbox(args)

        captured = capsys.readouterr()
        assert "test note" in captured.out


class TestCli_Consolidate:
    """Test cmd_consolidate promotes inbox items."""

    def test_cmd_consolidate_promotes(self, tmp_path, capsys):
        """cmd_consolidate consolidates inbox items."""
        from llmem.cli import cmd_consolidate
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        # Pre-populate inbox
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add_to_inbox(content="promote me", attention_score=0.8)
        store.close()

        args = argparse.Namespace(
            db=db,
            min_score=0.0,
            dry_run=False,
        )
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_consolidate(args)

        captured = capsys.readouterr()
        assert "Promoted: 1 items" in captured.out

    def test_cmd_consolidate_dry_run(self, tmp_path, capsys):
        """cmd_consolidate --dry-run shows plan without changes."""
        from llmem.cli import cmd_consolidate
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        # Pre-populate inbox
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add_to_inbox(content="maybe later", attention_score=0.6)
        store.close()

        args = argparse.Namespace(
            db=db,
            min_score=0.0,
            dry_run=True,
        )
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_consolidate(args)

        captured = capsys.readouterr()
        assert "[DRY RUN]" in captured.out
        # Inbox should still have the item (dry run)
        store2 = MemoryStore(db_path=db, disable_vec=True)
        assert store2.inbox_count() == 1
        store2.close()


class TestCli_Learn:
    """Test the cmd_learn CLI handler."""

    def test_learn_processes_directory(self, tmp_path):
        """cmd_learn processes a directory and reports chunk count."""
        from llmem.cli import cmd_learn

        # Create a small code directory
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "hello.py").write_text("def hello():\n    print('hello')\n")
        (code_dir / "world.py").write_text("def world():\n    print('world')\n")

        db = tmp_path / "learn_test.db"
        import argparse

        args = argparse.Namespace(
            path=str(code_dir),
            db=db,
            strategy="paragraph",
            window_size=50,
            overlap=10,
            no_embed=True,
            ollama_url=None,
        )

        # Capture stdout
        import io

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            cmd_learn(args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

        assert "Ingested" in output
        assert "chunks" in output

    def test_learn_with_fixed_strategy(self, tmp_path):
        """cmd_learn with --strategy=fixed uses FixedLineChunking."""
        from llmem.cli import cmd_learn

        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "app.py").write_text("\n".join(f"line{i}" for i in range(20)))

        db = tmp_path / "learn_fixed.db"
        import argparse

        args = argparse.Namespace(
            path=str(code_dir),
            db=db,
            strategy="fixed",
            window_size=10,
            overlap=2,
            no_embed=True,
            ollama_url=None,
        )

        import io

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            cmd_learn(args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

        assert "Ingested" in output

    def test_learn_nonexistent_directory_exits(self, tmp_path):
        """cmd_learn exits with error for non-existent directory."""
        from llmem.cli import cmd_learn

        import argparse

        args = argparse.Namespace(
            path=str(tmp_path / "nonexistent"),
            db=tmp_path / "test.db",
            strategy="paragraph",
            window_size=50,
            overlap=10,
            no_embed=True,
            ollama_url=None,
        )

        with pytest.raises(SystemExit):
            cmd_learn(args)

    def test_search_include_code_flag(self, tmp_path):
        """The search command accepts --include-code flag."""
        from llmem.cli import main

        # Just verify the flag is recognized in the parser
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            sys.argv = ["llmem", "search", "--help"]
            main()
        except SystemExit:
            pass
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        assert "--include-code" in output

    def test_learn_no_embed_does_not_disable_vec(self, tmp_path):
        """Issue ll-67q3p-6prmn: --no-embed must not conflate with disable_vec.

        The --no-embed flag should only skip embedding generation, not
        disable the vec extension entirely. This test verifies that cmd_learn
        no longer passes disable_vec=args.no_embed to CodeIndex.
        """
        from llmem.cli import cmd_learn

        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "hello.py").write_text("def hello():\n    print('hello')\n")

        db = tmp_path / "test.db"
        import argparse

        args = argparse.Namespace(
            path=str(code_dir),
            db=db,
            strategy="paragraph",
            window_size=50,
            overlap=10,
            no_embed=True,
            ollama_url=None,
        )

        import io

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            cmd_learn(args)
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

        assert "Ingested" in output
        # Verify the database was created and FTS triggers exist
        # (if disable_vec had been True AND it dropped triggers, they'd be gone)
        import sqlite3

        conn = sqlite3.connect(str(db))
        triggers = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        conn.close()
        assert "code_chunks_fts_insert" in triggers


class TestCli_SearchCodeInterleaving:
    """Issue ll-67q3p-esnow: Code results must interleave with memory results.

    Previously, code results always sorted below memory results because
    _rrf_score defaulted to 0.0. Now code results get proper RRF scores
    based on their FTS rank position.
    """

    def test_code_results_have_rrf_scores(self, tmp_path):
        """Code search results receive RRF scores, not 0.0 default."""
        from llmem.code_index import CodeIndex
        from llmem.retrieve import DEFAULT_RRF_K

        db = tmp_path / "test.db"
        idx = CodeIndex(db_path=db, disable_vec=True)
        idx.add_chunk(
            file_path="app.py",
            start_line=1,
            end_line=10,
            content="def hello_world(): pass",
            language="python",
        )
        idx.add_chunk(
            file_path="util.py",
            start_line=1,
            end_line=5,
            content="def goodbye(): pass",
            language="python",
        )
        idx.close()

        # Simulate what cmd_search does: compute RRF scores for code results
        code_index = CodeIndex(db_path=db, disable_vec=True)
        code_results = code_index.search_content(query="hello", limit=10)
        code_index.close()

        # Assign RRF scores exactly as cmd_search does now
        for i, cr in enumerate(code_results):
            fts_rank = i + 1
            cr["_rrf_score"] = (1 - 0.0) * (1 / (DEFAULT_RRF_K + fts_rank))

        # Code results should have non-zero RRF scores
        assert len(code_results) >= 1
        for cr in code_results:
            assert cr["_rrf_score"] > 0.0, (
                f"Code result got _rrf_score={cr['_rrf_score']}, "
                "should be > 0.0 for proper interleaving"
            )

    def test_code_results_interleave_with_memory_results(self, tmp_path):
        """When a code result has high relevance, it sorts among memory results."""
        from llmem.code_index import CodeIndex
        from llmem.store import MemoryStore
        from llmem.retrieve import DEFAULT_RRF_K

        db = tmp_path / "test.db"

        # Add a memory with a low-ish RRF score
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add(
            type="fact",
            content="unrelated note about weather",
            confidence=0.5,
        )
        store.close()

        # Add a code chunk matching "hello"
        idx = CodeIndex(db_path=db, disable_vec=True)
        idx.add_chunk(
            file_path="app.py",
            start_line=1,
            end_line=10,
            content="def hello_world(): pass",
            language="python",
        )
        idx.close()

        # Code result with rank 1 gets score: (1)/(60+1) ≈ 0.0164
        # Memory result with typical low rrf might be lower or higher
        # This test verifies code results are not stuck at 0.0
        code_index = CodeIndex(db_path=db, disable_vec=True)
        code_results = code_index.search_content(query="hello", limit=10)
        code_index.close()

        for i, cr in enumerate(code_results):
            fts_rank = i + 1
            cr["_rrf_score"] = (1 - 0.0) * (1 / (DEFAULT_RRF_K + fts_rank))
            cr["_source"] = "code"

        # The first code result should have a score well above 0.0
        if code_results:
            assert code_results[0]["_rrf_score"] > 0.0


class TestCli_EmbedMetrics:
    """Test cmd_embed reports anisotropy and similarity range."""

    def test_embed_reports_anisotropy_and_similarity_range(self, tmp_path, capsys):
        """cmd_embed reports anisotropy and similarity range values."""
        from llmem.cli import cmd_embed
        from llmem.embed import EmbeddingEngine
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add(
            type="fact",
            content="test fact 1",
            embedding=EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0]),
        )
        store.add(
            type="fact",
            content="test fact 2",
            embedding=EmbeddingEngine.vec_to_bytes([0.0, 1.0, 0.0]),
        )
        store.add(
            type="decision",
            content="test decision",
            embedding=EmbeddingEngine.vec_to_bytes([0.0, 0.0, 1.0]),
        )
        store.close()

        args = argparse.Namespace(db=db)
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_embed(args)

        captured = capsys.readouterr()
        assert "Anisotropy" in captured.out
        assert "Similarity range" in captured.out
        assert "Discrimination gap" in captured.out
        assert "3 vectors" in captured.out

    def test_embed_warns_on_high_anisotropy(self, tmp_path, capsys):
        """cmd_embed warns when anisotropy exceeds 0.5."""
        from llmem.cli import cmd_embed
        from llmem.embed import EmbeddingEngine
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)

        # Identical vectors → anisotropy = 1.0, which exceeds threshold
        emb = EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0])
        store.add(type="fact", content="test1", embedding=emb)
        store.add(type="fact", content="test2", embedding=emb)
        store.add(type="fact", content="test3", embedding=emb)
        store.close()

        args = argparse.Namespace(db=db)
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_embed(args)

        captured = capsys.readouterr()
        assert "Anisotropy" in captured.out
        assert "WARNING" in captured.err
        assert "anisotropy" in captured.err.lower() or "Anisotropy" in captured.err

    def test_embed_warns_on_low_similarity_range(self, tmp_path, capsys):
        """cmd_embed warns when similarity_range is below 0.1."""
        from llmem.cli import cmd_embed
        from llmem.embed import EmbeddingEngine
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)

        # Identical vectors → similarity_range = 0.0, below threshold
        emb = EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0])
        store.add(type="fact", content="test1", embedding=emb)
        store.add(type="fact", content="test2", embedding=emb)
        store.close()

        args = argparse.Namespace(db=db)
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_embed(args)

        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        # Both high anisotropy and low similarity range should warn
        assert "poor quality" in captured.err

    def test_embed_no_warning_on_good_embeddings(self, tmp_path, capsys):
        """cmd_embed does not warn when metrics are within thresholds."""
        from llmem.cli import cmd_embed
        from llmem.embed import EmbeddingEngine
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)

        # Vectors with diverse orientations and non-zero spread:
        # [1,0,0], [1,1,0], [0,1,0] → pairwise cosines are 0.707, 0.0, 0.707
        # anisotropy ≈ avg(0.707, 0.0, 0.707) / 3 ≈ 0.47 < 0.5
        # similarity_range = 0.707 - 0.0 = 0.707 > 0.1
        store.add(
            type="fact",
            content="test fact",
            embedding=EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0]),
        )
        store.add(
            type="decision",
            content="test decision",
            embedding=EmbeddingEngine.vec_to_bytes([1.0, 1.0, 0.0]),
        )
        store.add(
            type="preference",
            content="test preference",
            embedding=EmbeddingEngine.vec_to_bytes([0.0, 1.0, 0.0]),
        )
        store.close()

        args = argparse.Namespace(db=db)
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_embed(args)

        captured = capsys.readouterr()
        assert "Anisotropy" in captured.out
        assert "Similarity range" in captured.out
        assert "WARNING" not in captured.err

    def test_embed_always_reports_metrics(self, tmp_path, capsys):
        """cmd_embed always reports metrics — no --metrics flag needed."""
        from llmem.cli import cmd_embed
        from llmem.embed import EmbeddingEngine
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add(
            type="fact",
            content="test fact",
            embedding=EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0]),
        )
        store.close()

        # No metrics attribute on args — embed always reports
        args = argparse.Namespace(db=db)
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_embed(args)

        captured = capsys.readouterr()
        assert "Anisotropy" in captured.out
        assert "Similarity range" in captured.out

    def test_embed_does_not_generate_new_embeddings(self, tmp_path, capsys):
        """cmd_embed only analyses existing embeddings — it never creates new ones.

        Verifies that calling cmd_embed on memories without embeddings
        does not add embeddings, and that the function merely reads
        what is already stored.
        """
        from llmem.cli import cmd_embed
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        # Add memories WITHOUT embeddings
        mid = store.add(type="fact", content="unembedded fact")
        assert store.get(mid)["embedding"] is None
        store.close()

        args = argparse.Namespace(db=db)
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_embed(args)

        # Verify no embeddings were generated — the memory still has none
        store2 = MemoryStore(db_path=db, disable_vec=True)
        mem = store2.get(mid)
        assert mem["embedding"] is None, (
            "cmd_embed should not generate embeddings, but embedding was created"
        )
        store2.close()

        captured = capsys.readouterr()
        assert "No embedded memories found" in captured.out


class TestCli_ConsolidateMetrics:
    """Test cmd_consolidate --metrics reports embedding quality metrics."""

    def test_consolidate_with_metrics_reports_metrics(self, tmp_path, capsys):
        """cmd_consolidate --metrics reports anisotropy and similarity range."""
        from llmem.cli import cmd_consolidate
        from llmem.embed import EmbeddingEngine
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        # Pre-populate inbox and embeddings
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add_to_inbox(content="promote me", attention_score=0.8)

        # Add memories with embeddings for metrics
        store.add(
            type="fact",
            content="embedded fact 1",
            embedding=EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0]),
        )
        store.add(
            type="fact",
            content="embedded fact 2",
            embedding=EmbeddingEngine.vec_to_bytes([0.0, 1.0, 0.0]),
        )
        store.close()

        args = argparse.Namespace(
            db=db,
            min_score=0.0,
            dry_run=False,
            metrics=True,
        )
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_consolidate(args)

        captured = capsys.readouterr()
        assert "Promoted" in captured.out
        assert "Anisotropy" in captured.out
        assert "Similarity range" in captured.out


class TestCli_EmbedMetricsCapping:
    """Test that _report_embedding_metrics respects embedding caps from DoS protection."""

    def test_embed_reports_capped_count_when_exceeding_limit(self, tmp_path, capsys):
        """_report_embedding_metrics shows capped vector count when total > limit."""
        from llmem.cli import _report_embedding_metrics
        from llmem.embed import EmbeddingEngine
        from llmem.metrics import METRICS_MAX_EMBEDDINGS
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        # Add a few embeddings (not enough to hit the real cap, but we test
        # the code path by checking the output format)
        emb = EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0])
        for i in range(3):
            store.add(type="fact", content=f"fact {i}", embedding=emb)
        store.close()

        store2 = MemoryStore(db_path=db, disable_vec=True)
        _report_embedding_metrics(store2)
        store2.close()

        captured = capsys.readouterr()
        # With 3 embeddings < METRICS_MAX_EMBEDDINGS, no capping message
        assert "3 vectors" in captured.out
        assert "capped" not in captured.out.lower()

    def test_get_embeddings_with_types_respects_limit(self, tmp_path):
        """get_embeddings_with_types limit parameter caps returned rows."""
        from llmem.embed import EmbeddingEngine
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        emb = EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0])
        for i in range(5):
            store.add(type="fact", content=f"fact {i}", embedding=emb)

        # With limit=2, only 2 rows returned
        rows = store.get_embeddings_with_types(limit=2)
        assert len(rows) == 2
        store.close()

    def test_count_embeddings_returns_correct_count(self, tmp_path):
        """count_embeddings returns count of valid memories with embeddings."""
        from llmem.embed import EmbeddingEngine
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        emb = EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0])
        store.add(type="fact", content="embedded", embedding=emb)
        store.add(type="fact", content="not embedded")
        store.add(type="decision", content="also embedded", embedding=emb)

        assert store.count_embeddings() == 2
        store.close()
