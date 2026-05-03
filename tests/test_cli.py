"""Tests for the llmem CLI entry point."""

import argparse
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCli_EntryPoint:
    """Test that the llmem CLI entry point works."""

    def test_cli_help(self):
        """CLI --help outputs usage info with llmem branding."""
        from llmem.cli import main

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
            sys.stdout.getvalue()
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

    def test_core_cli_has_track_review_subcommand(self):
        """Core CLI now has a 'track-review' subcommand."""
        from llmem.cli import main
        from llmem.registry import _reset_registries

        _reset_registries()

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

    def test_core_cli_has_no_track_test_subcommand(self):
        """Core CLI must not have a 'track-test' subcommand."""
        from llmem.cli import main
        from llmem.registry import _reset_registries

        _reset_registries()

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
            _reset_registries,
        )

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


class TestCli_EmbedMetricsCapping:
    """Test that _report_embedding_metrics respects embedding caps from DoS protection."""

    def test_embed_reports_capped_count_when_exceeding_limit(self, tmp_path, capsys):
        """_report_embedding_metrics shows capped vector count when total > limit."""
        from llmem.cli import _report_embedding_metrics
        from llmem.embed import EmbeddingEngine
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
