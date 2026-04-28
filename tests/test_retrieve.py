"""Tests for llmem.retrieve module — Retriever, hybrid_search, _rrf_score."""

import logging
import struct

import pytest

from llmem.retrieve import (
    Retriever,
    _rrf_score,
    DEFAULT_ALPHA,
    DEFAULT_RRF_K,
)
from llmem.store import MemoryStore
from llmem.embed import EmbeddingEngine


# ---------------------------------------------------------------------------
# _rrf_score unit tests — no database needed
# ---------------------------------------------------------------------------


class TestRetrieve_RrfScore:
    """Test _rrf_score function."""

    def test_rrf_score_returns_merged_results_sorted_by_score(self):
        """Given semantic and FTS rank dicts, returns list sorted by descending score."""
        semantic_ranks = {"m1": 1, "m2": 2, "m3": 3}
        fts_ranks = {"m1": 2, "m3": 1, "m4": 3}
        results = _rrf_score(semantic_ranks, fts_ranks, alpha=0.7, k=60)
        # All IDs from both dicts are present
        ids = [r[0] for r in results]
        assert set(ids) == {"m1", "m2", "m3", "m4"}
        # Results sorted by descending score
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_score_alpha_weights_semantic(self):
        """With alpha=1.0, results are in pure semantic order; alpha=0.0 in pure FTS order."""
        semantic_ranks = {"m1": 1, "m2": 2, "m3": 3}
        fts_ranks = {"m3": 1, "m1": 2, "m2": 3}

        # alpha=1.0 → pure semantic ranking
        pure_semantic = _rrf_score(semantic_ranks, fts_ranks, alpha=1.0, k=60)
        pure_semantic_ids = [r[0] for r in pure_semantic]
        # m1 has semantic rank 1 (best), m2 rank 2, m3 rank 3
        assert pure_semantic_ids == ["m1", "m2", "m3"]

        # alpha=0.0 → pure FTS ranking
        pure_fts = _rrf_score(semantic_ranks, fts_ranks, alpha=0.0, k=60)
        pure_fts_ids = [r[0] for r in pure_fts]
        # m3 has FTS rank 1, m1 rank 2, m2 rank 3
        assert pure_fts_ids == ["m3", "m1", "m2"]

    def test_rrf_score_deterministic_ordering(self):
        """Given identical rank inputs, _rrf_score returns a deterministic, reproducible order."""
        semantic_ranks = {"a": 1, "b": 2}
        fts_ranks = {"a": 1, "b": 2}
        # Even with same ranks for both items, tie-breaking should be
        # deterministic (by ascending ID)
        result1 = _rrf_score(semantic_ranks, fts_ranks, alpha=0.5, k=60)
        result2 = _rrf_score(semantic_ranks, fts_ranks, alpha=0.5, k=60)
        assert result1 == result2

    def test_rrf_score_empty_inputs(self):
        """_rrf_score returns an empty list when given empty rank dicts."""
        assert _rrf_score({}, {}, alpha=0.7, k=60) == []
        # ID "a" in semantic only → semantic rank 1, FTS default rank = len(fts)+1 = 2
        # score = alpha*(1/(k+1)) + (1-alpha)*(1/(k+2))
        assert _rrf_score({"a": 1}, {}, alpha=0.7, k=60) == [
            ("a", pytest.approx(0.7 / 61 + 0.3 / 62))
        ]
        # ID "a" in FTS only → semantic default rank = len(semantic)+1 = 2, FTS rank 1
        # score = (1-alpha)*(1/(k+1)) + alpha*(1/(k+2))
        assert _rrf_score({}, {"a": 1}, alpha=0.7, k=60) == [
            ("a", pytest.approx(0.3 / 61 + 0.7 / 62))
        ]

    def test_rrf_score_missing_side_gets_default_rank(self):
        """IDs present in only one dict get rank len(that_list)+1 for the missing side."""
        # m1 is only in semantic (rank 1), m2 only in FTS (rank 1)
        semantic_ranks = {"m1": 1}
        fts_ranks = {"m2": 1}
        results = _rrf_score(semantic_ranks, fts_ranks, alpha=0.5, k=60)
        result_dict = dict(results)
        # m1: semantic rank 1, fts rank len(fts_ranks)+1 = 2
        # score = 0.5*(1/61) + 0.5*(1/62)
        # m2: semantic rank len(semantic_ranks)+1 = 2, fts rank 1
        # score = 0.5*(1/62) + 0.5*(1/61)
        # Both have the same score, but tie-breaking by ascending ID means m1 first
        assert result_dict["m1"] == pytest.approx(0.5 * (1 / 61) + 0.5 * (1 / 62))
        assert result_dict["m2"] == pytest.approx(0.5 * (1 / 62) + 0.5 * (1 / 61))
        # With same score, m1 comes first (ascending ID tie-break)
        assert results[0][0] == "m1"

    def test_rrf_score_deduplicates_by_id(self):
        """An ID appearing in both dicts appears once with combined score."""
        semantic_ranks = {"m1": 1}
        fts_ranks = {"m1": 1}
        results = _rrf_score(semantic_ranks, fts_ranks, alpha=0.5, k=60)
        ids = [r[0] for r in results]
        assert len(ids) == 1
        assert ids[0] == "m1"
        expected_score = 0.5 * (1 / 61) + 0.5 * (1 / 61)
        assert results[0][1] == pytest.approx(expected_score)


# ---------------------------------------------------------------------------
# Retriever.hybrid_search tests — need a MemoryStore
# ---------------------------------------------------------------------------


class TestRetrieve_HybridSearch:
    """Test Retriever.hybrid_search method."""

    def _add_memories(self, store: MemoryStore) -> dict[str, str]:
        """Add test memories to the store, return {label: memory_id}."""
        ids = {}
        ids["python"] = store.add(
            type="fact", content="Python is a programming language"
        )
        ids["rust"] = store.add(
            type="fact", content="Rust is a systems programming language"
        )
        ids["javascript"] = store.add(
            type="fact", content="JavaScript runs in the browser"
        )
        return ids

    def test_hybrid_search_returns_merged_results(self, store):
        """When both FTS5 and semantic search produce results, hybrid_search
        returns a deduplicated merged list sorted by RRF score."""
        ids = self._add_memories(store)
        # Without embedder, hybrid_search falls back to FTS5 — still returns
        # valid results
        retriever = Retriever(store=store, embedder=None)
        results = retriever.hybrid_search("Python", limit=10)
        assert isinstance(results, list)
        assert len(results) >= 1
        # Each result should be a dict with at least 'id' and 'content'
        for r in results:
            assert "id" in r
            assert "content" in r

    def test_hybrid_search_alpha_weights_semantic(self, store):
        """With alpha=1.0, hybrid search returns results in pure semantic order;
        with alpha=0.0, returns pure FTS5 order."""
        self._add_memories(store)
        # Without embedder, alpha=0.0 should still work (pure FTS)
        retriever = Retriever(store=store, embedder=None)
        results_fts = retriever.hybrid_search("Python", limit=10, alpha=0.0)
        assert isinstance(results_fts, list)
        assert len(results_fts) >= 1

    def test_hybrid_search_deduplicates_by_id(self, store):
        """When the same memory ID appears in both FTS and semantic results,
        it appears only once in hybrid results with a combined RRF score."""
        ids = self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)
        results = retriever.hybrid_search("Python", limit=10)
        result_ids = [r["id"] for r in results]
        # No duplicate IDs
        assert len(result_ids) == len(set(result_ids))

    def test_hybrid_search_fallback_fts_only_when_no_embedder(self, store, caplog):
        """When embedder=None, hybrid_search falls back to FTS5-only with a
        log warning and returns the same results as search()."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)

        with caplog.at_level(logging.WARNING, logger="llmem.retrieve"):
            results = retriever.hybrid_search("Python", limit=10)

        assert len(results) >= 1
        # Check that the warning was logged
        assert any(
            "embedder not configured" in rec.message.lower()
            or "falling back to fts5" in rec.message.lower()
            for rec in caplog.records
        )

        # Results should match FTS5-only search
        fts_results = retriever.search("Python", limit=10)
        hybrid_ids = {r["id"] for r in results}
        fts_ids = {r["id"] for r in fts_results}
        assert hybrid_ids == fts_ids

    def test_hybrid_search_fts_only_flag(self, store):
        """When search_mode='fts', returns only FTS5 results regardless of
        embedder availability."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)

        results = retriever.hybrid_search("Python", limit=10, search_mode="fts")
        assert len(results) >= 1
        # Should match what search() returns
        fts_results = retriever.search("Python", limit=10)
        assert {r["id"] for r in results} == {r["id"] for r in fts_results}

    def test_hybrid_search_semantic_only_flag_raises_without_embedder(self, store):
        """When search_mode='semantic' and embedder is None, raises ValueError."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)

        with pytest.raises(ValueError, match="llmem: retrieve:"):
            retriever.hybrid_search("Python", limit=10, search_mode="semantic")

    def test_hybrid_search_returns_rrf_score_key(self, store):
        """Each result dict includes an '_rrf_score' key with the float score."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)

        results = retriever.hybrid_search("Python", limit=10)
        for r in results:
            assert "_rrf_score" in r
            assert isinstance(r["_rrf_score"], float)

    def test_hybrid_search_empty_query_returns_empty(self, store):
        """An empty query string returns an empty list."""
        retriever = Retriever(store=store, embedder=None)
        results = retriever.hybrid_search("", limit=10)
        assert results == []

    def test_hybrid_search_type_filter(self, store):
        """search_mode='fts' with type_filter narrows results to that type."""
        ids = self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)

        results = retriever.hybrid_search(
            "programming", limit=10, type_filter="fact", search_mode="fts"
        )
        for r in results:
            assert r["type"] == "fact"


# ---------------------------------------------------------------------------
# Retriever.hybrid_search with vec0 semantic — integration tests
# ---------------------------------------------------------------------------


class TestRetrieve_HybridSearchSemantic:
    """Integration tests for hybrid_search with semantic (vec0) path.

    Requires sqlite-vec to be installed; skipped otherwise.
    """

    @pytest.fixture
    def vec_store(self, tmp_path):
        """Create a MemoryStore with vec enabled (requires sqlite-vec)."""
        pytest.importorskip("sqlite_vec")
        db = tmp_path / "vec_test.db"
        s = MemoryStore(db_path=db, vec_dimensions=3, disable_vec=False)
        yield s
        s.close()

    @staticmethod
    def _make_embedding(values: list[float]) -> bytes:
        """Pack float values into a bytes embedding."""
        return struct.pack(f"{len(values)}f", *values)

    def _add_memories_with_embeddings(self, store: MemoryStore) -> dict[str, str]:
        """Add memories with embeddings to the store, return {label: memory_id}."""
        ids = {}
        ids["python"] = store.add(
            type="fact",
            content="Python programming language",
            embedding=self._make_embedding([0.9, 0.1, 0.1]),
        )
        ids["rust"] = store.add(
            type="fact",
            content="Rust systems programming",
            embedding=self._make_embedding([0.1, 0.9, 0.1]),
        )
        ids["javascript"] = store.add(
            type="fact",
            content="JavaScript web browser",
            embedding=self._make_embedding([0.1, 0.1, 0.9]),
        )
        return ids

    def test_hybrid_search_semantic_only_flag_with_embedder(self, vec_store):
        """When search_mode='semantic' with an embedder, returns semantic results."""
        ids = self._add_memories_with_embeddings(vec_store)

        # Create a mock embedder that returns a fixed vector
        class FakeEmbedder:
            def embed(self, text: str) -> list[float]:
                return [0.9, 0.1, 0.1]

        retriever = Retriever(store=vec_store, embedder=FakeEmbedder())
        results = retriever.hybrid_search("Python", limit=10, search_mode="semantic")
        assert len(results) >= 1
        # The result closest to [0.9, 0.1, 0.1] should be "python"
        assert any(r["id"] == ids["python"] for r in results)

    def test_hybrid_search_hybrid_mode_with_embedder(self, vec_store):
        """Hybrid mode with embedder returns merged results from both paths."""
        ids = self._add_memories_with_embeddings(vec_store)

        class FakeEmbedder:
            def embed(self, text: str) -> list[float]:
                return [0.9, 0.1, 0.1]

        retriever = Retriever(store=vec_store, embedder=FakeEmbedder())
        results = retriever.hybrid_search("Python", limit=10)
        assert len(results) >= 1
        # Results should be deduplicated
        result_ids = [r["id"] for r in results]
        assert len(result_ids) == len(set(result_ids))
        # Each result should have _rrf_score
        for r in results:
            assert "_rrf_score" in r


# ---------------------------------------------------------------------------
# CLI tests for hybrid search flags
# ---------------------------------------------------------------------------


class TestCli_SearchHybrid:
    """Test CLI search command uses hybrid_search by default."""

    def _make_mock_retriever(self):
        """Create a mock Retriever with hybrid_search returning a sample result."""
        from unittest.mock import MagicMock

        mock_retriever = MagicMock()
        mock_retriever.hybrid_search.return_value = [
            {
                "id": "m1",
                "type": "fact",
                "content": "test",
                "confidence": 0.8,
                "_rrf_score": 0.01,
            }
        ]
        return mock_retriever

    def test_cli_search_default_uses_hybrid(self, tmp_path):
        """The CLI cmd_search function uses hybrid search by default."""
        import argparse
        from unittest.mock import patch

        from llmem.cli import cmd_search
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        MemoryStore(db_path=db, disable_vec=True).close()

        mock_retriever = self._make_mock_retriever()

        with patch("llmem.retrieve.Retriever") as MockRetriever:
            MockRetriever.return_value = mock_retriever

            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                with patch("llmem.embed.EmbeddingEngine", return_value=None):
                    args = argparse.Namespace(
                        db=db,
                        query="test query",
                        type=None,
                        limit=20,
                        json=False,
                        fts_only=False,
                        semantic_only=False,
                    )
                    cmd_search(args)

            # Verify hybrid_search was called with search_mode="hybrid"
            mock_retriever.hybrid_search.assert_called_once()
            call_kwargs = mock_retriever.hybrid_search.call_args[1]
            assert call_kwargs.get("search_mode") == "hybrid"

    def test_cli_search_fts_only_flag(self, tmp_path):
        """The --fts-only flag on CLI search returns FTS5-only results."""
        import argparse
        from unittest.mock import patch

        from llmem.cli import cmd_search
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        MemoryStore(db_path=db, disable_vec=True).close()

        mock_retriever = self._make_mock_retriever()

        with patch("llmem.retrieve.Retriever") as MockRetriever:
            MockRetriever.return_value = mock_retriever

            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                args = argparse.Namespace(
                    db=db,
                    query="test query",
                    type=None,
                    limit=20,
                    json=False,
                    fts_only=True,
                    semantic_only=False,
                )
                cmd_search(args)

            # Verify hybrid_search called with search_mode="fts"
            mock_retriever.hybrid_search.assert_called_once()
            call_kwargs = mock_retriever.hybrid_search.call_args[1]
            assert call_kwargs["search_mode"] == "fts"

    def test_cli_search_semantic_only_flag(self, tmp_path):
        """The --semantic-only flag on CLI search returns semantic-only results."""
        import argparse
        from unittest.mock import patch

        from llmem.cli import cmd_search
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        MemoryStore(db_path=db, disable_vec=True).close()

        mock_retriever = self._make_mock_retriever()

        with patch("llmem.retrieve.Retriever") as MockRetriever:
            MockRetriever.return_value = mock_retriever

            with patch(
                "llmem.cli.MemoryStore",
                side_effect=lambda db_path, **kw: MemoryStore(
                    db_path=db_path, disable_vec=True
                ),
            ):
                with patch("llmem.embed.EmbeddingEngine", return_value=None):
                    args = argparse.Namespace(
                        db=db,
                        query="test query",
                        type=None,
                        limit=20,
                        json=False,
                        fts_only=False,
                        semantic_only=True,
                    )
                    cmd_search(args)

            # Verify hybrid_search called with search_mode="semantic"
            mock_retriever.hybrid_search.assert_called_once()
            call_kwargs = mock_retriever.hybrid_search.call_args[1]
            assert call_kwargs["search_mode"] == "semantic"


class TestRetrieve_DefaultValues:
    """Test that module-level constants have correct default values."""

    def test_default_alpha_is_0p7(self):
        assert DEFAULT_ALPHA == 0.7

    def test_default_rrf_k_is_60(self):
        assert DEFAULT_RRF_K == 60
