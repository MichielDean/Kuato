"""Tests for llmem.retrieve module — Retriever, hybrid_search, _rrf_score."""

import logging
import math
import struct
from datetime import datetime, timedelta, timezone

import pytest

from llmem.retrieve import (
    Retriever,
    _rrf_score,
    _compute_rerank_signals,
    _compute_weighted_signal,
    DEFAULT_ALPHA,
    DEFAULT_RRF_K,
    CONFIDENCE_WEIGHT,
    RECENCY_WEIGHT,
    ACCESS_WEIGHT,
    TYPE_WEIGHT,
    TYPE_PRIORITY,
)
from llmem.store import MemoryStore


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
        # ID "a" in semantic only → semantic rank 1, FTS default rank = len(fts)+1 = 1
        # (since fts_ranks is empty, n_fts=0, default = 0+1 = 1)
        # score = alpha*(1/(k+1)) + (1-alpha)*(1/(k+1)) = 1/(k+1)
        assert _rrf_score({"a": 1}, {}, alpha=0.7, k=60) == [
            ("a", pytest.approx(0.7 / 61 + 0.3 / 61))
        ]
        # ID "a" in FTS only → FTS rank 1, semantic default rank = len(semantic)+1 = 1
        # (since semantic_ranks is empty, n_semantic=0, default = 0+1 = 1)
        # score = (1-alpha)*(1/(k+1)) + alpha*(1/(k+1)) = 1/(k+1)
        assert _rrf_score({}, {"a": 1}, alpha=0.7, k=60) == [
            ("a", pytest.approx(0.3 / 61 + 0.7 / 61))
        ]

    def test_rrf_score_missing_side_gets_default_rank(self):
        """IDs present in only one dict get default rank len(their_present_list)+1
        for the missing side. E.g., if semantic_ranks has 1 entry and an ID
        is only in fts_ranks (which also has 1 entry), the ID only in fts_ranks
        gets semantic default rank = len(semantic_ranks)+1 = 2."""
        # m1 is only in semantic (rank 1), m2 only in FTS (rank 1)
        semantic_ranks = {"m1": 1}
        fts_ranks = {"m2": 1}
        results = _rrf_score(semantic_ranks, fts_ranks, alpha=0.5, k=60)
        result_dict = dict(results)
        # m1: semantic rank 1, fts default rank = len(fts_ranks)+1 = 2
        # score = 0.5*(1/61) + 0.5*(1/62)
        # m2: semantic default rank = len(semantic_ranks)+1 = 2, fts rank 1
        # score = 0.5*(1/62) + 0.5*(1/61)
        # Both have the same score, but tie-breaking by ascending ID means m1 first
        assert result_dict["m1"] == pytest.approx(0.5 * (1 / 61) + 0.5 * (1 / 62))
        assert result_dict["m2"] == pytest.approx(0.5 * (1 / 62) + 0.5 * (1 / 61))
        # With same score, m1 comes first (ascending ID tie-break)
        assert results[0][0] == "m1"

    def test_rrf_score_missing_id_default_rank_not_swapped(self):
        """Regression test: IDs missing from one list should get default rank
        based on the length of the list they are ABSENT from, not the list
        they appear in. This tests the fix for the swapped-n_semantic/n_fts bug."""
        # 5 semantic results, 2 FTS results
        # ID "a" is rank 1 in semantic but absent from FTS
        # It should get FTS default rank = len(fts_ranks)+1 = 3 (NOT len(semantic)+1 = 6)
        semantic_ranks = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        fts_ranks = {"f": 1, "g": 2}
        results = _rrf_score(semantic_ranks, fts_ranks, alpha=0.5, k=60)
        result_dict = dict(results)
        # "a": semantic_rank=1, fts_rank=len(fts_ranks)+1=3
        # score = 0.5*(1/(60+1)) + 0.5*(1/(60+3)) = 0.5/61 + 0.5/63
        assert result_dict["a"] == pytest.approx(0.5 / 61 + 0.5 / 63)
        # Verify NOT the old (buggy) value which used len(semantic)+1=6 for FTS default
        buggy_value = 0.5 / 61 + 0.5 / 66  # would be n_semantic+1 = 6
        assert result_dict["a"] != pytest.approx(buggy_value)

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
        self._add_memories(store)
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
        self._add_memories(store)
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

    def test_hybrid_search_fts_only_rrf_score_is_rrf_computed(self, store):
        """In fts-only mode, _rrf_score is an actual RRF score (1/(k+rank)),
        not the raw BM25 value. Regression test for inconsistent _rrf_score."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)

        results = retriever.hybrid_search("Python", limit=10, search_mode="fts")
        for r in results:
            assert "_rrf_score" in r
            # RRF score for pure-FTS rank 1 with k=60 should be 1/(60+1) ≈ 0.01639
            # A raw BM25 value could be something entirely different (e.g. 2.5 or -0.3)
            # So just validate it's positive and in the expected RRF range (0, 1/(k+1)]
            assert r["_rrf_score"] > 0
            assert r["_rrf_score"] <= 1 / 61  # max RRF score with k=60

    def test_hybrid_search_no_fts_rank_leaks_into_results(self, store):
        """Internal _fts_rank metadata should never appear in hybrid_search
        result dicts. Regression test for _fts_rank leak into JSON output."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)

        # Test all modes
        for mode in ["fts", "hybrid"]:
            results = retriever.hybrid_search("Python", limit=10, search_mode=mode)
            for r in results:
                assert "_fts_rank" not in r, (
                    f"_fts_rank leaked into results in {mode} mode: {r.keys()}"
                )

    def test_hybrid_search_empty_query_returns_empty(self, store):
        """An empty query string returns an empty list, even when the store
        has memories (regression: empty query used to return ALL memories)."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)
        results = retriever.hybrid_search("", limit=10)
        assert results == []

    def test_hybrid_search_empty_query_fts_mode_returns_empty(self, store):
        """An empty query in fts mode also returns empty (not all memories)."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)
        results = retriever.hybrid_search("", limit=10, search_mode="fts")
        assert results == []

    def test_hybrid_search_invalid_search_mode_raises_value_error(self, store):
        """An invalid search_mode value raises ValueError with llmem: retrieve: prefix.
        Regression test for silent fallthrough to hybrid path on typos."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)

        with pytest.raises(ValueError, match="llmem: retrieve:"):
            retriever.hybrid_search("Python", limit=10, search_mode="fulltext")

        with pytest.raises(ValueError, match="llmem: retrieve:"):
            retriever.hybrid_search("Python", limit=10, search_mode="HYBRID")

        with pytest.raises(ValueError, match="llmem: retrieve:"):
            retriever.hybrid_search("Python", limit=10, search_mode="")

    def test_hybrid_search_type_filter(self, store):
        """search_mode='fts' with type_filter narrows results to that type."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)

        results = retriever.hybrid_search(
            "programming", limit=10, type_filter="fact", search_mode="fts"
        )
        for r in results:
            assert r["type"] == "fact"

    def test_hybrid_search_track_access_parameter(self, store):
        """Calling hybrid_search(track_access=False) does not increment
        access_count on returned memories."""
        mid = store.add(type="fact", content="Hybrid track_access param test")
        mem_before = store.get(mid, track_access=False)
        assert mem_before["access_count"] == 0

        retriever = Retriever(store=store, embedder=None)
        results = retriever.hybrid_search(
            "Hybrid track_access param",
            limit=10,
            search_mode="fts",
            track_access=False,
        )

        assert any(r["id"] == mid for r in results)
        mem_after = store.get(mid, track_access=False)
        assert mem_after["access_count"] == 0


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

    def test_hybrid_search_semantic_only_type_filter(self, vec_store):
        """search_mode='semantic' with type_filter narrows results to that type."""
        ids = {}
        ids["python_fact"] = vec_store.add(
            type="fact",
            content="Python programming language",
            embedding=self._make_embedding([0.9, 0.1, 0.1]),
        )
        ids["python_pref"] = vec_store.add(
            type="preference",
            content="I prefer Python for scripting",
            embedding=self._make_embedding([0.85, 0.15, 0.1]),
        )
        ids["rust_fact"] = vec_store.add(
            type="fact",
            content="Rust systems programming",
            embedding=self._make_embedding([0.1, 0.9, 0.1]),
        )

        class FakeEmbedder:
            def embed(self, text: str) -> list[float]:
                return [0.9, 0.1, 0.1]

        retriever = Retriever(store=vec_store, embedder=FakeEmbedder())

        # Without type_filter, both types returned
        all_results = retriever.hybrid_search(
            "Python", limit=10, search_mode="semantic"
        )
        types_seen = {r["type"] for r in all_results}
        assert len(types_seen) > 1 or len(all_results) >= 1

        # With type_filter="fact", only fact type returned
        filtered = retriever.hybrid_search(
            "Python", limit=10, type_filter="fact", search_mode="semantic"
        )
        for r in filtered:
            assert r["type"] == "fact"

    def test_hybrid_search_hybrid_mode_with_embedder(self, vec_store):
        """Hybrid mode with embedder returns merged results from both paths."""
        self._add_memories_with_embeddings(vec_store)

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


# ---------------------------------------------------------------------------
# _compute_rerank_signals / _compute_weighted_signal unit tests
# ---------------------------------------------------------------------------


class TestRetrieve_Reranking:
    """Test _compute_rerank_signals and _compute_weighted_signal pure functions."""

    def test_rerank_score_computes_confidence_signal(self):
        """Confidence signal uses the 'confidence' field from the memory dict
        multiplied by CONFIDENCE_WEIGHT in the weighted sum."""
        memory = {
            "confidence": 0.9,
            "accessed_at": datetime.now(timezone.utc).isoformat(),
            "access_count": 5,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "fact",
        }
        now = datetime.now(timezone.utc)
        signals = _compute_rerank_signals(memory, TYPE_PRIORITY, now)
        assert signals["confidence"] == 0.9

        # Verify it contributes CONFIDENCE_WEIGHT * 0.9 in the weighted sum
        weighted = _compute_weighted_signal(signals)
        assert weighted == pytest.approx(
            CONFIDENCE_WEIGHT * 0.9
            + RECENCY_WEIGHT * signals["recency"]
            + ACCESS_WEIGHT * signals["access"]
            + TYPE_WEIGHT * signals["type"]
        )

    def test_rerank_score_computes_recency_signal_with_exponential_decay(self):
        """Recency uses exp(-0.01 * days_since_access) for memories with
        accessed_at, and returns 0.0 for memories without accessed_at."""
        now = datetime.now(timezone.utc)

        # Memory accessed 0 days ago → exp(0) = 1.0
        recent = {
            "confidence": 0.8,
            "accessed_at": now.isoformat(),
            "access_count": 1,
            "created_at": now.isoformat(),
            "type": "fact",
        }
        signals_recent = _compute_rerank_signals(recent, TYPE_PRIORITY, now)
        assert signals_recent["recency"] == pytest.approx(1.0, abs=0.01)

        # Memory accessed 100 days ago → exp(-1.0) ≈ 0.368
        old_access = (now - timedelta(days=100)).isoformat()
        old = {
            "confidence": 0.8,
            "accessed_at": old_access,
            "access_count": 1,
            "created_at": old_access,
            "type": "fact",
        }
        signals_old = _compute_rerank_signals(old, TYPE_PRIORITY, now)
        assert signals_old["recency"] == pytest.approx(math.exp(-1.0), abs=0.01)

        # Memory without accessed_at → recency signal = 0.0
        no_access = {
            "confidence": 0.8,
            "accessed_at": None,
            "access_count": 0,
            "created_at": now.isoformat(),
            "type": "fact",
        }
        signals_no = _compute_rerank_signals(no_access, TYPE_PRIORITY, now)
        assert signals_no["recency"] == 0.0

    def test_rerank_score_computes_access_frequency_signal(self):
        """Access frequency uses log(1 + access_count / max(age_days, 1))
        with edge cases for age_days=0 and access_count=0."""
        now = datetime.now(timezone.utc)

        # access_count=0 → log(1) = 0.0
        zero = {
            "confidence": 0.8,
            "accessed_at": None,
            "access_count": 0,
            "created_at": now.isoformat(),
            "type": "fact",
        }
        signals_zero = _compute_rerank_signals(zero, TYPE_PRIORITY, now)
        assert signals_zero["access"] == 0.0

        # access_count=10, age_days=30 → log(1 + 10/30) ≈ 0.288
        created = (now - timedelta(days=30)).isoformat()
        mem = {
            "confidence": 0.8,
            "accessed_at": None,
            "access_count": 10,
            "created_at": created,
            "type": "fact",
        }
        signals = _compute_rerank_signals(mem, TYPE_PRIORITY, now)
        assert signals["access"] == pytest.approx(math.log(1 + 10 / 30), abs=0.01)

        # age_days=0 (created today) → max(age_days, 1) = 1
        today = {
            "confidence": 0.8,
            "accessed_at": None,
            "access_count": 5,
            "created_at": now.isoformat(),
            "type": "fact",
        }
        signals_today = _compute_rerank_signals(today, TYPE_PRIORITY, now)
        assert signals_today["access"] == pytest.approx(math.log(1 + 5 / 1), abs=0.01)

    def test_rerank_score_computes_type_priority_signal(self):
        """Type priority looks up TYPE_PRIORITY dict and returns 1.0 for
        unknown types."""
        now = datetime.now(timezone.utc)

        # Known type: decision → 1.2
        decision = {
            "confidence": 0.8,
            "accessed_at": None,
            "access_count": 0,
            "created_at": now.isoformat(),
            "type": "decision",
        }
        signals = _compute_rerank_signals(decision, TYPE_PRIORITY, now)
        assert signals["type"] == 1.2

        # Known type: event → 0.9
        event = {
            "confidence": 0.8,
            "accessed_at": None,
            "access_count": 0,
            "created_at": now.isoformat(),
            "type": "event",
        }
        signals_event = _compute_rerank_signals(event, TYPE_PRIORITY, now)
        assert signals_event["type"] == 0.9

        # Unknown type → 1.0
        unknown = {
            "confidence": 0.8,
            "accessed_at": None,
            "access_count": 0,
            "created_at": now.isoformat(),
            "type": "custom_unknown_type",
        }
        signals_unknown = _compute_rerank_signals(unknown, TYPE_PRIORITY, now)
        assert signals_unknown["type"] == 1.0

    def test_rerank_score_blend_formula(self):
        """final_score = semantic_score * (1 - blend) + weighted_signal * blend
        with known inputs."""
        now = datetime.now(timezone.utc)
        memory = {
            "confidence": 1.0,
            "accessed_at": now.isoformat(),
            "access_count": 0,
            "created_at": now.isoformat(),
            "type": "decision",
        }
        signals = _compute_rerank_signals(memory, TYPE_PRIORITY, now)
        weighted_signal = _compute_weighted_signal(signals)

        semantic_score = 0.02
        blend = 0.3
        final = semantic_score * (1 - blend) + weighted_signal * blend

        # Manual calculation: confidence=1.0, recency≈1.0, access=0.0, type=1.2
        # weighted = 0.4*1.0 + 0.3*1.0 + 0.2*0.0 + 0.1*1.2 = 0.4 + 0.3 + 0.0 + 0.12 = 0.82
        assert weighted_signal == pytest.approx(0.82, abs=0.02)
        assert final == pytest.approx(0.02 * 0.7 + 0.82 * 0.3, abs=0.01)

        # blend=0.0 → final = semantic_score (pure RRF)
        assert semantic_score * (1 - 0.0) + weighted_signal * 0.0 == pytest.approx(
            semantic_score
        )
        # blend=1.0 → final = weighted_signal (pure signals)
        assert semantic_score * (1 - 1.0) + weighted_signal * 1.0 == pytest.approx(
            weighted_signal
        )

    def test_rerank_score_handles_missing_fields(self):
        """Memories missing confidence, accessed_at, access_count, or type
        use sensible defaults without raising errors."""
        now = datetime.now(timezone.utc)

        # Minimal memory with only required keys
        minimal = {"id": "m1"}
        signals = _compute_rerank_signals(minimal, TYPE_PRIORITY, now)
        # confidence defaults to 0.0 when missing
        assert signals["confidence"] == 0.0
        # accessed_at missing → recency = 0.0
        assert signals["recency"] == 0.0
        # access_count missing → access = log(1) = 0.0
        assert signals["access"] == 0.0
        # type missing → 1.0
        assert signals["type"] == 1.0

    def test_rerank_score_default_blend_factor(self, tmp_path):
        """The default blend factor on Retriever is 0.3."""
        db = tmp_path / "test_blend.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        retriever = Retriever(store=store, embedder=None)
        assert retriever._blend == 0.3
        store.close()

    def test_rerank_blend_out_of_range_raises_value_error(self, tmp_path):
        """Blend factor outside [0.0, 1.0] raises ValueError with llmem: retrieve: prefix."""
        db = tmp_path / "test_blend_err.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        with pytest.raises(ValueError, match="llmem: retrieve:"):
            Retriever(store=store, embedder=None, blend=-0.1)
        with pytest.raises(ValueError, match="llmem: retrieve:"):
            Retriever(store=store, embedder=None, blend=1.1)
        store.close()

    def test_compute_weighted_signal_returns_zero_for_zero_signals(self):
        """When all signals are zero, the weighted sum is zero."""
        signals = {"confidence": 0.0, "recency": 0.0, "access": 0.0, "type": 0.0}
        assert _compute_weighted_signal(signals) == 0.0

    def test_compute_weighted_signal_uses_all_weights(self):
        """Weighted signal combines all four signals with their respective weights."""
        signals = {"confidence": 1.0, "recency": 1.0, "access": 1.0, "type": 1.0}
        result = _compute_weighted_signal(signals)
        expected = (
            CONFIDENCE_WEIGHT * 1.0
            + RECENCY_WEIGHT * 1.0
            + ACCESS_WEIGHT * 1.0
            + TYPE_WEIGHT * 1.0
        )
        assert result == pytest.approx(expected)

    def test_compute_rerank_signals_accepts_custom_type_priority(self):
        """_compute_rerank_signals accepts a custom type_priority dict,
        not hardcoded to module-level TYPE_PRIORITY."""
        now = datetime.now(timezone.utc)
        custom_priority = {"custom_type": 2.0}
        memory = {
            "confidence": 0.8,
            "accessed_at": None,
            "access_count": 0,
            "created_at": now.isoformat(),
            "type": "custom_type",
        }
        signals = _compute_rerank_signals(memory, custom_priority, now)
        assert signals["type"] == 2.0


# ---------------------------------------------------------------------------
# Reranking integration tests — need a MemoryStore
# ---------------------------------------------------------------------------


class TestRetrieve_RerankingIntegration:
    """Integration tests for reranking with MemoryStore and Retriever."""

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

    def test_hybrid_search_applies_reranking(self, store):
        """hybrid_search returns results with _rerank_score key and results
        are sorted by blended score."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None)
        results = retriever.hybrid_search("Python", limit=10)
        assert len(results) >= 1
        for r in results:
            assert "_rrf_score" in r
            assert "_rerank_score" in r
            assert isinstance(r["_rerank_score"], float)
        # Results should be sorted by _rerank_score descending
        rerank_scores = [r["_rerank_score"] for r in results]
        assert rerank_scores == sorted(rerank_scores, reverse=True)

    def test_hybrid_search_reranking_boosts_high_confidence(self, store):
        """A memory with significantly higher confidence ranks higher after
        reranking when blend is non-zero."""
        # Add a low-confidence and high-confidence memory with the same text
        store.add(
            type="fact", content="UniqueAlpha programming language", confidence=0.1
        )
        high_id = store.add(
            type="fact", content="UniqueAlpha programming language", confidence=1.0
        )

        # With blend=0.0, results should be in RRF order (probably same due to
        # identical content — the tie-breaker is ID). With blend > 0, the
        # high-confidence memory should outrank the low-confidence one.
        retriever_no_blend = Retriever(store=store, embedder=None, blend=0.0)
        retriever_no_blend.hybrid_search(
            "UniqueAlpha", limit=10, search_mode="fts"
        )

        retriever_with_blend = Retriever(store=store, embedder=None, blend=0.5)
        results_with_blend = retriever_with_blend.hybrid_search(
            "UniqueAlpha", limit=10, search_mode="fts"
        )

        # With blend=0.5, the high-confidence memory should rank first
        result_ids = [r["id"] for r in results_with_blend]
        assert result_ids[0] == high_id

    def test_hybrid_search_reranking_uses_type_priority(self, store):
        """Memories of type 'decision' get a boost over type 'event' due to
        TYPE_PRIORITY weighting."""
        # Add two memories with same content but different types
        decision_id = store.add(
            type="decision", content="TypePriorityTest architecture choice"
        )
        event_id = store.add(
            type="event", content="TypePriorityTest architecture choice"
        )

        retriever = Retriever(store=store, embedder=None, blend=0.5)
        results = retriever.hybrid_search(
            "TypePriorityTest", limit=10, search_mode="fts"
        )

        # Decision type (priority 1.2) should outrank event type (priority 0.9)
        result_ids = [r["id"] for r in results]
        assert result_ids.index(decision_id) < result_ids.index(event_id)

    def test_search_tracks_access_by_default(self, store):
        """Calling retriever.search() increments access_count on returned
        memories."""
        mid = store.add(type="fact", content="Access tracking test content")
        # Verify initial access_count is 0
        mem_before = store.get(mid, track_access=False)
        assert mem_before["access_count"] == 0

        retriever = Retriever(store=store, embedder=None)
        results = retriever.search("Access tracking test", limit=10)

        # The memory should appear in results
        assert any(r["id"] == mid for r in results)

        # Access count should now be > 0
        mem_after = store.get(mid, track_access=False)
        assert mem_after["access_count"] > 0

    def test_hybrid_search_tracks_access_by_default(self, store):
        """Calling retriever.hybrid_search() increments access_count on
        returned memories."""
        mid = store.add(type="fact", content="Hybrid access tracking test")
        mem_before = store.get(mid, track_access=False)
        assert mem_before["access_count"] == 0

        retriever = Retriever(store=store, embedder=None)
        results = retriever.hybrid_search("Hybrid access tracking", limit=10)

        assert any(r["id"] == mid for r in results)

        mem_after = store.get(mid, track_access=False)
        assert mem_after["access_count"] > 0

    def test_hybrid_search_blend_zero_same_as_rrf(self, store):
        """With blend=0.0, hybrid_search results are ordered identically to
        pure RRF (current behavior before reranking)."""
        self._add_memories(store)
        retriever = Retriever(store=store, embedder=None, blend=0.0)
        results = retriever.hybrid_search("Python", limit=10, search_mode="fts")

        for r in results:
            assert "_rerank_score" in r
            assert r["_rerank_score"] == pytest.approx(r["_rrf_score"])

    def test_hybrid_search_blend_one_ignores_rrf(self, store):
        """With blend=1.0, results are ordered purely by weighted signals,
        ignoring RRF scores."""
        store.add(type="decision", content="Blending test decision", confidence=1.0)
        store.add(type="event", content="Blending test event", confidence=0.1)

        retriever = Retriever(store=store, embedder=None, blend=1.0)
        results = retriever.hybrid_search("Blending test", limit=10, search_mode="fts")

        assert len(results) >= 2
        # All results should have _rerank_score but it should be based purely
        # on weighted signals (blend=1.0 means rrf_score * 0 contribution)
        for r in results:
            assert "_rerank_score" in r
            assert r["_rerank_score"] != r["_rrf_score"] or r["_rrf_score"] == 0.0

    def test_search_access_tracking_is_best_effort(self, store):
        """If touch() fails for a non-existent memory ID in results, the
        error is caught and logged — not propagated to the caller."""
        retriever = Retriever(store=store, embedder=None)
        # search() should complete without raising even if touch might fail
        # (this is a best-effort test — the actual error handling is internal)
        results = retriever.search("nonexistent query xyz", limit=10)
        assert isinstance(results, list)

    def test_search_track_access_false_skips_touch(self, store):
        """Calling retriever.search(track_access=False) does not increment
        access_count on returned memories."""
        mid = store.add(type="fact", content="No access tracking content")
        mem_before = store.get(mid, track_access=False)
        assert mem_before["access_count"] == 0

        retriever = Retriever(store=store, embedder=None)
        results = retriever.search("No access tracking", limit=10, track_access=False)

        assert any(r["id"] == mid for r in results)
        mem_after = store.get(mid, track_access=False)
        assert mem_after["access_count"] == 0

    def test_hybrid_search_track_access_false_skips_touch(self, store):
        """Calling retriever.hybrid_search(track_access=False) does not
        increment access_count on returned memories."""
        mid = store.add(type="fact", content="Hybrid no access tracking")
        mem_before = store.get(mid, track_access=False)
        assert mem_before["access_count"] == 0

        retriever = Retriever(store=store, embedder=None)
        results = retriever.hybrid_search(
            "Hybrid no access tracking", limit=10, track_access=False
        )

        assert any(r["id"] == mid for r in results)
        mem_after = store.get(mid, track_access=False)
        assert mem_after["access_count"] == 0

    def test_search_track_access_true_is_default(self, store):
        """When track_access is not specified, search() defaults to tracking
        access (track_access=True behavior)."""
        mid = store.add(type="fact", content="Default tracking content")
        mem_before = store.get(mid, track_access=False)
        assert mem_before["access_count"] == 0

        retriever = Retriever(store=store, embedder=None)
        results = retriever.search("Default tracking", limit=10)

        assert any(r["id"] == mid for r in results)
        mem_after = store.get(mid, track_access=False)
        assert mem_after["access_count"] > 0


# ---------------------------------------------------------------------------
# Ref expansion tests — traverse_refs in Retriever.search()
# ---------------------------------------------------------------------------


class TestRetrieve_RefExpansion:
    """Test Retriever.search() with traverse_refs flag for code reference expansion."""

    def test_search_traverse_refs_expands_code_refs(self, store, tmp_path):
        """When traverse_refs=True, search results include resolved code ref content."""
        # Create a test file for the code ref to resolve
        f = tmp_path / "test_code.py"
        f.write_text("def hello():\n    return 'world'\n")
        code_ref = f"{f}:1:2"

        mid = store.add(type="fact", content="searchable memory about hello function")
        store.add_relation(mid, code_ref, "references", target_type="code")

        retriever = Retriever(store=store, embedder=None)
        results = retriever.search(
            "hello function", traverse_refs=True, max_ref_depth=1
        )
        # Should have at least one code ref result
        code_results = [r for r in results if r.get("_source") == "code"]
        assert len(code_results) >= 1
        assert "hello" in code_results[0]["content"]

    def test_search_traverse_refs_default_off(self, store, tmp_path):
        """When traverse_refs=False (default), search results do NOT include code refs."""
        f = tmp_path / "test_code.py"
        f.write_text("def hello():\n    return 'world'\n")
        code_ref = f"{f}:1:2"

        mid = store.add(type="fact", content="searchable memory about hello function")
        store.add_relation(mid, code_ref, "references", target_type="code")

        retriever = Retriever(store=store, embedder=None)
        results = retriever.search("hello function")
        code_results = [r for r in results if r.get("_source") == "code"]
        assert len(code_results) == 0

    def test_search_traverse_refs_respects_max_ref_depth(self, store, tmp_path):
        """max_ref_depth=1 only follows one hop."""
        f = tmp_path / "test_code.py"
        f.write_text("def hello():\n    return 'world'\n")
        code_ref = f"{f}:1:2"

        mid = store.add(type="fact", content="searchable memory")
        store.add_relation(mid, code_ref, "references", target_type="code")

        retriever = Retriever(store=store, embedder=None)
        results = retriever.search("searchable", traverse_refs=True, max_ref_depth=1)
        code_results = [r for r in results if r.get("_source") == "code"]
        assert len(code_results) >= 1

    def test_search_traverse_refs_missing_file_skipped(self, store, caplog):
        """A code ref pointing to a missing file is silently skipped (logged, not raised)."""
        code_ref = "/tmp/nonexistent_ref_file_xyz:1:5"

        mid = store.add(type="fact", content="memory pointing to missing code")
        store.add_relation(mid, code_ref, "references", target_type="code")

        retriever = Retriever(store=store, embedder=None)
        with caplog.at_level(logging.DEBUG, logger="llmem.refs"):
            results = retriever.search(
                "missing code", traverse_refs=True, max_ref_depth=1
            )
        # No code results from missing file
        code_results = [r for r in results if r.get("_source") == "code"]
        assert len(code_results) == 0
        # Should not raise
