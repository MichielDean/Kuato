"""Tests for llmem.metrics module — embedding quality metrics."""

import math
import struct

import pytest

from llmem.embed import EmbeddingEngine
from llmem.metrics import (
    ANISOTROPY_WARNING_THRESHOLD,
    SIMILARITY_RANGE_WARNING_THRESHOLD,
    EmbeddingMetrics,
    anisotropy,
    bytes_to_vec,
    compute_metrics,
    cosine_similarity,
    discrimination_gap,
    similarity_range,
)


class TestMetrics_CosineSimilarity:
    """Test cosine_similarity function."""

    def test_cosine_similarity_known_vectors(self):
        """Known vector pairs produce expected cosine similarity values."""
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
        assert cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vectors(self):
        """Cosine similarity with zero vector returns 0.0 (no division by zero)."""
        assert cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0
        assert cosine_similarity([1, 0, 0], [0, 0, 0]) == 0.0
        assert cosine_similarity([0, 0, 0], [0, 0, 0]) == 0.0

    def test_cosine_similarity_magnitude_vectors(self):
        """Cosine similarity is scale-invariant (magnitude doesn't matter)."""
        assert cosine_similarity([1, 0, 0], [2, 0, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 1, 0], [2, 2, 0]) == pytest.approx(1.0)

    def test_cosine_similarity_negative_vectors(self):
        """Cosine similarity handles negative components correctly."""
        assert cosine_similarity([1, 1], [-1, -1]) == pytest.approx(-1.0)
        assert cosine_similarity([1, 2], [3, 4]) == pytest.approx(
            (1 * 3 + 2 * 4) / (math.sqrt(5) * math.sqrt(25))
        )


class TestMetrics_BytesToVec:
    """Test bytes_to_vec function."""

    def test_bytes_to_vec_roundtrip(self):
        """bytes_to_vec(vec_to_bytes(x)) should reproduce x within float precision."""
        original = [0.1, 0.2, 0.3]
        encoded = EmbeddingEngine.vec_to_bytes(original)
        decoded = bytes_to_vec(encoded)
        for a, b in zip(original, decoded):
            assert a == pytest.approx(b, abs=1e-6)

    def test_bytes_to_vec_single_float(self):
        """bytes_to_vec of a single float32 bytes returns [1.0]."""
        data = struct.pack("1f", 1.0)
        assert bytes_to_vec(data) == pytest.approx([1.0])

    def test_bytes_to_vec_empty_bytes(self):
        """bytes_to_vec of empty bytes returns empty list."""
        assert bytes_to_vec(b"") == []

    def test_bytes_to_vec_known_values(self):
        """bytes_to_vec correctly decodes known packed float32 values."""
        data = struct.pack("3f", 1.0, 2.0, 3.0)
        result = bytes_to_vec(data)
        assert result == pytest.approx([1.0, 2.0, 3.0])


class TestMetrics_Anisotropy:
    """Test anisotropy function."""

    def test_anisotropy_uniform_vectors(self):
        """Anisotropy of orthogonal vectors (identity matrix rows) should be near 0."""
        # Identity matrix rows: [1,0,0], [0,1,0], [0,0,1]
        # All pairwise cosine similarities are 0, so anisotropy ≈ 0
        identity_rows = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = anisotropy(identity_rows)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_anisotropy_identical_vectors(self):
        """Anisotropy of identical vectors should be near 1.0."""
        vecs = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        result = anisotropy(vecs)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_anisotropy_empty_input(self):
        """Anisotropy of empty input returns 0.0 (no crash)."""
        assert anisotropy([]) == 0.0

    def test_anisotropy_single_vector(self):
        """Anisotropy of a single vector returns 0.0 (cannot compute pairwise)."""
        assert anisotropy([[1, 0, 0]]) == 0.0

    def test_anisotropy_two_opposite_vectors(self):
        """Anisotropy of two opposite vectors (cos_sim = -1) is clamped to 0.0."""
        result = anisotropy([[1, 0], [-1, 0]])
        # Average pairwise similarity = -1.0, but clamped to [0.0, 1.0]
        assert result == pytest.approx(0.0)

    def test_anisotropy_always_in_zero_to_one_range(self):
        """Anisotropy always returns a value in [0.0, 1.0] per its contract."""
        # Opposite vectors produce raw average similarity = -1.0,
        # but the contract says [0.0, 1.0], so it must be clamped.
        result = anisotropy([[1, 0, 0], [-1, 0, 0]])
        assert 0.0 <= result <= 1.0

    def test_anisotropy_mixed_opposite_and_aligned(self):
        """Anisotropy of mixed opposite and aligned vectors stays in [0.0, 1.0]."""
        # [1,0], [1,0], [-1,0]: pairs are (1.0, -1.0, -1.0)
        # raw avg = -1/3, clamped to 0.0
        result = anisotropy([[1, 0], [1, 0], [-1, 0]])
        assert 0.0 <= result <= 1.0


class TestMetrics_SimilarityRange:
    """Test similarity_range function."""

    def test_similarity_range_spread_vectors(self):
        """Similarity range of diverse vectors should be > 0.3."""
        # Vectors with a spread of cosine similarities
        vecs = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
        result = similarity_range(vecs)
        assert result > 0.3

    def test_similarity_range_identical_vectors(self):
        """Similarity range of identical vectors should be 0.0."""
        vecs = [[1, 0, 0], [1, 0, 0]]
        assert similarity_range(vecs) == pytest.approx(0.0)

    def test_similarity_range_empty_input(self):
        """Similarity range of empty input returns 0.0."""
        assert similarity_range([]) == 0.0

    def test_similarity_range_single_vector(self):
        """Similarity range of single vector returns 0.0."""
        assert similarity_range([[1, 0, 0]]) == 0.0

    def test_similarity_range_opposite_and_aligned_vectors(self):
        """Similarity range with mix of aligned and opposite vectors."""
        # [1,0] and [1,0]: cos = 1.0, [1,0] and [-1,0]: cos = -1.0
        result = similarity_range([[1, 0], [1, 0], [-1, 0]])
        # max pairwise = 1.0, min pairwise = -1.0, range = 2.0
        assert result == pytest.approx(2.0)


class TestMetrics_DiscriminationGap:
    """Test discrimination_gap function."""

    def test_discrimination_gap_clear_labels(self):
        """Vectors with clear cluster separation should have gap > 0."""
        # Cluster A: vectors near [1,0]
        # Cluster B: vectors near [0,1]
        embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
        labels = ["a", "a", "b", "b"]
        result = discrimination_gap(embeddings, labels)
        assert result > 0

    def test_discrimination_gap_no_labels(self):
        """discrimination_gap with None labels returns 0.0."""
        result = discrimination_gap([[1, 0], [0, 1]], None)
        assert result == 0.0

    def test_discrimination_gap_empty_labels(self):
        """discrimination_gap with empty labels list returns 0.0."""
        result = discrimination_gap([[1, 0]], [])
        assert result == 0.0

    def test_discrimination_gap_single_label(self):
        """discrimination_gap with all same labels returns 0.0."""
        result = discrimination_gap([[1, 0], [0, 1]], ["a", "a"])
        assert result == 0.0

    def test_discrimination_gap_mismatched_length_raises(self):
        """discrimination_gap raises ValueError if labels length != embeddings length."""
        with pytest.raises(ValueError, match="labels length"):
            discrimination_gap([[1, 0]], ["a", "b"])


class TestMetrics_ComputeMetrics:
    """Test compute_metrics convenience function."""

    def test_compute_metrics_without_labels(self):
        """compute_metrics without labels returns EmbeddingMetrics with discrimination_gap=None."""
        vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = compute_metrics(vecs)
        assert isinstance(result, EmbeddingMetrics)
        assert result.anisotropy is not None
        assert result.similarity_range is not None
        assert result.discrimination_gap is None

    def test_compute_metrics_with_labels(self):
        """compute_metrics with labels returns EmbeddingMetrics with all fields populated."""
        vecs = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
        labels = ["a", "a", "b", "b"]
        result = compute_metrics(vecs, labels)
        assert isinstance(result, EmbeddingMetrics)
        assert result.anisotropy is not None
        assert result.similarity_range is not None
        assert result.discrimination_gap is not None

    def test_compute_metrics_empty_embeddings(self):
        """compute_metrics with empty embeddings returns zero values."""
        result = compute_metrics([])
        assert result.anisotropy == 0.0
        assert result.similarity_range == 0.0
        assert result.discrimination_gap is None


class TestMetrics_WarningThresholds:
    """Test that warning threshold constants have expected values."""

    def test_anisotropy_warning_threshold(self):
        """ANISOTROPY_WARNING_THRESHOLD should be 0.5."""
        assert ANISOTROPY_WARNING_THRESHOLD == 0.5

    def test_similarity_range_warning_threshold(self):
        """SIMILARITY_RANGE_WARNING_THRESHOLD should be 0.1."""
        assert SIMILARITY_RANGE_WARNING_THRESHOLD == 0.1


class TestMetrics_EmbeddingMetricsDataclass:
    """Test EmbeddingMetrics dataclass immutability and construction."""

    def test_embedding_metrics_is_frozen(self):
        """EmbeddingMetrics should be frozen (immutable)."""
        m = EmbeddingMetrics(
            anisotropy=0.3, similarity_range=0.5, discrimination_gap=None
        )
        with pytest.raises(AttributeError):
            m.anisotropy = 0.9  # type: ignore[misc]

    def test_embedding_metrics_construction(self):
        """EmbeddingMetrics can be constructed directly."""
        m = EmbeddingMetrics(
            anisotropy=0.4, similarity_range=0.6, discrimination_gap=0.2
        )
        assert m.anisotropy == 0.4
        assert m.similarity_range == 0.6
        assert m.discrimination_gap == 0.2

    def test_embedding_metrics_construction_without_discrimination(self):
        """EmbeddingMetrics can be constructed with discrimination_gap=None."""
        m = EmbeddingMetrics(
            anisotropy=0.4, similarity_range=0.6, discrimination_gap=None
        )
        assert m.discrimination_gap is None


class TestMetrics_StoreGetEmbeddingsWithTypes:
    """Test MemoryStore.get_embeddings_with_types public API."""

    def test_returns_embeddings_with_types(self, tmp_path):
        """get_embeddings_with_types returns (bytes, type) pairs for memories with embeddings."""
        from llmem.store import MemoryStore
        from llmem.embed import EmbeddingEngine

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add(
            type="fact",
            content="embedded fact",
            embedding=EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0]),
        )
        store.add(
            type="decision",
            content="embedded decision",
            embedding=EmbeddingEngine.vec_to_bytes([0.0, 1.0, 0.0]),
        )
        store.add(
            type="fact",
            content="fact without embedding",
        )

        rows = store.get_embeddings_with_types()
        assert len(rows) == 2
        assert all(isinstance(emb, bytes) for emb, _ in rows)
        types = [t for _, t in rows]
        assert "fact" in types
        assert "decision" in types
        store.close()

    def test_excludes_expired_memories(self, tmp_path):
        """get_embeddings_with_types excludes memories with valid_until set."""
        from llmem.store import MemoryStore
        from llmem.embed import EmbeddingEngine

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        # Add a valid memory with embedding
        store.add(
            type="fact",
            content="valid fact",
            embedding=EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0]),
        )
        # Add a memory, then invalidate it (which nulls the embedding)
        mem_id = store.add(
            type="decision",
            content="will be invalidated",
            embedding=EmbeddingEngine.vec_to_bytes([0.0, 1.0, 0.0]),
        )
        store.invalidate(mem_id)

        rows = store.get_embeddings_with_types()
        # Only the valid memory should appear; the invalidated one has embedding=NULL
        assert len(rows) == 1
        assert rows[0][1] == "fact"
        store.close()

    def test_excludes_memories_expired_via_update(self, tmp_path):
        """get_embeddings_with_types excludes memories with valid_until set via update().

        update() can set valid_until without clearing embedding, so this tests
        that the valid_until IS NULL filter works even when embedding is non-null.
        """
        from llmem.store import MemoryStore
        from llmem.embed import EmbeddingEngine

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        # Add a valid memory with embedding
        store.add(
            type="fact",
            content="valid fact",
            embedding=EmbeddingEngine.vec_to_bytes([1.0, 0.0, 0.0]),
        )
        # Add a memory, then set valid_until via update (keeps embedding)
        mem_id = store.add(
            type="decision",
            content="will be expired via update",
            embedding=EmbeddingEngine.vec_to_bytes([0.0, 1.0, 0.0]),
        )
        store.update(mem_id, valid_until="2025-01-01T00:00:00+00:00")

        rows = store.get_embeddings_with_types()
        # Only the valid memory should appear, even though the expired
        # one still has a non-null embedding in the database
        assert len(rows) == 1
        assert rows[0][1] == "fact"
        store.close()

    def test_returns_empty_when_no_embeddings(self, tmp_path):
        """get_embeddings_with_types returns empty list when no embeddings exist."""
        from llmem.store import MemoryStore

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        store.add(type="fact", content="no embedding here")

        rows = store.get_embeddings_with_types()
        assert rows == []
        store.close()
