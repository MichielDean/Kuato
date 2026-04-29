"""Embedding quality metrics for detecting poor embedding vectors.

Provides anisotropy, similarity_range, and discrimination_gap computations
over embedding matrices, plus threshold constants for quality warnings.
"""

import logging
import math
import struct
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Warning thresholds — when anisotropy exceeds or similarity_range falls
# below these values, embeddings may be poor quality.
ANISOTROPY_WARNING_THRESHOLD: float = 0.5
SIMILARITY_RANGE_WARNING_THRESHOLD: float = 0.1


@dataclass(frozen=True)
class EmbeddingMetrics:
    """Result container for embedding quality metrics.

    Attributes:
        anisotropy: Measures vector uniformity. Lower is better (more
            isotropic). Value in [0.0, 1.0] for 2+ vectors; 0.0 for
            empty or single-vector input.
        similarity_range: Spread between max and min pairwise cosine
            similarity. Higher is better. 0.0 for empty, single, or
            identical vector input.
        discrimination_gap: Average inter-class cosine distance minus
            average intra-class cosine distance. Higher is better.
            None when labels are not provided.
    """

    anisotropy: float
    similarity_range: float
    discrimination_gap: float | None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors.

    Returns 1.0 for any two equal non-zero vectors, 0.0 for any two
    orthogonal non-zero vectors, and 0.0 when either vector has zero
    magnitude (no ZeroDivisionError).

    Args:
        a: First vector as a list of floats.
        b: Second vector as a list of floats.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 when either
        vector is zero-length or has zero magnitude.
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def bytes_to_vec(data: bytes) -> list[float]:
    """Decode a bytes object (packed float32 format) into a list of floats.

    Complements :meth:`EmbeddingEngine.vec_to_bytes` which encodes in the
    opposite direction.

    Args:
        data: Bytes in packed float32 format (little-endian, as produced
            by ``struct.pack``).

    Returns:
        List of floats. Returns an empty list for empty input bytes.
    """
    if not data:
        return []
    dim = len(data) // 4
    return list(struct.unpack(f"{dim}f", data))


def anisotropy(embeddings: list[list[float]]) -> float:
    """Measure vector uniformity (anisotropy) of an embedding matrix.

    Lower values indicate more isotropic (uniform) embeddings. Higher
    values indicate anisotropic embeddings that cluster in a preferred
    direction, which degrades search quality.

    For 2+ identical non-zero vectors, returns 1.0. For 2+ orthogonal
    vectors, returns a value near 0.0.

    Args:
        embeddings: List of embedding vectors.

    Returns:
        Anisotropy value in [0.0, 1.0]. Returns 0.0 for empty or
        single-vector input.
    """
    if len(embeddings) <= 1:
        return 0.0

    # Anisotropy: 1 minus the average pairwise cosine similarity,
    # adjusted so that isotropic (uniform) distributions score low.
    # Using: anisotropy = max_cosine - min_cosine if they exist, but
    # the standard definition is average pairwise similarity centred
    # around the mean direction.  A simpler proxy: average pairwise
    # cosine similarity.  When all vectors are identical, average
    # pairwise cosine = 1.0, anisotropy = 1.0.  When all vectors are
    # orthogonal, average pairwise cosine ≈ 0.0, anisotropy ≈ 0.0.
    total = 0.0
    count = 0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            total += cosine_similarity(embeddings[i], embeddings[j])
            count += 1
    if count == 0:
        return 0.0
    avg_sim = total / count
    # Anisotropy is the average pairwise similarity for identical
    # vectors this is 1.0, for orthogonal it's ~0.0.
    return avg_sim


def similarity_range(embeddings: list[list[float]]) -> float:
    """Measure the spread of pairwise cosine similarities.

    Returns the difference between the maximum and minimum pairwise
    cosine similarity among all unique pairs. A higher value indicates
    more discriminative embeddings.

    Args:
        embeddings: List of embedding vectors.

    Returns:
        Similarity range as a float. Returns 0.0 for empty input,
        single vector, or identical vectors.
    """
    if len(embeddings) <= 1:
        return 0.0

    min_sim = 1.0
    max_sim = -1.0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim < min_sim:
                min_sim = sim
            if sim > max_sim:
                max_sim = sim
    return max_sim - min_sim


def discrimination_gap(
    embeddings: list[list[float]], labels: list[str] | None
) -> float:
    """Measure the separation between labelled groups of embeddings.

    Returns the average inter-class cosine distance minus the average
    intra-class cosine distance. Higher values indicate better
    discrimination between labelled groups.

    Args:
        embeddings: List of embedding vectors.
        labels: List of string labels, same length as embeddings.
            If None, empty, or all labels are the same, returns 0.0.

    Returns:
        Discrimination gap as a float. 0.0 when labels are not
        provided, empty, or all identical.

    Raises:
        ValueError: If labels is provided but has different length
            than embeddings.
    """
    if labels is None or len(labels) == 0:
        return 0.0

    if len(labels) != len(embeddings):
        raise ValueError(
            f"llmem: metrics: labels length {len(labels)} does not "
            f"match embeddings length {len(embeddings)}"
        )

    # If all labels are the same, no inter-class distance
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        return 0.0

    # Group embeddings by label
    groups: dict[str, list[list[float]]] = {}
    for emb, label in zip(embeddings, labels):
        groups.setdefault(label, []).append(emb)

    # Compute average intra-class similarity
    intra_total = 0.0
    intra_count = 0
    for label, group in groups.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                intra_total += cosine_similarity(group[i], group[j])
                intra_count += 1
    avg_intra = intra_total / intra_count if intra_count > 0 else 0.0

    # Compute average inter-class similarity
    inter_total = 0.0
    inter_count = 0
    label_list = list(groups.keys())
    for li in range(len(label_list)):
        for lj in range(li + 1, len(label_list)):
            group_a = groups[label_list[li]]
            group_b = groups[label_list[lj]]
            for a in group_a:
                for b in group_b:
                    inter_total += cosine_similarity(a, b)
                    inter_count += 1
    avg_inter = inter_total / inter_count if inter_count > 0 else 0.0

    # Discrimination gap: inter-class distance minus intra-class distance.
    # Distance is (1 - similarity), so gap = (1 - avg_inter) - (1 - avg_intra)
    # = avg_intra - avg_inter
    return avg_intra - avg_inter


def compute_metrics(
    embeddings: list[list[float]], labels: list[str] | None = None
) -> EmbeddingMetrics:
    """Compute all embedding quality metrics at once.

    Convenience wrapper that returns an :class:`EmbeddingMetrics` dataclass
    with ``anisotropy``, ``similarity_range``, and ``discrimination_gap``
    (None if labels are not provided).

    Args:
        embeddings: List of embedding vectors.
        labels: Optional list of string labels, same length as embeddings.

    Returns:
        An EmbeddingMetrics dataclass with computed values.
    """
    aniso = anisotropy(embeddings)
    sim_range = similarity_range(embeddings)
    disc_gap = discrimination_gap(embeddings, labels) if labels is not None else None
    return EmbeddingMetrics(
        anisotropy=aniso,
        similarity_range=sim_range,
        discrimination_gap=disc_gap,
    )
