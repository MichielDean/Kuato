// Package metrics provides embedding quality metrics for detecting
// poor embedding vectors. It computes anisotropy, similarity range,
// and discrimination gap over embedding matrices.
package metrics

import (
	"fmt"
	"log/slog"
	"math"
)

// Warning thresholds — when anisotropy exceeds or similarity_range falls
// below these values, embeddings may be poor quality.
const (
	AnisotropyWarningThreshold   = 0.5
	SimilarityRangeWarningThreshold = 0.1
	MetricsMaxEmbeddings         = 10000
)

// EmbeddingMetrics holds the result of computing all embedding quality metrics.
// No pointer fields — value struct.
type EmbeddingMetrics struct {
	Anisotropy        float64
	SimilarityRange   float64
	DiscriminationGap float64
}

// fmtErr wraps an error with the "llmem: metrics:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: metrics: "+format, args...)
}

// cosineSimilarity computes the cosine similarity between two float32 vectors.
// Returns 0.0 when either vector has zero magnitude.
// This is a package-internal implementation matching the algorithm in internal/store/helpers.go.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}
	var dot, magA, magB float64
	for i := range a {
		fa := float64(a[i])
		fb := float64(b[i])
		dot += fa * fb
		magA += fa * fa
		magB += fb * fb
	}
	if magA == 0 || magB == 0 {
		return 0.0
	}
	return dot / (math.Sqrt(magA) * math.Sqrt(magB))
}

// Anisotropy measures vector uniformity of an embedding matrix.
// Returns the average pairwise cosine similarity, clamped to [0.0, 1.0].
// Returns 0.0 for empty or single-vector input.
func Anisotropy(embeddings [][]float32) float64 {
	if len(embeddings) <= 1 {
		return 0.0
	}

	total := 0.0
	count := 0
	for i := range embeddings {
		for j := i + 1; j < len(embeddings); j++ {
			total += cosineSimilarity(embeddings[i], embeddings[j])
			count++
		}
	}
	if count == 0 {
		return 0.0
	}
	avgSim := total / float64(count)
	// Clamp to [0.0, 1.0]
	return math.Max(0.0, math.Min(1.0, avgSim))
}

// SimilarityRange returns the spread of pairwise cosine similarities.
// Returns max - min pairwise cosine similarity.
// Returns 0.0 for empty, single, or identical inputs.
func SimilarityRange(embeddings [][]float32) float64 {
	if len(embeddings) <= 1 {
		return 0.0
	}

	minSim := 1.0
	maxSim := -1.0
	for i := range embeddings {
		for j := i + 1; j < len(embeddings); j++ {
			sim := cosineSimilarity(embeddings[i], embeddings[j])
			if sim < minSim {
				minSim = sim
			}
			if sim > maxSim {
				maxSim = sim
			}
		}
	}
	return maxSim - minSim
}

// DiscriminationGap measures the separation between labelled groups of embeddings.
// Returns inter-class distance minus intra-class distance.
// Returns 0.0 with nil error when labels is nil or empty.
// Returns 0.0 with nil error when all labels are the same class.
// Returns an error when len(labels) != len(embeddings).
func DiscriminationGap(embeddings [][]float32, labels []string) (float64, error) {
	if labels == nil || len(labels) == 0 {
		return 0.0, nil
	}

	if len(labels) != len(embeddings) {
		return 0.0, fmtErr("labels length %d does not match embeddings length %d", len(labels), len(embeddings))
	}

	// If all labels are the same, no inter-class distance
	uniqueLabels := make(map[string]bool)
	for _, l := range labels {
		uniqueLabels[l] = true
	}
	if len(uniqueLabels) <= 1 {
		return 0.0, nil
	}

	// Group embeddings by label
	groups := make(map[string][][]float32)
	for i, label := range labels {
		groups[label] = append(groups[label], embeddings[i])
	}

	// Compute average intra-class similarity
	intraTotal := 0.0
	intraCount := 0
	for _, group := range groups {
		for i := range group {
			for j := i + 1; j < len(group); j++ {
				intraTotal += cosineSimilarity(group[i], group[j])
				intraCount++
			}
		}
	}
	avgIntra := 0.0
	if intraCount > 0 {
		avgIntra = intraTotal / float64(intraCount)
	}

	// Compute average inter-class similarity
	interTotal := 0.0
	interCount := 0
	labelList := make([]string, 0, len(groups))
	for l := range groups {
		labelList = append(labelList, l)
	}
	for li := range labelList {
		for lj := li + 1; lj < len(labelList); lj++ {
			groupA := groups[labelList[li]]
			groupB := groups[labelList[lj]]
			for _, a := range groupA {
				for _, b := range groupB {
					interTotal += cosineSimilarity(a, b)
					interCount++
				}
			}
		}
	}
	avgInter := 0.0
	if interCount > 0 {
		avgInter = interTotal / float64(interCount)
	}

	// Discrimination gap: (1 - avg_inter) - (1 - avg_intra) = avg_intra - avg_inter
	return avgIntra - avgInter, nil
}

// ComputeMetrics computes all embedding quality metrics at once.
// When maxEmbeddings <= 0, defaults to MetricsMaxEmbeddings.
// Truncates input if len(embeddings) > maxEmbeddings, logging a warning.
// Returns an EmbeddingMetrics struct.
func ComputeMetrics(embeddings [][]float32, labels []string, maxEmbeddings int) (*EmbeddingMetrics, error) {
	if maxEmbeddings <= 0 {
		maxEmbeddings = MetricsMaxEmbeddings
	}
	if len(embeddings) > maxEmbeddings {
		slog.Warn("llmem: metrics: capping embeddings for metrics computation", "original", len(embeddings), "cap", maxEmbeddings)
		embeddings = embeddings[:maxEmbeddings]
		if labels != nil {
			labelCap := maxEmbeddings
			if len(labels) < labelCap {
				labelCap = len(labels)
			}
			labels = labels[:labelCap]
			// Keep embeddings and labels in sync: if labels are shorter,
			// cap embeddings to the same count.
			if labelCap < len(embeddings) {
				embeddings = embeddings[:labelCap]
			}
		}
	}

	aniso := Anisotropy(embeddings)
	simRange := SimilarityRange(embeddings)
	discGap, err := DiscriminationGap(embeddings, labels)
	if err != nil {
		return nil, err
	}

	return &EmbeddingMetrics{
		Anisotropy:        aniso,
		SimilarityRange:   simRange,
		DiscriminationGap: discGap,
	}, nil
}