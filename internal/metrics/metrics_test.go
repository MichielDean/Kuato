package metrics

import (
	"math"
	"testing"
)

func TestAnisotropy_EmptyInput(t *testing.T) {
	result := Anisotropy(nil)
	if result != 0.0 {
		t.Errorf("expected 0.0 for nil input, got %f", result)
	}
	result = Anisotropy([][]float32{})
	if result != 0.0 {
		t.Errorf("expected 0.0 for empty input, got %f", result)
	}
}

func TestAnisotropy_SingleVector(t *testing.T) {
	result := Anisotropy([][]float32{{1.0, 0.0, 0.0}})
	if result != 0.0 {
		t.Errorf("expected 0.0 for single vector, got %f", result)
	}
}

func TestAnisotropy_IdenticalVectors(t *testing.T) {
	vec := []float32{1.0, 0.0, 0.0}
	result := Anisotropy([][]float32{vec, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}})
	if result != 1.0 {
		t.Errorf("expected 1.0 for identical vectors, got %f", result)
	}
}

func TestAnisotropy_OrthogonalVectors(t *testing.T) {
	result := Anisotropy([][]float32{
		{1.0, 0.0},
		{0.0, 1.0},
	})
	// Orthogonal vectors have cosine similarity 0.0, so anisotropy should be ~0.0
	// But math.Max(0.0, ...) ensures it's clamped to >= 0
	if math.Abs(result-0.0) > 0.01 {
		t.Errorf("expected near 0.0 for orthogonal vectors, got %f", result)
	}
}

func TestSimilarityRange_EmptyInput(t *testing.T) {
	result := SimilarityRange(nil)
	if result != 0.0 {
		t.Errorf("expected 0.0 for nil input, got %f", result)
	}
}

func TestSimilarityRange_SingleVector(t *testing.T) {
	result := SimilarityRange([][]float32{{1.0, 0.0}})
	if result != 0.0 {
		t.Errorf("expected 0.0 for single vector, got %f", result)
	}
}

func TestSimilarityRange_IdenticalVectors(t *testing.T) {
	result := SimilarityRange([][]float32{
		{1.0, 0.0},
		{1.0, 0.0},
	})
	if result != 0.0 {
		t.Errorf("expected 0.0 for identical vectors, got %f", result)
	}
}

func TestSimilarityRange_DiverseVectors(t *testing.T) {
	result := SimilarityRange([][]float32{
		{1.0, 0.0},
		{0.0, 1.0},
		{-1.0, 0.0},
	})
	// Should have positive range since similarity varies between pairs
	if result <= 0.0 {
		t.Errorf("expected positive range for diverse vectors, got %f", result)
	}
}

func TestDiscriminationGap_NoLabels(t *testing.T) {
	gap, err := DiscriminationGap([][]float32{{1.0, 0.0}}, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if gap != 0.0 {
		t.Errorf("expected 0.0 for nil labels, got %f", gap)
	}

	gap, err = DiscriminationGap([][]float32{{1.0, 0.0}}, []string{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if gap != 0.0 {
		t.Errorf("expected 0.0 for empty labels, got %f", gap)
	}
}

func TestDiscriminationGap_LabelLengthMismatch(t *testing.T) {
	_, err := DiscriminationGap([][]float32{{1.0, 0.0}}, []string{"a", "b"})
	if err == nil {
		t.Error("expected error for mismatched lengths")
	}
}

func TestDiscriminationGap_SingleClass(t *testing.T) {
	gap, err := DiscriminationGap(
		[][]float32{{1.0, 0.0}, {0.9, 0.1}},
		[]string{"a", "a"},
	)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if gap != 0.0 {
		t.Errorf("expected 0.0 for single class, got %f", gap)
	}
}

func TestDiscriminationGap_TwoClasses(t *testing.T) {
	gap, err := DiscriminationGap(
		[][]float32{
			{1.0, 0.0}, {0.9, 0.1}, // class A
			{0.0, 1.0}, {0.1, 0.9}, // class B
		},
		[]string{"a", "a", "b", "b"},
	)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Well-separated classes should have positive gap
	if gap <= 0.0 {
		t.Errorf("expected positive gap for separable classes, got %f", gap)
	}
}

func TestComputeMetrics_Truncation(t *testing.T) {
	// Create more embeddings than maxEmbeddings
	embeddings := make([][]float32, 200)
	for i := range embeddings {
		embeddings[i] = []float32{float32(i), 1.0}
	}
	labels := make([]string, 200)
	for i := range labels {
		labels[i] = "a"
	}

	result, err := ComputeMetrics(embeddings, labels, 100)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	// Truncation should have happened (slog.Warn was called)
	// Verify the result is still valid
	if result.Anisotropy < 0 || result.Anisotropy > 1.0 {
		t.Errorf("anisotropy out of range: %f", result.Anisotropy)
	}
}

func TestComputeMetrics_Defaults(t *testing.T) {
	// Test with maxEmbeddings=0 defaults to MetricsMaxEmbeddings
	result, err := ComputeMetrics([][]float32{{1.0, 0.0}, {0.0, 1.0}}, nil, 0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.DiscriminationGap != 0.0 {
		t.Errorf("expected 0.0 discrimination gap for nil labels, got %f", result.DiscriminationGap)
	}
}

func TestCosineSimilarity_ZeroVector(t *testing.T) {
	result := cosineSimilarity([]float32{0.0, 0.0}, []float32{1.0, 0.0})
	if result != 0.0 {
		t.Errorf("expected 0.0 for zero vector, got %f", result)
	}
}

func TestCosineSimilarity_IdenticalVectors(t *testing.T) {
	v := []float32{1.0, 2.0, 3.0}
	result := cosineSimilarity(v, v)
	// For identical non-zero vectors, cosine similarity should be 1.0
	if math.Abs(result-1.0) > 1e-6 {
		t.Errorf("expected 1.0 for identical vectors, got %f", result)
	}
}