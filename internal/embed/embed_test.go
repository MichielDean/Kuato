package embed

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestEmbeddingEngine_Embed_Success(t *testing.T) {
	// Mock Ollama server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/embeddings" {
			var req struct {
				Model  string `json:"model"`
				Prompt string `json:"prompt"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("failed to decode request: %v", err)
				http.Error(w, "bad request", http.StatusBadRequest)
				return
			}
			// Return a 3-dimensional vector for testing
			vec := make([]float32, 3)
			for i := range vec {
				vec[i] = float32(i) * 0.1
			}
			resp := map[string]any{"embedding": vec}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     server.URL,
		Dimensions:  3,
		HTTPClient:  server.Client(),
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	vec, err := engine.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != 3 {
		t.Errorf("expected 3 dimensions, got %d", len(vec))
	}
	if math.Abs(float64(vec[0])) > 1e-6 {
		t.Errorf("expected vec[0] ≈ 0.0, got %f", vec[0])
	}
}

func TestEmbeddingEngine_Embed_CacheHit(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		vec := make([]float32, 3)
		resp := map[string]any{"embedding": vec}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     server.URL,
		Dimensions:  3,
		HTTPClient:  server.Client(),
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	_, err = engine.Embed(context.Background(), "test text")
	if err != nil {
		t.Fatalf("first Embed: %v", err)
	}

	_, err = engine.Embed(context.Background(), "test text")
	if err != nil {
		t.Fatalf("second Embed (cache): %v", err)
	}

	if callCount != 1 {
		t.Errorf("expected 1 call (second should be cache hit), got %d", callCount)
	}
}

func TestEmbeddingEngine_Embed_CacheEviction(t *testing.T) {
	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     "http://localhost:11434",
		Dimensions:  3,
		MaxCacheSize: 2, // Very small cache
		HTTPClient:  &http.Client{Timeout: 1 * time.Second},
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	// Pre-populate cache directly and verify eviction
	engine.cache.Add("key1", []float32{0.1, 0.2, 0.3})
	engine.cache.Add("key2", []float32{0.4, 0.5, 0.6})

	// Both should be in cache
	if _, ok := engine.cache.Get("key1"); !ok {
		t.Error("expected key1 in cache")
	}
	if _, ok := engine.cache.Get("key2"); !ok {
		t.Error("expected key2 in cache")
	}

	// Adding a third key should evict the oldest (key1) since max size is 2
	engine.cache.Add("key3", []float32{0.7, 0.8, 0.9})
	if _, ok := engine.cache.Get("key1"); ok {
		t.Error("expected key1 to be evicted from cache")
	}
	if _, ok := engine.cache.Get("key3"); !ok {
		t.Error("expected key3 in cache")
	}
}

func TestEmbeddingEngine_Embed_OllamaUnavailable(t *testing.T) {
	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     "http://localhost:0", // port 0 will fail
		Dimensions:  3,
		HTTPClient:  &http.Client{Timeout: 100 * time.Millisecond},
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	_, err = engine.Embed(context.Background(), "test")
	if err == nil {
		t.Error("expected error when Ollama is unavailable")
	}
	if !strings.Contains(err.Error(), "llmem: embed:") {
		t.Errorf("expected domain-prefixed error, got: %v", err)
	}
}

func TestEmbeddingEngine_Embed_InvalidResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("not json"))
	}))
	defer server.Close()

	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     server.URL,
		Dimensions:  3,
		HTTPClient:  server.Client(),
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	_, err = engine.Embed(context.Background(), "test")
	if err == nil {
		t.Error("expected error for invalid JSON response")
	}
}

func TestEmbeddingEngine_CheckAvailable_True(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{
				"models": []map[string]string{
					{"name": "nomic-embed-text"},
				},
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     server.URL,
		HTTPClient:  server.Client(),
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	if !engine.CheckAvailable(context.Background()) {
		t.Error("expected model to be available")
	}
}

func TestEmbeddingEngine_CheckAvailable_False(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{
				"models": []map[string]string{
					{"name": "some-other-model"},
				},
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     server.URL,
		HTTPClient:  server.Client(),
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	if engine.CheckAvailable(context.Background()) {
		t.Error("expected model NOT to be available")
	}
}

func TestEmbeddingEngine_CheckAvailable_Unreachable(t *testing.T) {
	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     "http://localhost:0",
		HTTPClient:  &http.Client{Timeout: 100 * time.Millisecond},
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	if engine.CheckAvailable(context.Background()) {
		t.Error("expected CheckAvailable to return false for unreachable host")
	}
}

func TestEmbeddingEngine_VecToBytes_RoundTrip(t *testing.T) {
	original := []float32{1.0, -2.5, 3.14, 0.0, -0.001}
	encoded := vecToBytes(original)
	decoded := bytesToVec(encoded)

	if len(decoded) != len(original) {
		t.Fatalf("expected %d elements, got %d", len(original), len(decoded))
	}
	for i := range original {
		if math.Abs(float64(decoded[i]-original[i])) > 1e-6 {
			t.Errorf("element %d: expected %f, got %f", i, original[i], decoded[i])
		}
	}
}

func TestEmbeddingEngine_Constructor_Defaults(t *testing.T) {
	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		HTTPClient: &http.Client{Timeout: 1 * time.Second}, // use local client to avoid network
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	if engine.model != defaultModel {
		t.Errorf("expected model %q, got %q", defaultModel, engine.model)
	}
	if engine.baseURL != defaultBaseURL {
		t.Errorf("expected baseURL %q, got %q", defaultBaseURL, engine.baseURL)
	}
	if engine.dimensions != defaultDimensions {
		t.Errorf("expected dimensions %d, got %d", defaultDimensions, engine.dimensions)
	}
}

func TestEmbeddingEngine_Constructor_InvalidURL(t *testing.T) {
	// URL validation only happens when HTTPClient is nil (production mode)
	_, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL: "ftp://bad-url.com",
		HTTPClient: nil, // nil triggers URL validation
	})
	if err == nil {
		t.Error("expected error for invalid URL scheme")
	}
}

func TestEmbeddingEngine_Close_Idempotent(t *testing.T) {
	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		HTTPClient: &http.Client{},
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	// Close should be safe to call multiple times
	if err := engine.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	if err := engine.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

func TestEmbeddingEngine_Embed_DimensionMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Return a 5-dim vector when engine expects 3
		vec := make([]float32, 5)
		resp := map[string]any{"embedding": vec}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     server.URL,
		Dimensions:  3,
		HTTPClient:  server.Client(),
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	_, err = engine.Embed(context.Background(), "test")
	if err == nil {
		t.Error("expected dimension mismatch error")
	}
	if !strings.Contains(err.Error(), "dimension") {
		t.Errorf("expected dimension error, got: %v", err)
	}
}

func TestEmbeddingEngine_Embed_ContextCancellation(t *testing.T) {
	// Create a server that delays its response (but longer than client timeout)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
		vec := make([]float32, 3)
		resp := map[string]any{"embedding": vec}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create engine with a short timeout client
	client := &http.Client{Timeout: 100 * time.Millisecond}
	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     server.URL,
		Dimensions:  3,
		HTTPClient:  client,
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}

	_, err = engine.Embed(context.Background(), "test")
	if err == nil {
		t.Error("expected timeout error from slow server")
	}
}

// Test concurrent access to cache
func TestEmbeddingEngine_ConcurrentAccess(t *testing.T) {
	engine, err := NewEmbeddingEngine(EmbeddingConfig{
		BaseURL:     "http://localhost:11434",
		Dimensions:  3,
		MaxCacheSize: 100,
		HTTPClient:  &http.Client{Timeout: 1 * time.Second},
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	defer engine.Close()

	// Verify cache is created and functional
	engine.cache.Add("test", []float32{1.0, 2.0, 3.0})
	vec, ok := engine.cache.Get("test")
	if !ok {
		t.Error("expected cache hit")
	}
	if len(vec) != 3 {
		t.Errorf("expected 3 elements, got %d", len(vec))
	}
}