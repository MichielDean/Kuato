// Package embed provides an Ollama /api/embeddings client with LRU cache
// for the LLMem project. It supports the nomic-embed-text model (768 dimensions)
// by default and gracefully falls back when Ollama is unavailable.
package embed

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"

	"github.com/MichielDean/LLMem/internal/urlvalidate"
)

const (
	defaultModel       = "nomic-embed-text"
	defaultBaseURL     = "http://localhost:11434"
	defaultMaxCache    = 2048
	defaultDimensions  = 768
	defaultTimeout     = 30 * time.Second
)

// EmbeddingConfig contains the configuration for creating an EmbeddingEngine.
type EmbeddingConfig struct {
	// Model is the embedding model name. Defaults to "nomic-embed-text".
	Model string

	// BaseURL is the Ollama API base URL. Defaults to "http://localhost:11434".
	BaseURL string

	// MaxCacheSize is the LRU cache max entries. Defaults to 2048.
	MaxCacheSize int

	// Dimensions is the expected vector dimensions. Defaults to 768.
	Dimensions int

	// Timeout is the HTTP client timeout. Defaults to 30s.
	// Timeout defaults to 30s if zero.
	Timeout time.Duration

	// HTTPClient is an optional pre-configured HTTP client (for testing with httptest.NewServer).
	HTTPClient *http.Client
}

// EmbeddingEngine generates embedding vectors using a local Ollama server.
// It caches results in an LRU cache to avoid redundant API calls.
// This is specific to Ollama embedding and will not be reused.
type EmbeddingEngine struct {
	model      string
	baseURL    string
	dimensions int
	httpClient *http.Client
	cache      *lru.Cache[string, []float32]
	mu         sync.RWMutex
}

// fmtErr wraps an error with the "llmem: embed:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: embed: "+format, args...)
}

// NewEmbeddingEngine creates and initializes an EmbeddingEngine.
// Validates baseURL via urlvalidate.ValidateBaseURL.
// If cfg.Model == "", defaults to "nomic-embed-text".
// If cfg.BaseURL == "", defaults to "http://localhost:11434".
// If cfg.MaxCacheSize <= 0, defaults to 2048.
// If cfg.Dimensions <= 0, defaults to 768.
// If cfg.HTTPClient == nil, creates a new http.Client with cfg.Timeout.
// The constructor leaves the engine in a fully usable state.
func NewEmbeddingEngine(cfg EmbeddingConfig) (*EmbeddingEngine, error) {
	// Apply defaults
	model := cfg.Model
	if model == "" {
		model = defaultModel
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	dimensions := cfg.Dimensions
	if dimensions <= 0 {
		dimensions = defaultDimensions
	}

	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = defaultTimeout
	}

	// Create or use provided HTTP client
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		// Validate baseURL only for production (non-test) clients.
		// When an HTTPClient is provided (e.g., from httptest.NewServer),
		// the caller controls the transport and URL validation is relaxed.
		validatedURL, err := urlvalidate.ValidateBaseURL(baseURL, "embed")
		if err != nil {
			return nil, fmtErr("unsafe Ollama URL: %w", err)
		}
		baseURL = validatedURL

		httpClient = &http.Client{
			Timeout: timeout,
		}
	} else {
		// For test environments, just strip trailing slash
		baseURL = strings.TrimRight(baseURL, "/")
	}

	maxCache := cfg.MaxCacheSize
	if maxCache <= 0 {
		maxCache = defaultMaxCache
	}

	cache, err := lru.New[string, []float32](maxCache)
	if err != nil {
		return nil, fmtErr("create LRU cache: %w", err)
	}

	return &EmbeddingEngine{
		model:      model,
		baseURL:    baseURL,
		dimensions: dimensions,
		httpClient: httpClient,
		cache:      cache,
	}, nil
}

// Embed returns the embedding vector for the given text.
// Returns cached result if available. On cache miss, makes HTTP POST to
// {baseURL}/api/embeddings with {"model": model, "prompt": text}.
// On HTTP error or JSON decode error, wraps error with domain prefix.
// Never panics. Context cancellation is respected via ctx.
// Callers MUST NOT modify the returned slice.
func (e *EmbeddingEngine) Embed(ctx context.Context, text string) ([]float32, error) {
	// Check cache first (read lock)
	e.mu.RLock()
	if val, ok := e.cache.Get(text); ok {
		e.mu.RUnlock()
		// Return a copy to prevent mutation
		result := make([]float32, len(val))
		copy(result, val)
		return result, nil
	}
	e.mu.RUnlock()

	// Cache miss — make API request
	reqBody := struct {
		Model  string `json:"model"`
		Prompt string `json:"prompt"`
	}{
		Model:  e.model,
		Prompt: text,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmtErr("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/api/embeddings", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmtErr("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmtErr("Ollama embedding request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmtErr("Ollama embedding request failed: HTTP %d", resp.StatusCode)
	}

	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmtErr("decode Ollama response: %w", err)
	}

	if len(result.Embedding) == 0 {
		return nil, fmtErr("Ollama returned empty embedding")
	}

	if len(result.Embedding) != e.dimensions {
		return nil, fmtErr("embedding dimension %d does not match expected %d", len(result.Embedding), e.dimensions)
	}

	// Store in cache (write lock). LRU eviction happens automatically.
	e.mu.Lock()
	e.cache.Add(text, result.Embedding)
	e.mu.Unlock()

	// Return a copy to prevent mutation
	out := make([]float32, len(result.Embedding))
	copy(out, result.Embedding)
	return out, nil
}

// CheckAvailable returns true if the configured Ollama model is available.
// Makes GET to {baseURL}/api/tags and checks if any model name starts
// with the configured model name. Returns false on any error.
// Never returns an error — logs at slog.Debug level.
func (e *EmbeddingEngine) CheckAvailable(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.baseURL+"/api/tags", nil)
	if err != nil {
		slog.Debug("llmem: embed: check available: create request failed", "error", err)
		return false
	}

	resp, err := e.httpClient.Do(req)
	if err != nil {
		slog.Debug("llmem: embed: check available: request failed", "error", err)
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		slog.Debug("llmem: embed: check available: unexpected status", "status", resp.StatusCode)
		return false
	}

	var tagsResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		slog.Debug("llmem: embed: check available: decode failed", "error", err)
		return false
	}

	for _, m := range tagsResp.Models {
		// Match exact name or name with tag suffix (e.g. "nomic-embed-text:latest").
		// Do NOT match longer model names that merely start with our model name.
		if m.Name == e.model || strings.HasPrefix(m.Name, e.model+":") {
			return true
		}
	}
	return false
}

// Close closes the underlying HTTP client's idle connections.
// Safe to call multiple times (idempotent).
func (e *EmbeddingEngine) Close() error {
	e.httpClient.CloseIdleConnections()
	return nil
}

// vecToBytes encodes a []float32 into packed little-endian bytes.
// Matches Python's struct.pack(f"{dim}f", *vec).
func vecToBytes(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// bytesToVec decodes a packed float32 byte slice into a []float32.
// Matches Python's struct.unpack(f"{dim}f", data).
func bytesToVec(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	dim := len(data) / 4
	result := make([]float32, dim)
	for i := 0; i < dim; i++ {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return result
}