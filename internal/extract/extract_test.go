package extract

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/MichielDean/LLMem/internal/ollama"
)

func newTestOllamaClient(server *httptest.Server) *ollama.OllamaClient {
	client, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		panic(err)
	}
	return client
}

func TestExtractionEngine_Extract_Success(t *testing.T) {
	expectedMemories := []map[string]any{
		{"type": "fact", "content": "Go is statically typed", "confidence": 0.9},
	}
	expectedJSON, _ := json.Marshal(expectedMemories)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{"response": string(expectedJSON)}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "glm-5.1:cloud"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	engine, err := NewExtractionEngine(ExtractionConfig{
		OllamaClient: newTestOllamaClient(server),
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}

	result := engine.Extract(context.Background(), "Go is statically typed")
	if len(result) != 1 {
		t.Fatalf("expected 1 memory, got %d", len(result))
	}
	if result[0]["type"] != "fact" {
		t.Errorf("expected type=fact, got %v", result[0]["type"])
	}
}

func TestExtractionEngine_Extract_GracefulDegradation(t *testing.T) {
	// Server that always returns errors
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	defer server.Close()

	engine, err := NewExtractionEngine(ExtractionConfig{
		OllamaClient: newTestOllamaClient(server),
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}

	result := engine.Extract(context.Background(), "some text")
	if result == nil {
		t.Error("expected empty slice, not nil")
	}
	if len(result) != 0 {
		t.Errorf("expected empty slice on failure, got %d items", len(result))
	}
}

func TestExtractionEngine_CheckAvailable_True(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "glm-5.1:cloud"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	engine, err := NewExtractionEngine(ExtractionConfig{
		OllamaClient: newTestOllamaClient(server),
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}

	if !engine.CheckAvailable(context.Background()) {
		t.Error("expected CheckAvailable to return true")
	}
}

func TestExtractionEngine_CheckAvailable_False(t *testing.T) {
	// Use unreachable server
	client, _ := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    "http://localhost:59998",
		HTTPClient: &http.Client{Timeout: 100 * time.Millisecond},
	})

	engine, err := NewExtractionEngine(ExtractionConfig{
		OllamaClient: client,
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}

	if engine.CheckAvailable(context.Background()) {
		t.Error("expected CheckAvailable to return false for unreachable server")
	}
}

func TestExtractionEngine_Extract_CodeBlock(t *testing.T) {
	memories := []map[string]any{
		{"type": "fact", "content": "test", "confidence": 0.8},
	}
	memJSON, _ := json.Marshal(memories)
	response := "Here are the extracted memories:\n```json\n" + string(memJSON) + "\n```\n"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{"response": response}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	engine, err := NewExtractionEngine(ExtractionConfig{
		OllamaClient: newTestOllamaClient(server),
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}

	result := engine.Extract(context.Background(), "test text")
	if len(result) != 1 {
		t.Fatalf("expected 1 memory from code block extraction, got %d", len(result))
	}
}

func TestTryParseJSONArray_EmptyArray(t *testing.T) {
	result := tryParseJSONArray("[]")
	if result == nil {
		t.Error("expected empty array, not nil")
	}
	if len(result) != 0 {
		t.Errorf("expected 0 items, got %d", len(result))
	}
}

func TestTryParseJSONArray_InvalidJSON(t *testing.T) {
	result := tryParseJSONArray("not json")
	if result != nil {
		t.Error("expected nil for invalid JSON")
	}
}

func TestExtractionEngine_HTTPClientInjection(t *testing.T) {
	// Verify that ExtractionConfig.HTTPClient is properly wired through to OllamaClient
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{"response": `[{"type":"fact","content":"test","confidence":0.9}]`}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	engine, err := NewExtractionEngine(ExtractionConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine with HTTPClient: %v", err)
	}

	result := engine.Extract(context.Background(), "test text")
	if len(result) != 1 {
		t.Errorf("expected 1 memory via HTTPClient injection, got %d", len(result))
	}
}

func TestNewExtractionEngine_Defaults(t *testing.T) {
	engine, err := NewExtractionEngine(ExtractionConfig{})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}
	if engine.model != defaultModel {
		t.Errorf("expected model %q, got %q", defaultModel, engine.model)
	}
}

func TestExtractionEngine_Extract_EmptyResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{"response": "No memories found."}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	engine, err := NewExtractionEngine(ExtractionConfig{
		OllamaClient: newTestOllamaClient(server),
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}

	result := engine.Extract(context.Background(), "irrelevant text")
	if len(result) != 0 {
		t.Errorf("expected empty result for non-JSON response, got %d items", len(result))
	}
}

func TestExtractionEngine_Extract_UnavailableOllama(t *testing.T) {
	client, _ := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    "http://localhost:59997",
		HTTPClient: &http.Client{Timeout: 100 * time.Millisecond},
	})

	engine, err := NewExtractionEngine(ExtractionConfig{
		OllamaClient: client,
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}

	// Context that times out immediately
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	result := engine.Extract(ctx, "test")
	if len(result) != 0 {
		t.Errorf("expected empty result on timeout, got %d items", len(result))
	}

	_ = strings.TrimSpace("") // suppress unused warning
}