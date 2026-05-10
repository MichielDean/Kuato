package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNewOllamaClient_Defaults(t *testing.T) {
	client, err := NewOllamaClient(OllamaClientConfig{})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}
	if client.baseURL != defaultBaseURL {
		t.Errorf("expected baseURL %q, got %q", defaultBaseURL, client.baseURL)
	}
	if client.timeout != defaultTimeout {
		t.Errorf("expected timeout %v, got %v", defaultTimeout, client.timeout)
	}
}

func TestNewOllamaClient_CustomBaseURL(t *testing.T) {
	client, err := NewOllamaClient(OllamaClientConfig{
		BaseURL: "http://localhost:11434",
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}
	if client.baseURL != "http://localhost:11434" {
		t.Errorf("expected baseURL http://localhost:11434, got %q", client.baseURL)
	}
}

func TestNewOllamaClient_TestServer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client, err := NewOllamaClient(OllamaClientConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}
	if client.baseURL != server.URL {
		t.Errorf("expected baseURL %q, got %q", server.URL, client.baseURL)
	}
}

func TestOllamaClient_Generate_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Errorf("expected /api/generate, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}

		var req struct {
			Model  string `json:"model"`
			Prompt string `json:"prompt"`
			Stream bool   `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if req.Stream {
			t.Error("expected stream:false")
		}

		resp := map[string]string{"response": "  Hello, world!  "}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client, err := NewOllamaClient(OllamaClientConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	result, err := client.Generate(context.Background(), "test prompt", "test-model")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result != "Hello, world!" {
		t.Errorf("expected 'Hello, world!', got %q", result)
	}
}

func TestOllamaClient_Generate_Unavailable(t *testing.T) {
	client, err := NewOllamaClient(OllamaClientConfig{
		BaseURL:    "http://localhost:59999",
		HTTPClient: &http.Client{Timeout: 100 * time.Millisecond},
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	_, err = client.Generate(context.Background(), "test", "model")
	if err == nil {
		t.Error("expected error for unavailable Ollama")
	}
	if !strings.Contains(err.Error(), "llmem: ollama:") {
		t.Errorf("error should have domain prefix: %v", err)
	}
}

func TestOllamaClient_IsAvailable_True(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{
				"models": []map[string]string{
					{"name": "nomic-embed-text:latest"},
				},
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	client, err := NewOllamaClient(OllamaClientConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	if !client.IsAvailable(context.Background()) {
		t.Error("expected IsAvailable to return true")
	}
}

func TestOllamaClient_IsAvailable_Unreachable(t *testing.T) {
	client, err := NewOllamaClient(OllamaClientConfig{
		BaseURL:    "http://localhost:59999",
		HTTPClient: &http.Client{Timeout: 100 * time.Millisecond},
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	if client.IsAvailable(context.Background()) {
		t.Error("expected IsAvailable to return false for unreachable server")
	}
}

func TestOllamaClient_PullModel_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/pull" {
			t.Errorf("expected /api/pull, got %s", r.URL.Path)
		}
		resp := map[string]string{"status": "success"}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client, err := NewOllamaClient(OllamaClientConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	ok, err := client.PullModel(context.Background(), "test-model")
	if err != nil {
		t.Fatalf("PullModel: %v", err)
	}
	if !ok {
		t.Error("expected PullModel to return true")
	}
}