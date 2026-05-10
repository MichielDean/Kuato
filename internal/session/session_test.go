package session

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/MichielDean/LLMem/internal/store"
)

func newTestStore(t *testing.T) *store.MemoryStore {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	ms, err := store.NewMemoryStore(store.StoreConfig{
		DBPath:     dbPath,
		DisableVec: true,
	})
	if err != nil {
		t.Fatalf("NewMemoryStore: %v", err)
	}
	t.Cleanup(func() { ms.Close() })
	return ms
}

func TestNewSessionHookCoordinator_NilStore(t *testing.T) {
	_, err := NewSessionHookCoordinator(SessionHookConfig{Store: nil})
	if err == nil {
		t.Error("expected error for nil store")
	}
}

func TestNewSessionHookCoordinator_Defaults(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}
	if coord.store == nil {
		t.Error("expected store to be set")
	}
	if coord.contextDir == "" {
		t.Error("expected default context dir")
	}
}

func TestSessionHookCoordinator_OnCreated(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnCreated(context.Background(), "session-123")
	if err != nil {
		t.Fatalf("OnCreated: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}
}

func TestSessionHookCoordinator_OnCreated_InvalidSessionID(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	_, err = coord.OnCreated(context.Background(), "../etc/passwd")
	if err == nil {
		t.Error("expected error for invalid session ID with traversal")
	}
}

func TestSessionHookCoordinator_OnIdle_Debounce(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	// First call should succeed
	result, err := coord.OnIdle(context.Background(), "session-456")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	// With no adapter, should return no_transcript
	if result != ResultNoTranscript {
		t.Errorf("expected %q, got %q", ResultNoTranscript, result)
	}

	// Second call within debounce window should return debounced
	result, err = coord.OnIdle(context.Background(), "session-456")
	if err != nil {
		t.Fatalf("OnIdle second: %v", err)
	}
	if result != ResultDebounced {
		t.Errorf("expected %q, got %q", ResultDebounced, result)
	}
}

func TestSessionHookCoordinator_OnCompacting(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:      ms,
		ContextDir: dir,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	resultType, contextPath, err := coord.OnCompacting(context.Background(), "session-789")
	if err != nil {
		t.Fatalf("OnCompacting: %v", err)
	}
	if resultType != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, resultType)
	}
	if contextPath == "" {
		t.Error("expected non-empty context path")
	}
}

func TestSessionHookCoordinator_OnEnding_NoAdapter(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnEnding(context.Background(), "session-end")
	if err != nil {
		t.Fatalf("OnEnding: %v", err)
	}
	if result != ResultNoTranscript {
		t.Errorf("expected %q, got %q", ResultNoTranscript, result)
	}
}

func TestOpenCodeAdapter_EmptyDBPath(t *testing.T) {
	adapter := NewOpenCodeAdapter("")
	content, err := adapter.ReadTranscript("test-session")
	if err != nil {
		t.Fatalf("ReadTranscript: %v", err)
	}
	if content != "" {
		t.Error("expected empty content for empty db path")
	}
}

func TestOpenCodeAdapter_ListSessions(t *testing.T) {
	adapter := NewOpenCodeAdapter("")
	sessions, err := adapter.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 0 {
		t.Errorf("expected 0 sessions, got %d", len(sessions))
	}
}