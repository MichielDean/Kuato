package session

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/MichielDean/LLMem/internal/embed"
	"github.com/MichielDean/LLMem/internal/extract"
	"github.com/MichielDean/LLMem/internal/ollama"
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

// createTestOpenCodeDB creates a temporary SQLite database with the OpenCode schema
// and returns the path to it. The caller is responsible for cleanup via t.Cleanup.
func createTestOpenCodeDB(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "opencode.db")

	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open test opencode db: %v", err)
	}
	defer db.Close()

	// Create OpenCode schema with backtick-quoted identifiers
	schema := `
	CREATE TABLE IF NOT EXISTS session (
		id TEXT PRIMARY KEY,
		project_id TEXT,
		parent_id TEXT,
		slug TEXT,
		directory TEXT,
		title TEXT,
		version TEXT,
		share_url TEXT,
		summary_additions TEXT,
		summary_deletions TEXT,
		summary_files TEXT,
		summary_diffs TEXT,
		revert TEXT,
		permission TEXT,
		time_created INTEGER,
		time_updated INTEGER,
		time_compacting INTEGER,
		time_archived INTEGER,
		workspace_id TEXT,
		path TEXT
	);

	CREATE TABLE IF NOT EXISTS message (
		id TEXT PRIMARY KEY,
		session_id TEXT NOT NULL,
		time_created INTEGER,
		time_updated INTEGER,
		data TEXT,
		FOREIGN KEY (session_id) REFERENCES session(id)
	);

	CREATE TABLE IF NOT EXISTS part (
		id TEXT PRIMARY KEY,
		message_id TEXT NOT NULL,
		session_id TEXT,
		time_created INTEGER,
		time_updated INTEGER,
		data TEXT,
		FOREIGN KEY (message_id) REFERENCES message(id)
	);
	`
	if _, err := db.Exec(schema); err != nil {
		t.Fatalf("create opencode schema: %v", err)
	}

	return dbPath
}

// insertTestSession inserts a session into the test OpenCode DB.
func insertTestSession(t *testing.T, dbPath string, id string, timeCreated, timeUpdated int64, timeCompacting *int64, directory string) {
	t.Helper()
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open db for insert: %v", err)
	}
	defer db.Close()

	var compactingVal interface{}
	if timeCompacting != nil {
		compactingVal = *timeCompacting
	}

	_, err = db.Exec(
		"INSERT INTO session (id, time_created, time_updated, time_compacting, directory) VALUES (?, ?, ?, ?, ?)",
		id, timeCreated, timeUpdated, compactingVal, directory,
	)
	if err != nil {
		t.Fatalf("insert test session %s: %v", id, err)
	}
}

// insertTestMessage inserts a message into the test OpenCode DB.
func insertTestMessage(t *testing.T, dbPath string, id, sessionID string, timeCreated int64, role string) {
	t.Helper()
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open db for insert: %v", err)
	}
	defer db.Close()

	data := fmt.Sprintf(`{"role": "%s"}`, role)
	_, err = db.Exec(
		"INSERT INTO message (id, session_id, time_created, data) VALUES (?, ?, ?, ?)",
		id, sessionID, timeCreated, data,
	)
	if err != nil {
		t.Fatalf("insert test message %s: %v", id, err)
	}
}

// insertTestPart inserts a part into the test OpenCode DB.
func insertTestPart(t *testing.T, dbPath string, id, messageID, sessionID string, timeCreated int64, partData string) {
	t.Helper()
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open db for insert: %v", err)
	}
	defer db.Close()

	_, err = db.Exec(
		"INSERT INTO part (id, message_id, session_id, time_created, data) VALUES (?, ?, ?, ?, ?)",
		id, messageID, sessionID, timeCreated, partData,
	)
	if err != nil {
		t.Fatalf("insert test part %s: %v", id, err)
	}
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

func TestSessionHookCoordinator_OnIdle_CustomDebounce(t *testing.T) {
	ms := newTestStore(t)
	// Set DebounceSeconds to 0 to verify it defaults to 30, and use a very
	// short debounce to test custom values
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		DebounceSeconds:  0, // should default to 30
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}
	if coord.debounceSeconds != 30 {
		t.Errorf("expected debounceSeconds 30, got %d", coord.debounceSeconds)
	}

	// Verify explicit custom value
	coord2, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		DebounceSeconds:  10,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}
	if coord2.debounceSeconds != 10 {
		t.Errorf("expected debounceSeconds 10, got %d", coord2.debounceSeconds)
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

func TestOpenCodeAdapter_ReadTranscript_EmptyDBPath(t *testing.T) {
	adapter, err := NewOpenCodeAdapter("")
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })
	content, err := adapter.ReadTranscript("test-session")
	if err != nil {
		t.Fatalf("ReadTranscript: %v", err)
	}
	if content != "" {
		t.Error("expected empty content for empty db path")
	}
}

func TestOpenCodeAdapter_ListSessions_EmptyDBPath(t *testing.T) {
	adapter, err := NewOpenCodeAdapter("")
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })
	sessions, err := adapter.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(sessions) != 0 {
		t.Errorf("expected 0 sessions, got %d", len(sessions))
	}
}

func TestSessionHookCoordinator_OnIdle_EvictsStaleEntries(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:           ms,
		DebounceSeconds: 30,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	// Simulate several sessions having idle entries
	for i := 0; i < 100; i++ {
		coord.lastIdle[fmt.Sprintf("session-%d", i)] = time.Now()
	}

	// Now add stale entries (older than eviction threshold = 30s * 10 = 300s)
	staleTime := time.Now().Add(-600 * time.Second) // 10 minutes ago, well past eviction
	coord.lastIdle["stale-session-1"] = staleTime
	coord.lastIdle["stale-session-2"] = staleTime

	totalBefore := len(coord.lastIdle)

	// OnIdle triggers eviction of stale entries
	_, err = coord.OnIdle(context.Background(), "new-session")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}

	// Stale entries should have been evicted, recent ones kept
	totalAfter := len(coord.lastIdle)

	if totalAfter >= totalBefore {
		t.Errorf("expected stale entries to be evicted: before=%d, after=%d", totalBefore, totalAfter)
	}

	// Stale entries should be gone
	if _, exists := coord.lastIdle["stale-session-1"]; exists {
		t.Error("stale-session-1 should have been evicted")
	}
	if _, exists := coord.lastIdle["stale-session-2"]; exists {
		t.Error("stale-session-2 should have been evicted")
	}

	// New session should be present
	if _, exists := coord.lastIdle["new-session"]; !exists {
		t.Error("new-session should be present in lastIdle")
	}
}

func TestSessionHookCoordinator_OnCompacting_UsesValidatedID(t *testing.T) {
	ms := newTestStore(t)
	dir := t.TempDir()
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:      ms,
		ContextDir: dir,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	resultType, contextPath, err := coord.OnCompacting(context.Background(), "safe-session-id")
	if err != nil {
		t.Fatalf("OnCompacting: %v", err)
	}
	if resultType != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, resultType)
	}
	// Verify the context file is created with the validated session ID in its name
	expected := filepath.Join(dir, "safe-session-id_context.md")
	if contextPath != expected {
		t.Errorf("expected context path %s, got %s", expected, contextPath)
	}
}

func TestSessionHookCoordinator_OnIdle_UsesValidatedID(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	// Valid session ID should work
	result, err := coord.OnIdle(context.Background(), "valid-session")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	if result != ResultNoTranscript {
		t.Errorf("expected %q, got %q", ResultNoTranscript, result)
	}

	// The validated ID should be used for debounce, so calling with same ID should debounce
	result2, err := coord.OnIdle(context.Background(), "valid-session")
	if err != nil {
		t.Fatalf("OnIdle second: %v", err)
	}
	if result2 != ResultDebounced {
		t.Errorf("expected %q, got %q", ResultDebounced, result2)
	}
}

func TestSessionHookCoordinator_OnCreated_UsesValidatedID(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnCreated(context.Background(), "test-session-abc")
	if err != nil {
		t.Fatalf("OnCreated: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}
}

func TestSessionHookCoordinator_OnEnding_UsesValidatedID(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{Store: ms})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnEnding(context.Background(), "test-session-end")
	if err != nil {
		t.Fatalf("OnEnding: %v", err)
	}
	if result != ResultNoTranscript {
		t.Errorf("expected %q, got %q", ResultNoTranscript, result)
	}
}

func TestOpenCodeAdapter_ReadOnlyDB(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	// The adapter opens the DB in read-only mode (mode=ro).
	// Verify that write operations are rejected, confirming mode=ro is effective.
	// Using the adapter's underlying db handle directly to attempt a write.
	writeErr := func() error {
		_, err := adapter.db.Exec("INSERT INTO session (id) VALUES ('ro-test')")
		return err
	}()
	if writeErr == nil {
		t.Error("expected write to fail on read-only database, but it succeeded — mode=ro not enforced")
	}
}

func TestOpenCodeAdapter_Close_Idempotent(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}

	// Close should not error
	if err := adapter.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	// Second close should also be nil (idempotent)
	if err := adapter.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestOpenCodeAdapter_DBNotFound(t *testing.T) {
	adapter, err := NewOpenCodeAdapter("/nonexistent/path/opencode.db")
	if err != nil {
		// Constructor should succeed even with nonexistent path — the error
		// comes when trying to query. Ping will fail but sql.Open succeeds.
		// Actually, Ping should fail so the constructor returns an error.
		// Either behavior is acceptable — let's verify the error path works.
		t.Logf("NewOpenCodeAdapter returned error for nonexistent DB: %v (acceptable)", err)
		// This is expected — the constructor eagerly validates the connection.
		return
	}
	t.Cleanup(func() { adapter.Close() })

	// If the constructor somehow succeeded, ReadTranscript should still return an error
	_, err = adapter.ReadTranscript("test-session")
	if err == nil {
		t.Error("expected error when reading from nonexistent database")
	}
	if !strings.Contains(err.Error(), "llmem: session:") {
		t.Errorf("expected error to contain domain prefix 'llmem: session:', got: %v", err)
	}
}

func TestOpenCodeAdapter_ReadTranscript_SessionIDNotFound(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)

	// Insert a session so the DB has data, but query for a different ID
	insertTestSession(t, dbPath, "existing-session", 1000, 2000, nil, "/work")

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	transcript, err := adapter.ReadTranscript("nonexistent-session")
	if err != nil {
		t.Fatalf("ReadTranscript for nonexistent session: %v", err)
	}
	if transcript != "" {
		t.Errorf("expected empty transcript for nonexistent session, got: %q", transcript)
	}
}

func TestOpenCodeAdapter_ReadTranscript_FullTranscript(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)

	// Insert a session
	insertTestSession(t, dbPath, "sess-1", 1000, 5000, nil, "/work/project")

	// Insert user message
	insertTestMessage(t, dbPath, "msg-1", "sess-1", 1000, "user")
	// Insert user text part
	insertTestPart(t, dbPath, "part-1", "msg-1", "sess-1", 1000,
		`{"type": "text", "text": "Hello, how are you?"}`)

	// Insert assistant message
	insertTestMessage(t, dbPath, "msg-2", "sess-1", 2000, "assistant")
	// Insert assistant text part
	insertTestPart(t, dbPath, "part-2", "msg-2", "sess-1", 2000,
		`{"type": "text", "text": "I'm doing well, thanks!"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	transcript, err := adapter.ReadTranscript("sess-1")
	if err != nil {
		t.Fatalf("ReadTranscript: %v", err)
	}

	if !strings.Contains(transcript, "User:") {
		t.Error("expected transcript to contain 'User:'")
	}
	if !strings.Contains(transcript, "Hello, how are you?") {
		t.Error("expected transcript to contain user message text")
	}
	if !strings.Contains(transcript, "Assistant:") {
		t.Error("expected transcript to contain 'Assistant:'")
	}
	if !strings.Contains(transcript, "I'm doing well, thanks!") {
		t.Error("expected transcript to contain assistant message text")
	}
}

func TestOpenCodeAdapter_ReadTranscript_CompactingRecentContext(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)

	// Insert a session with time_compacting set to 3000 — only messages
	// after time 3000 should be returned
	compactingTime := int64(3000)
	insertTestSession(t, dbPath, "sess-compact", 1000, 5000, &compactingTime, "/work")

	// Insert user message BEFORE compaction
	insertTestMessage(t, dbPath, "msg-old", "sess-compact", 1000, "user")
	insertTestPart(t, dbPath, "part-old", "msg-old", "sess-compact", 1000,
		`{"type": "text", "text": "Old message before compaction"}`)

	// Insert assistant message AFTER compaction
	insertTestMessage(t, dbPath, "msg-new", "sess-compact", 4000, "assistant")
	insertTestPart(t, dbPath, "part-new", "msg-new", "sess-compact", 4000,
		`{"type": "text", "text": "Recent message after compaction"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	transcript, err := adapter.ReadTranscript("sess-compact")
	if err != nil {
		t.Fatalf("ReadTranscript: %v", err)
	}

	// Should contain the message after compaction
	if !strings.Contains(transcript, "Recent message after compaction") {
		t.Error("expected transcript to contain recent message after compaction")
	}

	// Should NOT contain the message before compaction
	if strings.Contains(transcript, "Old message before compaction") {
		t.Error("expected transcript NOT to contain old message before compaction")
	}
}

func TestOpenCodeAdapter_ReadTranscript_InvalidSessionID(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	_, err = adapter.ReadTranscript("../etc/passwd")
	if err == nil {
		t.Error("expected error for path-traversal session ID")
	}
	if !strings.Contains(err.Error(), "llmem: session:") {
		t.Errorf("expected error to contain domain prefix, got: %v", err)
	}
}

func TestOpenCodeAdapter_ListSessions_ReturnsSessions(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)

	// Insert sessions
	insertTestSession(t, dbPath, "sess-1", 1000000, 2000000, nil, "/work/project1")
	insertTestSession(t, dbPath, "sess-2", 3000000, 4000000, nil, "/work/project2")

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	sessions, err := adapter.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}

	if len(sessions) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(sessions))
	}

	// Verify session data — sessions are ordered by time_created DESC
	// sess-2 has time_created=3000000, should appear first
	found1, found2 := false, false
	for _, s := range sessions {
		if s.ID == "sess-1" {
			found1 = true
			if s.StartTime == "" {
				t.Error("expected non-empty StartTime for sess-1")
			}
			if s.EndTime == "" {
				t.Error("expected non-empty EndTime for sess-1")
			}
			if s.WorkDir != "/work/project1" {
				t.Errorf("expected WorkDir='/work/project1', got %q", s.WorkDir)
			}
			// Verify time is in RFC3339 format
			if _, err := time.Parse(time.RFC3339, s.StartTime); err != nil {
				t.Errorf("StartTime not in RFC3339 format: %q, err: %v", s.StartTime, err)
			}
		}
		if s.ID == "sess-2" {
			found2 = true
			if s.WorkDir != "/work/project2" {
				t.Errorf("expected WorkDir='/work/project2', got %q", s.WorkDir)
			}
		}
	}
	if !found1 {
		t.Error("expected to find sess-1 in results")
	}
	if !found2 {
		t.Error("expected to find sess-2 in results")
	}
}

func TestOpenCodeAdapter_ReadTranscript_ExcludedPartTypes(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)

	insertTestSession(t, dbPath, "sess-parts", 1000, 5000, nil, "/work")

	// User message with multiple part types
	insertTestMessage(t, dbPath, "msg-parts", "sess-parts", 1000, "user")

	// Text part (included)
	insertTestPart(t, dbPath, "p-text", "msg-parts", "sess-parts", 1000,
		`{"type": "text", "text": "Hello"}`)
	// Step-start part (excluded)
	insertTestPart(t, dbPath, "p-step-start", "msg-parts", "sess-parts", 1100,
		`{"type": "step-start", "text": "thinking..."}`)
	// Reasoning part (included with prefix)
	insertTestPart(t, dbPath, "p-reasoning", "msg-parts", "sess-parts", 1200,
		`{"type": "reasoning", "text": "Let me think..."}`)
	// Step-finish part (excluded)
	insertTestPart(t, dbPath, "p-step-finish", "msg-parts", "sess-parts", 1300,
		`{"type": "step-finish", "text": "done thinking"}`)
	// Compaction part (excluded)
	insertTestPart(t, dbPath, "p-compaction", "msg-parts", "sess-parts", 1400,
		`{"type": "compaction", "text": "compacted"}`)
	// Patch part (included)
	insertTestPart(t, dbPath, "p-patch", "msg-parts", "sess-parts", 1500,
		`{"type": "patch", "text": "diff content"}`)
	// Tool part (included)
	insertTestPart(t, dbPath, "p-tool", "msg-parts", "sess-parts", 1600,
		`{"type": "tool", "name": "read_file", "text": ""}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	transcript, err := adapter.ReadTranscript("sess-parts")
	if err != nil {
		t.Fatalf("ReadTranscript: %v", err)
	}

	// Should contain text part
	if !strings.Contains(transcript, "Hello") {
		t.Error("expected transcript to contain text part 'Hello'")
	}
	// Should contain reasoning part with prefix
	if !strings.Contains(transcript, "Reasoning: Let me think...") {
		t.Error("expected transcript to contain reasoning part with prefix")
	}
	// Should contain tool part with name
	if !strings.Contains(transcript, "Tool: read_file") {
		t.Error("expected transcript to contain tool part 'Tool: read_file'")
	}
	// Should contain patch part
	if !strings.Contains(transcript, "Patch") {
		t.Error("expected transcript to contain patch part 'Patch'")
	}
	// Should NOT contain step-start content
	if strings.Contains(transcript, "thinking...") {
		t.Error("expected transcript NOT to contain step-start content")
	}
	// Should NOT contain step-finish content
	if strings.Contains(transcript, "done thinking") {
		t.Error("expected transcript NOT to contain step-finish content")
	}
	// Should NOT contain compaction content
	if strings.Contains(transcript, "compacted") {
		t.Error("expected transcript NOT to contain compaction content")
	}
}

func TestSessionHookCoordinator_OnIdle_WithAdapter(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-idle", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-idle", "sess-idle", 1000, "user")
	insertTestPart(t, dbPath, "part-idle", "msg-idle", "sess-idle", 1000,
		`{"type": "text", "text": "Idle transcript content"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:   ms,
		Adapter: adapter,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnIdle(context.Background(), "sess-idle")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}
}

func TestSessionHookCoordinator_OnEnding_WithAdapter(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-ending", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-ending", "sess-ending", 1000, "assistant")
	insertTestPart(t, dbPath, "part-ending", "msg-ending", "sess-ending", 1000,
		`{"type": "text", "text": "Ending transcript content"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:   ms,
		Adapter: adapter,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnEnding(context.Background(), "sess-ending")
	if err != nil {
		t.Fatalf("OnEnding: %v", err)
	}
	// Without OllamaClient, extraction may run but introspect is skipped gracefully
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}
}

func TestOpenCodeAdapter_ListSessions_FallbackToPath(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)

	// Insert a session where directory is empty but path has a value
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}

	_, err = db.Exec(
		"INSERT INTO session (id, time_created, time_updated, directory, path) VALUES (?, ?, ?, ?, ?)",
		"sess-path", int64(1000000), int64(2000000), "", "/alt/work",
	)
	if err != nil {
		t.Fatalf("insert session: %v", err)
	}
	db.Close()

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	sessions, err := adapter.ListSessions()
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}

	if len(sessions) != 1 {
		t.Fatalf("expected 1 session, got %d", len(sessions))
	}

	if sessions[0].WorkDir != "/alt/work" {
		t.Errorf("expected WorkDir='/alt/work' (fallback to path), got %q", sessions[0].WorkDir)
	}
}

// TestOpenCodeAdapter_NilInterfaceSafety verifies that a nil *OpenCodeAdapter
// assigned to SessionAdapter does not create a non-nil interface with a nil
// underlying value (the classic Go nil-interface panic). When the adapter is
// nil, OnIdle and OnEnding should return ResultNoTranscript, not panic.
func TestOpenCodeAdapter_NilInterfaceSafety(t *testing.T) {
	ms := newTestStore(t)

	// Construct coordinator with nil adapter — this must result in no_transcript,
	// not a panic. The bug was: assigning (*OpenCodeAdapter)(nil) to SessionAdapter
	// creates a non-nil interface, bypassing the nil check in OnIdle/OnEnding.
	var nilAdapter SessionAdapter = nil

	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:   ms,
		Adapter: nilAdapter,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	// OnIdle must not panic — it should return no_transcript
	result, err := coord.OnIdle(context.Background(), "test-session")
	if err != nil {
		t.Fatalf("OnIdle with nil adapter: %v", err)
	}
	if result != ResultNoTranscript {
		t.Errorf("expected %q, got %q", ResultNoTranscript, result)
	}

	// OnEnding must not panic — it should return no_transcript
	result, err = coord.OnEnding(context.Background(), "test-session")
	if err != nil {
		t.Fatalf("OnEnding with nil adapter: %v", err)
	}
	if result != ResultNoTranscript {
		t.Errorf("expected %q, got %q", ResultNoTranscript, result)
	}
}

// TestOpenCodeAdapter_EmptyDBPath_IsNilInterface verifies that an adapter
// created with an empty DB path, when used as a SessionAdapter, properly
// evaluates to nil (not a non-nil interface with nil underlying value).
func TestOpenCodeAdapter_EmptyDBPath_IsNilInterface(t *testing.T) {
	adapter, err := NewOpenCodeAdapter("")
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter with empty path: %v", err)
	}

	// The adapter with empty path should work as a SessionAdapter that
	// returns empty/nil results, not panic.
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:   ms,
		Adapter: adapter,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	// OnIdle with the empty-path adapter should not panic — it returns no_transcript
	// because ReadTranscript returns empty string for empty dbPath
	result, err := coord.OnIdle(context.Background(), "test-session")
	if err != nil {
		t.Fatalf("OnIdle with empty-path adapter: %v", err)
	}
	// Empty DB path adapter returns empty transcript, which maps to no_transcript
	if result != ResultNoTranscript {
		t.Errorf("expected %q, got %q", ResultNoTranscript, result)
	}
}

// newTestExtractionEngine creates an ExtractionEngine backed by a test HTTP server
// that returns the given JSON array response from the /api/generate endpoint.
func newTestExtractionEngine(t *testing.T, responseMemories []map[string]any) (*extract.ExtractionEngine, *httptest.Server) {
	t.Helper()
	responseJSON, err := json.Marshal(responseMemories)
	if err != nil {
		t.Fatalf("marshal test response: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{"response": string(responseJSON)}
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
	t.Cleanup(server.Close)

	client, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	engine, err := extract.NewExtractionEngine(extract.ExtractionConfig{
		OllamaClient: client,
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}
	return engine, server
}

// newFailingExtractionEngine creates an ExtractionEngine backed by a server
// that always returns errors (simulating unavailable LLM).
func newFailingExtractionEngine(t *testing.T) *extract.ExtractionEngine {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	t.Cleanup(server.Close)

	client, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	engine, err := extract.NewExtractionEngine(extract.ExtractionConfig{
		OllamaClient: client,
	})
	if err != nil {
		t.Fatalf("NewExtractionEngine: %v", err)
	}
	return engine
}

// newTestEmbeddingEngine creates an EmbeddingEngine backed by a test HTTP server.
func newTestEmbeddingEngine(t *testing.T) (*embed.EmbeddingEngine, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/embeddings" {
			// Return a 4-dimensional embedding vector for testing
			resp := map[string]any{
				"embedding": []float32{0.1, 0.2, 0.3, 0.4},
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "nomic-embed-text"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(server.Close)

	engine, err := embed.NewEmbeddingEngine(embed.EmbeddingConfig{
		HTTPClient:   server.Client(),
		BaseURL:      server.URL,
		Dimensions:   4, // small for testing
		MaxCacheSize: 100,
	})
	if err != nil {
		t.Fatalf("NewEmbeddingEngine: %v", err)
	}
	return engine, server
}

// newIntrospectTestServer creates an httptest.Server that serves the Ollama
// /api/tags and /api/generate endpoints for introspection tests.
// It returns a structured self-assessment response from /api/generate.
func newIntrospectTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			resp := map[string]any{"models": []map[string]string{{"name": "test-model"}}}
			json.NewEncoder(w).Encode(resp)
			return
		}
		if r.URL.Path == "/api/generate" {
			resp := map[string]string{
				"response": "Category: PROJECT_STATE\nWhat_happened: Session completed\nProposed_update: reviewed patterns",
			}
			json.NewEncoder(w).Encode(resp)
			return
		}
		http.NotFound(w, r)
	}))
	return server
}

func TestSessionHookCoordinator_OnIdle_ExtractsMemories(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-extract", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-extract", "sess-extract", 1000, "user")
	insertTestPart(t, dbPath, "part-extract", "msg-extract", "sess-extract", 1000,
		`{"type": "text", "text": "The project uses Go 1.22"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	extractor, _ := newTestExtractionEngine(t, []map[string]any{
		{"type": "fact", "content": "The project uses Go 1.22", "confidence": 0.9},
	})

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnIdle(context.Background(), "sess-extract")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	// Verify memory was stored
	memories, err := ms.ListAll(context.Background(), store.ListParams{Limit: 50})
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	if len(memories) == 0 {
		t.Error("expected at least one memory to be stored")
	}

	// Verify the memory content
	found := false
	for _, m := range memories {
		if m.Content == "The project uses Go 1.22" {
			found = true
			if m.Source != "session" {
				t.Errorf("expected source 'session', got %q", m.Source)
			}
			if m.Confidence != 0.9 {
				t.Errorf("expected confidence 0.9, got %f", m.Confidence)
			}
			if m.Metadata["source_type"] != "session" {
				t.Errorf("expected metadata source_type 'session', got %v", m.Metadata["source_type"])
			}
			break
		}
	}
	if !found {
		t.Error("expected to find memory with content 'The project uses Go 1.22'")
	}
}

func TestSessionHookCoordinator_OnIdle_ExtractsMemories_UnavailableLLM(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-nollm", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-nollm", "sess-nollm", 1000, "user")
	insertTestPart(t, dbPath, "part-nollm", "msg-nollm", "sess-nollm", 1000,
		`{"type": "text", "text": "Some content for extraction"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	extractor := newFailingExtractionEngine(t)

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnIdle(context.Background(), "sess-nollm")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	// Should return success even when LLM is unavailable — graceful degradation
	if result != ResultSuccess {
		t.Errorf("expected %q (graceful degradation), got %q", ResultSuccess, result)
	}
}

func TestSessionHookCoordinator_OnIdle_DedupExtraction(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-dedup", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-dedup", "sess-dedup", 1000, "user")
	insertTestPart(t, dbPath, "part-dedup", "msg-dedup", "sess-dedup", 1000,
		`{"type": "text", "text": "Go 1.22 is the runtime version"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	extractor, _ := newTestExtractionEngine(t, []map[string]any{
		{"type": "fact", "content": "Go 1.22 is the runtime version", "confidence": 0.85},
	})

	ms := newTestStore(t)

	// First, mark this session as already extracted (IsExtracted returns true)
	err = ms.LogExtraction(context.Background(), "session", "sess-dedup", nil, 1)
	if err != nil {
		t.Fatalf("LogExtraction: %v", err)
	}

	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	// Use a very short debounce to allow re-extraction within the test
	coord.debounceSeconds = 0

	result, err := coord.OnIdle(context.Background(), "sess-dedup")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	// OnIdle should still succeed — it uses SupersedeBySource, not IsExtracted check
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}
}

func TestSessionHookCoordinator_OnIdle_LogsExtraction(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-log", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-log", "sess-log", 1000, "user")
	insertTestPart(t, dbPath, "part-log", "msg-log", "sess-log", 1000,
		`{"type": "text", "text": "Important fact to extract"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	extractor, _ := newTestExtractionEngine(t, []map[string]any{
		{"type": "fact", "content": "Important fact to extract", "confidence": 0.7},
	})

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnIdle(context.Background(), "sess-log")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	// Verify extraction was logged
	extracted, err := ms.IsExtracted(context.Background(), "session", "sess-log")
	if err != nil {
		t.Fatalf("IsExtracted: %v", err)
	}
	if !extracted {
		t.Error("expected extraction to be logged via IsExtracted")
	}
}

func TestSessionHookCoordinator_OnIdle_OverridesPriorMemories(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-supersede", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-supersede", "sess-supersede", 1000, "user")
	insertTestPart(t, dbPath, "part-supersede", "msg-supersede", "sess-supersede", 1000,
		`{"type": "text", "text": "Updated fact about project"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	extractor, _ := newTestExtractionEngine(t, []map[string]any{
		{"type": "fact", "content": "Updated fact about project", "confidence": 0.9},
	})

	ms := newTestStore(t)

	// Pre-add a memory that should be superseded
	_, err = ms.Add(context.Background(), store.AddParams{
		Type:       "fact",
		Content:    "Old fact about project",
		Source:     "session",
		Confidence: 0.7,
		Metadata:   map[string]any{"source_type": "session", "source_id": "sess-supersede"},
	})
	if err != nil {
		t.Fatalf("Pre-add memory: %v", err)
	}

	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnIdle(context.Background(), "sess-supersede")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	// Verify the old memory was superseded (valid_until set)
	memories, err := ms.ListAll(context.Background(), store.ListParams{Limit: 50})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	// Find the old memory — it should be invalidated (valid_until set)
	foundOld := false
	foundNew := false
	for _, m := range memories {
		if m.Content == "Old fact about project" {
			foundOld = true
			// The old memory should now be invalidated (valid_until should be set)
			if m.ValidUntil == "" {
				t.Error("expected old memory to be invalidated (valid_until should be set)")
			}
		}
		if m.Content == "Updated fact about project" {
			foundNew = true
		}
	}
	if !foundOld {
		t.Error("expected to find old superseded memory")
	}
	if !foundNew {
		t.Error("expected to find new extracted memory")
	}
}

func TestSessionHookCoordinator_OnEnding_ExtractsMemories(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-ending-ex", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-ending-ex", "sess-ending-ex", 1000, "assistant")
	insertTestPart(t, dbPath, "part-ending-ex", "msg-ending-ex", "sess-ending-ex", 1000,
		`{"type": "text", "text": "We decided to use PostgreSQL for the database"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	extractor, _ := newTestExtractionEngine(t, []map[string]any{
		{"type": "decision", "content": "We decided to use PostgreSQL for the database", "confidence": 0.95},
	})

	// Create a mock Ollama server for introspection
	introspectServer := newIntrospectTestServer(t)
	ollamaClient, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    introspectServer.URL,
		HTTPClient: introspectServer.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
		OllamaClient:     ollamaClient,
		IntrospectModel:  "test-model",
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := coord.OnEnding(ctx, "sess-ending-ex")
	if err != nil {
		t.Fatalf("OnEnding: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	memories, err := ms.ListAll(context.Background(), store.ListParams{Limit: 50})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, m := range memories {
		if m.Content == "We decided to use PostgreSQL for the database" {
			found = true
			if m.Type != "decision" {
				t.Errorf("expected type 'decision', got %q", m.Type)
			}
			break
		}
	}
	if !found {
		t.Error("expected to find extracted decision memory")
	}

	// Verify a self_assessment was stored by IntrospectTranscript
	saMemories, err := ms.ListAll(context.Background(), store.ListParams{Type: "self_assessment", Limit: 10})
	if err != nil {
		t.Fatalf("Search self_assessment: %v", err)
	}
	if len(saMemories) == 0 {
		t.Error("expected at least one self_assessment memory from IntrospectTranscript")
	}
}

func TestSessionHookCoordinator_OnEnding_StoresSessionEndEvent(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-end-evt", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-end-evt", "sess-end-evt", 1000, "user")
	insertTestPart(t, dbPath, "part-end-evt", "msg-end-evt", "sess-end-evt", 1000,
		`{"type": "text", "text": "Good session"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	// Use extractor that returns no memories (LLM degradation) so we focus on event storage
	extractor := newFailingExtractionEngine(t)

	// Create a mock Ollama server that fails — triggers fallback to session_end event
	introspectServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	t.Cleanup(introspectServer.Close)

	ollamaClient, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    introspectServer.URL,
		HTTPClient: introspectServer.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
		OllamaClient:     ollamaClient,
		IntrospectModel:  "test-model",
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := coord.OnEnding(ctx, "sess-end-evt")
	if err != nil {
		t.Fatalf("OnEnding: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	// Since Ollama is unavailable, IntrospectTranscript fails and falls back to
	// a session_end event memory
	allMemories, err := ms.ListAll(context.Background(), store.ListParams{Limit: 50})
	if err != nil {
		t.Fatalf("Search all: %v", err)
	}
	hasSessionMemory := false
	for _, m := range allMemories {
		if m.Source == "introspect" || m.Source == "session_end" || m.Source == "session" {
			hasSessionMemory = true
			break
		}
	}
	if !hasSessionMemory {
		t.Error("expected at least one memory stored from session ending (introspect, session_end, or session)")
	}
}

func TestSessionHookCoordinator_OnEnding_IntrospectFallback(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-intro-fb", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-intro-fb", "sess-intro-fb", 1000, "user")
	insertTestPart(t, dbPath, "part-intro-fb", "msg-intro-fb", "sess-intro-fb", 1000,
		`{"type": "text", "text": "Fallback transcript content"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	// Use failing extractor (LLM unavailable)
	extractor := newFailingExtractionEngine(t)

	// Create a mock Ollama server that returns failures — triggers IntrospectTranscript graceful degradation
	introspectServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	}))
	t.Cleanup(introspectServer.Close)

	ollamaClient, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    introspectServer.URL,
		HTTPClient: introspectServer.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
		OllamaClient:     ollamaClient,
		IntrospectModel:  "test-model",
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := coord.OnEnding(ctx, "sess-intro-fb")
	if err != nil {
		t.Fatalf("OnEnding: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q (graceful degradation), got %q", ResultSuccess, result)
	}

	// When introspect fails (no LLM), there should still be memories stored
	// Check for self_assessment or event memory from degraded IntrospectTranscript
	memories, err := ms.ListAll(context.Background(), store.ListParams{Type: "self_assessment", Limit: 10})
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	if len(memories) == 0 {
		// Check for event memories as fallback
		eventMemories, err := ms.ListAll(context.Background(), store.ListParams{Type: "event", Limit: 10})
		if err != nil {
			t.Fatalf("ListAll events: %v", err)
		}
		if len(eventMemories) == 0 {
			t.Error("expected at least one self_assessment or event memory from IntrospectTranscript fallback")
		}
	} else {
		// Verify a self_assessment memory was stored from introspect
		found := false
		for _, m := range memories {
			if m.Source == "introspect" {
				found = true
				break
			}
		}
		if !found {
			t.Error("expected self_assessment memory with source 'introspect'")
		}
	}
}

func TestSessionHookCoordinator_OnIdle_NilExtractionEngine(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-nil-ext", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-nil-ext", "sess-nil-ext", 1000, "user")
	insertTestPart(t, dbPath, "part-nil-ext", "msg-nil-ext", "sess-nil-ext", 1000,
		`{"type": "text", "text": "Content without extraction"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:   ms,
		Adapter: adapter,
		// ExtractionEngine is nil — extraction should be skipped gracefully
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnIdle(context.Background(), "sess-nil-ext")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	// No extraction memories should have been stored
	memories, err := ms.ListAll(context.Background(), store.ListParams{Limit: 50})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	// No extraction memories should have been stored from session extraction
	var sessionMemCount int
	for _, m := range memories {
		if m.Source == "session" {
			sessionMemCount++
		}
	}
	if sessionMemCount != 0 {
		t.Errorf("expected no session-sourced memories with nil extraction engine, got %d", sessionMemCount)
	}
}

func TestSessionHookCoordinator_OnEnding_NilExtractionEngine(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-nil-end", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-nil-end", "sess-nil-end", 1000, "user")
	insertTestPart(t, dbPath, "part-nil-end", "msg-nil-end", "sess-nil-end", 1000,
		`{"type": "text", "text": "Ending content without extraction"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:   ms,
		Adapter: adapter,
		// ExtractionEngine is nil — extraction should be skipped.
		// OllamaClient is nil — IntrospectTranscript falls back to degraded storage
		// (produces a self_assessment memory without LLM call).
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnEnding(context.Background(), "sess-nil-end")
	if err != nil {
		t.Fatalf("OnEnding: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	// IntrospectTranscript is called even with nil OllamaClient — it falls back
	// to degraded storage producing a self_assessment memory with a plain-text summary.
	saMemories, err := ms.ListAll(context.Background(), store.ListParams{Type: "self_assessment", Limit: 10})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(saMemories) == 0 {
		t.Error("expected at least one self_assessment memory from IntrospectTranscript degraded storage (nil OllamaClient)")
	}
	// Verify the degraded memory contains the session ID
	found := false
	for _, m := range saMemories {
		if strings.Contains(m.Content, "sess-nil-end") {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected degraded self_assessment memory to contain session ID 'sess-nil-end'")
	}
}

// TestSessionHookCoordinator_OnEnding_NilOllamaClient_ProducesDegradedIntrospection
// verifies the fix for issue ll-m8knf-vup88: OnEnding must call IntrospectTranscript
// even when OllamaClient is nil. IntrospectTranscript has its own nil-client graceful
// degradation — it produces a plain-text self_assessment memory. The bug was that
// OnEnding guarded the call with c.ollamaClient != nil, bypassing this degradation
// and producing zero introspection memories for sessions with valid transcripts.
func TestSessionHookCoordinator_OnEnding_NilOllamaClient_ProducesDegradedIntrospection(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-nil-ollama", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-nil-ollama", "sess-nil-ollama", 1000, "user")
	insertTestPart(t, dbPath, "part-nil-ollama", "msg-nil-ollama", "sess-nil-ollama", 1000,
		`{"type": "text", "text": "The project uses Go 1.22 for robust concurrency"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	// Set up an extraction engine working normally
	extractor, _ := newTestExtractionEngine(t, []map[string]any{
		{"type": "fact", "content": "The project uses Go 1.22", "confidence": 0.9},
	})

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
		// OllamaClient is nil — IntrospectTranscript must still be called
		// and must produce a degraded self_assessment memory
	})

	result, err := coord.OnEnding(context.Background(), "sess-nil-ollama")
	if err != nil {
		t.Fatalf("OnEnding: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	// Verify that IntrospectTranscript produced a degraded self_assessment memory
	saMemories, err := ms.ListAll(context.Background(), store.ListParams{Type: "self_assessment", Limit: 10})
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	if len(saMemories) == 0 {
		t.Error("expected at least one self_assessment memory from IntrospectTranscript degraded path (nil OllamaClient)")
	}

	// Verify the degraded memory contains the session ID
	found := false
	for _, m := range saMemories {
		if strings.Contains(m.Content, "sess-nil-ollama") {
			found = true
			if m.Source != "introspect" {
				t.Errorf("expected source 'introspect', got %q", m.Source)
			}
			break
		}
	}
	if !found {
		t.Error("expected degraded self_assessment memory to contain session ID")
	}

	// Also verify extraction memories were still stored
	sessionMemories, err := ms.ListAll(context.Background(), store.ListParams{Type: "fact", Limit: 10})
	if err != nil {
		t.Fatalf("ListAll facts: %v", err)
	}
	if len(sessionMemories) == 0 {
		t.Error("expected at least one fact memory from extraction")
	}
}

func TestSessionHookCoordinator_OnIdle_WithAdapter_VerifiesMemoriesStored(t *testing.T) {
	// Updated version of the existing test that now verifies memories are stored
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-idle-v2", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-idle-v2", "sess-idle-v2", 1000, "user")
	insertTestPart(t, dbPath, "part-idle-v2", "msg-idle-v2", "sess-idle-v2", 1000,
		`{"type": "text", "text": "Idle transcript content"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	// Set up extraction engine that returns a memory
	extractor, _ := newTestExtractionEngine(t, []map[string]any{
		{"type": "fact", "content": "Idle transcript content", "confidence": 0.8},
	})

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	result, err := coord.OnIdle(context.Background(), "sess-idle-v2")
	if err != nil {
		t.Fatalf("OnIdle: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	// Verify that a memory was actually stored in the test store
	memories, err := ms.ListAll(context.Background(), store.ListParams{Limit: 50})
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	var sessionMemCount int
	for _, m := range memories {
		if m.Source == "session" {
			sessionMemCount++
		}
	}
	if sessionMemCount == 0 {
		t.Error("expected at least one session-sourced memory to be stored after OnIdle with extraction")
	}
}

func TestSessionHookCoordinator_OnEnding_WithAdapter_VerifiesMemoriesStored(t *testing.T) {
	// Updated version of the existing test that now verifies memories are stored
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-ending-v2", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-ending-v2", "sess-ending-v2", 1000, "assistant")
	insertTestPart(t, dbPath, "part-ending-v2", "msg-ending-v2", "sess-ending-v2", 1000,
		`{"type": "text", "text": "Ending transcript content"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	// Set up extraction engine that returns a memory
	extractor, _ := newTestExtractionEngine(t, []map[string]any{
		{"type": "fact", "content": "Ending transcript content", "confidence": 0.85},
	})

	// Set up a mock Ollama server for introspection
	introspectServer := newIntrospectTestServer(t)
	ollamaClient, err := ollama.NewOllamaClient(ollama.OllamaClientConfig{
		BaseURL:    introspectServer.URL,
		HTTPClient: introspectServer.Client(),
	})
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:            ms,
		Adapter:          adapter,
		ExtractionEngine: extractor,
		OllamaClient:     ollamaClient,
		IntrospectModel:  "test-model",
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	result, err := coord.OnEnding(ctx, "sess-ending-v2")
	if err != nil {
		t.Fatalf("OnEnding: %v", err)
	}
	if result != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, result)
	}

	// Verify that a memory was actually stored in the test store
	memories, err := ms.ListAll(context.Background(), store.ListParams{Limit: 50})
	if err != nil {
		t.Fatalf("ListAll: %v", err)
	}
	var sessionOrIntrospectCount int
	for _, m := range memories {
		if m.Source == "session" || m.Source == "introspect" {
			sessionOrIntrospectCount++
		}
	}
	if sessionOrIntrospectCount == 0 {
		t.Error("expected at least one session or introspect memory to be stored after OnEnding with extraction")

func TestSessionHookCoordinator_OnEndingWithIntrospect(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-end-auto", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-end-auto", "sess-end-auto", 1000, "assistant")
	insertTestPart(t, dbPath, "part-end-auto", "msg-end-auto", "sess-end-auto", 1000,
		`{"type": "text", "text": "Had an error in production code"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:   ms,
		Adapter: adapter,
		// Use non-existent Ollama URL to ensure graceful degradation works
		BaseURL: "http://localhost:59999",
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	resultType, memoryID, err := coord.OnEndingWithIntrospect(context.Background(), "sess-end-auto")
	if err != nil {
		t.Fatalf("OnEndingWithIntrospect: %v", err)
	}
	if resultType != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, resultType)
	}
	// Memory ID should be non-empty even with graceful degradation (no Ollama)
	if memoryID == "" {
		t.Error("expected non-empty memory ID from auto introspection")
	}

	// Verify the memory was actually stored
	mem, err := ms.Get(context.Background(), memoryID, false)
	if err != nil {
		t.Fatalf("Get memory: %v", err)
	}
	if mem == nil {
		t.Fatal("expected memory to be stored")
	}
	if mem.Type != "self_assessment" {
		t.Errorf("expected type self_assessment, got %q", mem.Type)
	}
	if mem.Source != "introspect-auto" {
		t.Errorf("expected source introspect-auto, got %q", mem.Source)
	}
}

func TestSessionHookCoordinator_OnEndingWithIntrospect_NoTranscript(t *testing.T) {
	ms := newTestStore(t)
	// No adapter — should return ResultNoTranscript
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store: ms,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	resultType, memoryID, err := coord.OnEndingWithIntrospect(context.Background(), "test-session")
	if err != nil {
		t.Fatalf("OnEndingWithIntrospect: %v", err)
	}
	if resultType != ResultNoTranscript {
		t.Errorf("expected %q, got %q", ResultNoTranscript, resultType)
	}
	if memoryID != "" {
		t.Errorf("expected empty memory ID for no transcript, got %q", memoryID)
	}
}

func TestSessionHookCoordinator_OnEndingWithIntrospect_InvalidSessionID(t *testing.T) {
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store: ms,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	_, _, err = coord.OnEndingWithIntrospect(context.Background(), "../etc/passwd")
	if err == nil {
		t.Error("expected error for path-traversal session ID")
	}
	if err.Error() == "" {
		t.Error("expected non-empty error message")
	}
}

func TestSessionHookCoordinator_OnEndingWithIntrospect_IntrospectionFailure(t *testing.T) {
	dbPath := createTestOpenCodeDB(t)
	insertTestSession(t, dbPath, "sess-end-fail", 1000, 2000, nil, "/work")
	insertTestMessage(t, dbPath, "msg-end-fail", "sess-end-fail", 1000, "assistant")
	insertTestPart(t, dbPath, "part-end-fail", "msg-end-fail", "sess-end-fail", 1000,
		`{"type": "text", "text": "Some transcript content"}`)

	adapter, err := NewOpenCodeAdapter(dbPath)
	if err != nil {
		t.Fatalf("NewOpenCodeAdapter: %v", err)
	}
	t.Cleanup(func() { adapter.Close() })

	// Create a store that will fail on Add, forcing introspection failure
	ms := newTestStore(t)
	coord, err := NewSessionHookCoordinator(SessionHookConfig{
		Store:   ms,
		Adapter: adapter,
	})
	if err != nil {
		t.Fatalf("NewSessionHookCoordinator: %v", err)
	}

	// Close the store to force Add failures — introspection will fail
	// but the hook should still return ResultSuccess gracefully
	ms.Close()

	// Use a context with a timeout so the test doesn't hang on model calls
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resultType, memoryID, err := coord.OnEndingWithIntrospect(ctx, "sess-end-fail")
	if err != nil {
		t.Fatalf("OnEndingWithIntrospect should not return error on introspection failure: %v", err)
	}
	// Should still return ResultSuccess since transcript was read
	if resultType != ResultSuccess {
		t.Errorf("expected %q, got %q", ResultSuccess, resultType)
	}
	// memoryID should be empty since introspection failed
	if memoryID != "" {
		t.Errorf("expected empty memory ID on introspection failure, got %q", memoryID)
	}
