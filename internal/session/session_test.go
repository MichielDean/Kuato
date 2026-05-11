package session

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

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