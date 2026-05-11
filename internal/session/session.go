// Package session provides session lifecycle hook coordination for LLMem.
// It handles created, idle, compacting, and ending events with debounced idle processing.
package session

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"

	"github.com/MichielDean/LLMem/internal/embed"
	"github.com/MichielDean/LLMem/internal/extract"
	"github.com/MichielDean/LLMem/internal/introspect"
	"github.com/MichielDean/LLMem/internal/ollama"
	"github.com/MichielDean/LLMem/internal/paths"
	"github.com/MichielDean/LLMem/internal/store"
)

// Result constants for session hook operations.
const (
	ResultSuccess          = "success"
	ResultAlreadyProcessed = "already_processed"
	ResultError            = "error"
	ResultDebounced        = "debounced"
	ResultNoTranscript     = "no_transcript"
)

// idleDebounceSeconds is the minimum interval between idle events for the same session.
const idleDebounceSeconds = 30

// idleEvictionFactor controls when stale entries are pruned from lastIdle.
// Entries older than idleDebounceSeconds * idleEvictionFactor are evicted.
const idleEvictionFactor = 10

// defaultIntrospectModel is the LLM model used for automatic session introspection.
const defaultIntrospectModel = "glm-5.1:cloud"

// defaultIntrospectBaseURL is the Ollama base URL used for automatic session introspection.
const defaultIntrospectBaseURL = "http://localhost:11434"

// Compacting key memory types and minimum confidence.
var compactingKeyTypes = []string{"decision", "preference", "procedure", "project_state"}

// SessionAdapter is the interface for reading session content.
// Implementations provide session transcript reading, listing, and cleanup capabilities.
type SessionAdapter interface {
	// ReadTranscript reads the full transcript for a session.
	ReadTranscript(sessionID string) (string, error)
	// ListSessions returns available session information.
	ListSessions() ([]SessionInfo, error)
	// Close releases any resources held by the adapter. Safe to call multiple times.
	Close() error
}

// SessionInfo holds metadata about a session.
type SessionInfo struct {
	ID        string
	StartTime string
	EndTime   string
	WorkDir   string
}

// OpenCodeAdapter reads session data from the OpenCode SQLite database.
// This is specific to OpenCode and will not be reused.
type OpenCodeAdapter struct {
	dbPath string
	db     *sql.DB
	closed bool
	mu     sync.Mutex
}

// NewOpenCodeAdapter creates an adapter that reads from the given DB path.
// The constructor opens the database connection eagerly and leaves the adapter
// in a fully usable state. If dbPath is empty, the adapter returns empty/nil
// results from all methods (safe no-op mode).
// The database is opened in read-only mode (mode=ro) to prevent any modifications
// to the external OpenCode database.
func NewOpenCodeAdapter(dbPath string) (*OpenCodeAdapter, error) {
	if dbPath == "" {
		return &OpenCodeAdapter{dbPath: "", db: nil, closed: false}, nil
	}

	// Open the database in read-only mode — we must not modify the external
	// OpenCode database. No WAL pragma or migrations are applied.
	// The "file:" URI prefix is required so modernc.org/sqlite processes the
	// mode=ro parameter via SQLite's URI mechanism. Without "file:", the
	// driver strips query parameters and opens with READWRITE|CREATE flags,
	// silently ignoring mode=ro.
	dsn := "file:" + dbPath + "?mode=ro"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmtErr("open opencode db %s: %w", dbPath, err)
	}

	// Verify the connection works
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmtErr("ping opencode db %s: %w", dbPath, err)
	}

	return &OpenCodeAdapter{dbPath: dbPath, db: db, closed: false}, nil
}

// Close closes the underlying database connection. Safe to call multiple times
// (idempotent, returns nil on subsequent calls).
func (a *OpenCodeAdapter) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	a.closed = true
	if a.db != nil {
		return a.db.Close()
	}
	return nil
}

// ReadTranscript reads the session transcript from the OpenCode database.
// If dbPath is empty, returns ("", nil). If sessionID fails validation,
// returns ("", error). If the session ID does not exist, returns ("", nil).
// When session.time_compacting is set (non-NULL), returns only messages
// created after the compaction time (recent context), not the full transcript.
// The transcript format is a text-based conversation log where each message
// shows the role ("User:" or "Assistant:") followed by content. Text parts
// are included, reasoning parts are prefixed with "Reasoning: ", tool call
// parts show tool name + input summary, patch parts show diff summary.
// Step-start, step-finish, and compaction parts are excluded.
func (a *OpenCodeAdapter) ReadTranscript(sessionID string) (string, error) {
	if a.dbPath == "" {
		return "", nil
	}
	// Validate session ID to prevent path traversal
	if _, err := paths.ValidateSessionID(sessionID); err != nil {
		return "", fmtErr("read transcript: %w", err)
	}

	// Check if session exists and get time_compacting
	var timeCompacting sql.NullInt64
	err := a.db.QueryRow(
		"SELECT `time_compacting` FROM `session` WHERE `id` = ?",
		sessionID,
	).Scan(&timeCompacting)
	if err != nil {
		if err == sql.ErrNoRows {
			// Session not found — return empty, no error
			return "", nil
		}
		return "", fmtErr("read transcript: query session %s: %w", sessionID, err)
	}

	// Build query based on whether compacting is set
	query := "SELECT `id`, `data` FROM `message` WHERE `session_id` = ? ORDER BY `time_created` ASC"
	args := []any{sessionID}

	if timeCompacting.Valid {
		// Only return messages after the compaction time (recent context)
		query = "SELECT `id`, `data` FROM `message` WHERE `session_id` = ? AND `time_created` > ? ORDER BY `time_created` ASC"
		args = []any{sessionID, timeCompacting.Int64}
	}

	rows, err := a.db.Query(query, args...)
	if err != nil {
		return "", fmtErr("read transcript: query messages for session %s: %w", sessionID, err)
	}
	defer rows.Close()

	var transcript strings.Builder
	for rows.Next() {
		var msgID, data string
		if err := rows.Scan(&msgID, &data); err != nil {
			return "", fmtErr("read transcript: scan message for session %s: %w", sessionID, err)
		}

		// Parse the message data JSON to extract role
		var msgData struct {
			Role string `json:"role"`
		}
		if err := json.Unmarshal([]byte(data), &msgData); err != nil {
			slog.Debug("llmem: session: skipping unparseable message data", "session_id", sessionID, "message_id", msgID, "error", err)
			continue
		}

		// Get parts for this message
		parts, err := a.readParts(msgID)
		if err != nil {
			slog.Debug("llmem: session: error reading parts", "session_id", sessionID, "message_id", msgID, "error", err)
			continue
		}

		if len(parts) == 0 {
			continue
		}

		// Determine role label
		roleLabel := "Assistant:"
		if msgData.Role == "user" {
			roleLabel = "User:"
		}

		transcript.WriteString(roleLabel)
		transcript.WriteString("\n")
		for _, part := range parts {
			transcript.WriteString(part)
			transcript.WriteString("\n")
		}
		transcript.WriteString("\n")
	}

	if err := rows.Err(); err != nil {
		return "", fmtErr("read transcript: iterate messages for session %s: %w", sessionID, err)
	}

	return transcript.String(), nil
}

// partContent holds the extracted text content and type from a message part.
type partContent struct {
	text     string
	partType string
	toolName string
}

// readParts reads all parts for a message, filtering out excluded types
// and returning formatted text for each included part.
func (a *OpenCodeAdapter) readParts(messageID string) ([]string, error) {
	rows, err := a.db.Query(
		"SELECT `data` FROM `part` WHERE `message_id` = ? ORDER BY `time_created` ASC",
		messageID,
	)
	if err != nil {
		return nil, fmtErr("read parts: query parts for message %s: %w", messageID, err)
	}
	defer rows.Close()

	var result []string
	for rows.Next() {
		var data string
		if err := rows.Scan(&data); err != nil {
			return nil, fmtErr("read parts: scan part for message %s: %w", messageID, err)
		}

		parsed, err := parsePartData(data)
		if err != nil {
			slog.Debug("llmem: session: skipping unparseable part data", "message_id", messageID, "error", err)
			continue
		}

		// Filter excluded part types
		switch parsed.partType {
		case "step-start", "step-finish", "compaction":
			continue
		}

		var formatted string
		switch parsed.partType {
		case "text":
			formatted = parsed.text
		case "reasoning":
			formatted = "Reasoning: " + parsed.text
		case "tool":
			if parsed.toolName != "" {
				formatted = "Tool: " + parsed.toolName
			}
		case "patch":
			formatted = "Patch"
		default:
			// Include unknown types as text if they have content
			if parsed.text != "" {
				formatted = parsed.text
			}
		}

		if formatted != "" {
			result = append(result, formatted)
		}
	}

	if err := rows.Err(); err != nil {
		return nil, fmtErr("read parts: iterate parts for message %s: %w", messageID, err)
	}

	return result, nil
}

// parsePartData extracts the part type, text, and tool name from a part's JSON data.
func parsePartData(data string) (partContent, error) {
	var raw map[string]any
	if err := json.Unmarshal([]byte(data), &raw); err != nil {
		return partContent{}, fmtErr("parse part data: %w", err)
	}

	pc := partContent{}

	// Extract type
	if typ, ok := raw["type"]; ok {
		pc.partType, _ = typ.(string)
	}

	// Extract text
	if text, ok := raw["text"]; ok {
		pc.text, _ = text.(string)
	}

	// Extract tool name from tool invocations
	if pc.partType == "tool" {
		// Tool name might be at "name" or "tool" key
		if name, ok := raw["name"]; ok {
			pc.toolName, _ = name.(string)
		}
	}

	return pc, nil
}

// ListSessions returns available session information from the OpenCode database.
// Times are converted from Unix milliseconds to RFC3339 strings.
// If dbPath is empty, returns ([], nil).
func (a *OpenCodeAdapter) ListSessions() ([]SessionInfo, error) {
	if a.dbPath == "" {
		return []SessionInfo{}, nil
	}

	rows, err := a.db.Query(
		"SELECT `id`, `time_created`, `time_updated`, `directory`, `path` FROM `session` ORDER BY `time_created` DESC",
	)
	if err != nil {
		return nil, fmtErr("list sessions: query sessions: %w", err)
	}
	defer rows.Close()

	var sessions []SessionInfo
	for rows.Next() {
		var id string
		var timeCreated, timeUpdated sql.NullInt64
		var directory, sessionPath sql.NullString
		if err := rows.Scan(&id, &timeCreated, &timeUpdated, &directory, &sessionPath); err != nil {
			return nil, fmtErr("list sessions: scan session: %w", err)
		}

		var startTime, endTime string
		if timeCreated.Valid {
			startTime = time.UnixMilli(timeCreated.Int64).UTC().Format(time.RFC3339)
		}
		if timeUpdated.Valid {
			endTime = time.UnixMilli(timeUpdated.Int64).UTC().Format(time.RFC3339)
		}

		// Prefer directory, fall back to path
		workDir := directory.String
		if workDir == "" {
			workDir = sessionPath.String
		}

		sessions = append(sessions, SessionInfo{
			ID:        id,
			StartTime: startTime,
			EndTime:   endTime,
			WorkDir:   workDir,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmtErr("list sessions: iterate sessions: %w", err)
	}

	if sessions == nil {
		sessions = []SessionInfo{}
	}

	return sessions, nil
}

// SessionHookConfig contains configuration for creating a SessionHookCoordinator.
type SessionHookConfig struct {
	// Store is required for all hook operations.
	Store *store.MemoryStore

	// Adapter provides session content. If nil, idle and ending return no_transcript.
	Adapter SessionAdapter

	// DebounceSeconds is the minimum interval between idle events. Defaults to 30.
	DebounceSeconds int

	// ContextDir is the directory for writing context files. Defaults from paths.GetContextDir().
	ContextDir string

	// Model is the LLM model name for introspection. Defaults to "glm-5.1:cloud" if zero.
	Model string

	// BaseURL is the Ollama base URL for introspection. Defaults to "http://localhost:11434" if zero.
	BaseURL string

	// ExtractionEngine extracts memories from text. If nil, idle and ending skip extraction.
	ExtractionEngine *extract.ExtractionEngine

	// Embedding generates embedding vectors. If nil, memories are stored without embeddings.
	Embedding *embed.EmbeddingEngine

	// OllamaClient is used for introspection in OnEnding. If nil, IntrospectTranscript
	// falls back to degraded storage immediately (no LLM call attempted) and still
	// produces a self_assessment memory with a plain-text session summary.
	OllamaClient *ollama.OllamaClient

	// IntrospectModel is the LLM model name for IntrospectTranscript.
	// Defaults to "glm-5.1:cloud" if empty.
	IntrospectModel string
}

// SessionHookCoordinator orchestrates memory operations for session lifecycle events.
type SessionHookCoordinator struct {
	store            *store.MemoryStore
	adapter          SessionAdapter
	contextDir       string
	debounceSeconds  int
	extractor        *extract.ExtractionEngine
	embedder         *embed.EmbeddingEngine
	ollamaClient     *ollama.OllamaClient
	introspectModel  string
	lastIdle         map[string]time.Time
	mu               sync.Mutex
	model            string
	baseURL          string
}

// fmtErr wraps an error with the "llmem: session:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: session: "+format, args...)
}

// NewSessionHookCoordinator creates and initializes a coordinator.
// The constructor leaves it in a fully usable state.
func NewSessionHookCoordinator(cfg SessionHookConfig) (*SessionHookCoordinator, error) {
	if cfg.Store == nil {
		return nil, fmtErr("store is required")
	}

	debounceSeconds := cfg.DebounceSeconds
	if debounceSeconds <= 0 {
		debounceSeconds = idleDebounceSeconds
	}

	contextDir := cfg.ContextDir
	if contextDir == "" {
		contextDir = paths.GetContextDir()
	}

	return &SessionHookCoordinator{
		store:            cfg.Store,
		adapter:          cfg.Adapter,
		contextDir:       contextDir,
		debounceSeconds:  debounceSeconds,
		extractor:        cfg.ExtractionEngine,
		embedder:         cfg.Embedding,
		ollamaClient:     cfg.OllamaClient,
		introspectModel:  cfg.IntrospectModel,
		lastIdle:         map[string]time.Time{},
		model:            cfg.Model,
		baseURL:          cfg.BaseURL,
	}, nil
}

// OnCreated handles the session.created event.
// Returns result type string: "success", "already_processed", or "error".
func (c *SessionHookCoordinator) OnCreated(ctx context.Context, sessionID string) (string, error) {
	validID, err := paths.ValidateSessionID(sessionID)
	if err != nil {
		return ResultError, fmtErr("on_created: validate session ID: %w", err)
	}

	// Check if already processed
	alreadyProcessed, err := c.store.IsExtracted(ctx, "session", validID)
	if err != nil {
		slog.Debug("llmem: session: on_created: check extracted failed", "error", err)
	} else if alreadyProcessed {
		return ResultAlreadyProcessed, nil
	}

	logSessionEvent("created", validID)
	return ResultSuccess, nil
}

// OnIdle handles the session.idle event with debounce (configurable interval, defaults to 30s).
// Returns result type string: "debounced", "no_transcript", "success", or "error".
func (c *SessionHookCoordinator) OnIdle(ctx context.Context, sessionID string) (string, error) {
	validID, err := paths.ValidateSessionID(sessionID)
	if err != nil {
		return ResultError, fmtErr("on_idle: validate session ID: %w", err)
	}

	// Check debounce
	c.mu.Lock()
	// Evict stale entries to prevent unbounded map growth.
	// Entries older than evictionThreshold are no longer useful for debounce.
	evictionThreshold := time.Now().Add(-time.Duration(c.debounceSeconds*idleEvictionFactor) * time.Second)
	for id, t := range c.lastIdle {
		if t.Before(evictionThreshold) {
			delete(c.lastIdle, id)
		}
	}

	lastTime, exists := c.lastIdle[validID]
	if exists && time.Since(lastTime).Seconds() < float64(c.debounceSeconds) {
		c.mu.Unlock()
		return ResultDebounced, nil
	}
	c.lastIdle[validID] = time.Now()
	c.mu.Unlock()

	// If no adapter, no transcript
	if c.adapter == nil {
		return ResultNoTranscript, nil
	}

	transcript, err := c.adapter.ReadTranscript(validID)
	if err != nil {
		return ResultError, fmtErr("on_idle: read transcript: %w", err)
	}
	if transcript == "" {
		return ResultNoTranscript, nil
	}

	// Extract memories from transcript if extraction engine is available
	if c.extractor != nil {
		extracted := c.extractMemories(ctx, transcript, validID)
		_ = extracted // extracted count logged by extractMemories
	}

	logSessionEvent("idle", validID)
	return ResultSuccess, nil
}

// OnCompacting handles the session.compacting event.
// Returns result type string, context file path (for injection), and error.
func (c *SessionHookCoordinator) OnCompacting(ctx context.Context, sessionID string) (string, string, error) {
	validID, err := paths.ValidateSessionID(sessionID)
	if err != nil {
		return ResultError, "", fmtErr("on_compacting: validate session ID: %w", err)
	}

	// Create context directory
	if err := os.MkdirAll(c.contextDir, 0700); err != nil {
		return ResultError, "", fmtErr("on_compacting: create context dir: %w", err)
	}

	// Write context file with key memories using validated session ID
	contextPath := filepath.Join(c.contextDir, validID+"_context.md")
	var contentBuilder strings.Builder
	contentBuilder.WriteString("# Context\n\n")

	// Search for key memories
	for _, memType := range compactingKeyTypes {
		memories, err := c.store.Search(ctx, store.SearchParams{
			Type:      memType,
			ValidOnly: true,
			Limit:    10,
		})
		if err != nil {
			slog.Debug("llmem: session: context search failed", "type", memType, "error", err)
			continue
		}
		for _, m := range memories {
			if m.Confidence >= 0.7 {
				contentBuilder.WriteString(fmt.Sprintf("## %s (%s)\n\n%s\n\n", m.Type, m.ID, m.Content))
			}
		}
	}

	if err := os.WriteFile(contextPath, []byte(contentBuilder.String()), 0600); err != nil {
		return ResultError, "", fmtErr("on_compacting: write context: %w", err)
	}

	logSessionEvent("compacting", validID)
	return ResultSuccess, contextPath, nil
}

// OnEnding handles the session.ending event.
// Returns result type string: "success", "no_transcript", or "error".
func (c *SessionHookCoordinator) OnEnding(ctx context.Context, sessionID string) (string, error) {
	validID, err := paths.ValidateSessionID(sessionID)
	if err != nil {
		return ResultError, fmtErr("on_ending: validate session ID: %w", err)
	}

	// If no adapter, no transcript
	if c.adapter == nil {
		return ResultNoTranscript, nil
	}

	transcript, err := c.adapter.ReadTranscript(validID)
	if err != nil {
		return ResultError, fmtErr("on_ending: read transcript: %w", err)
	}
	if transcript == "" {
		return ResultNoTranscript, nil
	}

	// Extract memories from transcript if extraction engine is available
	if c.extractor != nil {
		extracted := c.extractMemories(ctx, transcript, validID)
		_ = extracted // extracted count logged by extractMemories
	}

	// Run introspection on the full session transcript.
	// IntrospectTranscript handles nil OllamaClient gracefully by producing
	// degraded self_assessment content (no LLM call attempted). Do NOT guard
	// with c.ollamaClient != nil — that bypasses the function's own nil-client
	// degradation and results in zero introspection memories for sessions with
	// valid transcripts but no configured Ollama connection.
	introspectID, err := introspect.IntrospectTranscript(ctx, c.store, transcript, validID, c.ollamaClient, c.introspectModel)
	if err != nil {
		slog.Warn("llmem: session: on_ending: introspect_transcript failed, storing degraded event", "error", err, "session_id", validID)
		// Fall back to storing a simple session-end event.
		// Use context.Background() for the fallback store operation since the
		// calling context may have expired during the LLM call.
		_, storeErr := c.store.Add(context.Background(), store.AddParams{
			Type:       "event",
			Content:    fmt.Sprintf("Session ended: %s", validID),
			Source:     "session_end",
			Confidence: 0.7,
			Metadata:   map[string]any{"source_type": "session", "source_id": validID},
		})
		if storeErr != nil {
			slog.Debug("llmem: session: on_ending: store session_end event failed", "error", storeErr, "session_id", validID)
		}
	} else {
		slog.Debug("llmem: session: on_ending: stored introspect_transcript memory", "id", introspectID, "session_id", validID)
	}

	logSessionEvent("ending", validID)
	return ResultSuccess, nil
}

// OnEndingWithIntrospect handles the session.ending event with automatic introspection.
// It reads the session transcript and generates a self_assessment memory via
// introspect.IntrospectAuto. Returns (resultType, memoryID, error).
// When adapter is nil or transcript is empty, returns (ResultNoTranscript, "", nil).
// On success, returns (ResultSuccess, memoryID, nil).
// If introspection fails but transcript was read, logs a warning and returns
// (ResultSuccess, "", nil) — the session hook should not fail the ending event.
func (c *SessionHookCoordinator) OnEndingWithIntrospect(ctx context.Context, sessionID string) (string, string, error) {
	validID, err := paths.ValidateSessionID(sessionID)
	if err != nil {
		return ResultError, "", fmtErr("on_ending_with_introspect: validate session ID: %w", err)
	}

	// If no adapter, no transcript
	if c.adapter == nil {
		return ResultNoTranscript, "", nil
	}

	transcript, err := c.adapter.ReadTranscript(validID)
	if err != nil {
		return ResultError, "", fmtErr("on_ending_with_introspect: read transcript: %w", err)
	}
	if transcript == "" {
		return ResultNoTranscript, "", nil
	}

	model := c.model
	if model == "" {
		model = defaultIntrospectModel
	}
	baseURL := c.baseURL
	if baseURL == "" {
		baseURL = defaultIntrospectBaseURL
	}

	memoryID, err := introspect.IntrospectAuto(ctx, c.store, transcript, model, baseURL)
	if err != nil {
		// Introspection failed, but the session hook should not crash the ending event.
		// Log a warning and return success with empty memoryID.
		slog.Warn("llmem: session: on_ending_with_introspect: introspection failed", "error", err)
		return ResultSuccess, "", nil
	}

	logSessionEvent("ending_with_introspect", validID)
	return ResultSuccess, memoryID, nil
}

// extractMemories extracts memories from a transcript, handles dedup via SupersedeBySource,
// stores each extracted memory, embeds if possible, and logs the extraction.
// Returns the number of memories successfully stored.
// Errors are logged and gracefully degraded — this method never returns an error.
func (c *SessionHookCoordinator) extractMemories(ctx context.Context, transcript, validID string) int {
	// Supersede prior memories from this session before extracting fresh ones.
	// OnIdle intentionally re-extracts as the conversation grows, so we
	// invalidate old memories rather than checking IsExtracted.
	superceded, err := c.store.SupersedeBySource(ctx, "session", validID)
	if err != nil {
		slog.Debug("llmem: session: supersede_by_source failed", "error", err, "session_id", validID)
	} else if superceded > 0 {
		slog.Debug("llmem: session: superseded prior session memories", "count", superceded, "session_id", validID)
	}

	extractedMaps := c.extractor.Extract(ctx, transcript)
	if len(extractedMaps) == 0 {
		slog.Debug("llmem: session: extraction returned no memories", "session_id", validID)
		// Log the extraction event even if no memories were extracted
		if logErr := c.store.LogExtraction(ctx, "session", validID, &transcript, 0); logErr != nil {
			slog.Debug("llmem: session: log_extraction failed", "error", logErr, "session_id", validID)
		}
		return 0
	}

	storedCount := 0
	for _, m := range extractedMaps {
		memType, _ := m["type"].(string)
		content, _ := m["content"].(string)
		confidence, _ := m["confidence"].(float64)

		// Default values per the spec
		if memType == "" {
			memType = "fact"
		}
		if content == "" {
			continue // skip entries without content
		}
		if confidence == 0 {
			confidence = 0.8
		}

		addParams := store.AddParams{
			Type:       memType,
			Content:    content,
			Source:     "session",
			Confidence: confidence,
			Metadata:   map[string]any{"source_type": "session", "source_id": validID},
		}

		// Generate embedding if embedder is available
		if c.embedder != nil {
			embedding, embedErr := c.embedder.Embed(ctx, content)
			if embedErr != nil {
				slog.Debug("llmem: session: embedding failed, storing without embedding", "error", embedErr, "session_id", validID)
			} else if len(embedding) > 0 {
				addParams.Embedding = store.VecToBytes(embedding)
			}
		}

		id, addErr := c.store.Add(ctx, addParams)
		if addErr != nil {
			slog.Debug("llmem: session: store extracted memory failed", "error", addErr, "session_id", validID)
			continue
		}
		storedCount++

		_ = id // ID is not needed further, but the store returned it
	}

	// Log the extraction event
	if logErr := c.store.LogExtraction(ctx, "session", validID, &transcript, storedCount); logErr != nil {
		slog.Debug("llmem: session: log_extraction failed", "error", logErr, "session_id", validID)
	}

	slog.Debug("llmem: session: extracted and stored memories", "count", storedCount, "session_id", validID)
	return storedCount
}

// logSessionEvent logs a session event at debug level.
func logSessionEvent(eventType, sessionID string) {
	slog.Debug("llmem: session: event", "type", eventType, "session_id", sessionID)
}