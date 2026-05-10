// Package session provides session lifecycle hook coordination for LLMem.
// It handles created, idle, compacting, and ending events with debounced idle processing.
package session

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

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

// Compacting key memory types and minimum confidence.
var compactingKeyTypes = []string{"decision", "preference", "procedure", "project_state"}

// SessionAdapter is the interface for reading session content.
// Implementations provide session transcript reading and listing capabilities.
type SessionAdapter interface {
	// ReadTranscript reads the full transcript for a session.
	ReadTranscript(sessionID string) (string, error)
	// ListSessions returns available session information.
	ListSessions() ([]SessionInfo, error)
}

// SessionInfo holds metadata about a session.
type SessionInfo struct {
	ID        string
	StartTime string
	EndTime   string
	WorkDir   string
}

// OpenCodeAdapter reads session data from the OpenCode SQLite database.
type OpenCodeAdapter struct {
	dbPath string
}

// NewOpenCodeAdapter creates an adapter that reads from the given DB path.
func NewOpenCodeAdapter(dbPath string) *OpenCodeAdapter {
	return &OpenCodeAdapter{dbPath: dbPath}
}

// ReadTranscript reads the session transcript from the OpenCode database.
// Currently returns an empty string — the full implementation requires SQLite
// queries against the opencode schema.
func (a *OpenCodeAdapter) ReadTranscript(sessionID string) (string, error) {
	if a.dbPath == "" {
		return "", nil
	}
	// Validate session ID
	if _, err := paths.ValidateSessionID(sessionID); err != nil {
		return "", fmtErr("read transcript: %w", err)
	}
	// OpenCode SQLite integration is not yet implemented
	slog.Debug("llmem: session: OpenCode transcript reading not yet implemented", "session_id", sessionID)
	return "", nil
}

// ListSessions returns available session information.
// Currently returns an empty list — the full implementation requires SQLite queries.
func (a *OpenCodeAdapter) ListSessions() ([]SessionInfo, error) {
	if a.dbPath == "" {
		return []SessionInfo{}, nil
	}
	// OpenCode SQLite integration is not yet implemented
	slog.Debug("llmem: session: OpenCode session listing not yet implemented")
	return []SessionInfo{}, nil
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
}

// SessionHookCoordinator orchestrates memory operations for session lifecycle events.
type SessionHookCoordinator struct {
	store            *store.MemoryStore
	adapter          SessionAdapter
	contextDir       string
	debounceSeconds  int
	lastIdle         map[string]time.Time
	mu               sync.Mutex
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
		lastIdle:         map[string]time.Time{},
	}, nil
}

// OnCreated handles the session.created event.
// Returns result type string: "success", "already_processed", or "error".
func (c *SessionHookCoordinator) OnCreated(ctx context.Context, sessionID string) (string, error) {
	validID, err := paths.ValidateSessionID(sessionID)
	if err != nil {
		return ResultError, fmtErr("on_created: validate session ID: %w", err)
	}
	_ = validID

	// Check if already processed
	alreadyProcessed, err := c.store.IsExtracted(ctx, "session", sessionID)
	if err != nil {
		slog.Debug("llmem: session: on_created: check extracted failed", "error", err)
	} else if alreadyProcessed {
		return ResultAlreadyProcessed, nil
	}

	logSessionEvent("created", sessionID)
	return ResultSuccess, nil
}

// OnIdle handles the session.idle event with debounce (configurable interval, defaults to 30s).
// Returns result type string: "debounced", "no_transcript", "success", or "error".
func (c *SessionHookCoordinator) OnIdle(ctx context.Context, sessionID string) (string, error) {
	validID, err := paths.ValidateSessionID(sessionID)
	if err != nil {
		return ResultError, fmtErr("on_idle: validate session ID: %w", err)
	}
	_ = validID

	// Check debounce
	c.mu.Lock()
	lastTime, exists := c.lastIdle[sessionID]
	if exists && time.Since(lastTime).Seconds() < float64(c.debounceSeconds) {
		c.mu.Unlock()
		return ResultDebounced, nil
	}
	c.lastIdle[sessionID] = time.Now()
	c.mu.Unlock()

	// If no adapter, no transcript
	if c.adapter == nil {
		return ResultNoTranscript, nil
	}

	transcript, err := c.adapter.ReadTranscript(sessionID)
	if err != nil {
		return ResultError, fmtErr("on_idle: read transcript: %w", err)
	}
	if transcript == "" {
		return ResultNoTranscript, nil
	}

	_ = transcript // transcript content would be used for extraction
	logSessionEvent("idle", sessionID)
	return ResultSuccess, nil
}

// OnCompacting handles the session.compacting event.
// Returns result type string, context file path (for injection), and error.
func (c *SessionHookCoordinator) OnCompacting(ctx context.Context, sessionID string) (string, string, error) {
	validID, err := paths.ValidateSessionID(sessionID)
	if err != nil {
		return ResultError, "", fmtErr("on_compacting: validate session ID: %w", err)
	}
	_ = validID

	// Create context directory
	if err := os.MkdirAll(c.contextDir, 0700); err != nil {
		return ResultError, "", fmtErr("on_compacting: create context dir: %w", err)
	}

	// Write context file with key memories
	contextPath := filepath.Join(c.contextDir, sessionID+"_context.md")
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

	logSessionEvent("compacting", sessionID)
	return ResultSuccess, contextPath, nil
}

// OnEnding handles the session.ending event.
// Returns result type string: "success", "no_transcript", or "error".
func (c *SessionHookCoordinator) OnEnding(ctx context.Context, sessionID string) (string, error) {
	validID, err := paths.ValidateSessionID(sessionID)
	if err != nil {
		return ResultError, fmtErr("on_ending: validate session ID: %w", err)
	}
	_ = validID

	// If no adapter, no transcript
	if c.adapter == nil {
		return ResultNoTranscript, nil
	}

	transcript, err := c.adapter.ReadTranscript(sessionID)
	if err != nil {
		return ResultError, fmtErr("on_ending: read transcript: %w", err)
	}
	if transcript == "" {
		return ResultNoTranscript, nil
	}

	_ = transcript // transcript content would be used for extraction
	logSessionEvent("ending", sessionID)
	return ResultSuccess, nil
}

// logSessionEvent logs a session event at debug level.
func logSessionEvent(eventType, sessionID string) {
	slog.Debug("llmem: session: event", "type", eventType, "session_id", sessionID)
}