package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	_ "modernc.org/sqlite"

	"github.com/google/uuid"
)

const (
	defaultVecDimensions     = 768
	defaultBruteForceMaxRows = 10000
	defaultConfidence         = 0.8
	defaultExportLimit        = 10000
	maxIDLength               = 256
	maxEmbeddingBytes         = 1024 * 1024 // 1 MB per embedding vector
)

// MemoryStore is a SQLite-backed memory store with FTS5 full-text search
// and optional vec0 vector search.
type MemoryStore struct {
	dbPath          string
	vecDimensions   int
	vecAvailable    bool
	disableVec      bool
	registeredTypes map[string]struct{}
	db              *sql.DB
	mu              sync.RWMutex
	closed          bool
}

// NewMemoryStore creates and initializes a MemoryStore.
// It opens the database, applies migrations, and optionally sets up vec0.
// If cfg.DBPath is empty, defaults to ~/.config/llmem/memory.db.
// If cfg.VecDimensions is 0, defaults to 768.
// The constructor leaves the store in a fully usable state.
func NewMemoryStore(cfg StoreConfig) (*MemoryStore, error) {
	if cfg.VecDimensions < 0 {
		return nil, fmtErr("vec_dimensions must be non-negative, got %d", cfg.VecDimensions)
	}
	if cfg.VecDimensions == 0 {
		cfg.VecDimensions = defaultVecDimensions
	}

	dbPath := cfg.DBPath
	if dbPath == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, fmtErr("default db path: %w", err)
		}
		dbPath = filepath.Join(homeDir, ".config", "llmem", "memory.db")
	}

	// Create parent directory with 0700 permissions
	if dbPath != ":memory:" {
		dir := filepath.Dir(dbPath)
		if err := os.MkdirAll(dir, 0700); err != nil {
			return nil, fmtErr("create db directory: %w", err)
		}
	}

	// Build registered types set with defensive copy
	registeredTypes := make(map[string]struct{})
	types := cfg.RegisteredTypes
	if len(types) == 0 {
		types = DefaultRegisteredTypes()
	}
	for _, t := range types {
		registeredTypes[t] = struct{}{}
	}

	// Set restrictive umask (0o177) before creating the DB file so it
	// is created with owner-only permissions (0600). On Unix this uses
	// syscall.Umask; on other platforms it's a no-op and the 0700 parent
	// directory serves as the primary access control.
	var oldMask int
	if dbPath != ":memory:" {
		oldMask = setUmask(0o177)
	}

	db, err := openDB(dbPath)
	if err != nil {
		if dbPath != ":memory:" {
			resetUmask(oldMask)
		}
		return nil, fmtErr("open db: %w", err)
	}

	if dbPath != ":memory:" {
		resetUmask(oldMask)
	}

	ms := &MemoryStore{
		dbPath:          dbPath,
		vecDimensions:   cfg.VecDimensions,
		vecAvailable:    false,
		disableVec:      cfg.DisableVec,
		registeredTypes: registeredTypes,
		db:              db,
	}

	// Run migrations
	if err := runMigrationsFromDB(db); err != nil {
		db.Close()
		return nil, fmtErr("run migrations: %w", err)
	}

	// Fix FTS index if empty but memories exist
	if err := ms.rebuildFTSIfEmpty(); err != nil {
		slog.Warn("llmem: store: failed to rebuild FTS index", "error", err)
	}

	// Register vec virtual table module if available
	if !cfg.DisableVec {
		// Try to register the vec0 virtual table module.
		// If sqlite-vec is available, it will be registered via the
		// modernc.org/sqlite extension mechanism. The CREATE VIRTUAL TABLE
		// statement is attempted in initVecTable, and if it fails, we
		// fall back to brute-force search.
		ms.vecAvailable = true
		if err := ms.initVecTable(); err != nil {
			slog.Warn("llmem: store: sqlite-vec unavailable, using brute-force fallback", "error", err)
			ms.vecAvailable = false
		}
	}

	// Chmod DB files for security
	if dbPath != ":memory:" {
		if err := chmodDBFiles(dbPath); err != nil {
			slog.Warn("llmem: store: failed to set db file permissions", "error", err)
		}
	}

	return ms, nil
}

// Close closes the database connection. Safe to call multiple times.
func (ms *MemoryStore) Close() error {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	if ms.closed {
		return nil
	}
	ms.closed = true
	return ms.db.Close()
}

// RegisterMemoryType adds a custom memory type to the instance's type registry.
// The type name must match ^[a-z][a-z0-9_]*$ and be at most 64 characters.
// Returns an error for duplicate registration or invalid names.
func (ms *MemoryStore) RegisterMemoryType(typeName string) error {
	if !isValidTypeName(typeName) {
		return fmtErr("register_memory_type: invalid type name %q: must match ^[a-z][a-z0-9_]*$ and be at most 64 chars", typeName)
	}
	ms.mu.Lock()
	defer ms.mu.Unlock()
	if _, exists := ms.registeredTypes[typeName]; exists {
		return fmtErr("register_memory_type: type %q is already registered", typeName)
	}
	ms.registeredTypes[typeName] = struct{}{}
	return nil
}

// Add creates a new memory and returns its ID.
// Returns an error if the type is not registered or embedding dimensions don't match.
func (ms *MemoryStore) Add(ctx context.Context, params AddParams) (string, error) {
	ms.mu.RLock()
	_, typeOK := ms.registeredTypes[params.Type]
	ms.mu.RUnlock()
	if !typeOK {
		return "", fmtErr("add: unregistered type %q: register it with RegisterMemoryType first", params.Type)
	}

	// Validate embedding size limit (DoS protection)
	if len(params.Embedding) > maxEmbeddingBytes {
		return "", fmtErr("add: embedding size %d bytes exceeds maximum %d bytes", len(params.Embedding), maxEmbeddingBytes)
	}

	// Validate embedding dimensions
	if len(params.Embedding) > 0 && !ms.disableVec {
		actualDim := len(params.Embedding) / 4
		if actualDim != ms.vecDimensions {
			return "", fmtErr("add: embedding dimension %d does not match vec_dimensions %d — use %d-dimensional embeddings", actualDim, ms.vecDimensions, ms.vecDimensions)
		}
	}

	memID := params.ID
	if memID == "" {
		memID = uuid.New().String()
	}

	now := nowUTC()
	confidence := params.Confidence
	if confidence == 0 {
		confidence = defaultConfidence
	}

	hintsJSON, err := json.Marshal(params.Hints)
	if err != nil {
		return "", fmtErr("add: marshal hints: %w", err)
	}
	if string(hintsJSON) == "null" {
		hintsJSON = []byte("[]")
	}

	metadataJSON, err := json.Marshal(params.Metadata)
	if err != nil {
		return "", fmtErr("add: marshal metadata: %w", err)
	}
	if string(metadataJSON) == "null" {
		metadataJSON = []byte("{}")
	}

	source := params.Source
	if source == "" {
		source = "manual"
	}

	// Convert empty strings to nil for nullable columns
	var validUntil any
	if params.ValidUntil != "" {
		validUntil = params.ValidUntil
	}
	var summary any
	if params.Summary != "" {
		summary = params.Summary
	}
	var embedding any
	if params.Embedding != nil {
		embedding = params.Embedding
	}

	_, err = ms.db.ExecContext(ctx,
		`INSERT INTO "memories" ("id", "type", "content", "summary", "hints", "source", "confidence", "valid_from", "valid_until", "created_at", "updated_at", "metadata", "embedding") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		memID, params.Type, params.Content, summary, string(hintsJSON), source, confidence, now, validUntil, now, now, string(metadataJSON), embedding,
	)
	if err != nil {
		return "", fmtErr("add: insert memory: %w", err)
	}
	return memID, nil
}

// Get retrieves a memory by ID. Returns nil if not found.
// If trackAccess is true, increments access_count and updates accessed_at.
func (ms *MemoryStore) Get(ctx context.Context, id string, trackAccess bool) (*Memory, error) {
	row := ms.db.QueryRowContext(ctx,
		`SELECT "id", "type", "content", "summary", "hints", "source", "confidence", "valid_from", "valid_until", "created_at", "updated_at", "accessed_at", "access_count", "metadata", "embedding" FROM "memories" WHERE "id" = ?`,
		id,
	)
	m, err := scanMemory(row)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmtErr("get: query: %w", err)
	}

	if trackAccess {
		now := nowUTC()
		_, err = ms.db.ExecContext(ctx,
			`UPDATE "memories" SET "accessed_at" = ?, "access_count" = "access_count" + 1 WHERE "id" = ?`,
			now, id,
		)
		if err != nil {
			slog.Debug("llmem: store: get: failed to update access count", "error", err)
		} else {
			m.AccessCount++
			m.AccessedAt = now
		}
	}

	return m, nil
}

// GetBatch retrieves multiple memories by their IDs.
// Returns a map keyed by memory ID. Returns empty map for empty input.
// If validOnly is true, only returns memories with valid_until IS NULL.
func (ms *MemoryStore) GetBatch(ctx context.Context, ids []string, validOnly bool) (map[string]*Memory, error) {
	if len(ids) == 0 {
		return map[string]*Memory{}, nil
	}

	ph := placeholders(len(ids))
	args := make([]any, len(ids))
	for i, id := range ids {
		args[i] = id
	}

	query := fmt.Sprintf(`SELECT "id", "type", "content", "summary", "hints", "source", "confidence", "valid_from", "valid_until", "created_at", "updated_at", "accessed_at", "access_count", "metadata", "embedding" FROM "memories" WHERE "id" IN (%s)`, ph)
	if validOnly {
		query += ` AND "valid_until" IS NULL`
	}

	rows, err := ms.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmtErr("get_batch: query: %w", err)
	}
	defer rows.Close()

	result := make(map[string]*Memory)
	for rows.Next() {
		m, err := scanMemoryFromRows(rows)
		if err != nil {
			return nil, fmtErr("get_batch: scan: %w", err)
		}
		result[m.ID] = m
	}
	return result, rows.Err()
}

// Update modifies a memory's fields. Returns false if not found, true on success.
// Rejects ClearEmbedding=true AND Embedding != nil (conflict).
func (ms *MemoryStore) Update(ctx context.Context, params UpdateParams) (bool, error) {
	if params.ClearEmbedding && params.Embedding != nil {
		return false, fmtErr("update: cannot specify both embedding and clear_embedding=true")
	}

	// Validate embedding size limit (DoS protection)
	if len(params.Embedding) > maxEmbeddingBytes {
		return false, fmtErr("update: embedding size %d bytes exceeds maximum %d bytes", len(params.Embedding), maxEmbeddingBytes)
	}

	// Check existence
	var exists bool
	err := ms.db.QueryRowContext(ctx, `SELECT 1 FROM "memories" WHERE "id" = ?`, params.ID).Scan(&exists)
	if err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, fmtErr("update: check existence: %w", err)
	}
	if !exists {
		return false, nil
	}

	now := nowUTC()
	sets := []string{`"updated_at" = ?`}
	vals := []any{now}

	if params.Content != nil {
		sets = append(sets, `"content" = ?`)
		vals = append(vals, *params.Content)
		// Clear embedding when content changes unless explicitly providing new embedding
		if !params.ClearEmbedding && params.Embedding == nil {
			sets = append(sets, `"embedding" = NULL`)
		}
	}
	if params.Summary != nil {
		sets = append(sets, `"summary" = ?`)
		vals = append(vals, *params.Summary)
	}
	if params.Confidence != nil {
		sets = append(sets, `"confidence" = ?`)
		vals = append(vals, *params.Confidence)
	}
	if params.ValidUntil != nil {
		sets = append(sets, `"valid_until" = ?`)
		vals = append(vals, *params.ValidUntil)
	}
	if params.Metadata != nil {
		metadataJSON, err := json.Marshal(params.Metadata)
		if err != nil {
			return false, fmtErr("update: marshal metadata: %w", err)
		}
		sets = append(sets, `"metadata" = ?`)
		vals = append(vals, string(metadataJSON))
	}
	if params.Hints != nil {
		hintsJSON, err := json.Marshal(params.Hints)
		if err != nil {
			return false, fmtErr("update: marshal hints: %w", err)
		}
		sets = append(sets, `"hints" = ?`)
		vals = append(vals, string(hintsJSON))
	}
	if params.ClearEmbedding {
		sets = append(sets, `"embedding" = NULL`)
	} else if params.Embedding != nil {
		sets = append(sets, `"embedding" = ?`)
		vals = append(vals, params.Embedding)
	}

	vals = append(vals, params.ID)
	query := fmt.Sprintf(`UPDATE "memories" SET %s WHERE "id" = ?`, strings.Join(sets, ", "))
	result, err := ms.db.ExecContext(ctx, query, vals...)
	if err != nil {
		return false, fmtErr("update: %w", err)
	}
	n, _ := result.RowsAffected()
	return n > 0, nil
}

// Invalidate sets valid_until on a memory, effectively expiring it.
// Also clears the embedding and updates metadata with invalidation_reason.
// Returns false if not found.
func (ms *MemoryStore) Invalidate(ctx context.Context, id string, reason string) (bool, error) {
	// Fetch current metadata
	var metadataStr string
	err := ms.db.QueryRowContext(ctx, `SELECT "metadata" FROM "memories" WHERE "id" = ?`, id).Scan(&metadataStr)
	if err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, fmtErr("invalidate: fetch metadata: %w", err)
	}

	metadata := map[string]any{}
	if err := json.Unmarshal([]byte(metadataStr), &metadata); err != nil {
		metadata = map[string]any{}
	}
	if reason != "" {
		metadata["invalidation_reason"] = reason
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return false, fmtErr("invalidate: marshal metadata: %w", err)
	}

	now := nowUTC()
	result, err := ms.db.ExecContext(ctx,
		`UPDATE "memories" SET "valid_until" = ?, "metadata" = ?, "embedding" = NULL, "updated_at" = ? WHERE "id" = ?`,
		now, string(metadataJSON), now, id,
	)
	if err != nil {
		return false, fmtErr("invalidate: %w", err)
	}
	n, _ := result.RowsAffected()
	return n > 0, nil
}

// Delete removes a memory by ID and cascades to its relations.
// Returns false if not found.
func (ms *MemoryStore) Delete(ctx context.Context, id string) (bool, error) {
	// Delete relations where target_id matches (no FK cascade on target_id)
	if _, err := ms.db.ExecContext(ctx, `DELETE FROM "relations" WHERE "target_id" = ?`, id); err != nil {
		return false, fmtErr("delete: cleanup target relations: %w", err)
	}

	result, err := ms.db.ExecContext(ctx, `DELETE FROM "memories" WHERE "id" = ?`, id)
	if err != nil {
		return false, fmtErr("delete: %w", err)
	}
	n, _ := result.RowsAffected()
	return n > 0, nil
}

// Search performs FTS5 full-text search with fallback to LIKE.
// Returns memories ranked by BM25 (FTS) or updated_at DESC (LIKE).
func (ms *MemoryStore) Search(ctx context.Context, params SearchParams) ([]*Memory, error) {
	limit := params.Limit
	if limit <= 0 {
		limit = 20
	}
	offset := params.Offset
	if offset < 0 {
		offset = 0
	}

	if params.Query != "" {
		return ms.searchFTS(ctx, params, limit, offset)
	}
	return ms.searchNoQuery(ctx, params, limit, offset)
}

func (ms *MemoryStore) searchFTS(ctx context.Context, params SearchParams, limit, offset int) ([]*Memory, error) {
	ftsQuery := sanitizeFTSQuery(params.Query)

	var clauses []string
	var ftsVals []any
	if params.ValidOnly {
		clauses = append(clauses, `m."valid_until" IS NULL`)
	}
	if params.Type != "" {
		clauses = append(clauses, `m."type" = ?`)
		ftsVals = append(ftsVals, params.Type)
	}
	where := ""
	if len(clauses) > 0 {
		where = " AND " + strings.Join(clauses, " AND ")
	}

	query := fmt.Sprintf(
		`SELECT m."id", m."type", m."content", m."summary", m."hints", m."source", m."confidence", m."valid_from", m."valid_until", m."created_at", m."updated_at", m."accessed_at", m."access_count", m."metadata", m."embedding" FROM "memories_fts" AS fts JOIN "memories" AS m ON m."rowid" = fts."rowid" WHERE "memories_fts" MATCH ?%s ORDER BY rank DESC LIMIT ? OFFSET ?`,
		where,
	)
	args := append([]any{ftsQuery}, ftsVals...)
	args = append(args, limit, offset)

	rows, err := ms.db.QueryContext(ctx, query, args...)
	if err != nil {
		// Fallback to LIKE search
		return ms.fallbackLikeSearch(ctx, params, limit, offset)
	}
	defer rows.Close()

	var results []*Memory
	for rows.Next() {
		m, err := scanMemoryFromRows(rows)
		if err != nil {
			return nil, fmtErr("search: scan: %w", err)
		}
		results = append(results, m)
	}
	return results, rows.Err()
}

func (ms *MemoryStore) searchNoQuery(ctx context.Context, params SearchParams, limit, offset int) ([]*Memory, error) {
	var clauses []string
	var vals []any
	if params.ValidOnly {
		clauses = append(clauses, `"valid_until" IS NULL`)
	}
	if params.Type != "" {
		clauses = append(clauses, `"type" = ?`)
		vals = append(vals, params.Type)
	}
	where := "1=1"
	if len(clauses) > 0 {
		where = strings.Join(clauses, " AND ")
	}

	query := fmt.Sprintf(
		`SELECT "id", "type", "content", "summary", "hints", "source", "confidence", "valid_from", "valid_until", "created_at", "updated_at", "accessed_at", "access_count", "metadata", "embedding" FROM "memories" WHERE %s ORDER BY "updated_at" DESC LIMIT ? OFFSET ?`,
		where,
	)
	vals = append(vals, limit, offset)

	rows, err := ms.db.QueryContext(ctx, query, vals...)
	if err != nil {
		return nil, fmtErr("search: query: %w", err)
	}
	defer rows.Close()

	var results []*Memory
	for rows.Next() {
		m, err := scanMemoryFromRows(rows)
		if err != nil {
			return nil, fmtErr("search: scan: %w", err)
		}
		results = append(results, m)
	}
	return results, rows.Err()
}

func (ms *MemoryStore) fallbackLikeSearch(ctx context.Context, params SearchParams, limit, offset int) ([]*Memory, error) {
	clauses, vals := ms.buildLikeClauses(params.Query, params.Type, params.ValidOnly)

	rows, err := ms.db.QueryContext(ctx,
		fmt.Sprintf(`SELECT "id", "type", "content", "summary", "hints", "source", "confidence", "valid_from", "valid_until", "created_at", "updated_at", "accessed_at", "access_count", "metadata", "embedding" FROM "memories" AS m WHERE %s ORDER BY m."updated_at" DESC LIMIT ? OFFSET ?`, clauses),
		append(vals, limit, offset)...,
	)
	if err != nil {
		return nil, fmtErr("search: like fallback: %w", err)
	}
	defer rows.Close()

	var results []*Memory
	for rows.Next() {
		m, err := scanMemoryFromRows(rows)
		if err != nil {
			return nil, fmtErr("search: like scan: %w", err)
		}
		results = append(results, m)
	}
	return results, rows.Err()
}

func (ms *MemoryStore) buildLikeClauses(query, typeFilter string, validOnly bool) (string, []any) {
	var clauses []string
	var vals []any

	escaped := escapeLike(query)
	clauses = append(clauses, `(m."content" LIKE ? ESCAPE '\' OR m."summary" LIKE ? ESCAPE '\' OR m."hints" LIKE ? ESCAPE '\')`)
	vals = append(vals, "%"+escaped+"%", "%"+escaped+"%", "%"+escaped+"%")

	if validOnly {
		clauses = append(clauses, `m."valid_until" IS NULL`)
	}
	if typeFilter != "" {
		clauses = append(clauses, `m."type" = ?`)
		vals = append(vals, typeFilter)
	}
	return strings.Join(clauses, " AND "), vals
}

// SearchCount returns the number of matching memories.
func (ms *MemoryStore) SearchCount(ctx context.Context, params SearchCountParams) (int, error) {
	if params.Query != "" {
		return ms.searchCountFTS(ctx, params)
	}
	return ms.searchCountNoQuery(ctx, params)
}

func (ms *MemoryStore) searchCountFTS(ctx context.Context, params SearchCountParams) (int, error) {
	ftsQuery := sanitizeFTSQuery(params.Query)

	var clauses []string
	var ftsVals []any
	if params.ValidOnly {
		clauses = append(clauses, `m."valid_until" IS NULL`)
	}
	if params.Type != "" {
		clauses = append(clauses, `m."type" = ?`)
		ftsVals = append(ftsVals, params.Type)
	}
	where := ""
	if len(clauses) > 0 {
		where = " AND " + strings.Join(clauses, " AND ")
	}

	query := fmt.Sprintf(
		`SELECT COUNT(*) FROM "memories_fts" AS fts JOIN "memories" AS m ON m."rowid" = fts."rowid" WHERE "memories_fts" MATCH ?%s`,
		where,
	)
	args := append([]any{ftsQuery}, ftsVals...)

	var count int
	err := ms.db.QueryRowContext(ctx, query, args...).Scan(&count)
	if err != nil {
		// Fallback to LIKE count
		clauses, vals := ms.buildLikeClauses(params.Query, params.Type, params.ValidOnly)
		err = ms.db.QueryRowContext(ctx,
			fmt.Sprintf(`SELECT COUNT(*) FROM "memories" AS m WHERE %s`, clauses),
			vals...,
		).Scan(&count)
		if err != nil {
			return 0, fmtErr("search_count: like fallback: %w", err)
		}
	}
	return count, nil
}

func (ms *MemoryStore) searchCountNoQuery(ctx context.Context, params SearchCountParams) (int, error) {
	var clauses []string
	var vals []any
	if params.ValidOnly {
		clauses = append(clauses, `"valid_until" IS NULL`)
	}
	if params.Type != "" {
		clauses = append(clauses, `"type" = ?`)
		vals = append(vals, params.Type)
	}
	where := "1=1"
	if len(clauses) > 0 {
		where = strings.Join(clauses, " AND ")
	}

	var count int
	err := ms.db.QueryRowContext(ctx,
		fmt.Sprintf(`SELECT COUNT(*) FROM "memories" WHERE %s`, where),
		vals...,
	).Scan(&count)
	if err != nil {
		return 0, fmtErr("search_count: %w", err)
	}
	return count, nil
}

// SearchByEmbedding searches memories by vector similarity.
// Uses vec0 if available, brute-force fallback otherwise.
// If limit <= 0, defaults to 20.
func (ms *MemoryStore) SearchByEmbedding(ctx context.Context, queryVec []float32, validOnly bool, limit int, threshold float64) ([]*ScoredMemory, error) {
	if limit <= 0 {
		limit = 20
	}
	if ms.vecAvailable {
		return ms.searchByEmbeddingVec(ctx, queryVec, validOnly, limit, threshold)
	}
	return ms.searchByEmbeddingBrute(ctx, queryVec, validOnly, limit, threshold)
}

func (ms *MemoryStore) searchByEmbeddingVec(ctx context.Context, queryVec []float32, validOnly bool, limit int, threshold float64) ([]*ScoredMemory, error) {
	queryBytes := vecToBytes(queryVec)

	multipliers := []int{3, 10, 50, 0}
	var scored []*ScoredMemory

	for _, multiplier := range multipliers {
		var searchLimit int
		if multiplier == 0 {
			var totalRows int
			err := ms.db.QueryRowContext(ctx, `SELECT count(*) FROM "memories_vec"`).Scan(&totalRows)
			if err != nil {
				return nil, fmtErr("search_by_embedding: vec count: %w", err)
			}
			searchLimit = totalRows
		} else {
			searchLimit = limit*multiplier + 1
		}

		rows, err := ms.db.QueryContext(ctx,
			`SELECT "rowid", "distance" FROM "memories_vec" WHERE "embedding" MATCH ? AND k = ? ORDER BY "distance"`,
			queryBytes, searchLimit,
		)
		if err != nil {
			// Fall back to brute-force
			return ms.searchByEmbeddingBrute(ctx, queryVec, validOnly, limit, threshold)
		}

		type vecRow struct {
			rowid    int64
			distance float64
		}
		var vecRows []vecRow
		for rows.Next() {
			var r vecRow
			if err := rows.Scan(&r.rowid, &r.distance); err != nil {
				rows.Close()
				return nil, fmtErr("search_by_embedding: vec scan: %w", err)
			}
			vecRows = append(vecRows, r)
		}
		rows.Close()

		if len(vecRows) == 0 {
			return nil, nil
		}

		// Fetch matching memory IDs
		rowids := make([]any, len(vecRows))
		for i, r := range vecRows {
			rowids[i] = r.rowid
		}
		ph := placeholders(len(rowids))
		where := fmt.Sprintf(`"rowid" IN (%s)`, ph)
		if validOnly {
			where += ` AND "valid_until" IS NULL`
		}

		memRows, err := ms.db.QueryContext(ctx,
			fmt.Sprintf(`SELECT "id", "rowid" FROM "memories" WHERE %s`, where),
			rowids...,
		)
		if err != nil {
			return nil, fmtErr("search_by_embedding: fetch memories: %w", err)
		}

		rowidToMemID := make(map[int64]string)
		for memRows.Next() {
			var memID string
			var rid int64
			if err := memRows.Scan(&memID, &rid); err != nil {
				memRows.Close()
				return nil, fmtErr("search_by_embedding: scan mem: %w", err)
			}
			rowidToMemID[rid] = memID
		}
		memRows.Close()

		// Build scored list
		scored = scored[:0]
		for _, r := range vecRows {
			cosineSim := 1.0 - r.distance
			if memID, ok := rowidToMemID[r.rowid]; ok && cosineSim >= threshold {
				scored = append(scored, &ScoredMemory{Memory: &Memory{ID: memID}, Score: cosineSim})
			}
		}

		if len(scored) >= limit || multiplier == 0 {
			break
		}
	}

	// Truncate to limit
	if len(scored) > limit {
		scored = scored[:limit]
	}

	if len(scored) == 0 {
		return nil, nil
	}

	// Fetch full memories
	topIDs := make([]any, len(scored))
	for i, sm := range scored {
		topIDs[i] = sm.Memory.ID
	}
	ph := placeholders(len(topIDs))
	memRows, err := ms.db.QueryContext(ctx,
		fmt.Sprintf(`SELECT "id", "type", "content", "summary", "hints", "source", "confidence", "valid_from", "valid_until", "created_at", "updated_at", "accessed_at", "access_count", "metadata", "embedding" FROM "memories" WHERE "id" IN (%s)`, ph),
		topIDs...,
	)
	if err != nil {
		return nil, fmtErr("search_by_embedding: fetch full: %w", err)
	}
	defer memRows.Close()

	memMap := make(map[string]*Memory)
	for memRows.Next() {
		m, err := scanMemoryFromRows(memRows)
		if err != nil {
			return nil, fmtErr("search_by_embedding: scan: %w", err)
		}
		memMap[m.ID] = m
	}

	var results []*ScoredMemory
	for _, sm := range scored {
		if m, ok := memMap[sm.Memory.ID]; ok {
			results = append(results, &ScoredMemory{Memory: m, Score: sm.Score})
		}
	}
	return results, nil
}

func (ms *MemoryStore) searchByEmbeddingBrute(ctx context.Context, queryVec []float32, validOnly bool, limit int, threshold float64) ([]*ScoredMemory, error) {
	where := `WHERE "embedding" IS NOT NULL`
	args := []any{}
	if validOnly {
		where += ` AND "valid_until" IS NULL`
	}
	where += ` LIMIT ?`
	args = append(args, defaultBruteForceMaxRows)

	rows, err := ms.db.QueryContext(ctx,
		fmt.Sprintf(`SELECT "id", "embedding" FROM "memories" %s`, where),
		args...,
	)
	if err != nil {
		return nil, fmtErr("search_by_embedding: brute query: %w", err)
	}
	defer rows.Close()

	type scoredEntry struct {
		id    string
		score float64
	}
	var entries []scoredEntry

	for rows.Next() {
		var id string
		var embBytes []byte
		if err := rows.Scan(&id, &embBytes); err != nil {
			return nil, fmtErr("search_by_embedding: brute scan: %w", err)
		}
		vec := bytesToVec(embBytes)
		if len(vec) != len(queryVec) {
			continue
		}
		score := cosineSimilarity(queryVec, vec)
		if score >= threshold {
			entries = append(entries, scoredEntry{id: id, score: score})
		}
	}
	if rows.Err() != nil {
		return nil, fmtErr("search_by_embedding: brute rows: %w", rows.Err())
	}

	// Sort descending by score
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].score > entries[j].score
	})

	if len(entries) == 0 {
		return nil, nil
	}

	// Cap at limit
	if len(entries) > limit {
		entries = entries[:limit]
	}

	// Fetch full memories
	topIDs := make([]any, len(entries))
	for i, e := range entries {
		topIDs[i] = e.id
	}
	ph := placeholders(len(topIDs))
	memRows, err := ms.db.QueryContext(ctx,
		fmt.Sprintf(`SELECT "id", "type", "content", "summary", "hints", "source", "confidence", "valid_from", "valid_until", "created_at", "updated_at", "accessed_at", "access_count", "metadata", "embedding" FROM "memories" WHERE "id" IN (%s)`, ph),
		topIDs...,
	)
	if err != nil {
		return nil, fmtErr("search_by_embedding: brute fetch: %w", err)
	}
	defer memRows.Close()

	memMap := make(map[string]*Memory)
	for memRows.Next() {
		m, err := scanMemoryFromRows(memRows)
		if err != nil {
			return nil, fmtErr("search_by_embedding: brute scan mem: %w", err)
		}
		memMap[m.ID] = m
	}

	var results []*ScoredMemory
	for _, e := range entries {
		if m, ok := memMap[e.id]; ok {
			results = append(results, &ScoredMemory{Memory: m, Score: e.score})
		}
	}
	return results, nil
}

// ListAll lists memories with optional type filter. Delegates to Search with no query.
func (ms *MemoryStore) ListAll(ctx context.Context, params ListParams) ([]*Memory, error) {
	limit := params.Limit
	if limit <= 0 {
		limit = 100
	}
	return ms.Search(ctx, SearchParams{
		Type:      params.Type,
		ValidOnly: params.ValidOnly,
		Limit:     limit,
	})
}

// Count returns total number of memories, optionally limited to valid only.
func (ms *MemoryStore) Count(ctx context.Context, validOnly bool) (int, error) {
	clause := ""
	if validOnly {
		clause = `WHERE "valid_until" IS NULL`
	}
	var count int
	err := ms.db.QueryRowContext(ctx,
		fmt.Sprintf(`SELECT COUNT(*) FROM "memories" %s`, clause),
	).Scan(&count)
	if err != nil {
		return 0, fmtErr("count: %w", err)
	}
	return count, nil
}

// CountByType returns a map of type to count, sorted by count descending.
func (ms *MemoryStore) CountByType(ctx context.Context, validOnly bool) (map[string]int, error) {
	clause := ""
	if validOnly {
		clause = `WHERE "valid_until" IS NULL`
	}
	rows, err := ms.db.QueryContext(ctx,
		fmt.Sprintf(`SELECT "type", COUNT(*) as cnt FROM "memories" %s GROUP BY "type" ORDER BY cnt DESC`, clause),
	)
	if err != nil {
		return nil, fmtErr("count_by_type: %w", err)
	}
	defer rows.Close()

	result := make(map[string]int)
	for rows.Next() {
		var mtype string
		var cnt int
		if err := rows.Scan(&mtype, &cnt); err != nil {
			return nil, fmtErr("count_by_type: scan: %w", err)
		}
		result[mtype] = cnt
	}
	return result, rows.Err()
}

// CountEmbeddings returns the count of valid memories with non-null embeddings.
func (ms *MemoryStore) CountEmbeddings(ctx context.Context) (int, error) {
	var count int
	err := ms.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM "memories" WHERE "embedding" IS NOT NULL AND "valid_until" IS NULL`,
	).Scan(&count)
	if err != nil {
		return 0, fmtErr("count_embeddings: %w", err)
	}
	return count, nil
}

// AddRelation adds a relation between two memories.
// Returns the relation UUID. Validates relation_type against allowed types.
func (ms *MemoryStore) AddRelation(ctx context.Context, sourceID, targetID, relationType string) (string, error) {
	if !isValidRelationType(relationType) {
		return "", fmtErr("add_relation: invalid relation_type %q, must be one of %v", relationType, ValidRelationTypes())
	}

	relID := uuid.New().String()
	now := nowUTC()

	_, err := ms.db.ExecContext(ctx,
		`INSERT INTO "relations" ("id", "source_id", "target_id", "relation_type", "created_at") VALUES (?,?,?,?,?)`,
		relID, sourceID, targetID, relationType, now,
	)
	if err != nil {
		return "", fmtErr("add_relation: insert: %w", err)
	}
	return relID, nil
}

// GetRelations returns all relations for a memory (both source and target).
func (ms *MemoryStore) GetRelations(ctx context.Context, memID string) ([]*Relation, error) {
	rows, err := ms.db.QueryContext(ctx,
		`SELECT "id", "source_id", "target_id", "relation_type", "created_at" FROM "relations" WHERE "source_id" = ? OR "target_id" = ?`,
		memID, memID,
	)
	if err != nil {
		return nil, fmtErr("get_relations: %w", err)
	}
	defer rows.Close()

	var relations []*Relation
	for rows.Next() {
		r, err := scanRelation(rows)
		if err != nil {
			return nil, fmtErr("get_relations: scan: %w", err)
		}
		relations = append(relations, r)
	}
	return relations, rows.Err()
}

// GetRelationsBatch retrieves relations for multiple memory IDs.
func (ms *MemoryStore) GetRelationsBatch(ctx context.Context, memIDs []string) ([]*Relation, error) {
	if len(memIDs) == 0 {
		return []*Relation{}, nil
	}
	ph := placeholders(len(memIDs))
	args := make([]any, len(memIDs)*2)
	for i, id := range memIDs {
		args[i] = id
		args[len(memIDs)+i] = id
	}

	rows, err := ms.db.QueryContext(ctx,
		fmt.Sprintf(`SELECT "id", "source_id", "target_id", "relation_type", "created_at" FROM "relations" WHERE "source_id" IN (%s) OR "target_id" IN (%s)`, ph, ph),
		args...,
	)
	if err != nil {
		return nil, fmtErr("get_relations_batch: %w", err)
	}
	defer rows.Close()

	var relations []*Relation
	for rows.Next() {
		r, err := scanRelation(rows)
		if err != nil {
			return nil, fmtErr("get_relations_batch: scan: %w", err)
		}
		relations = append(relations, r)
	}
	return relations, rows.Err()
}

// TraverseRelations performs bidirectional relation traversal using recursive CTE.
// Caps maxDepth at 5. Returns deduplicated results with distance and relationScore.
func (ms *MemoryStore) TraverseRelations(ctx context.Context, startIDs []string, maxDepth int) ([]*TraversedRelation, error) {
	if len(startIDs) == 0 || maxDepth < 1 {
		return []*TraversedRelation{}, nil
	}
	if maxDepth > 5 {
		maxDepth = 5
	}

	ph := placeholders(len(startIDs))
	args := make([]any, len(startIDs)*2+2+len(startIDs))
	for i, id := range startIDs {
		args[i] = id
		args[len(startIDs)+i] = id
	}
	args[len(startIDs)*2] = maxDepth
	args[len(startIDs)*2+1] = maxDepth
	for i, id := range startIDs {
		args[len(startIDs)*2+2+i] = id
	}

	cteSQL := fmt.Sprintf(
		`WITH RECURSIVE "rel_traverse"("node_id", "reached_id", "rel_type", "dist") AS (
			SELECT "source_id", "target_id", "relation_type", 1 FROM "relations" WHERE "source_id" IN (%s)
			UNION ALL
			SELECT "target_id", "source_id", "relation_type", 1 FROM "relations" WHERE "target_id" IN (%s)
			UNION ALL
			SELECT r."source_id", r."target_id", r."relation_type", rt."dist" + 1 FROM "relations" r
			INNER JOIN "rel_traverse" rt ON rt."reached_id" = r."source_id"
			WHERE rt."dist" < ?
			UNION ALL
			SELECT r."target_id", r."source_id", r."relation_type", rt."dist" + 1 FROM "relations" r
			INNER JOIN "rel_traverse" rt ON rt."reached_id" = r."target_id"
			WHERE rt."dist" < ?
		)
		SELECT "reached_id", "rel_type", MIN("dist") as "dist" FROM "rel_traverse"
		WHERE "reached_id" NOT IN (%s)
		GROUP BY "reached_id", "rel_type"
		ORDER BY "dist"`,
		ph, ph, ph,
	)

	rows, err := ms.db.QueryContext(ctx, cteSQL, args...)
	if err != nil {
		return nil, fmtErr("traverse_relations: %w", err)
	}
	defer rows.Close()

	var results []*TraversedRelation
	seen := make(map[string]bool)
	for rows.Next() {
		var reachedID, relType string
		var dist int
		if err := rows.Scan(&reachedID, &relType, &dist); err != nil {
			return nil, fmtErr("traverse_relations: scan: %w", err)
		}
		key := fmt.Sprintf("%s:%s", reachedID, relType)
		if seen[key] {
			continue
		}
		seen[key] = true

		// relationScore decays as 0.5^distance (matches Python)
		relationScore := 0.5
		for i := 1; i < dist; i++ {
			relationScore *= 0.5
		}

		results = append(results, &TraversedRelation{
			TargetID:      reachedID,
			RelationType:  relType,
			Distance:      dist,
			RelationScore: relationScore,
		})
	}
	return results, rows.Err()
}

// LogExtraction upserts an extraction log entry.
func (ms *MemoryStore) LogExtraction(ctx context.Context, sourceType, sourceID string, rawText *string, extractedCount int) error {
	_, err := ms.db.ExecContext(ctx,
		`INSERT INTO "extraction_log" ("source_type", "source_id", "raw_text", "extracted_count", "created_at") VALUES (?,?,?,?,datetime('now')) ON CONFLICT("source_type", "source_id") DO UPDATE SET "raw_text"=excluded."raw_text", "extracted_count"=excluded."extracted_count"`,
		sourceType, sourceID, rawText, extractedCount,
	)
	if err != nil {
		return fmtErr("log_extraction: %w", err)
	}
	return nil
}

// SupersedeBySource invalidates all memories matching source metadata.
func (ms *MemoryStore) SupersedeBySource(ctx context.Context, sourceType, sourceID string) (int, error) {
	now := nowUTC()
	result, err := ms.db.ExecContext(ctx,
		`UPDATE "memories" SET "valid_until" = ?, "updated_at" = ? WHERE "source" = ? AND "valid_until" IS NULL AND json_extract("metadata", '$.source_id') = ?`,
		now, now, sourceType, sourceID,
	)
	if err != nil {
		return 0, fmtErr("supersede_by_source: %w", err)
	}
	n, _ := result.RowsAffected()
	return int(n), nil
}

// IsExtracted checks whether a source has been extracted.
func (ms *MemoryStore) IsExtracted(ctx context.Context, sourceType, sourceID string) (bool, error) {
	var exists bool
	err := ms.db.QueryRowContext(ctx,
		`SELECT 1 FROM "extraction_log" WHERE "source_type" = ? AND "source_id" = ?`,
		sourceType, sourceID,
	).Scan(&exists)
	if err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, fmtErr("is_extracted: %w", err)
	}
	return true, nil
}

// RemoveExtractionLog removes an extraction log entry. Returns false if not found.
func (ms *MemoryStore) RemoveExtractionLog(ctx context.Context, sourceType, sourceID string) (bool, error) {
	result, err := ms.db.ExecContext(ctx,
		`DELETE FROM "extraction_log" WHERE "source_type" = ? AND "source_id" = ?`,
		sourceType, sourceID,
	)
	if err != nil {
		return false, fmtErr("remove_extraction_log: %w", err)
	}
	n, _ := result.RowsAffected()
	return n > 0, nil
}

// FindSimilar finds similar memories using vector or text search.
func (ms *MemoryStore) FindSimilar(ctx context.Context, params FindSimilarParams) ([]*ScoredMemory, error) {
	if len(params.QueryVec) > 0 {
		return ms.SearchByEmbedding(ctx, params.QueryVec, false, params.Limit, params.Threshold)
	}
	if params.Content != "" {
		results, err := ms.Search(ctx, SearchParams{
			Query:     params.Content,
			ValidOnly: false,
			Limit:     params.Limit,
		})
		if err != nil {
			return nil, err
		}
		var scored []*ScoredMemory
		for _, m := range results {
			scored = append(scored, &ScoredMemory{Memory: m, Score: 0.0})
		}
		return scored, nil
	}
	return []*ScoredMemory{}, nil
}

// ConsolidateDuplicates finds near-duplicate pairs by cosine similarity.
func (ms *MemoryStore) ConsolidateDuplicates(ctx context.Context, threshold float64, limit int) ([]*DuplicatePair, error) {
	if limit <= 0 {
		limit = 500
	}

	rows, err := ms.db.QueryContext(ctx,
		`SELECT "id", "content", "embedding" FROM "memories" WHERE "embedding" IS NOT NULL AND "valid_until" IS NULL LIMIT ?`,
		limit,
	)
	if err != nil {
		return nil, fmtErr("consolidate_duplicates: %w", err)
	}
	defer rows.Close()

	type entry struct {
		id      string
		content string
		emb     []float32
	}
	var entries []entry
	for rows.Next() {
		var id, content string
		var embBytes []byte
		if err := rows.Scan(&id, &content, &embBytes); err != nil {
			return nil, fmtErr("consolidate_duplicates: scan: %w", err)
		}
		vec := bytesToVec(embBytes)
		entries = append(entries, entry{id: id, content: content, emb: vec})
	}
	if rows.Err() != nil {
		return nil, fmtErr("consolidate_duplicates: rows: %w", rows.Err())
	}

	if len(entries) == 0 {
		return []*DuplicatePair{}, nil
	}

	var pairs []*DuplicatePair
	seen := make(map[string]bool)
	for i, e1 := range entries {
		if seen[e1.id] {
			continue
		}
		for j := i + 1; j < len(entries); j++ {
			e2 := entries[j]
			if seen[e2.id] || e2.id == e1.id {
				continue
			}
			if len(e1.emb) != len(e2.emb) {
				continue
			}
			score := cosineSimilarity(e1.emb, e2.emb)
			if score >= threshold {
				pairs = append(pairs, &DuplicatePair{
					SourceID: e1.id,
					TargetID: e2.id,
					Score:    score,
				})
				seen[e1.id] = true
				seen[e2.id] = true
			}
		}
	}
	return pairs, nil
}

// ExportAll exports all memories ordered by created_at.
// If limit is nil, defaults to 10000. Pass 0 for no limit.
func (ms *MemoryStore) ExportAll(ctx context.Context, limit *int) ([]*Memory, error) {
	effectiveLimit := defaultExportLimit
	if limit != nil {
		effectiveLimit = *limit
	}

	var rows *sql.Rows
	var err error
	if effectiveLimit == 0 {
		rows, err = ms.db.QueryContext(ctx,
			`SELECT "id", "type", "content", "summary", "hints", "source", "confidence", "valid_from", "valid_until", "created_at", "updated_at", "accessed_at", "access_count", "metadata", "embedding" FROM "memories" ORDER BY "created_at"`,
		)
	} else {
		rows, err = ms.db.QueryContext(ctx,
			`SELECT "id", "type", "content", "summary", "hints", "source", "confidence", "valid_from", "valid_until", "created_at", "updated_at", "accessed_at", "access_count", "metadata", "embedding" FROM "memories" ORDER BY "created_at" LIMIT ?`,
			effectiveLimit,
		)
	}
	if err != nil {
		return nil, fmtErr("export_all: %w", err)
	}
	defer rows.Close()

	var memories []*Memory
	for rows.Next() {
		m, err := scanMemoryFromRows(rows)
		if err != nil {
			return nil, fmtErr("export_all: scan: %w", err)
		}
		memories = append(memories, m)
	}
	return memories, rows.Err()
}

// ImportMemories imports a list of memories with per-entry validation.
// Skips invalid entries with slog.Warn. Returns count of successfully imported memories.
func (ms *MemoryStore) ImportMemories(ctx context.Context, memories []ImportMemory) (int, error) {
	count := 0
	for i, m := range memories {
		// Validate type and content are strings
		if m.Type == "" {
			slog.Warn("llmem: store: import: skipping entry: missing type", "index", i)
			continue
		}
		if m.Content == "" {
			slog.Warn("llmem: store: import: skipping entry: missing content", "index", i)
			continue
		}

		// Validate ID length
		if len(m.ID) > maxIDLength {
			slog.Warn("llmem: store: import: skipping entry: id too long", "index", i, "id_length", len(m.ID))
			continue
		}

		// Validate embedding size limit (DoS protection)
		if len(m.Embedding) > maxEmbeddingBytes {
			slog.Warn("llmem: store: import: skipping entry: embedding exceeds size limit", "index", i, "got", len(m.Embedding), "max", maxEmbeddingBytes)
			continue
		}

		// Validate embedding dimensions
		if len(m.Embedding) > 0 && !ms.disableVec {
			expectedLen := ms.vecDimensions * 4
			if len(m.Embedding) != expectedLen {
				slog.Warn("llmem: store: import: skipping entry: embedding dimension mismatch", "index", i, "got", len(m.Embedding), "expected", expectedLen)
				continue
			}
		}

		confidence := m.Confidence
		if confidence == 0 {
			confidence = defaultConfidence
		}
		if confidence < 0 || confidence > 1 {
			slog.Warn("llmem: store: import: skipping entry: confidence out of range", "index", i, "confidence", confidence)
			continue
		}

		_, err := ms.Add(ctx, AddParams{
			ID:         m.ID,
			Type:       m.Type,
			Content:    m.Content,
			Summary:    m.Summary,
			Source:     m.Source,
			Confidence: confidence,
			Metadata:   m.Metadata,
			Embedding:  m.Embedding,
			Hints:      m.Hints,
		})
		if err != nil {
			slog.Warn("llmem: store: import: skipping entry", "index", i, "error", err)
			continue
		}
		count++
	}
	return count, nil
}

// Touch increments access_count and updates accessed_at for a memory.
// Returns false if not found.
func (ms *MemoryStore) Touch(ctx context.Context, id string) (bool, error) {
	var exists bool
	err := ms.db.QueryRowContext(ctx, `SELECT 1 FROM "memories" WHERE "id" = ?`, id).Scan(&exists)
	if err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, fmtErr("touch: check: %w", err)
	}
	if !exists {
		return false, nil
	}

	now := nowUTC()
	_, err = ms.db.ExecContext(ctx,
		`UPDATE "memories" SET "accessed_at" = ?, "access_count" = "access_count" + 1 WHERE "id" = ?`,
		now, id,
	)
	if err != nil {
		return false, fmtErr("touch: update: %w", err)
	}
	return true, nil
}

// TouchBatch increments access_count and updates accessed_at for multiple memories.
// Returns the number of rows affected.
func (ms *MemoryStore) TouchBatch(ctx context.Context, ids []string) (int, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	now := nowUTC()
	ph := placeholders(len(ids))
	args := []any{now}
	for _, id := range ids {
		args = append(args, id)
	}
	result, err := ms.db.ExecContext(ctx,
		fmt.Sprintf(`UPDATE "memories" SET "accessed_at" = ?, "access_count" = "access_count" + 1 WHERE "id" IN (%s)`, ph),
		args...,
	)
	if err != nil {
		return 0, fmtErr("touch_batch: %w", err)
	}
	n, _ := result.RowsAffected()
	return int(n), nil
}

// GetEmbeddingsWithTypes returns (embedding_bytes, type) tuples for valid memories with embeddings.
// If limit is 0, returns all rows (no limit). If limit < 0, defaults to defaultBruteForceMaxRows.
func (ms *MemoryStore) GetEmbeddingsWithTypes(ctx context.Context, limit int) ([]*EmbeddingWithType, error) {
	effectiveLimit := limit
	if effectiveLimit < 0 {
		effectiveLimit = defaultBruteForceMaxRows
	}

	var rows *sql.Rows
	var err error
	if effectiveLimit > 0 {
		rows, err = ms.db.QueryContext(ctx,
			`SELECT "embedding", "type" FROM "memories" WHERE "embedding" IS NOT NULL AND "valid_until" IS NULL LIMIT ?`,
			effectiveLimit,
		)
	} else {
		rows, err = ms.db.QueryContext(ctx,
			`SELECT "embedding", "type" FROM "memories" WHERE "embedding" IS NOT NULL AND "valid_until" IS NULL`,
		)
	}
	if err != nil {
		return nil, fmtErr("get_embeddings_with_types: %w", err)
	}
	defer rows.Close()

	var result []*EmbeddingWithType
	for rows.Next() {
		var emb []byte
		var mtype string
		if err := rows.Scan(&emb, &mtype); err != nil {
			return nil, fmtErr("get_embeddings_with_types: scan: %w", err)
		}
		result = append(result, &EmbeddingWithType{Embedding: emb, Type: mtype})
	}
	return result, rows.Err()
}

// openDB opens a SQLite database connection using the modernc.org/sqlite driver.
func openDB(dbPath string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmtErr("open db: %w", err)
	}
	// Enable WAL mode and foreign keys
	_, err = db.Exec("PRAGMA journal_mode=WAL")
	if err != nil {
		db.Close()
		return nil, fmtErr("enable WAL: %w", err)
	}
	_, err = db.Exec("PRAGMA foreign_keys=ON")
	if err != nil {
		db.Close()
		return nil, fmtErr("enable foreign keys: %w", err)
	}
	return db, nil
}

// rebuildFTSIfEmpty rebuilds the FTS index if it's empty but memories exist.
func (ms *MemoryStore) rebuildFTSIfEmpty() error {
	var ftsCount, memCount int
	err := ms.db.QueryRow(`SELECT count(*) FROM "memories_fts"`).Scan(&ftsCount)
	if err != nil {
		return fmtErr("rebuild FTS: count fts: %w", err)
	}
	err = ms.db.QueryRow(`SELECT count(*) FROM "memories"`).Scan(&memCount)
	if err != nil {
		return fmtErr("rebuild FTS: count memories: %w", err)
	}
	if ftsCount == 0 && memCount > 0 {
		_, err = ms.db.Exec(
			`INSERT INTO "memories_fts"("rowid", "content", "summary", "hints") SELECT "rowid", "content", "summary", "hints" FROM "memories" WHERE "content" IS NOT NULL`,
		)
		if err != nil {
			return fmtErr("rebuild FTS: insert: %w", err)
		}
	}
	return nil
}

// initVecTable creates the vec0 virtual table and triggers if sqlite-vec is available.
func (ms *MemoryStore) initVecTable() error {
	// Try to create the vec0 virtual table directly.
	// If sqlite-vec extension is not available, this will fail gracefully.
	_, err := ms.db.Exec(fmt.Sprintf(
		`CREATE VIRTUAL TABLE IF NOT EXISTS "memories_vec" USING vec0(rowid INTEGER PRIMARY KEY, embedding float[%d] distance_metric=cosine)`,
		ms.vecDimensions,
	))
	if err != nil {
		return fmtErr("init_vec: vec0 virtual table unavailable: %w", err)
	}

	// Check if vec table exists and has matching dimensions
	var tableName string
	err = ms.db.QueryRow(`SELECT name FROM sqlite_master WHERE type='table' AND name='memories_vec'`).Scan(&tableName)
	if err == nil {
		// Table exists, verify dimensions
		var sql string
		err = ms.db.QueryRow(`SELECT sql FROM sqlite_master WHERE type='table' AND name='memories_vec'`).Scan(&sql)
		if err == nil {
			matches := reVecDimensions.FindStringSubmatch(sql)
			if len(matches) > 1 {
				var existingDim int
				if _, err := fmt.Sscanf(matches[1], "%d", &existingDim); err != nil {
					return fmtErr("init_vec: parse dimension from vec0 table schema: %w", err)
				}
				if existingDim != ms.vecDimensions {
					return fmtErr("init_vec: existing vec0 table has dimensions=%d but vec_dimensions=%d", existingDim, ms.vecDimensions)
				}
			}
		}
	}

	// Populate vec table if behind
	var vecCount, memEmbCount int
	if err := ms.db.QueryRow(`SELECT count(*) FROM "memories_vec"`).Scan(&vecCount); err != nil {
		return fmtErr("init_vec: count vec rows: %w", err)
	}
	if err := ms.db.QueryRow(`SELECT count(*) FROM "memories" WHERE "embedding" IS NOT NULL`).Scan(&memEmbCount); err != nil {
		return fmtErr("init_vec: count memory embeddings: %w", err)
	}

	if vecCount < memEmbCount {
		if _, err := ms.db.Exec(`DELETE FROM "memories_vec"`); err != nil {
			return fmtErr("init_vec: clear vec table for repopulation: %w", err)
		}
		rows, err := ms.db.Query(`SELECT "rowid", "embedding" FROM "memories" WHERE "embedding" IS NOT NULL`)
		if err != nil {
			return fmtErr("init_vec: fetch embeddings for repopulation: %w", err)
		}
		for rows.Next() {
			var rowid int64
			var emb []byte
			if err := rows.Scan(&rowid, &emb); err != nil {
				rows.Close()
				return fmtErr("init_vec: scan embedding for repopulation: %w", err)
			}
			if _, err := ms.db.Exec(`INSERT INTO "memories_vec"("rowid", "embedding") VALUES (?, ?)`, rowid, emb); err != nil {
				rows.Close()
				return fmtErr("init_vec: insert embedding into vec table for rowid %d: %w", rowid, err)
			}
		}
		rows.Close()
	}

	// Create triggers
	triggerDefs := []struct {
		name string
		sql  string
	}{
		{
			"memories_vec_insert",
			`CREATE TRIGGER IF NOT EXISTS "memories_vec_insert" AFTER INSERT ON "memories" WHEN new."embedding" IS NOT NULL BEGIN INSERT INTO "memories_vec"("rowid", "embedding") VALUES (new."rowid", new."embedding"); END`,
		},
		{
			"memories_vec_update",
			`CREATE TRIGGER IF NOT EXISTS "memories_vec_update" AFTER UPDATE ON "memories" WHEN new."embedding" IS NOT NULL BEGIN DELETE FROM "memories_vec" WHERE "rowid" = old."rowid"; INSERT INTO "memories_vec"("rowid", "embedding") VALUES (new."rowid", new."embedding"); END`,
		},
		{
			"memories_vec_update_null",
			`CREATE TRIGGER IF NOT EXISTS "memories_vec_update_null" AFTER UPDATE ON "memories" WHEN new."embedding" IS NULL AND old."embedding" IS NOT NULL BEGIN DELETE FROM "memories_vec" WHERE "rowid" = old."rowid"; END`,
		},
		{
			"memories_vec_delete",
			`CREATE TRIGGER IF NOT EXISTS "memories_vec_delete" AFTER DELETE ON "memories" WHEN old."embedding" IS NOT NULL BEGIN DELETE FROM "memories_vec" WHERE "rowid" = old."rowid"; END`,
		},
	}

	for _, td := range triggerDefs {
		_, err := ms.db.Exec(td.sql)
		if err != nil {
			slog.Debug("llmem: store: failed to create vec trigger", "trigger", td.name, "error", err)
		}
	}

	return nil
}

// dropVecTriggers removes vec-related triggers when vec is disabled.
// Logs errors but does not fail — triggers may not exist if vec was never enabled.
func (ms *MemoryStore) dropVecTriggers() {
	for _, name := range []string{"memories_vec_insert", "memories_vec_update", "memories_vec_update_null", "memories_vec_delete"} {
		if _, err := ms.db.Exec(fmt.Sprintf(`DROP TRIGGER IF EXISTS "%s"`, name)); err != nil {
			slog.Debug("llmem: store: failed to drop vec trigger", "trigger", name, "error", err)
		}
	}
}

// chmodDBFiles sets 0600 permissions on the DB file and its WAL/SHM sidecars.
func chmodDBFiles(dbPath string) error {
	for _, suffix := range []string{"", "-wal", "-shm"} {
		path := dbPath + suffix
		if _, err := os.Stat(path); err == nil {
			if err := os.Chmod(path, 0600); err != nil {
				slog.Warn("llmem: store: failed to set permissions on db file", "path", path, "error", err)
			}
		}
	}
	return nil
}



// scanFunc is a function type that scans row columns into dest pointers.
type scanFunc func(dest ...any) error

// scanMemoryFields scans columns from a row into a Memory struct using the provided scan function.
// This is the shared implementation for both scanMemory (single row) and scanMemoryFromRows (iterator).
func scanMemoryFields(scan scanFunc) (*Memory, error) {
	m := &Memory{}
	var hintsJSON, metadataJSON string
	var summary, validUntil, accessedAt sql.NullString
	var embedding []byte

	err := scan(
		&m.ID, &m.Type, &m.Content, &summary, &hintsJSON, &m.Source, &m.Confidence,
		&m.ValidFrom, &validUntil, &m.CreatedAt, &m.UpdatedAt, &accessedAt,
		&m.AccessCount, &metadataJSON, &embedding,
	)
	if err != nil {
		return nil, err
	}

	m.Summary = summary.String
	m.ValidUntil = validUntil.String
	m.AccessedAt = accessedAt.String
	m.Embedding = embedding

	if err := json.Unmarshal([]byte(hintsJSON), &m.Hints); err != nil {
		m.Hints = []string{}
	}
	if m.Hints == nil {
		m.Hints = []string{}
	}
	if err := json.Unmarshal([]byte(metadataJSON), &m.Metadata); err != nil {
		m.Metadata = map[string]any{}
	}
	if m.Metadata == nil {
		m.Metadata = map[string]any{}
	}

	return m, nil
}

// scanMemory scans a single row from a *sql.Row into a Memory struct.
func scanMemory(row *sql.Row) (*Memory, error) {
	return scanMemoryFields(row.Scan)
}

// scanMemoryFromRows scans a single row from sql.Rows into a Memory struct.
func scanMemoryFromRows(rows *sql.Rows) (*Memory, error) {
	return scanMemoryFields(rows.Scan)
}

// scanRelation scans a single row from sql.Rows into a Relation struct.
func scanRelation(rows *sql.Rows) (*Relation, error) {
	r := &Relation{}
	err := rows.Scan(&r.ID, &r.SourceID, &r.TargetID, &r.RelationType, &r.CreatedAt)
	return r, err
}