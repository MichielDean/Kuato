-- 001_initial_schema.sql — DDL only
-- Initial schema for llmem memory store.
-- All identifiers are double-quoted for SQLite compatibility.
-- NO CHECK constraint on "type" — type validation is in application code.

CREATE TABLE IF NOT EXISTS "memories" (
    "id" TEXT PRIMARY KEY,
    "type" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "summary" TEXT,
    "hints" TEXT NOT NULL DEFAULT '[]',
    "source" TEXT NOT NULL DEFAULT 'manual',
    "confidence" REAL NOT NULL DEFAULT 0.8 CHECK("confidence" >= 0 AND "confidence" <= 1),
    "valid_from" TEXT NOT NULL DEFAULT (datetime('now')),
    "valid_until" TEXT,
    "created_at" TEXT NOT NULL DEFAULT (datetime('now')),
    "updated_at" TEXT NOT NULL DEFAULT (datetime('now')),
    "accessed_at" TEXT,
    "access_count" INTEGER NOT NULL DEFAULT 0,
    "metadata" TEXT NOT NULL DEFAULT '{}',
    "embedding" BLOB
);

CREATE TABLE IF NOT EXISTS "relations" (
    "id" TEXT PRIMARY KEY,
    "source_id" TEXT NOT NULL REFERENCES "memories"("id") ON DELETE CASCADE,
    "target_id" TEXT NOT NULL REFERENCES "memories"("id") ON DELETE CASCADE,
    "relation_type" TEXT NOT NULL CHECK("relation_type" IN (
        'supersedes','contradicts','depends_on','related_to','derived_from'
    )),
    "created_at" TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS "extraction_log" (
    "id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "source_type" TEXT NOT NULL,
    "source_id" TEXT NOT NULL,
    "raw_text" TEXT,
    "extracted_count" INTEGER NOT NULL DEFAULT 0,
    "created_at" TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE("source_type", "source_id")
);

CREATE TABLE IF NOT EXISTS "memory_types" (
    "name" TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS "_schema_migrations" (
    "version" INTEGER PRIMARY KEY,
    "applied_at" TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS "idx_memories_type" ON "memories"("type");
CREATE INDEX IF NOT EXISTS "idx_memories_valid" ON "memories"("valid_until");
CREATE INDEX IF NOT EXISTS "idx_memories_source" ON "memories"("source");
CREATE INDEX IF NOT EXISTS "idx_memories_updated_at" ON "memories"("updated_at");
CREATE INDEX IF NOT EXISTS "idx_relations_source" ON "relations"("source_id");
CREATE INDEX IF NOT EXISTS "idx_relations_target" ON "relations"("target_id");
CREATE INDEX IF NOT EXISTS "idx_extraction_log_source" ON "extraction_log"("source_type", "source_id");

CREATE VIRTUAL TABLE IF NOT EXISTS "memories_fts" USING fts5("content", "summary", "hints", content="memories", content_rowid="rowid");

CREATE TRIGGER IF NOT EXISTS "memories_fts_insert" AFTER INSERT ON "memories" BEGIN
    INSERT INTO "memories_fts"("rowid", "content", "summary", "hints") VALUES (new."rowid", new."content", new."summary", new."hints");
END;

CREATE TRIGGER IF NOT EXISTS "memories_fts_update" AFTER UPDATE ON "memories" BEGIN
    INSERT INTO "memories_fts"("memories_fts", "rowid", "content", "summary", "hints") VALUES ('delete', old."rowid", old."content", old."summary", old."hints");
    INSERT INTO "memories_fts"("rowid", "content", "summary", "hints") VALUES (new."rowid", new."content", new."summary", new."hints");
END;

CREATE TRIGGER IF NOT EXISTS "memories_fts_delete" AFTER DELETE ON "memories" BEGIN
    INSERT INTO "memories_fts"("memories_fts", "rowid", "content", "summary", "hints") VALUES ('delete', old."rowid", old."content", old."summary", old."hints");
END;

INSERT INTO "_schema_migrations" ("version") VALUES (1);