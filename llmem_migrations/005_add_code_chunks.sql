-- 005_add_code_chunks.sql — DDL only
-- Add code_chunks table and indexes for code indexing.
-- The code_chunks_vec and code_chunks_fts virtual tables are created
-- conditionally in CodeIndex._init_db() since they depend on optional
-- extensions (sqlite-vec and FTS5 respectively).

CREATE TABLE IF NOT EXISTS "code_chunks" (
    "id" TEXT PRIMARY KEY,
    "file_path" TEXT NOT NULL,
    "start_line" INTEGER NOT NULL,
    "end_line" INTEGER NOT NULL,
    "content" TEXT NOT NULL,
    "embedding" BLOB,
    "language" TEXT,
    "chunk_type" TEXT NOT NULL DEFAULT 'paragraph',
    "created_at" TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS "idx_code_chunks_file_path" ON "code_chunks"("file_path");
CREATE INDEX IF NOT EXISTS "idx_code_chunks_language" ON "code_chunks"("language");
CREATE INDEX IF NOT EXISTS "idx_code_chunks_chunk_type" ON "code_chunks"("chunk_type");

INSERT INTO "_schema_migrations" ("version") VALUES (5);