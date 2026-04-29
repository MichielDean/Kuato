-- 004_add_inbox.sql — DDL only
-- Working memory inbox: capacity-limited staging area with attention scoring.
-- Items enter via 'lobmem note' and are promoted to long-term memory
-- via 'lobmem consolidate' or the dream deep phase.

CREATE TABLE IF NOT EXISTS "inbox" (
    "id" TEXT PRIMARY KEY,
    "content" TEXT NOT NULL,
    "source" TEXT NOT NULL DEFAULT 'note' CHECK("source" IN ('note', 'learn', 'extract', 'consolidation')),
    "attention_score" REAL NOT NULL DEFAULT 0.5 CHECK("attention_score" >= 0.0 AND "attention_score" <= 1.0),
    "created_at" TEXT NOT NULL DEFAULT (datetime('now')),
    "metadata" TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS "idx_inbox_attention" ON "inbox"("attention_score");
CREATE INDEX IF NOT EXISTS "idx_inbox_created" ON "inbox"("created_at");

INSERT INTO "_schema_migrations" ("version") VALUES (4);