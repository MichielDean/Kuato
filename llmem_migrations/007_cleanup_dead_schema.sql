-- 007_cleanup_dead_schema.sql — DDL only
-- Drop unused inbox table and narrow relation_type CHECK constraint
-- to only the relation types actually used in production.
-- SQLite does not support ALTER TABLE DROP CONSTRAINT, so the relations
-- table must be recreated. Transaction control is handled by the Python
-- migration runner (_run_migrations), which wraps the entire migration
-- in BEGIN/COMMIT for crash safety.

-- Step 1: Drop the unused inbox table
DROP TABLE IF EXISTS "inbox";
DROP INDEX IF EXISTS "idx_inbox_attention";
DROP INDEX IF EXISTS "idx_inbox_created";

-- Step 2: Create new relations table with narrowed CHECK constraint
-- Removes unused types: contradicts, depends_on, references
-- Keeps only: supersedes, related_to, derived_from
-- Also removes the unused target_type column
CREATE TABLE IF NOT EXISTS "relations_new" (
    "id" TEXT PRIMARY KEY,
    "source_id" TEXT NOT NULL REFERENCES "memories"("id") ON DELETE CASCADE,
    "target_id" TEXT NOT NULL,
    "relation_type" TEXT NOT NULL CHECK("relation_type" IN (
        'supersedes','related_to','derived_from'
    )),
    "created_at" TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Step 3: Copy data from old table, keeping only valid relation types
-- Any rows with contradicts/depends_on/references would be dropped.
-- There are zero such rows in production.
INSERT INTO "relations_new" ("id", "source_id", "target_id", "relation_type", "created_at")
SELECT "id", "source_id", "target_id", "relation_type", "created_at"
FROM "relations"
WHERE "relation_type" IN ('supersedes', 'related_to', 'derived_from');

-- Step 4: Drop old table
DROP TABLE "relations";

-- Step 5: Rename new table
ALTER TABLE "relations_new" RENAME TO "relations";

-- Step 6: Recreate indexes
CREATE INDEX IF NOT EXISTS "idx_relations_source" ON "relations"("source_id");
CREATE INDEX IF NOT EXISTS "idx_relations_target" ON "relations"("target_id");

-- Step 7: Drop the now-unused target_type index
DROP INDEX IF EXISTS "idx_relations_target_type";

INSERT INTO "_schema_migrations" ("version") VALUES (7);