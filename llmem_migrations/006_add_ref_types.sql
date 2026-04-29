-- 006_add_ref_types.sql — DDL only
-- Add target_type column to relations table (default 'memory') and extend
-- relation_type CHECK to include 'references'. SQLite does not support
-- ALTER TABLE DROP CONSTRAINT, so the relations table must be recreated.

-- Step 1: Create new relations table with target_type column and extended CHECK
CREATE TABLE IF NOT EXISTS "relations_new" (
    "id" TEXT PRIMARY KEY,
    "source_id" TEXT NOT NULL REFERENCES "memories"("id") ON DELETE CASCADE,
    "target_id" TEXT NOT NULL,
    "relation_type" TEXT NOT NULL CHECK("relation_type" IN (
        'supersedes','contradicts','depends_on','related_to','derived_from','references'
    )),
    "target_type" TEXT NOT NULL DEFAULT 'memory' CHECK("target_type" IN ('memory', 'code')),
    "created_at" TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Step 2: Copy data from old table (target_type defaults to 'memory' for existing rows)
INSERT INTO "relations_new" ("id", "source_id", "target_id", "relation_type", "target_type", "created_at")
SELECT "id", "source_id", "target_id", "relation_type", 'memory', "created_at"
FROM "relations";

-- Step 3: Drop old table
DROP TABLE "relations";

-- Step 4: Rename new table
ALTER TABLE "relations_new" RENAME TO "relations";

-- Step 5: Recreate indexes from 001_initial_schema.sql
CREATE INDEX IF NOT EXISTS "idx_relations_source" ON "relations"("source_id");
CREATE INDEX IF NOT EXISTS "idx_relations_target" ON "relations"("target_id");

-- Step 6: Add new index for target_type
CREATE INDEX IF NOT EXISTS "idx_relations_target_type" ON "relations"("target_type");

INSERT INTO "_schema_migrations" ("version") VALUES (6);