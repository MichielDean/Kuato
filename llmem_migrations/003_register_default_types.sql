-- 003_register_default_types.sql — DML only, wrapped in transaction
-- Registers all default memory types in the memory_types table.
-- The memory_types table was created in 001_initial_schema.sql.

BEGIN TRANSACTION;
INSERT OR IGNORE INTO "memory_types" ("name") VALUES ('fact');
INSERT OR IGNORE INTO "memory_types" ("name") VALUES ('decision');
INSERT OR IGNORE INTO "memory_types" ("name") VALUES ('preference');
INSERT OR IGNORE INTO "memory_types" ("name") VALUES ('event');
INSERT OR IGNORE INTO "memory_types" ("name") VALUES ('project_state');
INSERT OR IGNORE INTO "memory_types" ("name") VALUES ('procedure');
INSERT OR IGNORE INTO "memory_types" ("name") VALUES ('conversation');
INSERT OR IGNORE INTO "memory_types" ("name") VALUES ('self_assessment');
COMMIT;

INSERT INTO "_schema_migrations" ("version") VALUES (3);