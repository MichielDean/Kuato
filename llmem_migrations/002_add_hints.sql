-- 002_add_hints.sql — DDL only
-- The hints column is already in the initial schema with DEFAULT '[]'.
-- This migration exists for forward-compatibility with databases that
-- were created using the old inline SCHEMA constant that lacked hints.

-- No-op: hints already included in 001_initial_schema.sql FTS definition.
INSERT INTO "_schema_migrations" ("version") VALUES (2);