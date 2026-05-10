-- +goose Up
-- 002_add_hints.sql — DDL only (no-op: hints already in initial schema)
-- The hints column is already in the initial schema with DEFAULT '[]'.
-- This migration exists for forward-compatibility with databases that
-- were created using the old inline SCHEMA constant that lacked hints.

-- No schema changes needed: hints column and FTS definition already present.

-- +goose Down
-- No schema changes to revert.