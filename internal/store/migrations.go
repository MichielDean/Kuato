package store

import (
	"context"
	"database/sql"

	"github.com/MichielDean/LLMem/migrations"
	"github.com/pressly/goose/v3"
)

// runMigrationsFromDB applies all pending Goose migrations to the database.
// Migrations are loaded from the embedded migrations.FS at the repository root.
func runMigrationsFromDB(db *sql.DB) error {
	goose.SetDialect("sqlite3")
	provider, err := goose.NewProvider(goose.DialectSQLite3, db, migrations.FS)
	if err != nil {
		return fmtErr("run migrations: new provider: %w", err)
	}
	_, err = provider.Up(context.Background())
	if err != nil {
		return fmtErr("run migrations: up: %w", err)
	}
	return nil
}