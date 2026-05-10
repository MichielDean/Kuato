// Package migrations provides embedded SQL migration files for Goose.
package migrations

import "embed"

// FS embeds the migration SQL files from the migrations/ directory.
// These are used by the store package via Goose to set up the database schema.
//
//go:embed *.sql
var FS embed.FS