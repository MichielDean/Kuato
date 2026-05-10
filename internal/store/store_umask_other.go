//go:build !unix && !linux && !darwin

package store

// setUmask is a no-op on non-Unix platforms (Windows, etc.) where
// syscall.Umask is unavailable. On these platforms, the 0700 parent
// directory (created by os.MkdirAll) is the primary defense — the
// DB file is only accessible to the owner of the containing directory.
// After creation, chmodDBFiles applies 0600 permissions to the DB
// file and its WAL/SHM sidecars.
func setUmask(mask int) int {
	// No-op: 0700 parent directory is the primary defense on Windows.
	return 0
}

// resetUmask is a no-op on non-Unix platforms.
func resetUmask(mask int) {
	// No-op: paired with setUmask no-op above.
}