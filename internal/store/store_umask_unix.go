//go:build unix || linux || darwin

package store

import "syscall"

// setUmask sets the file-creation umask and returns the previous mask.
// On Unix systems, this uses syscall.Umask to ensure the SQLite DB file
// is created with restrictive permissions (owner-only read/write).
// The umask 0o177 blocks group/other read/write, so new files
// get mode 0600 (rw-------).
func setUmask(mask int) int {
	return syscall.Umask(mask)
}

// resetUmask restores the previous umask after DB file creation.
func resetUmask(mask int) {
	syscall.Umask(mask)
}