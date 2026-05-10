// Package systemd generates systemd service and timer unit files for the LLMem dream schedule.
package systemd

import (
	"bytes"
	"embed"
	"fmt"
	"strings"
	"text/template"

	"github.com/MichielDean/LLMem/internal/paths"
)

//go:embed templates/*.service templates/*.timer
var templateFS embed.FS

// fmtErr wraps an error with the "llmem: systemd:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: systemd: "+format, args...)
}

// defaultDreamSchedule is the default systemd OnCalendar schedule for dream cycles.
const defaultDreamSchedule = "*-*-* 03:00:00"

// GenerateServiceUnit generates a systemd .service unit content for the dream schedule.
// The schedule parameter is used for documentation purposes (describing when the timer fires).
// Uses embed.FS for template loading.
func GenerateServiceUnit(dreamSchedule string) (string, error) {
	if dreamSchedule == "" {
		dreamSchedule = defaultDreamSchedule
	}

	homeDir := paths.GetHomeDir()
	dbPath := paths.GetDBPath()

	data := map[string]string{
		"HomeDir":       homeDir,
		"DBPath":         dbPath,
		"DreamSchedule": dreamSchedule,
	}

	tmpl, err := template.ParseFS(templateFS, "templates/llmem-dream.service")
	if err != nil {
		// Fallback to hardcoded template if embed fails
		return generateServiceUnitFallback(data), nil
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmtErr("execute service template: %w", err)
	}
	return buf.String(), nil
}

// GenerateTimerUnit generates a systemd .timer unit content for the dream schedule.
// The schedule parameter is used for the OnCalendar directive.
// Uses embed.FS for template loading. Validates schedule and rejects shell metacharacters.
func GenerateTimerUnit(dreamSchedule string) (string, error) {
	if dreamSchedule == "" {
		dreamSchedule = defaultDreamSchedule
	}

	if !ValidateSchedule(dreamSchedule) {
		return "", fmtErr("invalid schedule %q: contains forbidden characters", dreamSchedule)
	}

	data := map[string]string{
		"DreamSchedule": dreamSchedule,
	}

	tmpl, err := template.ParseFS(templateFS, "templates/llmem-dream.timer")
	if err != nil {
		// Fallback to hardcoded template if embed fails
		return generateTimerUnitFallback(dreamSchedule), nil
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmtErr("execute timer template: %w", err)
	}
	return buf.String(), nil
}

// generateServiceUnitFallback produces a service unit when templates aren't available.
func generateServiceUnitFallback(data map[string]string) string {
	return fmt.Sprintf(`[Unit]
Description=LLMem Dream Consolidation
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/llmem dream --apply
WorkingDirectory=%s
Environment=LMEM_HOME=%s

[Install]
WantedBy=multi-user.target
`, data["HomeDir"], data["HomeDir"])
}

// generateTimerUnitFallback produces a timer unit when templates aren't available.
func generateTimerUnitFallback(schedule string) string {
	return fmt.Sprintf(`[Unit]
Description=LLMem Dream Consolidation Timer

[Timer]
OnCalendar=%s
Persistent=true

[Install]
WantedBy=timers.target
`, schedule)
}

// ValidateSchedule checks if a systemd OnCalendar schedule string looks valid.
// This is a basic check — it doesn't validate all systemd calendar syntax.
func ValidateSchedule(schedule string) bool {
	if schedule == "" {
		return false
	}
	// Must not contain shell metacharacters
	forbidden := []string{";", "&", "|", "$", "`", "(", ")", "{", "}", "<", ">", "!", "#"}
	for _, f := range forbidden {
		if strings.Contains(schedule, f) {
			return false
		}
	}
	return true
}