package systemd

import (
	"strings"
	"testing"
)

func TestGenerateServiceUnit_Default(t *testing.T) {
	content, err := GenerateServiceUnit("")
	if err != nil {
		t.Fatalf("GenerateServiceUnit: %v", err)
	}
	if !strings.Contains(content, "[Unit]") {
		t.Error("expected [Unit] section")
	}
	if !strings.Contains(content, "[Service]") {
		t.Error("expected [Service] section")
	}
	if !strings.Contains(content, "llmem dream") {
		t.Error("expected 'llmem dream' in ExecStart")
	}
}

func TestGenerateTimerUnit_Default(t *testing.T) {
	content, err := GenerateTimerUnit("")
	if err != nil {
		t.Fatalf("GenerateTimerUnit: %v", err)
	}
	if !strings.Contains(content, "[Unit]") {
		t.Error("expected [Unit] section")
	}
	if !strings.Contains(content, "[Timer]") {
		t.Error("expected [Timer] section")
	}
	if !strings.Contains(content, "OnCalendar=*-*-* 03:00:00") {
		t.Error("expected default OnCalendar schedule")
	}
}

func TestGenerateTimerUnit_CustomSchedule(t *testing.T) {
	content, err := GenerateTimerUnit("daily")
	if err != nil {
		t.Fatalf("GenerateTimerUnit: %v", err)
	}
	if !strings.Contains(content, "OnCalendar=daily") {
		t.Error("expected custom OnCalendar schedule")
	}
}

func TestValidateSchedule_Valid(t *testing.T) {
	schedules := []string{
		"*-*-* 03:00:00",
		"hourly",
		"daily",
		"weekly",
		"monthly",
		"Mon *-*-* 03:00:00",
	}
	for _, s := range schedules {
		if !ValidateSchedule(s) {
			t.Errorf("expected schedule %q to be valid", s)
		}
	}
}

func TestValidateSchedule_Invalid(t *testing.T) {
	schedules := []string{
		"",
		"*-*-*; rm -rf /",
		"*-*-* && echo pwned",
		"*-*-* `rm -rf /`",
	}
	for _, s := range schedules {
		if ValidateSchedule(s) {
			t.Errorf("expected schedule %q to be invalid", s)
		}
	}
}

func TestGenerateServiceUnit_ContainsExpectedFields(t *testing.T) {
	content, err := GenerateServiceUnit("hourly")
	if err != nil {
		t.Fatalf("GenerateServiceUnit: %v", err)
	}
	if !strings.Contains(content, "Type=oneshot") {
		t.Error("expected Type=oneshot")
	}
	if !strings.Contains(content, "After=network.target") {
		t.Error("expected After=network.target")
	}
}

func TestGenerateTimerUnit_RejectsShellMetacharacters(t *testing.T) {
	// ValidateSchedule is now called inside GenerateTimerUnit,
	// so shell metacharacters in the schedule string should cause an error.
	maliciousSchedules := []string{
		"*-*-*; rm -rf /",
		"*-*-* && echo pwned",
		"*-*-* `rm -rf /`",
		"*-*-* $(whoami)",
		"*-*-* | cat /etc/passwd",
	}
	for _, s := range maliciousSchedules {
		_, err := GenerateTimerUnit(s)
		if err == nil {
			t.Errorf("expected error for malicious schedule %q, got nil", s)
		}
	}
}

func TestGenerateTimerUnit_ValidScheduleAccepted(t *testing.T) {
	// Valid schedules should still be accepted
	validSchedules := []string{
		"",
		"*-*-* 03:00:00",
		"hourly",
		"daily",
		"Mon *-*-* 03:00:00",
	}
	for _, s := range validSchedules {
		_, err := GenerateTimerUnit(s)
		if err != nil {
			t.Errorf("expected no error for valid schedule %q, got %v", s, err)
		}
	}
}