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