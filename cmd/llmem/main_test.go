package main

import (
	"testing"
)

// TestIntrospectCmd_Flags tests that --no-llm and --timeout flags are registered.
func TestIntrospectCmd_Flags(t *testing.T) {
	cmd := introspectCmd()

	noLLMFlag := cmd.Flags().Lookup("no-llm")
	if noLLMFlag == nil {
		t.Error("expected --no-llm flag to be registered on introspect command")
	}
	if noLLMFlag != nil && noLLMFlag.DefValue != "false" {
		t.Errorf("expected --no-llm default false, got %q", noLLMFlag.DefValue)
	}

	timeoutFlag := cmd.Flags().Lookup("timeout")
	if timeoutFlag == nil {
		t.Error("expected --timeout flag to be registered on introspect command")
	}
	if timeoutFlag != nil && timeoutFlag.DefValue != "" {
		t.Errorf("expected --timeout default empty, got %q", timeoutFlag.DefValue)
	}
}

// TestLearnCmd_Flags tests that --no-llm and --timeout flags are registered.
func TestLearnCmd_Flags(t *testing.T) {
	cmd := learnCmd()

	noLLMFlag := cmd.Flags().Lookup("no-llm")
	if noLLMFlag == nil {
		t.Error("expected --no-llm flag to be registered on learn command")
	}
	if noLLMFlag != nil && noLLMFlag.DefValue != "false" {
		t.Errorf("expected --no-llm default false, got %q", noLLMFlag.DefValue)
	}

	timeoutFlag := cmd.Flags().Lookup("timeout")
	if timeoutFlag == nil {
		t.Error("expected --timeout flag to be registered on learn command")
	}
	if timeoutFlag != nil && timeoutFlag.DefValue != "" {
		t.Errorf("expected --timeout default empty, got %q", timeoutFlag.DefValue)
	}
}

// TestIntrospectCmd_AutoFlags tests that --auto, --text, --session, --model, and --base-url
// flags are registered on the introspect command.
func TestIntrospectCmd_AutoFlags(t *testing.T) {
	cmd := introspectCmd()

	autoFlag := cmd.Flags().Lookup("auto")
	if autoFlag == nil {
		t.Error("expected --auto flag to be registered on introspect command")
	}
	if autoFlag != nil && autoFlag.DefValue != "false" {
		t.Errorf("expected --auto default false, got %q", autoFlag.DefValue)
	}

	textFlag := cmd.Flags().Lookup("text")
	if textFlag == nil {
		t.Error("expected --text flag to be registered on introspect command")
	}
	if textFlag != nil && textFlag.DefValue != "" {
		t.Errorf("expected --text default empty, got %q", textFlag.DefValue)
	}

	sessionFlag := cmd.Flags().Lookup("session")
	if sessionFlag == nil {
		t.Error("expected --session flag to be registered on introspect command")
	}
	if sessionFlag != nil && sessionFlag.DefValue != "" {
		t.Errorf("expected --session default empty, got %q", sessionFlag.DefValue)
	}

	modelFlag := cmd.Flags().Lookup("model")
	if modelFlag == nil {
		t.Error("expected --model flag to be registered on introspect command")
	}
	if modelFlag != nil && modelFlag.DefValue != "" {
		t.Errorf("expected --model default empty, got %q", modelFlag.DefValue)
	}

	baseURLFlag := cmd.Flags().Lookup("base-url")
	if baseURLFlag == nil {
		t.Error("expected --base-url flag to be registered on introspect command")
	}
	if baseURLFlag != nil && baseURLFlag.DefValue != "" {
		t.Errorf("expected --base-url default empty, got %q", baseURLFlag.DefValue)
	}
}