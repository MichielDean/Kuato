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