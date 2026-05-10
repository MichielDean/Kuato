package main

import (
	"context"
	"fmt"
	"os"

	"github.com/MichielDean/LLMem/internal/store"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: llmem <command>")
		os.Exit(1)
	}

	cfg := store.StoreConfig{
		DBPath:     "",  // use default
		DisableVec: true, // disable vec by default for CLI
	}

	ms, err := store.NewMemoryStore(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "llmem: failed to initialize store: %v\n", err)
		os.Exit(1)
	}
	defer ms.Close()

	ctx := context.Background()
	_ = ctx

	fmt.Println("llmem: store initialized successfully")
}