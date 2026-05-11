package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/MichielDean/LLMem/internal/config"
	"github.com/MichielDean/LLMem/internal/dream"
	"github.com/MichielDean/LLMem/internal/introspect"
	"github.com/MichielDean/LLMem/internal/paths"
	"github.com/MichielDean/LLMem/internal/session"
	"github.com/MichielDean/LLMem/internal/store"
	"github.com/MichielDean/LLMem/internal/taxonomy"
	"github.com/spf13/cobra"
)

var (
	dbPath     string
	jsonOutput  bool
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "llmem",
		Short: "LLMem — structured memory for LLM agents",
		Long:  "LLMem provides persistent memory storage, search, and consolidation for LLM agents.",
	}

	rootCmd.PersistentFlags().StringVar(&dbPath, "db", "", "Path to the memory database (default: ~/.config/llmem/memory.db)")
	rootCmd.PersistentFlags().BoolVar(&jsonOutput, "json", false, "Output results as JSON")

	rootCmd.AddCommand(
		addCmd(),
		getCmd(),
		searchCmd(),
		listCmd(),
		statsCmd(),
		updateCmd(),
		invalidateCmd(),
		deleteCmd(),
		exportCmd(),
		importCmd(),
		initCmd(),
		metricsCmd(),
		dreamCmd(),
		introspectCmd(),
		learnCmd(),
		trackReviewCmd(),
		contextCmd(),
		hookCmd(),
	)

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

// resolveDBPath returns the configured database path.
func resolveDBPath() string {
	if dbPath != "" {
		return dbPath
	}
	return paths.GetDBPath()
}

// loadConfig loads the LLMem configuration from the default config path.
func loadConfig() (*config.Config, error) {
	cfg, err := config.LoadConfig(paths.GetConfigPath())
	if err != nil {
		return nil, fmt.Errorf("llmem: load config: %w", err)
	}
	return cfg, nil
}

// openStore creates a MemoryStore and returns it with a cleanup function.
func openStore() (*store.MemoryStore, error) {
	cfg := store.StoreConfig{
		DBPath:     resolveDBPath(),
		DisableVec: true,
	}
	ms, err := store.NewMemoryStore(cfg)
	if err != nil {
		return nil, fmt.Errorf("llmem: failed to initialize store: %w", err)
	}
	return ms, nil
}

func addCmd() *cobra.Command {
	var (
		typeVal      string
		contentVal   string
		summaryVal   string
		sourceVal    string
		confidenceVal float64
		validUntilVal string
		metadataVal  string
		fileVal      string
	)
	cmd := &cobra.Command{
		Use:   "add",
		Short: "Add a new memory",
		RunE: func(cmd *cobra.Command, args []string) error {
			if fileVal != "" {
				resolvedFile, err := filepath.Abs(fileVal)
				if err != nil {
					return fmt.Errorf("llmem: add: resolve file path: %w", err)
				}
				if paths.IsBlockedPath(resolvedFile) {
					return fmt.Errorf("llmem: add: file path targets a blocked system directory: %s", resolvedFile)
				}
				data, err := os.ReadFile(resolvedFile)
				if err != nil {
					return fmt.Errorf("llmem: add: read file: %w", err)
				}
				contentVal = string(data)
			}
			if contentVal == "" {
				return fmt.Errorf("llmem: add: content is required (use --content or --file)")
			}

			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			var metadata map[string]any
			if metadataVal != "" {
				if err := json.Unmarshal([]byte(metadataVal), &metadata); err != nil {
					return fmt.Errorf("llmem: add: invalid metadata JSON: %w", err)
				}
			}

			id, err := ms.Add(context.Background(), store.AddParams{
				Type:       typeVal,
				Content:    contentVal,
				Summary:    summaryVal,
				Source:     sourceVal,
				Confidence: confidenceVal,
				ValidUntil: validUntilVal,
				Metadata:   metadata,
			})
			if err != nil {
				return err
			}
			fmt.Println(id)
			return nil
		},
	}
	cmd.Flags().StringVar(&typeVal, "type", "fact", "Memory type")
	cmd.Flags().StringVar(&contentVal, "content", "", "Memory content")
	cmd.Flags().StringVar(&summaryVal, "summary", "", "Memory summary")
	cmd.Flags().StringVar(&sourceVal, "source", "manual", "Memory source")
	cmd.Flags().Float64Var(&confidenceVal, "confidence", 0.8, "Confidence score (0.0-1.0)")
	cmd.Flags().StringVar(&validUntilVal, "valid-until", "", "ISO 8601 timestamp for validity expiration")
	cmd.Flags().StringVar(&metadataVal, "metadata", "", "JSON metadata")
	cmd.Flags().StringVar(&fileVal, "file", "", "Read content from file")
	return cmd
}

func getCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "get <id>",
		Short: "Get a memory by ID",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			mem, err := ms.Get(context.Background(), args[0], false)
			if err != nil {
				return err
			}
			if mem == nil {
				return fmt.Errorf("llmem: get: memory %s not found", args[0])
			}
			data, err := json.MarshalIndent(mem, "", "  ")
			if err != nil {
				return err
			}
			fmt.Println(string(data))
			return nil
		},
	}
}

func searchCmd() *cobra.Command {
	var (
		typeVal    string
		limitVal   int
		validOnly  bool
		ftsOnly    bool
		semanticOnly bool
	)
	cmd := &cobra.Command{
		Use:   "search <query>",
		Short: "Search memories",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			results, err := ms.Search(context.Background(), store.SearchParams{
				Query:        args[0],
				Type:         typeVal,
				ValidOnly:    validOnly,
				Limit:        limitVal,
				FTSOnly:      ftsOnly,
				SemanticOnly: semanticOnly,
			})
			if err != nil {
				return err
			}
			for _, m := range results {
				if jsonOutput {
					data, _ := json.MarshalIndent(m, "", "  ")
					fmt.Println(string(data))
				} else {
					fmt.Printf("%s [%s] %.2f %s\n", m.ID, m.Type, m.Confidence, m.Content)
				}
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&typeVal, "type", "", "Filter by memory type")
	cmd.Flags().IntVar(&limitVal, "limit", 20, "Maximum results")
	cmd.Flags().BoolVar(&validOnly, "valid-only", false, "Only show valid memories")
	cmd.Flags().BoolVar(&ftsOnly, "fts-only", false, "FTS search only")
	cmd.Flags().BoolVar(&semanticOnly, "semantic-only", false, "Semantic search only")
	return cmd
}

func listCmd() *cobra.Command {
	var (
		typeVal   string
		limitVal  int
		allVal    bool
	)
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List memories",
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			validOnly := !allVal
			results, err := ms.Search(context.Background(), store.SearchParams{
				Type:      typeVal,
				ValidOnly: validOnly,
				Limit:     limitVal,
			})
			if err != nil {
				return err
			}
			for _, m := range results {
				if jsonOutput {
					data, _ := json.MarshalIndent(m, "", "  ")
					fmt.Println(string(data))
				} else {
					fmt.Printf("%s [%s] %.2f %s\n", m.ID, m.Type, m.Confidence, m.Content)
				}
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&typeVal, "type", "", "Filter by memory type")
	cmd.Flags().IntVar(&limitVal, "limit", 100, "Maximum results")
	cmd.Flags().BoolVar(&allVal, "all", false, "Show all memories including expired")
	return cmd
}

func statsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "stats",
		Short: "Show memory statistics",
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			total, err := ms.Count(context.Background(), false)
			if err != nil {
				return err
			}
			active, err := ms.Count(context.Background(), true)
			if err != nil {
				return err
			}
			byType, err := ms.CountByType(context.Background(), true)
			if err != nil {
				return err
			}
			expired := total - active

			if jsonOutput {
				data := map[string]any{
					"total":   total,
					"active":  active,
					"expired": expired,
					"by_type": byType,
				}
				enc := json.NewEncoder(os.Stdout)
				enc.SetIndent("", "  ")
				return enc.Encode(data)
			}

			fmt.Printf("Total: %d\nActive: %d\nExpired: %d\n\nBy type:\n", total, active, expired)
			for typ, cnt := range byType {
				fmt.Printf("  %s: %d\n", typ, cnt)
			}
			return nil
		},
	}
}

func updateCmd() *cobra.Command {
	var (
		contentVal    string
		summaryVal    string
		confidenceVal float64
		validUntilVal string
		metadataVal   string
	)
	cmd := &cobra.Command{
		Use:   "update <id>",
		Short: "Update a memory",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			params := store.UpdateParams{ID: args[0]}
			if cmd.Flags().Changed("content") {
				params.Content = &contentVal
			}
			if cmd.Flags().Changed("summary") {
				params.Summary = &summaryVal
			}
			if cmd.Flags().Changed("confidence") {
				params.Confidence = &confidenceVal
			}
			if cmd.Flags().Changed("valid-until") {
				params.ValidUntil = &validUntilVal
			}
			if metadataVal != "" {
				var metadata map[string]any
				if err := json.Unmarshal([]byte(metadataVal), &metadata); err != nil {
					return fmt.Errorf("llmem: update: invalid metadata JSON: %w", err)
				}
				params.Metadata = metadata
			}

			ok, err := ms.Update(context.Background(), params)
			if err != nil {
				return err
			}
			if !ok {
				return fmt.Errorf("llmem: update: memory %s not found", args[0])
			}
			fmt.Printf("Updated %s\n", args[0])
			return nil
		},
	}
	cmd.Flags().StringVar(&contentVal, "content", "", "New content")
	cmd.Flags().StringVar(&summaryVal, "summary", "", "New summary")
	cmd.Flags().Float64Var(&confidenceVal, "confidence", 0, "New confidence score")
	cmd.Flags().StringVar(&validUntilVal, "valid-until", "", "New validity expiration")
	cmd.Flags().StringVar(&metadataVal, "metadata", "", "New JSON metadata")
	return cmd
}

func invalidateCmd() *cobra.Command {
	var reasonVal string
	cmd := &cobra.Command{
		Use:   "invalidate <id>",
		Short: "Invalidate a memory",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			ok, err := ms.Invalidate(context.Background(), args[0], reasonVal)
			if err != nil {
				return err
			}
			if !ok {
				return fmt.Errorf("llmem: invalidate: memory %s not found", args[0])
			}
			fmt.Printf("Invalidated %s\n", args[0])
			return nil
		},
	}
	cmd.Flags().StringVar(&reasonVal, "reason", "", "Invalidation reason")
	return cmd
}

func deleteCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "delete <id>",
		Short: "Delete a memory",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			ok, err := ms.Delete(context.Background(), args[0])
			if err != nil {
				return err
			}
			if !ok {
				return fmt.Errorf("llmem: delete: memory %s not found", args[0])
			}
			fmt.Printf("Deleted %s\n", args[0])
			return nil
		},
	}
}

func exportCmd() *cobra.Command {
	var (
		outputVal string
		limitVal  int
	)
	cmd := &cobra.Command{
		Use:   "export",
		Short: "Export memories as JSON",
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			var limit *int
			if cmd.Flags().Changed("limit") {
				limit = &limitVal
			}

			memories, err := ms.ExportAll(context.Background(), limit)
			if err != nil {
				return err
			}

			data, err := json.MarshalIndent(memories, "", "  ")
			if err != nil {
				return err
			}

			if outputVal != "" {
				resolved, err := paths.ValidateWritePath(outputVal, "export output")
				if err != nil {
					return fmt.Errorf("llmem: export: %w", err)
				}
				if err := os.WriteFile(resolved, data, 0600); err != nil {
					return fmt.Errorf("llmem: export: write: %w", err)
				}
				fmt.Printf("Exported %d memories to %s\n", len(memories), resolved)
			} else {
				fmt.Println(string(data))
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&outputVal, "output", "", "Output file path")
	cmd.Flags().IntVar(&limitVal, "limit", 10000, "Maximum memories to export")
	return cmd
}

func importCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "import <file>",
		Short: "Import memories from JSON file",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			// Validate file path
			resolved, err := paths.ValidateWritePath(args[0], "import file")
			if err != nil {
				return fmt.Errorf("llmem: import: %w", err)
			}

			// Check file size (max 10 MiB)
			info, err := os.Stat(resolved)
			if err != nil {
				return fmt.Errorf("llmem: import: stat: %w", err)
			}
			if info.Size() > 10*1024*1024 {
				return fmt.Errorf("llmem: import: file too large (max 10 MiB)")
			}

			data, err := os.ReadFile(resolved)
			if err != nil {
				return fmt.Errorf("llmem: import: read: %w", err)
			}

			var memories []store.ImportMemory
			if err := json.Unmarshal(data, &memories); err != nil {
				return fmt.Errorf("llmem: import: parse JSON: %w", err)
			}

			count, err := ms.ImportMemories(context.Background(), memories)
			if err != nil {
				return err
			}
			fmt.Printf("Imported %d memories\n", count)
			return nil
		},
	}
}

func initCmd() *cobra.Command {
	var (
		ollamaURLVal string
		forceVal     bool
	)
	cmd := &cobra.Command{
		Use:   "init",
		Short: "Initialize LLMem configuration and database",
		RunE: func(cmd *cobra.Command, args []string) error {
			// Migrate from ~/.lobsterdog/ if applicable
			migrated, err := paths.MigrateFromLobsterdog()
			if err != nil {
				slog.Warn("llmem: init: migration check failed", "error", err)
			}
			if migrated {
				fmt.Println("Migrated data from ~/.lobsterdog/ to ~/.config/llmem/")
			}

			// Create home directory
			homeDir := paths.GetHomeDir()
			if err := os.MkdirAll(homeDir, 0700); err != nil {
				return fmt.Errorf("llmem: init: create home directory: %w", err)
			}

			// Write config
			configPath := paths.GetConfigPath()
			defaultCfg := map[string]any{
				"memory": map[string]any{
					"ollama_url": defaultIfEmpty(ollamaURLVal, "http://localhost:11434"),
					"embed_model": "nomic-embed-text",
				},
			}
			written, err := config.WriteConfigYAML(configPath, defaultCfg, forceVal)
			if err != nil {
				return fmt.Errorf("llmem: init: write config: %w", err)
			}
			if written {
				fmt.Printf("Created config at %s\n", configPath)
			} else {
				fmt.Printf("Config already exists at %s (use --force to overwrite)\n", configPath)
			}

			// Initialize database
			dbPathVal := paths.GetDBPath()
			ms, err := store.NewMemoryStore(store.StoreConfig{
				DBPath:     dbPathVal,
				DisableVec: true,
			})
			if err != nil {
				return fmt.Errorf("llmem: init: create database: %w", err)
			}
			ms.Close()
			fmt.Printf("Created database at %s\n", dbPathVal)

			return nil
		},
	}
	cmd.Flags().StringVar(&ollamaURLVal, "ollama-url", "", "Ollama base URL")
	cmd.Flags().BoolVar(&forceVal, "force", false, "Overwrite existing config")
	return cmd
}

func metricsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "metrics",
		Short: "Report embedding quality metrics",
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			count, err := ms.CountEmbeddings(context.Background())
			if err != nil {
				return err
			}
			fmt.Printf("Embeddings: %d\n", count)
			return nil
		},
	}
}

func dreamCmd() *cobra.Command {
	var (
		applyVal   bool
		dryRunVal  bool
		phaseVal   string
		reportVal  string
	)
	cmd := &cobra.Command{
		Use:   "dream",
		Short: "Run dream consolidation cycle",
		RunE: func(cmd *cobra.Command, args []string) error {
			if dryRunVal {
				applyVal = false
			}
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			// Load config to populate dream settings
			cfg, err := loadConfig()
			if err != nil {
				return fmt.Errorf("llmem: dream: load config: %w", err)
			}

			dreamerCfg := cfg.DreamerConfig()
			dreamerCfg.Store = ms

			d, err := dream.NewDreamer(dreamerCfg)
			if err != nil {
				return err
			}

			result, err := d.Run(context.Background(), applyVal, phaseVal)
			if err != nil {
				return err
			}

			if result.Light != nil {
				fmt.Printf("Light: %d duplicate pairs\n", result.Light.DuplicatePairs)
			}
			if result.Deep != nil {
				fmt.Printf("Deep: %d decayed, %d boosted, %d merged, %d auto-linked\n",
					result.Deep.DecayedCount, result.Deep.BoostedCount,
					result.Deep.MergedCount, result.Deep.AutoLinkedCount)
			}
			if result.Rem != nil {
				fmt.Printf("REM: %d total memories, %d active\n",
					result.Rem.TotalMemories, result.Rem.ActiveMemories)
				for _, theme := range result.Rem.Themes {
					fmt.Printf("  Theme: %s\n", theme)
				}
			}

			if reportVal != "" {
				if err := d.GenerateDreamReport(result, reportVal); err != nil {
					return err
				}
				fmt.Printf("Dream report written to %s\n", reportVal)
			}

			return nil
		},
	}
	cmd.Flags().BoolVar(&applyVal, "apply", false, "Apply changes (default: dry run)")
	cmd.Flags().BoolVar(&dryRunVal, "dry-run", false, "Dry run only (default: true). Shorthand for omitting --apply.")
	cmd.Flags().StringVar(&phaseVal, "phase", "", "Run specific phase: light, deep, rem (default: all)")
	cmd.Flags().StringVar(&reportVal, "report", "", "Generate HTML dream report at this path")
	return cmd
}

func introspectCmd() *cobra.Command {
	var (
		whatHappened string
		categoryVal  string
		contextVal   string
		caughtByVal  string
		proposedFix  string
	)
	cmd := &cobra.Command{
		Use:   "introspect",
		Short: "Analyze a failure and store self_assessment memory",
		RunE: func(cmd *cobra.Command, args []string) error {
			if whatHappened == "" {
				return fmt.Errorf("llmem: introspect: --what-happened is required")
			}
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			id, err := introspect.IntrospectFailure(context.Background(), ms, introspect.IntrospectFailureParams{
				WhatHappened: whatHappened,
				Category:     categoryVal,
				Context:      contextVal,
				CaughtBy:     caughtByVal,
				ProposedFix:  proposedFix,
			})
			if err != nil {
				return err
			}
			fmt.Printf("Stored self_assessment: %s\n", id)
			return nil
		},
	}
	cmd.Flags().StringVar(&whatHappened, "what-happened", "", "What went wrong")
	cmd.Flags().StringVar(&categoryVal, "category", "", "Error category")
	cmd.Flags().StringVar(&contextVal, "context", "", "Context where it happened")
	cmd.Flags().StringVar(&caughtByVal, "caught-by", "", "How it was caught")
	cmd.Flags().StringVar(&proposedFix, "proposed-fix", "", "Proposed fix")
	cmd.MarkFlagRequired("what-happened")
	return cmd
}

func learnCmd() *cobra.Command {
	var (
		wrongVal  string
		rightVal  string
		contextVal string
	)
	cmd := &cobra.Command{
		Use:   "learn",
		Short: "Learn a lesson from a wrong→right correction",
		RunE: func(cmd *cobra.Command, args []string) error {
			if wrongVal == "" || rightVal == "" {
				return fmt.Errorf("llmem: learn: --wrong and --right are required")
			}
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			id, err := introspect.LearnLesson(context.Background(), ms, introspect.LearnLessonParams{
				WhatWasWrong:  wrongVal,
				WhatIsCorrect: rightVal,
				Context:       contextVal,
			})
			if err != nil {
				return err
			}
			fmt.Printf("Stored procedure: %s\n", id)
			return nil
		},
	}
	cmd.Flags().StringVar(&wrongVal, "wrong", "", "What was wrong")
	cmd.Flags().StringVar(&rightVal, "right", "", "What is correct")
	cmd.Flags().StringVar(&contextVal, "context", "", "Context")
	return cmd
}

func trackReviewCmd() *cobra.Command {
	var (
		singleVal   bool
		batchVal    bool
		cleanVal    bool
		findingsVal string
	)
	cmd := &cobra.Command{
		Use:   "track-review",
		Short: "Persist code review findings as self_assessment memories",
		RunE: func(cmd *cobra.Command, args []string) error {
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			if cleanVal {
				// Invalidate all self_assessment memories with source="track-review"
				// This is a bulk invalidation — fetch and invalidate each
				memories, err := ms.Search(context.Background(), store.SearchParams{
					Type:      "self_assessment",
					ValidOnly: true,
					Limit:    10000,
				})
				if err != nil {
					return err
				}
				count := 0
				for _, m := range memories {
					if m.Source == "track-review" {
						ok, err := ms.Invalidate(context.Background(), m.ID, "track-review clean")
						if err != nil {
							return fmt.Errorf("llmem: track-review: invalidate %s: %w", m.ID, err)
						}
						if ok {
							count++
						}
					}
				}
				fmt.Printf("Invalidated %d track-review memories\n", count)
			}

			if singleVal || batchVal {
				var input []byte
				if findingsVal != "" {
					resolvedFindings, rerr := filepath.Abs(findingsVal)
					if rerr != nil {
						return fmt.Errorf("llmem: track-review: resolve findings path: %w", rerr)
					}
					if paths.IsBlockedPath(resolvedFindings) {
						return fmt.Errorf("llmem: track-review: findings path targets a blocked system directory: %s", resolvedFindings)
					}
					input, err = os.ReadFile(resolvedFindings)
					if err != nil {
						return fmt.Errorf("llmem: track-review: read findings: %w", err)
					}
				} else {
					stat, _ := os.Stdin.Stat()
					if stat.Mode()&os.ModeCharDevice != 0 {
						return fmt.Errorf("llmem: track-review: provide --findings or pipe input")
					}
					input, err = io.ReadAll(os.Stdin)
					if err != nil {
						return fmt.Errorf("llmem: track-review: read stdin: %w", err)
					}
				}

				lines := strings.Split(strings.TrimSpace(string(input)), "\n")
				count := 0
				for _, line := range lines {
					line = strings.TrimSpace(line)
					if line == "" {
						continue
					}
					// Parse "Category: value" lines
					parsed := taxonomy.ParseSelfAssessment(line)
					category := parsed["Category"]
					if category == "" {
						category = "REVIEW_PASSED"
					}
					if _, ok := taxonomy.ErrorTaxonomy[category]; !ok {
						slog.Warn("llmem: track-review: unknown category, proceeding anyway", "category", category)
					}

					id, err := ms.Add(context.Background(), store.AddParams{
						Type:       "self_assessment",
						Content:    line,
						Source:     "track-review",
						Confidence: 0.9,
						Metadata:   map[string]any{"category": category, "source": "track-review"},
					})
					if err != nil {
						slog.Warn("llmem: track-review: failed to store finding", "error", err)
						continue
					}
					count++
					_ = id
				}
				fmt.Printf("Stored %d track-review findings\n", count)
			}

			return nil
		},
	}
	cmd.Flags().BoolVar(&singleVal, "single", false, "Store a single finding")
	cmd.Flags().BoolVar(&batchVal, "batch", false, "Store multiple findings")
	cmd.Flags().BoolVar(&cleanVal, "clean", false, "Invalidate all existing track-review memories")
	cmd.Flags().StringVar(&findingsVal, "findings", "", "Path to findings file (or stdin)")
	return cmd
}

func contextCmd() *cobra.Command {
	var (
		sessionIDVal string
		compacting   bool
	)
	cmd := &cobra.Command{
		Use:   "context",
		Short: "Inject context from session hooks",
		RunE: func(cmd *cobra.Command, args []string) error {
			if sessionIDVal == "" {
				return fmt.Errorf("llmem: context: --session-id is required")
			}
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			coord, err := session.NewSessionHookCoordinator(session.SessionHookConfig{
				Store: ms,
			})
			if err != nil {
				return err
			}

			if compacting {
				resultType, contextPath, err := coord.OnCompacting(context.Background(), sessionIDVal)
				if err != nil {
					return err
				}
				fmt.Printf("Compacting result: %s, context: %s\n", resultType, contextPath)
			} else {
				result, err := coord.OnCreated(context.Background(), sessionIDVal)
				if err != nil {
					return err
				}
				fmt.Printf("Created result: %s\n", result)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&sessionIDVal, "session-id", "", "Session ID")
	cmd.Flags().BoolVar(&compacting, "compacting", false, "Generate context for compacting")
	cmd.MarkFlagRequired("session-id")
	return cmd
}

func hookCmd() *cobra.Command {
	var (
		hookType     string
		sessionIDVal string
	)
	cmd := &cobra.Command{
		Use:   "hook [type] [session-id]",
		Short: "Handle session lifecycle hook events",
		Args:  cobra.MaximumNArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			if hookType == "" && len(args) > 0 {
				hookType = args[0]
			}
			if sessionIDVal == "" && len(args) > 1 {
				sessionIDVal = args[1]
			}
			if hookType == "" {
				return fmt.Errorf("llmem: hook: hook type is required (use --type or positional arg)")
			}
			if sessionIDVal == "" {
				return fmt.Errorf("llmem: hook: session-id is required (use --session-id or positional arg)")
			}
			ms, err := openStore()
			if err != nil {
				return err
			}
			defer ms.Close()

			coord, err := session.NewSessionHookCoordinator(session.SessionHookConfig{
				Store: ms,
			})
			if err != nil {
				return err
			}

			switch hookType {
			case "created":
				result, err := coord.OnCreated(context.Background(), sessionIDVal)
				if err != nil {
					return err
				}
				fmt.Printf("Hook result: %s\n", result)
			case "idle":
				result, err := coord.OnIdle(context.Background(), sessionIDVal)
				if err != nil {
					return err
				}
				fmt.Printf("Hook result: %s\n", result)
			case "compacting":
				resultType, contextPath, err := coord.OnCompacting(context.Background(), sessionIDVal)
				if err != nil {
					return err
				}
				fmt.Printf("Hook result: %s, context: %s\n", resultType, contextPath)
			case "ending":
				result, err := coord.OnEnding(context.Background(), sessionIDVal)
				if err != nil {
					return err
				}
				fmt.Printf("Hook result: %s\n", result)
			default:
				return fmt.Errorf("llmem: hook: unknown hook type %q (must be created, idle, compacting, or ending)", hookType)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&hookType, "type", "", "Hook type: created, idle, compacting, ending")
	cmd.Flags().StringVar(&sessionIDVal, "session-id", "", "Session ID")
	return cmd
}

func defaultIfEmpty(val, defaultVal string) string {
	if val == "" {
		return defaultVal
	}
	return val
}