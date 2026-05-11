// Package config provides configuration loading and writing for LLMem.
// It handles YAML config files with path resolution, defaults, and validation.
package config

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/MichielDean/LLMem/internal/dream"
	"github.com/MichielDean/LLMem/internal/paths"
	"github.com/MichielDean/LLMem/internal/skillpatch"
	"github.com/MichielDean/LLMem/internal/store"
	"github.com/MichielDean/LLMem/internal/urlvalidate"
	"gopkg.in/yaml.v3"
)

// DreamConfig holds dream consolidation settings.
// Only fields that are wired through to DreamerConfig are included.
// Removed dead fields: MinScore, MinRecallCount, MinUniqueQueries,
// BoostOnPromote, MergeModel, CalibrationEnabled,
// CalibrationLookbackDays, Enabled, Schedule — these were defined in
// config but never read by any method, creating a contract violation.
// Enabled and Schedule control systemd timer behaviour, not dream
// algorithm parameters; they are handled by internal/systemd directly.
type DreamConfig struct {
	SimilarityThreshold    float64 `yaml:"similarity_threshold"`
	DecayRate              float64 `yaml:"decay_rate"`
	DecayIntervalDays      int     `yaml:"decay_interval_days"`
	DecayFloor             float64 `yaml:"decay_floor"`
	ConfidenceFloor        float64 `yaml:"confidence_floor"`
	BoostThreshold         int     `yaml:"boost_threshold"`
	BoostAmount            float64 `yaml:"boost_amount"`
	DiaryPath              string  `yaml:"diary_path"`
	ReportPath             string  `yaml:"report_path"`
	BehavioralThreshold    int     `yaml:"behavioral_threshold"`
	BehavioralLookbackDays int     `yaml:"behavioral_lookback_days"`
	AutoLinkThreshold      float64 `yaml:"auto_link_threshold"`
	StaleProcedureDays     int     `yaml:"stale_procedure_days"`
	OllamaURL              string  `yaml:"ollama_url"`
	Model                  string  `yaml:"model"`
	// ModelTimeout is the timeout for each LLM call during REM behavioral insight generation.
	// Parsed as a Go duration string (e.g. "5m", "120s"). Defaults to "5m".
	ModelTimeout        string `yaml:"model_timeout"`
	ProposedChangesPath string  `yaml:"proposed_changes_path"`
	// SkillPatchDir is the root directory for skill files. Defaults to paths.GetSkillDir() if empty.
	SkillPatchDir       string  `yaml:"skill_patch_dir"`
}

// SessionConfig holds session lifecycle settings.
type SessionConfig struct {
	Adapter         string `yaml:"adapter"`
	DebounceSeconds int   `yaml:"debounce_seconds"`
}

// OpenCodeConfig holds OpenCode adapter settings.
type OpenCodeConfig struct {
	DBPath      string `yaml:"db_path"`
	ContextDir  string `yaml:"context_dir"`
}

// SkillPatchConfig holds skill patch settings.
type SkillPatchConfig struct {
	// Dir is the root directory for skill files. Defaults to paths.GetSkillDir() if empty.
	Dir string `yaml:"dir"`
}

// Config holds the full LLMem configuration.
type Config struct {
	Memory    MemoryConfig    `yaml:"memory"`
	Dream     DreamConfig     `yaml:"dream"`
	SkillPatch SkillPatchConfig `yaml:"skill_patch"`
	OpenCode  OpenCodeConfig  `yaml:"opencode"`
	Session   SessionConfig   `yaml:"session"`
}

// MemoryConfig holds memory store settings.
type MemoryConfig struct {
	DBPath            string `yaml:"db"`
	OllamaURL         string `yaml:"ollama_url"`
	EmbedModel        string `yaml:"embed_model"`
	ExtractModel      string `yaml:"extract_model"`
	ContextBudget     int    `yaml:"context_budget"`
	AutoExtract       bool   `yaml:"auto_extract"`
	MaxFileSize       int64  `yaml:"max_file_size"`
	// CallModelTimeout is the timeout for LLM calls in introspect/learn.
	// Parsed as a Go duration string (e.g. "5m", "120s"). Defaults to "5m".
	CallModelTimeout string `yaml:"call_model_timeout"`
}

// fmtErr wraps an error with the "llmem: config:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: config: "+format, args...)
}

// DefaultConfig returns a Config with all defaults applied.
func DefaultConfig() Config {
	return Config{
		Memory: MemoryConfig{
			DBPath:             paths.GetDBPath(),
			OllamaURL:          "http://localhost:11434",
			EmbedModel:         "nomic-embed-text",
			ExtractModel:       "glm-5.1:cloud",
			ContextBudget:      4000,
			AutoExtract:        true,
			MaxFileSize:        10 * 1024 * 1024,
			CallModelTimeout:   "5m",
		},
		Dream: DreamConfig{
			SimilarityThreshold:    0.92,
			DecayRate:              0.05,
			DecayIntervalDays:      30,
			DecayFloor:             0.3,
			ConfidenceFloor:         0.3,
			BoostThreshold:         5,
			BoostAmount:             0.05,
			DiaryPath:              paths.GetDreamDiaryPath(),
			ReportPath:             paths.GetDreamReportPath(),
			BehavioralThreshold:    3,
			BehavioralLookbackDays: 30,
			AutoLinkThreshold:      0.85,
			StaleProcedureDays:     30,
			ModelTimeout:           "5m",
		},
		SkillPatch: SkillPatchConfig{
			Dir: paths.GetSkillDir(),
		},
		OpenCode: OpenCodeConfig{
			DBPath:      openCodeDefaultDBPath(),
			ContextDir:  paths.GetContextDir(),
		},
		Session: SessionConfig{
			Adapter:          "opencode",
			DebounceSeconds:  30,
		},
	}
}

// LoadConfig loads configuration from a YAML file.
// If the file doesn't exist, returns default config.
// If the file exists but is malformed, returns an error.
func LoadConfig(configPath string) (*Config, error) {
	if configPath == "" {
		configPath = paths.GetConfigPath()
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			cfg := DefaultConfig()
			return &cfg, nil
		}
		return nil, fmtErr("read config %s: %w", configPath, err)
	}

	cfg := DefaultConfig()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		slog.Warn("llmem: config: failed to parse config file, using defaults", "path", configPath, "error", err)
		return &cfg, nil
	}

	// Apply path resolution for empty DB path
	if cfg.Memory.DBPath == "" {
		cfg.Memory.DBPath = paths.GetDBPath()
	}

	return &cfg, nil
}

// DBPath returns the resolved database path.
func (c *Config) DBPath() string {
	if c.Memory.DBPath != "" {
		return expandHome(c.Memory.DBPath)
	}
	return paths.GetDBPath()
}

// OllamaURL returns the validated Ollama URL.
func (c *Config) OllamaURL() (string, error) {
	url := c.Memory.OllamaURL
	if url == "" {
		url = "http://localhost:11434"
	}
	validated, err := urlvalidate.ValidateBaseURL(url, "config")
	if err != nil {
		return "", fmtErr("invalid Ollama URL: %w", err)
	}
	return validated, nil
}

// CallModelTimeoutDuration parses the CallModelTimeout config value as a time.Duration.
// Returns the default of 5 minutes if the value is empty or invalid, logging a warning.
func (c *Config) CallModelTimeoutDuration() time.Duration {
	if c.Memory.CallModelTimeout == "" {
		return 5 * time.Minute
	}
	parsed, err := time.ParseDuration(c.Memory.CallModelTimeout)
	if err != nil {
		slog.Warn("llmem: config: invalid call_model_timeout, using default 5m", "value", c.Memory.CallModelTimeout, "error", err)
		return 5 * time.Minute
	}
	return parsed
}

// DreamerConfig returns a dream.DreamerConfig populated from the config.
// Maps DreamConfig fields to their corresponding DreamerConfig fields.
// Store must be set by the caller before passing to dream.NewDreamer.
// If OllamaURL is configured, it is passed as BaseURL so Dreamer can attempt
// to create an OllamaClient. If OllamaClient creation fails inside
// dream.NewDreamer, behavioral insights fall back to count-based summaries.
func (c *Config) DreamerConfig() dream.DreamerConfig {
	ollamaURL := c.Dream.OllamaURL
	if ollamaURL == "" {
		ollamaURL = c.Memory.OllamaURL
	}
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}

	model := c.Dream.Model
	if model == "" {
		model = c.Memory.ExtractModel
	}
	if model == "" {
		model = "glm-5.1:cloud"
	}

	// Parse dream model timeout from config
	var modelTimeout time.Duration
	if c.Dream.ModelTimeout != "" {
		parsed, err := time.ParseDuration(c.Dream.ModelTimeout)
		if err != nil {
			slog.Warn("llmem: config: invalid dream model_timeout, using default 5m", "value", c.Dream.ModelTimeout, "error", err)
			modelTimeout = 5 * time.Minute
		} else {
			modelTimeout = parsed
		}
	}

	return dream.DreamerConfig{
		SimilarityThreshold:    c.Dream.SimilarityThreshold,
		DecayRate:              c.Dream.DecayRate,
		DecayIntervalDays:      c.Dream.DecayIntervalDays,
		DecayFloor:             c.Dream.DecayFloor,
		ConfidenceFloor:         c.Dream.ConfidenceFloor,
		BoostThreshold:         c.Dream.BoostThreshold,
		BoostAmount:            c.Dream.BoostAmount,
		AutoLinkThreshold:      c.Dream.AutoLinkThreshold,
		BehavioralThreshold:    c.Dream.BehavioralThreshold,
		BehavioralLookbackDays: c.Dream.BehavioralLookbackDays,
		StaleProcedureDays:     c.Dream.StaleProcedureDays,
		DiaryPath:              c.Dream.DiaryPath,
		ReportPath:             c.Dream.ReportPath,
		ProposedChangesPath:    c.Dream.ProposedChangesPath,
		BaseURL:                ollamaURL,
		Model:                  model,
		ModelTimeout:           modelTimeout,
	}
}

// DreamConfigResolved returns the fully-resolved dream configuration.
// Deprecated: use DreamerConfig() which returns a dream.DreamerConfig
// wired to the actual Dreamer implementation.
func (c *Config) DreamConfigResolved() DreamConfig {
	return c.Dream
}

// SessionConfigResolved returns the session configuration.
func (c *Config) SessionConfigResolved() SessionConfig {
	return c.Session
}

// NewSkillPatcher creates a SkillPatcher using the SkillPatch config and the given store.
// Returns nil without error if patching should be skipped (graceful degradation for callers).
// The caller provides the store since it's the same store used for dreaming.
func (c *Config) NewSkillPatcher(ms *store.MemoryStore) (*skillpatch.SkillPatcher, error) {
	if ms == nil {
		return nil, nil
	}

	skillDir := c.SkillPatch.Dir
	if skillDir == "" {
		skillDir = paths.GetSkillDir()
	}

	return skillpatch.NewSkillPatcher(skillpatch.SkillPatchConfig{
		SkillDir: skillDir,
		Store:    ms,
	})
}

// WriteConfigYAML writes config as YAML to the given path with 0600 permissions.
// Creates parent directories with 0700 permissions.
// Returns false if file exists and force is false.
func WriteConfigYAML(path string, config map[string]any, force bool) (bool, error) {
	if !force {
		if _, err := os.Stat(path); err == nil {
			slog.Info("llmem: config: file already exists, skipping", "path", path)
			return false, nil
		}
	}

	// Create parent directory with 0700 permissions
	dir := paths.GetDirFromPath(path)
	if dir != "" {
		if err := os.MkdirAll(dir, 0700); err != nil {
			return false, fmtErr("create config directory: %w", err)
		}
	}

	data, err := yaml.Marshal(config)
	if err != nil {
		return false, fmtErr("marshal config: %w", err)
	}

	// Write with 0600 permissions to protect API keys
	if err := os.WriteFile(path, data, 0600); err != nil {
		return false, fmtErr("write config: %w", err)
	}

	return true, nil
}

// openCodeDefaultDBPath returns the default OpenCode database path using filepath.Join.
func openCodeDefaultDBPath() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(paths.GetHomeDir(), "..", ".local", "share", "opencode", "opencode.db")
	}
	return filepath.Join(homeDir, ".local", "share", "opencode", "opencode.db")
}

// expandHome expands ~ to the user's home directory.
func expandHome(pathStr string) string {
	if strings.HasPrefix(pathStr, "~/") {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return pathStr
		}
		return filepath.Join(homeDir, pathStr[2:])
	}
	return pathStr
}