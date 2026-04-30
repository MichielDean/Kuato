# LLMem Configuration

Configuration reference for paths, providers, and the dream cycle. [Back to README](../README.md)

## General Configuration

LLMem looks for configuration at `~/.config/llmem/config.yaml`. If this file doesn't exist, sensible defaults are used.

### Path Resolution

| Path | Default | Override |
|------|---------|----------|
| Home directory | `~/.config/llmem/` | `LMEM_HOME` env var |
| Database | `~/.config/llmem/memory.db` | `config.yaml: memory.db` |
| Config file | `~/.config/llmem/config.yaml` | — |
| Dream diary | `~/.config/llmem/dream-diary.md` | `config.yaml: dream.diary_path` |
| Proposed changes | `~/.config/llmem/proposed-changes.md` | `config.yaml: dream.proposed_changes_path` |

**Backward compatibility:** If `~/.lobsterdog/` exists and `~/.config/llmem/` doesn't, LLMem will use the legacy path. Call `migrate_from_lobsterdog()` to copy data to the new location.

**`LMEM_HOME` env var:** Set this to override the home directory entirely. The path is validated against directory traversal, system directories, and symlink attacks.

### config.yaml Reference

```yaml
memory:
  context_budget: 4000
  auto_extract: true
  max_file_size: 10485760     # 10MB
  inbox_capacity: 7            # Miller's 7±2; items above this count are evicted

dream:
  enabled: true
  schedule: "*-*-* 03:00:00"
  similarity_threshold: 0.92
  decay_rate: 0.05
  decay_interval_days: 30
  decay_floor: 0.3
  confidence_floor: 0.3
  boost_threshold: 5
  boost_amount: 0.05
  min_score: 0.5
  min_recall_count: 3
  min_unique_queries: 1
  boost_on_promote: 0.1
  merge_model: qwen2.5:1.5b
  diary_path: null            # Auto-resolved from get_dream_diary_path()
  report_path: null              # Auto-resolved from get_dream_report_path()
  behavioral_threshold: 3
  behavioral_lookback_days: 30
  proposed_changes_path: null # Auto-resolved from get_proposed_changes_path()
  calibration_enabled: true
  stale_procedure_days: 30
  calibration_lookback_days: 90
  auto_link_threshold: 0.85  # Cosine similarity threshold for auto-linking related memories

opencode:
  context_dir: null           # Auto-resolved from get_context_dir()
  db_path: ~/.local/share/opencode/opencode.db

correction_detection:
  enabled: true
```