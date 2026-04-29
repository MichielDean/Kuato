"""Tests for llmem init command, ProviderDetector, is_ollama_running, and write_config_yaml."""

import argparse
import os
import stat
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from llmem.config import write_config_yaml
from llmem.ollama import is_ollama_running, ProviderDetector


# ---------------------------------------------------------------------------
# Helper: create an argparse Namespace matching what cmd_init expects
# ---------------------------------------------------------------------------
def _make_args(**overrides):
    """Build an argparse Namespace with sensible defaults for cmd_init."""
    defaults = {
        "ollama_url": None,
        "non_interactive": True,
        "force": False,
        "db": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ===========================================================================
# is_ollama_running
# ===========================================================================


class TestOllama_IsOllamaRunning:
    """Test is_ollama_running() probes and validation."""

    def test_returns_true_when_ollama_responds(self):
        """When Ollama responds with a models list, return True."""
        fake_data = '{"models": [{"name": "nomic-embed-text"}]}'
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_data.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("llmem.url_validate.safe_urlopen", return_value=mock_resp):
            result = is_ollama_running("http://localhost:11434")
        assert result is True

    def test_returns_false_when_ollama_unreachable(self):
        """When Ollama is unreachable, return False."""
        from urllib.error import URLError

        with patch(
            "llmem.url_validate.safe_urlopen",
            side_effect=URLError("connection refused"),
        ):
            result = is_ollama_running("http://localhost:11434")
        assert result is False

    def test_returns_false_on_invalid_json(self):
        """When Ollama returns invalid JSON, return False."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("llmem.url_validate.safe_urlopen", return_value=mock_resp):
            result = is_ollama_running("http://localhost:11434")
        assert result is False

    def test_returns_false_when_no_models_key(self):
        """When Ollama responds but JSON lacks 'models' key, return False."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"status": "ok"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("llmem.url_validate.safe_urlopen", return_value=mock_resp):
            result = is_ollama_running("http://localhost:11434")
        assert result is False

    def test_raises_value_error_for_empty_url(self):
        """Empty base_url raises ValueError."""
        with pytest.raises(ValueError, match="invalid base_url"):
            is_ollama_running("")

    def test_raises_value_error_for_unsafe_url(self):
        """Unsafe URL (e.g. file://) raises ValueError."""
        with pytest.raises(ValueError, match="invalid base_url|unsafe URL"):
            is_ollama_running("file:///etc/passwd")


# ===========================================================================
# ProviderDetector
# ===========================================================================


class TestOllama_ProviderDetector:
    """Test ProviderDetector.detect() classification logic."""

    def test_ollama_detected_when_running(self):
        """When Ollama is reachable, provider should be 'ollama'."""
        detector = ProviderDetector()
        with patch("llmem.ollama.is_ollama_running", return_value=True):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("ANTHROPIC_API_KEY", None)
                result = detector.detect()
        assert result["provider"] == "ollama"

    def test_ollama_not_detected_when_unavailable(self):
        """When no provider is available, provider should be 'none'."""
        detector = ProviderDetector()
        with patch("llmem.ollama.is_ollama_running", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("ANTHROPIC_API_KEY", None)
                result = detector.detect()
        assert result["provider"] == "none"

    def test_openai_api_key_detected(self):
        """When OPENAI_API_KEY is set, provider should be 'openai'."""
        detector = ProviderDetector()
        with patch("llmem.ollama.is_ollama_running", return_value=False):
            with patch.dict(
                os.environ, {"OPENAI_API_KEY": "sk-test-key-123"}, clear=False
            ):
                result = detector.detect()
        assert result["provider"] == "openai"
    def test_anthropic_api_key_detected(self):
        """When ANTHROPIC_API_KEY is set (and no OPENAI), provider should be 'anthropic'."""
        detector = ProviderDetector()
        with patch("llmem.ollama.is_ollama_running", return_value=False):
            env = {"ANTHROPIC_API_KEY": "sk-ant-test-key"}
            with patch.dict(os.environ, env, clear=False):
                # Remove OPENAI key if present
                os.environ.pop("OPENAI_API_KEY", None)
                result = detector.detect()
        assert result["provider"] == "anthropic"
    def test_ollama_takes_precedence_over_api_keys(self):
        """When Ollama is running, provider should be 'ollama' even if API keys exist."""
        detector = ProviderDetector()
        with patch("llmem.ollama.is_ollama_running", return_value=True):
            env = {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-ant-test"}
            with patch.dict(os.environ, env, clear=False):
                result = detector.detect()
        assert result["provider"] == "ollama"

    def test_openai_takes_precedence_over_anthropic(self):
        """When both keys are set (but no Ollama), OpenAI should win."""
        detector = ProviderDetector()
        with patch("llmem.ollama.is_ollama_running", return_value=False):
            env = {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-ant-test"}
            with patch.dict(os.environ, env, clear=False):
                result = detector.detect()
        assert result["provider"] == "openai"

    def test_ollama_url_override(self):
        """Custom Ollama URL is reflected in detection results."""
        detector = ProviderDetector()
        with patch("llmem.ollama.is_ollama_running", return_value=True) as mock_check:
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("ANTHROPIC_API_KEY", None)
                result = detector.detect(ollama_url="http://my-server:11434")
        assert result["ollama_url"] == "http://my-server:11434"
        mock_check.assert_called_once_with("http://my-server:11434")

    def test_detect_returns_all_keys(self):
        """Detection results always contain required keys."""
        detector = ProviderDetector()
        with patch("llmem.ollama.is_ollama_running", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("ANTHROPIC_API_KEY", None)
                result = detector.detect()
        assert "provider" in result
        assert "ollama_url" in result

# ===========================================================================
# write_config_yaml
# ===========================================================================


class TestConfig_WriteConfigYaml:
    """Test write_config_yaml() file creation, idempotency, and permissions."""

    def test_creates_config_yaml(self, tmp_path):
        """write_config_yaml creates a YAML file at the specified path."""
        config_path = tmp_path / "llmem" / "config.yaml"
        config = {"memory": {"ollama_url": "http://localhost:11434"}}
        result = write_config_yaml(config_path, config)
        assert result is True
        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert data["memory"]["ollama_url"] == "http://localhost:11434"

    def test_writes_human_readable_yaml(self, tmp_path):
        """YAML output uses default_flow_style=False for readability."""
        config_path = tmp_path / "config.yaml"
        config = {
            "memory": {
                "ollama_url": "http://localhost:11434",
                "embed_model": "nomic-embed-text",
            }
        }
        write_config_yaml(config_path, config)
        content = config_path.read_text()
        # Block style means no inline flow like {ollama_url: ...}
        assert "ollama_url:" in content

    def test_returns_false_when_file_exists_no_force(self, tmp_path):
        """If config.yaml exists and force=False, return False without overwriting."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("original: true")
        result = write_config_yaml(config_path, {"new": True}, force=False)
        assert result is False
        data = yaml.safe_load(config_path.read_text())
        assert data == {"original": True}

    def test_force_overwrites_existing_config(self, tmp_path):
        """With force=True, overwrite existing config.yaml."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("original: true")
        result = write_config_yaml(config_path, {"new": True}, force=True)
        assert result is True
        data = yaml.safe_load(config_path.read_text())
        assert data == {"new": True}

    def test_creates_parent_directories(self, tmp_path):
        """write_config_yaml creates parent directories as needed."""
        config_path = tmp_path / "deep" / "nested" / "config.yaml"
        write_config_yaml(config_path, {"memory": {"db": "/tmp/test.db"}})
        assert config_path.exists()

    def test_directory_permissions_0700(self, tmp_path):
        """Parent directory is created with 0o700 permissions."""
        deep_dir = tmp_path / "perm_test"
        config_path = deep_dir / "config.yaml"
        write_config_yaml(config_path, {"memory": {}})
        mode = stat.S_IMODE(os.stat(str(deep_dir)).st_mode)
        assert mode == 0o700

    def test_config_yaml_contains_expected_keys(self, tmp_path):
        """Written config.yaml has memory.ollama_url, embed_model, extract_model."""
        config_path = tmp_path / "config.yaml"
        config = {
            "memory": {
                "ollama_url": "http://localhost:11434",
                "embed_model": "nomic-embed-text",
                "extract_model": "qwen2.5:1.5b",
            },
            "dream": {"enabled": True},
        }
        write_config_yaml(config_path, config)
        data = yaml.safe_load(config_path.read_text())
        assert "ollama_url" in data["memory"]
        assert "embed_model" in data["memory"]
        assert "extract_model" in data["memory"]


# ===========================================================================
# cmd_init — CLI handler
# ===========================================================================


class TestInit_CmdInit:
    """Test cmd_init() — the llmem init CLI handler."""

    def test_creates_config_yaml(self, tmp_path):
        """cmd_init creates config.yaml in the llmem home directory."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "ollama",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        config_path = home / "config.yaml"
        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert data["memory"]["ollama_url"] == "http://localhost:11434"

    def test_creates_memory_db(self, tmp_path):
        """cmd_init creates memory.db with _schema_migrations entries."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        db_path = home / "memory.db"
        assert db_path.exists()
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute('SELECT "version" FROM "_schema_migrations"').fetchall()
        conn.close()
        assert len(rows) > 0, "memory.db should have _schema_migrations entries"

    def test_config_yaml_defaults(self, tmp_path):
        """Written config.yaml contains memory.ollama_url, embed_model, extract_model."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "ollama",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        config_path = home / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert "ollama_url" in data["memory"]
        assert "embed_model" in data["memory"]
        assert "extract_model" in data["memory"]

    def test_idempotent(self, tmp_path):
        """Calling init twice without --force succeeds (no error, no overwrite)."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "ollama",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)
                        # Second call should not fail
                        cmd_init(args)

        # Config should still exist and not be duplicated
        config_path = home / "config.yaml"
        assert config_path.exists()

    def test_force_overwrites_existing_config(self, tmp_path):
        """With --force, existing config.yaml is overwritten."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        config_path = home / "config.yaml"

        # First init
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=config_path):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        # Write initial config
                        home.mkdir(parents=True, exist_ok=True)
                        write_config_yaml(config_path, {"old": True})

                        # Force overwrite
                        args = _make_args(non_interactive=True, force=True)
                        cmd_init(args)

        data = yaml.safe_load(config_path.read_text())
        assert "old" not in data
        assert "memory" in data

    def test_no_force_preserves_existing_config(self, tmp_path):
        """Without --force, existing config.yaml is NOT overwritten."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        config_path = home / "config.yaml"

        # Write initial config
        home.mkdir(parents=True, exist_ok=True)
        write_config_yaml(config_path, {"old_config": True}, force=True)

        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=config_path):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True, force=False)
                        cmd_init(args)

        data = yaml.safe_load(config_path.read_text())
        assert "old_config" in data

    def test_directory_permissions_0700(self, tmp_path):
        """The llmem home directory is created with 0o700 permissions."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_perm_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        assert home.exists()
        mode = stat.S_IMODE(os.stat(str(home)).st_mode)
        assert mode == 0o700

    def test_ollama_detected_when_running(self, tmp_path):
        """When ProviderDetector detects Ollama, config has provider: ollama."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "ollama",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        config_path = home / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert data["memory"]["provider"] == "ollama"

    def test_ollama_not_detected_when_unavailable(self, tmp_path):
        """When no provider is detected, config has no provider key (or 'none' value)."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        config_path = home / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        # When provider is "none", it should not be in the config
        assert (
            data["memory"].get("provider", "none") == "none"
            or "provider" not in data["memory"]
        )

    def test_openai_api_key_detected(self, tmp_path):
        """When ProviderDetector detects OpenAI, config has provider: openai."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "openai",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        config_path = home / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert data["memory"]["provider"] == "openai"

    def test_anthropic_api_key_detected(self, tmp_path):
        """When ProviderDetector detects Anthropic, config has provider: anthropic."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "anthropic",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        config_path = home / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert data["memory"]["provider"] == "anthropic"

    def test_ollama_url_override(self, tmp_path):
        """--ollama-url flag overrides the default Ollama URL in config."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "ollama",
                            "ollama_url": "http://my-remote:11434",
                                                                                }
                        args = _make_args(
                            non_interactive=True, ollama_url="http://my-remote:11434"
                        )
                        cmd_init(args)

        config_path = home / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert data["memory"]["ollama_url"] == "http://my-remote:11434"

    def test_llmem_home_env_override(self, tmp_path):
        """LMEM_HOME env var changes the init target directory."""
        from llmem.cli import cmd_init

        home = tmp_path / "custom_llmem"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        assert (home / "config.yaml").exists()
        assert (home / "memory.db").exists()

    def test_non_interactive_mode(self, tmp_path):
        """--non-interactive skips prompts and uses defaults."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "ollama",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        # No input() should be called — this must not hang
                        cmd_init(args)

        config_path = home / "config.yaml"
        assert config_path.exists()

    def test_interactive_mode_keyboard_interrupt(self, tmp_path):
        """Ctrl+C during interactive init prints cancel message and exits."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=False)
                        with patch("builtins.input", side_effect=KeyboardInterrupt):
                            with pytest.raises(SystemExit):
                                cmd_init(args)

    def test_init_does_not_create_identity_or_skills(self, tmp_path):
        """Init does NOT create identity files, skills, or agents."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        # Verify ONLY config.yaml and memory.db exist in home
        files_in_home = [p.name for p in home.iterdir()]
        assert "config.yaml" in files_in_home
        assert "memory.db" in files_in_home
        assert "identity.md" not in files_in_home
        assert "rules.md" not in files_in_home
        assert "user.md" not in files_in_home
        assert "skills" not in files_in_home
        assert "agents" not in files_in_home

    def test_detect_value_error_exits_cleanly(self, tmp_path):
        """cmd_init exits with code 1 when detector.detect() raises ValueError."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.side_effect = ValueError(
                            "llmem: ollama: unsafe URL: 'ftp://bad'"
                        )
                        args = _make_args(non_interactive=True, ollama_url="ftp://bad")
                        with pytest.raises(SystemExit) as exc_info:
                            cmd_init(args)
                        assert exc_info.value.code == 1

    def test_detect_empty_url_exits_cleanly(self, tmp_path):
        """cmd_init exits with code 1 when --ollama-url is empty string."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.side_effect = ValueError(
                            "llmem: ollama: invalid base_url"
                        )
                        args = _make_args(non_interactive=True, ollama_url="")
                        with pytest.raises(SystemExit) as exc_info:
                            cmd_init(args)
                        assert exc_info.value.code == 1

    def test_interactive_unsafe_url_rejected(self, tmp_path):
        """Interactive-mode URL input that fails is_safe_url() is rejected."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        # User types an unsafe URL (file://) in interactive mode
                        args = _make_args(non_interactive=False)
                        with patch("builtins.input", return_value="file:///etc/passwd"):
                            with pytest.raises(SystemExit) as exc_info:
                                cmd_init(args)
                            assert exc_info.value.code == 1

    def test_interactive_non_http_url_rejected(self, tmp_path):
        """Interactive-mode URL input without http(s) scheme is rejected."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        # User types ftp:// URL in interactive mode
                        args = _make_args(non_interactive=False)
                        with patch(
                            "builtins.input", return_value="ftp://evil.server:11434"
                        ):
                            with pytest.raises(SystemExit) as exc_info:
                                cmd_init(args)
                            assert exc_info.value.code == 1

    def test_interactive_valid_url_accepted(self, tmp_path):
        """Interactive-mode URL input with a valid http(s) URL is accepted."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        custom_url = "http://my-ollama-server:11434"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "ollama",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        # User types a valid URL, then accepts default for dream
                        args = _make_args(non_interactive=False)
                        with patch("builtins.input", side_effect=[custom_url, ""]):
                            cmd_init(args)

        config_path = home / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert data["memory"]["ollama_url"] == custom_url


# ===========================================================================
# Integration tests
# ===========================================================================


class TestInit_Integration:
    """End-to-end integration tests for llmem init."""

    def test_init_then_add_memory(self, tmp_path):
        """After init, llmem add --type fact --content 'test' works."""
        from llmem.cli import cmd_init, cmd_add
        from llmem.store import MemoryStore, register_memory_type

        home = tmp_path / "llmem_home"
        config_path = home / "config.yaml"
        db_path = home / "memory.db"

        # Init
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=config_path):
                with patch("llmem.cli.get_db_path", return_value=db_path):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        init_args = _make_args(non_interactive=True)
                        cmd_init(init_args)

        assert db_path.exists()

        # Add a memory
        add_args = argparse.Namespace(
            db=db_path,
            type="fact",
            content="test integration memory",
            file=None,
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )
        # Patch MemoryStore to use disable_vec
        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_add(add_args)

    def test_init_creates_expected_artifacts(self, tmp_path):
        """On successful execution, config.yaml, memory.db, and directory with 0o700 exist."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "ollama",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        args = _make_args(non_interactive=True)
                        cmd_init(args)

        # (1) config.yaml exists with expected keys
        config_path = home / "config.yaml"
        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert "ollama_url" in data["memory"]
        assert "embed_model" in data["memory"]
        assert "extract_model" in data["memory"]

        # (2) memory.db exists with _schema_migrations
        db_path = home / "memory.db"
        assert db_path.exists()
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute('SELECT "version" FROM "_schema_migrations"').fetchall()
        conn.close()
        versions = {r[0] for r in rows}
        assert 1 in versions

        # (3) home directory has 0o700 permissions
        mode = stat.S_IMODE(os.stat(str(home)).st_mode)
        assert mode == 0o700

    def test_migrate_from_lobsterdog_then_init(self, tmp_path):
        """If ~/.lobsterdog/ exists, init should detect and attempt migration."""
        from llmem.cli import cmd_init

        home = tmp_path / "llmem_home"
        old_home = tmp_path / ".lobsterdog"
        old_home.mkdir()
        (old_home / "config.yaml").write_text("memory: {}")

        with patch("llmem.cli.get_llmem_home", return_value=home):
            with patch("llmem.cli.get_config_path", return_value=home / "config.yaml"):
                with patch("llmem.cli.get_db_path", return_value=home / "memory.db"):
                    with patch("llmem.cli.ProviderDetector") as MockDetector:
                        MockDetector.return_value.detect.return_value = {
                            "provider": "none",
                            "ollama_url": "http://localhost:11434",
                                                                                }
                        # Patch Path.home() for migrate_from_lobsterdog
                        with patch("llmem.cli.Path") as MockPath:
                            MockPath.home.return_value = tmp_path
                            MockPath.side_effect = lambda *a, **kw: (
                                Path(*a, **kw) if a else Path()
                            )
                            # Also patch paths in migrate_from_lobsterdog
                            with patch("llmem.paths.Path") as MockPathInner:
                                MockPathInner.home.return_value = tmp_path
                                MockPathInner.side_effect = lambda *a, **kw: (
                                    Path(*a, **kw) if a else Path()
                                )
                                args = _make_args(non_interactive=True)
                                cmd_init(args)

        # After init, the home dir should have config and db
        assert (home / "config.yaml").exists()
        assert (home / "memory.db").exists()


# ===========================================================================
# Serialization boundary test — ProviderDetector.detect returns no None values
# ===========================================================================


class TestOllama_ProviderDetector_Serialization:
    """Ensure ProviderDetector.detect() returns serializable data with no null values."""

    def test_detect_result_no_null_values(self):
        """Detection result dict has no None/null values for string fields."""
        detector = ProviderDetector()
        with patch("llmem.ollama.is_ollama_running", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                # Ensure env vars are absent
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("ANTHROPIC_API_KEY", None)
                result = detector.detect()

        # All values should be strings, no None
        for key, val in result.items():
            assert val is not None, f"{key} should not be None"
            assert isinstance(val, str), f"{key} should be a string, got {type(val)}"
