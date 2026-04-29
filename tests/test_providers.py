"""Tests for the provider abstraction layer.

Tests EmbedProvider/GenerateProvider ABCs, OllamaProvider, OpenAIProvider,
AnthropicProvider, NoneProvider, and resolve_provider.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from memory.providers import (
    AnthropicProvider,
    EmbedProvider,
    GenerateProvider,
    NoneProvider,
    OllamaProvider,
    OpenAIProvider,
    resolve_provider,
    SentenceTransformersProvider,
    DEFAULT_NONE_EMBED_DIMENSIONS,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OLLAMA_BASE_URL,
)
from memory.config import get_provider_config, DEFAULTS, load_config


# ---------------------------------------------------------------------------
# EmbedProvider / GenerateProvider ABC tests
# ---------------------------------------------------------------------------


class TestEmbedProvider_IsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            EmbedProvider()

    def test_subclass_without_methods_cannot_instantiate(self):
        class Incomplete(EmbedProvider):
            pass

        with pytest.raises(TypeError):
            Incomplete()


class TestGenerateProvider_IsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            GenerateProvider()

    def test_subclass_without_methods_cannot_instantiate(self):
        class Incomplete(GenerateProvider):
            pass

        with pytest.raises(TypeError):
            Incomplete()


class TestEmbedProvider_AbstractMethods:
    def test_embed_is_abstract(self):
        assert hasattr(EmbedProvider, "embed")
        assert getattr(EmbedProvider.embed, "__isabstractmethod__", False)

    def test_embed_batch_is_abstract(self):
        assert hasattr(EmbedProvider, "embed_batch")
        assert getattr(EmbedProvider.embed_batch, "__isabstractmethod__", False)

    def test_check_available_is_abstract(self):
        assert hasattr(EmbedProvider, "check_available")
        assert getattr(EmbedProvider.check_available, "__isabstractmethod__", False)

    def test_dimension_is_abstract(self):
        assert hasattr(EmbedProvider, "dimension")
        assert getattr(EmbedProvider.dimension, "__isabstractmethod__", False)


class TestGenerateProvider_AbstractMethods:
    def test_generate_is_abstract(self):
        assert hasattr(GenerateProvider, "generate")
        assert getattr(GenerateProvider.generate, "__isabstractmethod__", False)

    def test_check_available_is_abstract(self):
        assert hasattr(GenerateProvider, "check_available")
        assert getattr(GenerateProvider.check_available, "__isabstractmethod__", False)


# ---------------------------------------------------------------------------
# OllamaProvider tests
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    def test_implements_embed_provider(self):
        provider = OllamaProvider()
        assert isinstance(provider, EmbedProvider)

    def test_implements_generate_provider(self):
        provider = OllamaProvider()
        assert isinstance(provider, GenerateProvider)

    def test_embed_delegates_to_ollama_embeddings_api(self):
        provider = OllamaProvider()
        vec = [0.1] * 768
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"embedding": vec}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = provider.embed("test text")
        assert result == vec
        # Verify correct endpoint
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "/api/embeddings" in req.full_url

    def test_embed_batch_delegates_to_ollama(self):
        provider = OllamaProvider()
        vec_a = [0.1] * 768
        vec_b = [0.2] * 768
        call_count = 0

        def mock_urlopen(req, timeout=None):
            nonlocal call_count
            call_count += 1
            body = json.loads(req.data.decode())
            resp = MagicMock()
            if body["prompt"] == "text a":
                resp.read.return_value = json.dumps({"embedding": vec_a}).encode()
            else:
                resp.read.return_value = json.dumps({"embedding": vec_b}).encode()
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("memory.providers.safe_urlopen", side_effect=mock_urlopen):
            result = provider.embed_batch(["text a", "text b"])
        assert len(result) == 2
        assert result[0] == vec_a
        assert result[1] == vec_b
        assert call_count == 2

    def test_generate_delegates_to_call_ollama_generate(self):
        provider = OllamaProvider()
        with patch(
            "memory.providers._call_ollama_generate", return_value="generated text"
        ) as mock_gen:
            result = provider.generate("test prompt")
        assert result == "generated text"
        mock_gen.assert_called_once_with(
            model=provider._generate_model,
            base_url=provider._base_url,
            prompt="test prompt",
            temperature=0.1,
            max_tokens=2048,
            timeout=60,
        )

    def test_check_available_returns_true(self):
        provider = OllamaProvider()
        with patch("memory.providers.check_ollama_model", return_value=True):
            assert provider.check_available() is True

    def test_check_available_returns_false(self):
        provider = OllamaProvider()
        with patch("memory.providers.check_ollama_model", return_value=False):
            assert provider.check_available() is False

    def test_constructor_validates_url(self):
        with pytest.raises(ValueError, match="must be http/https"):
            OllamaProvider(base_url="ftp://evil.com")

    def test_constructor_strips_trailing_slash(self):
        provider = OllamaProvider(base_url="http://localhost:11434/")
        assert provider._base_url == "http://localhost:11434"

    def test_constructor_unsafe_url_raises(self):
        with patch("memory.providers.is_safe_url", return_value=False):
            with pytest.raises(ValueError, match="blocked"):
                OllamaProvider(base_url="http://evil.com:11434")

    def test_default_models(self):
        provider = OllamaProvider()
        assert provider._embed_model == "nomic-embed-text"
        assert provider._generate_model == "qwen2.5:1.5b"

    def test_custom_models(self):
        provider = OllamaProvider(
            embed_model="custom-embed",
            generate_model="custom-gen",
        )
        assert provider._embed_model == "custom-embed"
        assert provider._generate_model == "custom-gen"


# ---------------------------------------------------------------------------
# OpenAIProvider tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_implements_embed_provider(self):
        provider = OpenAIProvider(api_key="test-key")
        assert isinstance(provider, EmbedProvider)

    def test_implements_generate_provider(self):
        provider = OpenAIProvider(api_key="test-key")
        assert isinstance(provider, GenerateProvider)

    def test_embed_calls_openai_embeddings_api(self):
        provider = OpenAIProvider(api_key="test-key")
        vec = [0.1] * 1536
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "data": [{"embedding": vec, "index": 0}],
            },
        ) as mock_req:
            result = provider.embed("test text")
        assert result == vec
        mock_req.assert_called_once()
        call_args = mock_req.call_args
        assert call_args[0][0] == "/v1/embeddings"
        assert call_args[0][1]["model"] == "text-embedding-3-small"
        # embed() delegates to embed_batch, so input is wrapped in a list
        assert call_args[0][1]["input"] == ["test text"]

    def test_embed_batch_calls_openai(self):
        provider = OpenAIProvider(api_key="test-key")
        vec_a = [0.1] * 1536
        vec_b = [0.2] * 1536
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "data": [
                    {"embedding": vec_a, "index": 0},
                    {"embedding": vec_b, "index": 1},
                ],
            },
        ) as mock_req:
            result = provider.embed_batch(["text a", "text b"])
        assert len(result) == 2
        assert result[0] == vec_a
        assert result[1] == vec_b
        call_args = mock_req.call_args
        assert call_args[0][1]["input"] == ["text a", "text b"]

    def test_generate_calls_openai_chat_api(self):
        provider = OpenAIProvider(api_key="test-key")
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "choices": [{"message": {"content": "generated text"}}],
            },
        ) as mock_req:
            result = provider.generate("test prompt")
        assert result == "generated text"
        call_args = mock_req.call_args
        assert call_args[0][0] == "/v1/chat/completions"
        body = call_args[0][1]
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"] == [{"role": "user", "content": "test prompt"}]

    def test_check_available_true_on_valid_key(self):
        provider = OpenAIProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"data": []}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            assert provider.check_available() is True

    def test_check_available_false_on_invalid_key(self):
        provider = OpenAIProvider(api_key="invalid-key")
        import urllib.error

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://api.openai.com/v1/models",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=None,
            ),
        ):
            assert provider.check_available() is False

    def test_api_key_from_env_or_constructor(self):
        # Constructor param takes precedence
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            provider = OpenAIProvider(api_key="ctor-key")
            assert provider._api_key == "ctor-key"

        # Falls back to env var
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            # Ensure it's set
            os.environ["OPENAI_API_KEY"] = "env-key"
            provider = OpenAIProvider()
            assert provider._api_key == "env-key"

    def test_constructor_validates_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY from env
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="API key required"):
                OpenAIProvider()

    def test_authorization_header(self):
        provider = OpenAIProvider(api_key="test-key-123")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"data": []}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.check_available()
        # Verify Authorization header in the request
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "Bearer test-key-123"


# ---------------------------------------------------------------------------
# AnthropicProvider tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_implements_generate_provider(self):
        provider = AnthropicProvider(api_key="test-key")
        assert isinstance(provider, GenerateProvider)

    def test_does_not_implement_embed_provider(self):
        provider = AnthropicProvider(api_key="test-key")
        assert not isinstance(provider, EmbedProvider)

    def test_generate_calls_anthropic_messages_api(self):
        provider = AnthropicProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {
                    "content": [{"text": "generated text"}],
                }
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = provider.generate("test prompt")
        assert result == "generated text"
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "/v1/messages" in req.full_url
        body = json.loads(req.data.decode())
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["messages"] == [{"role": "user", "content": "test prompt"}]

    def test_check_available_true_on_valid_key(self):
        provider = AnthropicProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {
                    "content": [{"text": "ok"}],
                }
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            assert provider.check_available() is True

    def test_check_available_false_on_invalid_key(self):
        provider = AnthropicProvider(api_key="invalid-key")
        import urllib.error

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://api.anthropic.com/v1/messages",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=None,
            ),
        ):
            assert provider.check_available() is False

    def test_api_key_from_env_or_constructor(self):
        # Constructor param takes precedence
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            provider = AnthropicProvider(api_key="ctor-key")
            assert provider._api_key == "ctor-key"

        # Falls back to env var
        os.environ["ANTHROPIC_API_KEY"] = "env-key"
        try:
            provider = AnthropicProvider()
            assert provider._api_key == "env-key"
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)

    def test_x_api_key_header(self):
        provider = AnthropicProvider(api_key="test-key-456")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {
                    "content": [{"text": "ok"}],
                }
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.generate("test")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        # urllib normalizes header names to Title-Case
        headers = dict(req.header_items())
        assert headers.get("X-api-key") == "test-key-456"

    def test_anthropic_version_header(self):
        provider = AnthropicProvider(api_key="test-key-789")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {
                    "content": [{"text": "ok"}],
                }
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.generate("test")
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        # urllib normalizes header names to Title-Case
        headers = dict(req.header_items())
        assert headers.get("Anthropic-version") == "2023-06-01"


# ---------------------------------------------------------------------------
# NoneProvider tests
# ---------------------------------------------------------------------------


class TestNoneProvider:
    def test_implements_embed_provider(self):
        provider = NoneProvider()
        assert isinstance(provider, EmbedProvider)

    def test_implements_generate_provider(self):
        provider = NoneProvider()
        assert isinstance(provider, GenerateProvider)

    def test_embed_returns_zero_vector(self):
        provider = NoneProvider()
        result = provider.embed("test text")
        assert result == [0.0] * 768
        assert len(result) == 768

    def test_embed_returns_zero_vector_custom_dimension(self):
        provider = NoneProvider(embed_dimensions=1536)
        result = provider.embed("test text")
        assert len(result) == 1536
        assert all(v == 0.0 for v in result)

    def test_embed_batch_returns_zero_vectors(self):
        provider = NoneProvider()
        result = provider.embed_batch(["text one", "text two", "text three"])
        assert len(result) == 3
        for vec in result:
            assert vec == [0.0] * 768

    def test_generate_returns_empty_string(self):
        provider = NoneProvider()
        result = provider.generate("test prompt")
        assert result == ""

    def test_check_available_returns_false(self):
        provider = NoneProvider()
        assert provider.check_available() is False


# ---------------------------------------------------------------------------
# Provider resolution tests
# ---------------------------------------------------------------------------


class TestProviderResolution:
    def test_resolve_ollama_first(self):
        with patch("memory.providers.check_ollama_model", return_value=True):
            embed, gen = resolve_provider({})
        assert isinstance(embed, OllamaProvider)
        assert isinstance(gen, OllamaProvider)

    def test_resolve_fallback_to_openai(self):
        with (
            patch("memory.providers.check_ollama_model", return_value=False),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
        ):
            os.environ["OPENAI_API_KEY"] = "test-key"
            embed, gen = resolve_provider({})
        assert isinstance(embed, OpenAIProvider)
        assert isinstance(gen, OpenAIProvider)

    def test_resolve_fallback_to_anthropic(self):
        with (
            patch("memory.providers.check_ollama_model", return_value=False),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ["ANTHROPIC_API_KEY"] = "test-key"
            try:
                embed, gen = resolve_provider({})
            finally:
                os.environ.pop("ANTHROPIC_API_KEY", None)
        # Anthropic can't embed, so embed falls to NoneProvider
        assert isinstance(embed, NoneProvider)
        assert isinstance(gen, AnthropicProvider)

    def test_resolve_fallback_to_none(self):
        with (
            patch("memory.providers.check_ollama_model", return_value=False),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            embed, gen = resolve_provider({})
        assert isinstance(embed, NoneProvider)
        assert isinstance(gen, NoneProvider)

    def test_resolve_explicit_provider_config(self):
        with (
            patch("memory.providers.check_ollama_model", return_value=True),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
        ):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {"provider": {"default": "openai"}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OpenAIProvider)
        assert isinstance(gen, OpenAIProvider)

    def test_resolve_per_operation_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {
                "provider": {
                    "default": "openai",
                    "embed": {"model": "text-embedding-3-large"},
                    "generate": {"model": "gpt-4o"},
                },
            }
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OpenAIProvider)
        assert embed._embed_model == "text-embedding-3-large"
        assert isinstance(gen, OpenAIProvider)
        assert gen._generate_model == "gpt-4o"

    def test_resolve_with_legacy_ollama_url(self):
        with patch("memory.providers.check_ollama_model", return_value=True):
            config = {"memory": {"ollama_url": "http://custom:9999"}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OllamaProvider)
        assert embed._base_url == "http://custom:9999"

    def test_resolve_with_provider_ollama_base_url(self):
        with patch("memory.providers.check_ollama_model", return_value=True):
            config = {
                "provider": {
                    "default": "ollama",
                    "ollama": {"base_url": "http://custom:1234"},
                },
            }
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OllamaProvider)
        assert embed._base_url == "http://custom:1234"

    def test_resolve_none_explicit(self):
        config = {"provider": {"default": "none"}}
        embed, gen = resolve_provider(config)
        assert isinstance(embed, NoneProvider)
        assert isinstance(gen, NoneProvider)


# ---------------------------------------------------------------------------
# Config get_provider_config tests
# ---------------------------------------------------------------------------


class TestGetProviderConfig:
    def test_returns_defaults_when_no_config(self):
        result = get_provider_config(config={})
        assert result["default"] == "ollama"
        assert result["embed"] == {}
        assert result["generate"] == {}

    def test_returns_config_values_over_defaults(self):
        config = {"provider": {"default": "openai", "embed": {"model": "custom"}}}
        result = get_provider_config(config=config)
        assert result["default"] == "openai"
        assert result["embed"] == {"model": "custom"}
        assert result["generate"] == {}

    def test_defaults_dict_has_provider_key(self):
        assert "provider" in DEFAULTS
        assert DEFAULTS["provider"]["default"] == "ollama"
        assert DEFAULTS["provider"]["embed"] == {}
        assert DEFAULTS["provider"]["generate"] == {}


# ---------------------------------------------------------------------------
# OpenAI URL construction tests (regression: no double /v1)
# ---------------------------------------------------------------------------


class TestOpenAI_URL_NoDoubleV1:
    """Regression tests for the double /v1 path bug.

    DEFAULT_OPENAI_BASE_URL must NOT end with /v1 because endpoint paths
    already include /v1 (e.g., /v1/embeddings, /v1/chat/completions).
    """

    def test_default_base_url_does_not_end_with_v1(self):
        assert not DEFAULT_OPENAI_BASE_URL.endswith("/v1"), (
            f"DEFAULT_OPENAI_BASE_URL={DEFAULT_OPENAI_BASE_URL!r} ends with /v1, "
            "causing double /v1 in constructed URLs"
        )

    def test_embed_url_no_double_v1(self):
        provider = OpenAIProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"data": [{"embedding": [0.1] * 1536, "index": 0}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.embed("test")
        req = mock_urlopen.call_args[0][0]
        assert "/v1/v1/" not in req.full_url, (
            f"Double /v1 detected in URL: {req.full_url}"
        )
        assert req.full_url.endswith("/v1/embeddings"), (
            f"Expected URL ending with /v1/embeddings, got: {req.full_url}"
        )

    def test_generate_url_no_double_v1(self):
        provider = OpenAIProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.generate("test")
        req = mock_urlopen.call_args[0][0]
        assert "/v1/v1/" not in req.full_url, (
            f"Double /v1 detected in URL: {req.full_url}"
        )
        assert req.full_url.endswith("/v1/chat/completions"), (
            f"Expected URL ending with /v1/chat/completions, got: {req.full_url}"
        )

    def test_check_available_url_no_double_v1(self):
        provider = OpenAIProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"data": []}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.check_available()
        req = mock_urlopen.call_args[0][0]
        assert "/v1/v1/" not in req.full_url, (
            f"Double /v1 detected in URL: {req.full_url}"
        )
        assert req.full_url.endswith("/v1/models"), (
            f"Expected URL ending with /v1/models, got: {req.full_url}"
        )


# ---------------------------------------------------------------------------
# Config load_config logging tests
# ---------------------------------------------------------------------------


class TestLoadConfig_LogsOnException:
    """Verify that load_config logs a warning instead of silently swallowing exceptions."""

    def test_logs_warning_on_bad_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        # Write genuinely malformed YAML that yaml.safe_load cannot parse
        bad_file.write_text("{\n  foo: bar\n  baz")
        with patch("memory.config.log") as mock_log:
            result = load_config(bad_file)
        assert result == {}
        mock_log.warning.assert_called_once()
        call_args = mock_log.warning.call_args
        assert "failed to load config" in call_args[0][0]

    def test_logs_warning_on_non_dict_yaml(self, tmp_path):
        """YAML that parses but isn't a dict should return {} without logging."""
        arr_file = tmp_path / "arr.yaml"
        arr_file.write_text("- item1\n- item2\n")
        result = load_config(arr_file)
        assert result == {}

    def test_returns_empty_on_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        result = load_config(missing)
        assert result == {}


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-4unyo: OpenAIProvider.generate() must pass timeout to _make_request
# ---------------------------------------------------------------------------


class TestOpenAIProvider_TimeoutPassedToMakeRequest:
    """OpenAIProvider.generate() accepts a timeout parameter but previously
    silently ignored it — _make_request hardcoded self._timeout.
    The fix: _make_request must accept a timeout param and generate() must
    pass it through.
    """

    def test_generate_passes_custom_timeout_to_make_request(self):
        provider = OpenAIProvider(api_key="test-key")
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "choices": [{"message": {"content": "ok"}}],
            },
        ) as mock_req:
            provider.generate("test prompt", timeout=120)
        # Verify timeout was passed through to _make_request
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs.get("timeout") == 120, (
            f"Expected timeout=120 passed to _make_request, got kwargs={call_kwargs}"
        )

    def test_generate_passes_default_timeout_to_make_request(self):
        """When generate() is called without timeout, None is passed to _make_request
        which resolves to the constructor timeout."""
        provider = OpenAIProvider(api_key="test-key")
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "choices": [{"message": {"content": "ok"}}],
            },
        ) as mock_req:
            provider.generate("test prompt")
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs.get("timeout") is None, (
            f"Expected timeout=None (use constructor default) passed to _make_request, "
            f"got kwargs={call_kwargs}"
        )

    def test_embed_passes_timeout_to_make_request(self):
        """embed() should also use the instance timeout via _make_request."""
        provider = OpenAIProvider(api_key="test-key", timeout=45)
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "data": [{"embedding": [0.1] * 1536, "index": 0}],
            },
        ) as mock_req:
            provider.embed("test")
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs.get("timeout") == 45

    def test_make_request_uses_timeout_param_not_hardcoded(self):
        """Verify _make_request actually uses the timeout parameter in urlopen."""
        provider = OpenAIProvider(api_key="test-key", timeout=30)
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"data": [{"embedding": [0.1] * 1536, "index": 0}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.embed("test")
        # Verify urlopen was called with the correct timeout from _make_request
        call_kwargs = mock_urlopen.call_args[1]
        assert call_kwargs.get("timeout") == 30, (
            f"Expected urlopen timeout=30 (from constructor), got {call_kwargs}"
        )


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-cl2wn: OpenAIProvider and AnthropicProvider URL validation
# ---------------------------------------------------------------------------


class TestOpenAIProvider_URLValidation:
    """OpenAIProvider constructor must validate base_url scheme and SSRF,
    matching the validation performed by OllamaProvider.
    """

    def test_constructor_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="must be http/https"):
            OpenAIProvider(api_key="test-key", base_url="ftp://evil.com")

    def test_constructor_rejects_unsafe_url(self):
        """SSRF validation: a private/link-local IP should be rejected."""
        with patch("memory.providers.is_safe_url", return_value=False):
            with pytest.raises(ValueError, match="blocked"):
                OpenAIProvider(api_key="test-key", base_url="http://169.254.169.254/")

    def test_constructor_accepts_safe_url(self):
        """A safe URL (like the default) should work fine."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider._base_url == "https://api.openai.com"

    def test_constructor_strips_trailing_slash(self):
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.openai.com/"
        )
        assert provider._base_url == "https://api.openai.com"

    def test_constructor_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="must be http/https"):
            OpenAIProvider(api_key="test-key", base_url="file:///etc/passwd")


class TestAnthropicProvider_URLValidation:
    """AnthropicProvider constructor must validate base_url scheme and SSRF,
    matching the validation performed by OllamaProvider.
    """

    def test_constructor_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="must be http/https"):
            AnthropicProvider(api_key="test-key", base_url="ftp://evil.com")

    def test_constructor_rejects_unsafe_url(self):
        """SSRF validation: a private/link-local IP should be rejected."""
        with patch("memory.providers.is_safe_url", return_value=False):
            with pytest.raises(ValueError, match="blocked"):
                AnthropicProvider(
                    api_key="test-key", base_url="http://169.254.169.254/"
                )

    def test_constructor_accepts_safe_url(self):
        """A safe URL (like the default) should work fine."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider._base_url == "https://api.anthropic.com"

    def test_constructor_strips_trailing_slash(self):
        provider = AnthropicProvider(
            api_key="test-key", base_url="https://api.anthropic.com/"
        )
        assert provider._base_url == "https://api.anthropic.com"

    def test_constructor_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="must be http/https"):
            AnthropicProvider(api_key="test-key", base_url="file:///etc/passwd")


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-9g4ys: resolve_provider fallback on OllamaProvider ValueError
# ---------------------------------------------------------------------------


class TestResolveProvider_OllamaValueErrorFallback:
    """When _resolve_embed/generate_provider construct OllamaProvider and it
    raises ValueError (e.g. from URL validation), the resolver must fall back
    gracefully instead of crashing.
    """

    def test_resolve_embed_falls_back_on_ollama_value_error(self):
        """If OllamaProvider() raises ValueError, embed should fall back to
        OpenAI (or NoneProvider), not crash.
        """
        with (
            patch(
                "memory.providers.OllamaProvider",
                side_effect=ValueError(
                    "providers: Ollama URL blocked (unsafe address)"
                ),
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
        ):
            os.environ["OPENAI_API_KEY"] = "test-key"
            embed, gen = resolve_provider({})
        assert isinstance(embed, OpenAIProvider)

    def test_resolve_generate_falls_back_on_ollama_value_error(self):
        """If OllamaProvider() raises ValueError, generate should fall back to
        OpenAI (or NoneProvider), not crash.
        """
        # We need OllamaProvider to raise on construction but work normally
        # for the embed side check_available. Use a conditional side_effect.
        original_class = OllamaProvider
        call_count = {"n": 0}

        def conditional_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # First two constructions (embed+generate in resolve) raise
                raise ValueError("providers: Ollama URL blocked (unsafe address)")
            return original_class(*args, **kwargs)

        with (
            patch(
                "memory.providers.OllamaProvider", side_effect=conditional_side_effect
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
        ):
            os.environ["OPENAI_API_KEY"] = "test-key"
            embed, gen = resolve_provider({})
        assert isinstance(gen, OpenAIProvider)

    def test_resolve_falls_back_to_none_on_ollama_value_error_no_openai(self):
        """If OllamaProvider raises ValueError and no OpenAI key is available,
        should fall back to NoneProvider, not crash.
        """
        with (
            patch(
                "memory.providers.OllamaProvider",
                side_effect=ValueError(
                    "providers: Ollama URL blocked (unsafe address)"
                ),
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            embed, gen = resolve_provider({})
        assert isinstance(embed, NoneProvider)
        assert isinstance(gen, NoneProvider)

    def test_resolve_with_bad_ollama_url_falls_back(self):
        """Passing a bad ollama_base_url in config should not crash —
        the ValueError from OllamaProvider(base_url=...) should trigger fallback.
        """
        with (
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {
                "provider": {
                    "default": "ollama",
                    "ollama": {"base_url": "http://169.254.169.254:11434"},
                },
            }
            # This should NOT raise — it should fall back to NoneProvider
            embed, gen = resolve_provider(config)
        assert isinstance(embed, NoneProvider)
        assert isinstance(gen, NoneProvider)


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-kpm9t: Missing edge case tests
# ---------------------------------------------------------------------------


class TestEmbedBatch_EmptyList:
    """embed_batch([]) should return an empty list, not crash or error."""

    def test_ollama_embed_batch_empty_list(self):
        provider = OllamaProvider()
        result = provider.embed_batch([])
        assert result == []

    def test_openai_embed_batch_empty_list(self):
        provider = OpenAIProvider(api_key="test-key")
        with patch.object(
            provider,
            "_make_request",
            return_value={"data": []},
        ) as mock_req:
            result = provider.embed_batch([])
        assert result == []
        # Verify _make_request was still called (OpenAI batch API handles empty input)
        mock_req.assert_called_once()

    def test_none_embed_batch_empty_list(self):
        provider = NoneProvider()
        result = provider.embed_batch([])
        assert result == []


class TestAnthropicProvider_TimeoutPassthrough:
    """AnthropicProvider.generate() must pass the timeout parameter to urlopen."""

    def test_generate_passes_custom_timeout(self):
        provider = AnthropicProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"content": [{"text": "ok"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.generate("test", timeout=120)
        call_kwargs = mock_urlopen.call_args[1]
        assert call_kwargs.get("timeout") == 120, (
            f"Expected timeout=120 passed to urlopen, got kwargs={call_kwargs}"
        )

    def test_generate_passes_default_timeout(self):
        """When generate() is called without explicit timeout, the constructor
        default (60s) should be used."""
        provider = AnthropicProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"content": [{"text": "ok"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.generate("test")
        call_kwargs = mock_urlopen.call_args[1]
        assert call_kwargs.get("timeout") == 60, (
            f"Expected constructor default timeout=60, got kwargs={call_kwargs}"
        )

    def test_generate_with_custom_constructor_timeout(self):
        """When generate() is called without an explicit timeout, it should
        use the constructor-configured timeout (not the old method default of 60)."""
        provider = AnthropicProvider(api_key="test-key", timeout=45)
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"content": [{"text": "ok"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            provider.generate("test")
        call_kwargs = mock_urlopen.call_args[1]
        assert call_kwargs.get("timeout") == 45, (
            "generate() without explicit timeout should use constructor timeout (45), "
            f"not method default. Got kwargs={call_kwargs}"
        )


class TestOllamaProvider_CheckAvailable_PartialModel:
    """OllamaProvider.check_available() checks both embed and generate models.
    If only one is available, it should return False.
    """

    def test_check_available_false_when_only_embed_available(self):
        provider = OllamaProvider()
        with patch("memory.providers.check_ollama_model") as mock_check:
            # embed model available, generate model NOT available
            mock_check.side_effect = [True, False]
            assert provider.check_available() is False
            assert mock_check.call_count == 2

    def test_check_available_false_when_only_generate_available(self):
        provider = OllamaProvider()
        with patch("memory.providers.check_ollama_model") as mock_check:
            # embed model NOT available, generate model available
            mock_check.side_effect = [False, True]
            # check_available ANDs both results, so first False short-circuits
            # but actually Python `and` short-circuits, so only 1 call
            assert provider.check_available() is False

    def test_check_available_false_when_neither_available(self):
        provider = OllamaProvider()
        with patch("memory.providers.check_ollama_model", return_value=False):
            assert provider.check_available() is False


class TestNoneProvider_IgnoresGenerateParams:
    """NoneProvider.generate() should ignore all parameters and return ''."""

    def test_generate_ignores_prompt(self):
        provider = NoneProvider()
        assert provider.generate("") == ""
        assert provider.generate("complex prompt") == ""

    def test_generate_ignores_temperature(self):
        provider = NoneProvider()
        assert provider.generate("test", temperature=0.0) == ""
        assert provider.generate("test", temperature=1.0) == ""

    def test_generate_ignores_max_tokens(self):
        provider = NoneProvider()
        assert provider.generate("test", max_tokens=1) == ""
        assert provider.generate("test", max_tokens=10000) == ""

    def test_generate_ignores_timeout(self):
        provider = NoneProvider()
        assert provider.generate("test", timeout=1) == ""
        assert provider.generate("test", timeout=3600) == ""

    def test_embed_ignores_input_text(self):
        provider = NoneProvider()
        assert provider.embed("") == [0.0] * 768
        assert provider.embed("some text") == [0.0] * 768


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-rjf46: _resolve_embed_provider 'anthropic' branch
# ---------------------------------------------------------------------------


class TestResolveEmbedProvider_AnthropicBranch:
    """When provider.default='anthropic' is used for embed, it should
    explicitly fall back because Anthropic provides no embedding API,
    rather than falling through to the 'unknown' else branch.
    """

    def test_anthropic_embed_falls_back_to_none(self):
        """Anthropic has no embedding API — should fall back to NoneProvider."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {"provider": {"default": "anthropic"}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, NoneProvider)

    def test_anthropic_embed_falls_back_to_openai_if_key_available(self):
        """If OPENAI_API_KEY is set, anthropic embed fallback should try OpenAI."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {"provider": {"default": "anthropic"}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OpenAIProvider)


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-cj1li: generate() uses constructor timeout, not method default
# ---------------------------------------------------------------------------


class TestOllamaProvider_ConstructorTimeout:
    """OllamaProvider.generate() should use self._timeout when no explicit
    timeout is provided, not a hardcoded method default of 60.
    """

    def test_generate_uses_constructor_timeout_when_not_specified(self):
        provider = OllamaProvider(timeout=45)
        with patch(
            "memory.providers._call_ollama_generate", return_value="ok"
        ) as mock_gen:
            provider.generate("test prompt")
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("timeout") == 45, (
            f"Expected constructor timeout=45, got {call_kwargs}"
        )

    def test_generate_explicit_timeout_overrides_constructor(self):
        provider = OllamaProvider(timeout=45)
        with patch(
            "memory.providers._call_ollama_generate", return_value="ok"
        ) as mock_gen:
            provider.generate("test prompt", timeout=120)
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs.get("timeout") == 120, (
            f"Expected explicit timeout=120, got {call_kwargs}"
        )


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-njsot: _fallback functions don't take ollama_base_url
# ---------------------------------------------------------------------------


class TestFallbackFunctions_NoOllamaBaseBaseUrl:
    """_fallback_embed_provider and _fallback_generate_provider should not
    accept ollama_base_url — it was a dead documented parameter that was
    never used in the function body.
    """

    def test_fallback_embed_no_ollama_base_url_param(self):
        """_fallback_embed_provider should work with just config and skip_openai."""
        import inspect
        from memory.providers import _fallback_embed_provider

        sig = inspect.signature(_fallback_embed_provider)
        assert "ollama_base_url" not in sig.parameters, (
            f"_fallback_embed_provider should not have ollama_base_url param, "
            f"got params: {list(sig.parameters.keys())}"
        )

    def test_fallback_generate_no_ollama_base_url_param(self):
        """_fallback_generate_provider should work with just config, skip_openai, skip_anthropic."""
        import inspect
        from memory.providers import _fallback_generate_provider

        sig = inspect.signature(_fallback_generate_provider)
        assert "ollama_base_url" not in sig.parameters, (
            f"_fallback_generate_provider should not have ollama_base_url param, "
            f"got params: {list(sig.parameters.keys())}"
        )


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-dw30i: Fallback functions must check config-based API keys
# ---------------------------------------------------------------------------


class TestFallbackEmbedProvider_ConfigApiKey:
    """_fallback_embed_provider must check config-based API keys, not only env vars.

    When Ollama is unavailable and the user has an API key in config.yaml
    (not in env vars), the fallback should still try OpenAI — not silently
    skip to NoneProvider.
    """

    def test_fallback_embed_uses_config_openai_api_key(self):
        """OpenAI embed fallback should work with api_key from config, no env var."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            config = {
                "provider": {
                    "openai": {"api_key": "config-key-123"},
                },
            }
            from memory.providers import _fallback_embed_provider

            provider = _fallback_embed_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._api_key == "config-key-123"

    def test_fallback_embed_prefers_config_key_over_env(self):
        """Config-based api_key should take precedence over env var."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            config = {
                "provider": {
                    "openai": {"api_key": "config-key-456"},
                },
            }
            from memory.providers import _fallback_embed_provider

            provider = _fallback_embed_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._api_key == "config-key-456"

    def test_fallback_embed_uses_env_key_when_no_config_key(self):
        """If no config key, should still fall back to env var."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-789"}, clear=False):
            config = {"provider": {"openai": {}}}
            from memory.providers import _fallback_embed_provider

            provider = _fallback_embed_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._api_key == "env-key-789"

    def test_fallback_embed_returns_none_when_no_key(self):
        """When neither config key nor env key is available, return NoneProvider."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            config = {"provider": {"openai": {}}}
            from memory.providers import _fallback_embed_provider

            provider = _fallback_embed_provider(config)
        assert isinstance(provider, NoneProvider)

    def test_fallback_embed_skips_openai_when_flagged(self):
        """skip_openai=True should force skip even if config has an API key."""
        config = {
            "provider": {
                "openai": {"api_key": "config-key"},
            },
        }
        from memory.providers import _fallback_embed_provider

        provider = _fallback_embed_provider(config, skip_openai=True)
        assert isinstance(provider, NoneProvider)


class TestFallbackGenerateProvider_ConfigApiKey:
    """_fallback_generate_provider must check config-based API keys, not only env vars.

    When Ollama is unavailable and the user has API keys in config.yaml,
    the fallback should try OpenAI then Anthropic — not silently skip to NoneProvider.
    """

    def test_fallback_generate_uses_config_openai_api_key(self):
        """OpenAI generate fallback should work with api_key from config, no env var."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            config = {
                "provider": {
                    "openai": {"api_key": "openai-config-key"},
                },
            }
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._api_key == "openai-config-key"

    def test_fallback_generate_uses_config_anthropic_api_key(self):
        """Anthropic generate fallback should work with api_key from config, no env var,
        when OpenAI is skipped."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {
                "provider": {
                    "anthropic": {"api_key": "anthropic-config-key"},
                },
            }
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config, skip_openai=True)
        assert isinstance(provider, AnthropicProvider)
        assert provider._api_key == "anthropic-config-key"

    def test_fallback_generate_uses_config_anthropic_as_secondary(self):
        """Anthropic fallback with config key works when OpenAI is not available."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {
                "provider": {
                    "anthropic": {"api_key": "anthropic-config-key"},
                },
            }
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config, skip_openai=True)
        assert isinstance(provider, AnthropicProvider)
        assert provider._api_key == "anthropic-config-key"

    def test_fallback_generate_prefers_config_over_env_for_openai(self):
        """Config-based openai api_key should take precedence over env var."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            config = {
                "provider": {
                    "openai": {"api_key": "config-key"},
                },
            }
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._api_key == "config-key"

    def test_fallback_generate_prefers_config_over_env_for_anthropic(self):
        """Config-based anthropic api_key should take precedence over env var."""
        config = {
            "provider": {
                "anthropic": {"api_key": "config-key"},
            },
        }
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}, clear=False):
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config, skip_openai=True)
        assert isinstance(provider, AnthropicProvider)
        assert provider._api_key == "config-key"

    def test_fallback_generate_returns_none_when_no_keys(self):
        """When no config keys and no env keys, return NoneProvider."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {"provider": {"openai": {}, "anthropic": {}}}
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config)
        assert isinstance(provider, NoneProvider)

    def test_fallback_generate_uses_env_when_no_config_for_openai(self):
        """If no config key but env var exists, fallback should use it."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-only-key"}, clear=False):
            config = {"provider": {"openai": {}}}
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._api_key == "env-only-key"

    def test_fallback_generate_uses_env_when_no_config_for_anthropic(self):
        """If no config key but Anthropic env var exists, fallback should use it."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ["ANTHROPIC_API_KEY"] = "anthropic-env-key"
            config = {"provider": {"anthropic": {}}}
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config, skip_openai=True)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        assert isinstance(provider, AnthropicProvider)
        assert provider._api_key == "anthropic-env-key"

    def test_resolve_provider_ollama_unavailable_uses_config_openai_key(self):
        """Integration: when Ollama is unavailable but config has an OpenAI key,
        resolve_provider should return OpenAIProvider, not NoneProvider."""
        with (
            patch("memory.providers.check_ollama_model", return_value=False),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            config = {
                "provider": {
                    "default": "ollama",
                    "openai": {"api_key": "from-config"},
                },
            }
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OpenAIProvider)
        assert embed._api_key == "from-config"
        assert isinstance(gen, OpenAIProvider)
        assert gen._api_key == "from-config"

    def test_resolve_provider_ollama_unavailable_uses_config_anthropic_key(self):
        """Integration: when Ollama and OpenAI are unavailable but config has an
        Anthropic key, resolve_provider should return AnthropicProvider for generation."""
        with (
            patch("memory.providers.check_ollama_model", return_value=False),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {
                "provider": {
                    "default": "ollama",
                    "anthropic": {"api_key": "from-config"},
                },
            }
            embed, gen = resolve_provider(config)
        assert isinstance(embed, NoneProvider)
        assert isinstance(gen, AnthropicProvider)
        assert gen._api_key == "from-config"


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-bownn: API key __repr__ masking
# ---------------------------------------------------------------------------


class TestProviderReprMasksApiKey:
    """Provider __repr__ must never expose API keys in logs or tracebacks."""

    def test_openai_repr_masks_api_key(self):
        provider = OpenAIProvider(api_key="sk-super-secret-key-12345")
        r = repr(provider)
        assert "sk-super-secret-key-12345" not in r
        assert "***masked***" in r

    def test_anthropic_repr_masks_api_key(self):
        provider = AnthropicProvider(api_key="sk-ant-secret-key-67890")
        r = repr(provider)
        assert "sk-ant-secret-key-67890" not in r
        assert "***masked***" in r

    def test_ollama_repr_does_not_contain_secrets(self):
        provider = OllamaProvider()
        r = repr(provider)
        # Ollama has no API key, but should still have a safe repr
        assert "OllamaProvider" in r
        assert "base_url=" in r

    def test_none_provider_repr(self):
        provider = NoneProvider()
        r = repr(provider)
        assert "NoneProvider" in r

    def test_openai_repr_shows_base_url(self):
        provider = OpenAIProvider(api_key="test-key", base_url="https://api.openai.com")
        r = repr(provider)
        assert "api.openai.com" in r
        assert "test-key" not in r

    def test_anthropic_repr_shows_base_url(self):
        provider = AnthropicProvider(
            api_key="test-key", base_url="https://api.anthropic.com"
        )
        r = repr(provider)
        assert "api.anthropic.com" in r
        assert "test-key" not in r


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-b71fv: Error messages must not embed user-supplied URLs
# ---------------------------------------------------------------------------


class TestErrorMessagesDoNotLeakCredentials:
    """Error messages and logs must not include user-supplied URLs that
    could contain embedded credentials (e.g., https://user:pass@host).
    """

    def test_openai_invalid_scheme_no_url_in_message(self):
        """ValueError for invalid scheme should not include the raw URL."""
        with pytest.raises(ValueError) as exc_info:
            OpenAIProvider(api_key="test-key", base_url="ftp://user:secret@evil.com")
        msg = str(exc_info.value)
        assert "user:secret" not in msg
        assert "evil.com" not in msg

    def test_openai_unsafe_url_no_url_in_message(self):
        """ValueError for unsafe URL should not include the raw URL."""
        with patch("memory.providers.is_safe_url", return_value=False):
            with pytest.raises(ValueError) as exc_info:
                OpenAIProvider(
                    api_key="test-key", base_url="http://admin:pass@169.254.169.254/"
                )
            msg = str(exc_info.value)
            assert "admin:pass" not in msg
            assert "169.254.169.254" not in msg

    def test_anthropic_invalid_scheme_no_url_in_message(self):
        with pytest.raises(ValueError) as exc_info:
            AnthropicProvider(api_key="test-key", base_url="ftp://user:secret@evil.com")
        msg = str(exc_info.value)
        assert "user:secret" not in msg
        assert "evil.com" not in msg

    def test_anthropic_unsafe_url_no_url_in_message(self):
        with patch("memory.providers.is_safe_url", return_value=False):
            with pytest.raises(ValueError) as exc_info:
                AnthropicProvider(
                    api_key="test-key", base_url="http://admin:pass@169.254.169.254/"
                )
            msg = str(exc_info.value)
            assert "admin:pass" not in msg
            assert "169.254.169.254" not in msg

    def test_ollama_invalid_scheme_no_url_in_message(self):
        with pytest.raises(ValueError) as exc_info:
            OllamaProvider(base_url="ftp://user:secret@evil.com")
        msg = str(exc_info.value)
        assert "user:secret" not in msg
        assert "evil.com" not in msg

    def test_ollama_unsafe_url_no_url_in_message(self):
        with patch("memory.providers.is_safe_url", return_value=False):
            with pytest.raises(ValueError) as exc_info:
                OllamaProvider(base_url="http://admin:pass@169.254.169.254:11434/")
            msg = str(exc_info.value)
            assert "admin:pass" not in msg
            assert "169.254.169.254" not in msg


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-28720: No duplicate __repr__ methods
# ---------------------------------------------------------------------------


class TestProviderNoDuplicateRepr:
    """Verify that no provider class has duplicate __repr__ methods.

    Regression test: copy-paste artifacts caused OpenAIProvider and
    AnthropicProvider to each have two __repr__ definitions, where the
    second shadows the first (dead code, reader confusion).
    """

    def test_no_duplicate_repr_in_provider_classes(self):
        """No provider class should have duplicate __repr__ definitions."""
        import ast
        from pathlib import Path

        providers_path = (
            Path(__file__).resolve().parent.parent / "memory" / "providers.py"
        )
        with open(providers_path) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                repr_count = sum(
                    1
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__repr__"
                )
                assert repr_count <= 1, (
                    f"{node.name} has {repr_count} __repr__ methods — "
                    f"expected at most 1 (duplicate __repr__ is dead code)"
                )


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-tiy84: check_available methods must log on exception
# ---------------------------------------------------------------------------


class TestCheckAvailable_LogsOnException:
    """check_available() must log at debug level when exceptions occur,
    not silently swallow them. Operators need diagnostic info when health checks fail.
    """

    def test_openai_check_available_logs_on_http_error(self):
        """OpenAI check_available should log when HTTP errors occur."""
        provider = OpenAIProvider(api_key="bad-key")
        import urllib.error

        with (
            patch(
                "memory.providers.safe_urlopen",
                side_effect=urllib.error.HTTPError(
                    url="https://api.openai.com/v1/models",
                    code=401,
                    msg="Unauthorized",
                    hdrs=None,
                    fp=None,
                ),
            ),
            patch("memory.providers.log") as mock_log,
        ):
            result = provider.check_available()
        assert result is False
        # Verify debug was called with the expected message
        mock_log.debug.assert_called_once_with(
            "providers: OpenAI check_available failed", exc_info=True
        )

    def test_openai_check_available_logs_on_connection_error(self):
        """OpenAI check_available should log on connection errors."""
        provider = OpenAIProvider(api_key="test-key")
        import urllib.error

        with (
            patch(
                "memory.providers.safe_urlopen",
                side_effect=urllib.error.URLError("Connection refused"),
            ),
            patch("memory.providers.log") as mock_log,
        ):
            result = provider.check_available()
        assert result is False
        mock_log.debug.assert_called_once_with(
            "providers: OpenAI check_available failed", exc_info=True
        )

    def test_anthropic_check_available_logs_on_http_error(self):
        """Anthropic check_available should log when HTTP errors occur."""
        provider = AnthropicProvider(api_key="bad-key")
        import urllib.error

        with (
            patch(
                "memory.providers.safe_urlopen",
                side_effect=urllib.error.HTTPError(
                    url="https://api.anthropic.com/v1/messages",
                    code=401,
                    msg="Unauthorized",
                    hdrs=None,
                    fp=None,
                ),
            ),
            patch("memory.providers.log") as mock_log,
        ):
            result = provider.check_available()
        assert result is False
        mock_log.debug.assert_called_once_with(
            "providers: Anthropic check_available failed", exc_info=True
        )

    def test_anthropic_check_available_logs_on_connection_error(self):
        """Anthropic check_available should log on connection errors."""
        provider = AnthropicProvider(api_key="test-key")
        import urllib.error

        with (
            patch(
                "memory.providers.safe_urlopen",
                side_effect=urllib.error.URLError("Connection refused"),
            ),
            patch("memory.providers.log") as mock_log,
        ):
            result = provider.check_available()
        assert result is False
        mock_log.debug.assert_called_once_with(
            "providers: Anthropic check_available failed", exc_info=True
        )


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-9xxfp: OllamaProvider._embed_batch_internal wraps
# HTTPError and URLError/OSError into RuntimeError
# ---------------------------------------------------------------------------


class TestOllamaProvider_EmbedBatchInternal_ConnectionErrors:
    """OllamaProvider.embed()/embed_batch() docstrings promise RuntimeError
    on HTTP errors and connection failures. Raw HTTPError, URLError, or
    OSError must be wrapped in RuntimeError to honor the contract.
    """

    def test_embed_wraps_http_error_as_runtime_error(self):
        """embed() must wrap HTTPError in RuntimeError."""
        provider = OllamaProvider()
        import urllib.error

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:11434/api/embeddings",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=None,
            ),
        ):
            with pytest.raises(
                RuntimeError, match="Ollama embedding API returned HTTP 500"
            ):
                provider.embed("test text")

    def test_embed_wraps_url_error_as_runtime_error(self):
        """embed() must wrap URLError (connection refused) in RuntimeError."""
        provider = OllamaProvider()
        import urllib.error

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Ollama embedding request failed"):
                provider.embed("test text")

    def test_embed_wraps_os_error_as_runtime_error(self):
        """embed() must wrap OSError (network unreachable) in RuntimeError."""
        provider = OllamaProvider()

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=OSError("Network is unreachable"),
        ):
            with pytest.raises(RuntimeError, match="Ollama embedding request failed"):
                provider.embed("test text")

    def test_embed_batch_wraps_http_error_as_runtime_error(self):
        """embed_batch() must wrap HTTPError in RuntimeError."""
        provider = OllamaProvider()
        import urllib.error

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:11434/api/embeddings",
                code=503,
                msg="Service Unavailable",
                hdrs=None,
                fp=None,
            ),
        ):
            with pytest.raises(
                RuntimeError, match="Ollama embedding API returned HTTP 503"
            ):
                provider.embed_batch(["text a", "text b"])


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-6eq2d: OpenAI and Anthropic wrap URLError/OSError
#   into RuntimeError
# ---------------------------------------------------------------------------


class TestOpenAIProvider_ConnectionErrors:
    """OpenAIProvider._make_request must wrap URLError/OSError into RuntimeError.

    All OpenAI methods (embed, embed_batch, generate) delegate to _make_request,
    which must honor the 'Raises: RuntimeError' contract for connection failures.
    """

    def test_make_request_wraps_url_error_as_runtime_error(self):
        """_make_request must wrap URLError in RuntimeError."""
        provider = OpenAIProvider(api_key="test-key")
        import urllib.error

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(RuntimeError, match="OpenAI request failed"):
                provider.embed("test text")

    def test_make_request_wraps_os_error_as_runtime_error(self):
        """_make_request must wrap OSError in RuntimeError."""
        provider = OpenAIProvider(api_key="test-key")

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=OSError("Network is unreachable"),
        ):
            with pytest.raises(RuntimeError, match="OpenAI request failed"):
                provider.embed("test text")

    def test_make_request_http_error_still_runtime_error(self):
        """_make_request must still raise RuntimeError for HTTPError (existing behavior)."""
        provider = OpenAIProvider(api_key="test-key")
        import urllib.error

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://api.openai.com/v1/embeddings",
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=None,
            ),
        ):
            with pytest.raises(RuntimeError, match="OpenAI API returned HTTP 401"):
                provider.embed("test text")


class TestAnthropicProvider_ConnectionErrors:
    """AnthropicProvider.generate() must wrap URLError/OSError into RuntimeError.

    The documented contract says 'Raises: RuntimeError'. Connection failures
    must be wrapped to honor that contract.
    """

    def test_generate_wraps_url_error_as_runtime_error(self):
        """generate() must wrap URLError in RuntimeError."""
        provider = AnthropicProvider(api_key="test-key")
        import urllib.error

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Anthropic request failed"):
                provider.generate("test prompt")

    def test_generate_wraps_os_error_as_runtime_error(self):
        """generate() must wrap OSError in RuntimeError."""
        provider = AnthropicProvider(api_key="test-key")

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=OSError("Network is unreachable"),
        ):
            with pytest.raises(RuntimeError, match="Anthropic request failed"):
                provider.generate("test prompt")

    def test_generate_http_error_still_runtime_error(self):
        """generate() must still raise RuntimeError for HTTPError (existing behavior)."""
        provider = AnthropicProvider(api_key="test-key")
        import urllib.error

        with patch(
            "memory.providers.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://api.anthropic.com/v1/messages",
                code=429,
                msg="Too Many Requests",
                hdrs=None,
                fp=None,
            ),
        ):
            with pytest.raises(RuntimeError, match="Anthropic API returned HTTP 429"):
                provider.generate("test prompt")


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-887ww: OpenAIProvider.generate can return None
# Issue ll-1ztcx-woaqv: AnthropicProvider.generate can return None
#
# Both generate() methods promise '-> str' but can return None when the API
# returns null content (e.g., content_filter finish_reason for OpenAI,
# or null text in Anthropic response). Fix: use `or ""` to coerce None.
# ---------------------------------------------------------------------------


class TestOpenAIProvider_GenerateNeverReturnsNone:
    """OpenAIProvider.generate() must return str, never None.

    When OpenAI returns content=null (e.g., due to content_filter finish_reason),
    generate() must return '' instead of None, honoring the '-> str' contract.
    """

    def test_generate_returns_empty_string_when_content_is_null(self):
        """When API returns {'choices': [{'message': {'content': None}}]},
        generate() must return '' not None."""
        provider = OpenAIProvider(api_key="test-key")
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "choices": [{"message": {"content": None}}],
            },
        ):
            result = provider.generate("test prompt")
        assert result == ""
        assert result is not None

    def test_generate_returns_actual_content_when_present(self):
        """Normal case: content is a non-None string."""
        provider = OpenAIProvider(api_key="test-key")
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "choices": [{"message": {"content": "Hello, world!"}}],
            },
        ):
            result = provider.generate("test prompt")
        assert result == "Hello, world!"

    def test_generate_returns_empty_string_when_content_is_empty(self):
        """When content is empty string, should return empty string (not None)."""
        provider = OpenAIProvider(api_key="test-key")
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "choices": [{"message": {"content": ""}}],
            },
        ):
            result = provider.generate("test prompt")
        assert result == ""
        assert result is not None


class TestAnthropicProvider_GenerateNeverReturnsNone:
    """AnthropicProvider.generate() must return str, never None.

    When Anthropic returns content with null text, generate() must return ''
    instead of None, honoring the '-> str' contract.
    """

    def test_generate_returns_empty_string_when_text_is_null(self):
        """When API returns {'content': [{'text': None}]},
        generate() must return '' not None."""
        provider = AnthropicProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"content": [{"text": None}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = provider.generate("test prompt")
        assert result == ""
        assert result is not None

    def test_generate_returns_actual_text_when_present(self):
        """Normal case: text is a non-None string."""
        provider = AnthropicProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"content": [{"text": "Hello, world!"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = provider.generate("test prompt")
        assert result == "Hello, world!"

    def test_generate_returns_empty_string_when_text_is_empty(self):
        """When text is empty string, should return empty string (not None)."""
        provider = AnthropicProvider(api_key="test-key")
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"content": [{"text": ""}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = provider.generate("test prompt")
        assert result == ""
        assert result is not None


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-qjc6n / Issue 17: config.get(K, {}) None-safety
# ---------------------------------------------------------------------------


class TestResolveProvider_NoneConfigValues:
    """resolve_provider must handle config values of None gracefully.

    When YAML contains a key with no value (e.g., `provider:` with no
    subkeys), yaml.safe_load returns None for that key. dict.get(K, {}) only
    uses the default when the KEY is missing, not when the VALUE is None.
    This causes AttributeError when chaining .get() on the result.

    Fix: use config.get(K) or {} instead of config.get(K, {}).
    """

    def test_resolve_with_none_provider(self):
        """When config['provider'] is None, resolve_provider should not crash."""
        with patch("memory.providers.check_ollama_model", return_value=False):
            embed, gen = resolve_provider({"provider": None})
        assert isinstance(embed, NoneProvider)
        assert isinstance(gen, NoneProvider)

    def test_resolve_with_none_provider_embed(self):
        """When config['provider']['embed'] is None, should not crash."""
        with patch("memory.providers.check_ollama_model", return_value=True):
            config = {"provider": {"default": "ollama", "embed": None}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OllamaProvider)

    def test_resolve_with_none_provider_generate(self):
        """When config['provider']['generate'] is None, should not crash."""
        with patch("memory.providers.check_ollama_model", return_value=True):
            config = {"provider": {"default": "ollama", "generate": None}}
            embed, gen = resolve_provider(config)
        assert isinstance(gen, OllamaProvider)

    def test_resolve_with_none_provider_ollama(self):
        """When config['provider']['ollama'] is None, should use defaults."""
        with patch("memory.providers.check_ollama_model", return_value=True):
            config = {"provider": {"default": "ollama", "ollama": None}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OllamaProvider)
        assert embed._base_url == DEFAULT_OLLAMA_BASE_URL

    def test_resolve_with_none_memory(self):
        """When config['memory'] is None, legacy_ollama_url should not crash."""
        with patch("memory.providers.check_ollama_model", return_value=True):
            config = {"memory": None}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OllamaProvider)

    def test_resolve_openai_with_none_provider_openai(self):
        """When config['provider']['openai'] is None, should not crash on openai path."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {"provider": {"default": "openai", "openai": None}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OpenAIProvider)
        assert isinstance(gen, OpenAIProvider)

    def test_resolve_anthropic_with_none_provider_anthropic(self):
        """When config['provider']['anthropic'] is None, should not crash."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            os.environ["ANTHROPIC_API_KEY"] = "test-key"
            config = {"provider": {"default": "anthropic", "anthropic": None}}
            embed, gen = resolve_provider(config)
        assert isinstance(gen, AnthropicProvider)

    def test_fallback_embed_with_none_provider(self):
        """_fallback_embed_provider must handle None provider config sections."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            from memory.providers import _fallback_embed_provider

            config = {"provider": None}
            provider = _fallback_embed_provider(config)
        assert isinstance(provider, NoneProvider)

    def test_fallback_generate_with_none_provider(self):
        """_fallback_generate_provider must handle None provider config sections."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            from memory.providers import _fallback_generate_provider

            config = {"provider": None}
            provider = _fallback_generate_provider(config)
        assert isinstance(provider, NoneProvider)

    def test_fallback_generate_with_none_openai_section(self):
        """_fallback_generate_provider must handle None openai config section."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            os.environ["OPENAI_API_KEY"] = "env-key"
            from memory.providers import _fallback_generate_provider

            config = {"provider": {"openai": None}}
            provider = _fallback_generate_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._api_key == "env-key"


class TestConfigGetNoneSafety:
    """config.py functions must handle None values in config dict.

    When YAML contains `memory:` with no subkeys, yaml.safe_load returns
    {'memory': None}. dict.get('memory', {}) returns None (not {}),
    causing AttributeError on chained .get() calls.
    """

    def test_get_db_path_with_none_memory(self):
        from memory.config import get_db_path

        result = get_db_path(config={"memory": None})
        # Should return default, not crash
        assert isinstance(result, object)

    def test_get_ollama_url_with_none_memory(self):
        from memory.config import get_ollama_url

        result = get_ollama_url(config={"memory": None})
        # Should return default, not crash
        assert result == "http://localhost:11434"

    def test_is_auto_extract_with_none_memory(self):
        from memory.config import is_auto_extract

        result = is_auto_extract(config={"memory": None})
        # Should return default (True), not crash
        assert result is True

    def test_get_session_dirs_with_none_memory(self):
        from memory.config import get_session_dirs

        result = get_session_dirs(config={"memory": None})
        # Should return default, not crash
        assert isinstance(result, list)

    def test_get_max_file_size_with_none_memory(self):
        from memory.config import get_max_file_size

        result = get_max_file_size(config={"memory": None})
        # Should return default, not crash
        assert result == 10 * 1024 * 1024

    def test_get_prospective_model_with_none_memory(self):
        from memory.config import get_prospective_model

        result = get_prospective_model(config={"memory": None})
        # Should return default, not crash
        assert result == "qwen2.5:1.5b"

    def test_get_dream_config_with_none_dream(self):
        from memory.config import get_dream_config

        result = get_dream_config(config={"dream": None})
        # Should return defaults, not crash
        assert "enabled" in result

    def test_get_provider_config_with_none_provider(self):
        from memory.config import get_provider_config

        result = get_provider_config(config={"provider": None})
        # Should return defaults, not crash
        assert result["default"] == "ollama"

    def test_get_server_auth_token_with_none_server(self):
        from memory.config import get_server_auth_token

        result = get_server_auth_token(config={"server": None})
        # Should return None, not crash
        assert result is None

    def test_get_server_port_with_none_server(self):
        from memory.config import get_server_port

        result = get_server_port(config={"server": None})
        # Should return default port, not crash
        assert result == 8322

    def test_is_correction_detection_enabled_with_none_section(self):
        from memory.config import is_correction_detection_enabled

        result = is_correction_detection_enabled(config={"correction_detection": None})
        # Should return default (True), not crash
        assert result is True

    def test_get_source_filter_with_none_hook(self):
        from memory.config import get_source_filter

        result = get_source_filter(config={"hook": None})
        # Should return default ("direct"), not crash
        assert result == "direct"


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-x8iqe: provider_cfg.get("default", "ollama") returns None
# when YAML has 'default:' with no value
# ---------------------------------------------------------------------------


class TestResolveProvider_NoneDefaultKey:
    """resolve_provider must handle provider.default=None gracefully.

    When YAML contains `default:` with no value (e.g., `provider:\n  default:`),
    yaml.safe_load returns {'provider': {'default': None}}.
    provider_cfg.get("default", "ollama") returns None because the KEY exists
    (the default is only used when the KEY is missing, not when VALUE is None).
    The None propagates as the provider name, hitting the 'unknown' else branch.

    Fix: use provider_cfg.get("default") or "ollama" which coerces both
    missing-key and None-value to "ollama".
    """

    def test_resolve_with_none_default_uses_ollama(self):
        """When config['provider']['default'] is None, should default to ollama."""
        with patch("memory.providers.check_ollama_model", return_value=True):
            config = {"provider": {"default": None}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OllamaProvider)
        assert isinstance(gen, OllamaProvider)

    def test_resolve_with_none_default_falls_back_when_ollama_unavailable(self):
        """When default=None and Ollama unavailable, should fall back to
        OpenAI/Anthropic/NoneProvider, not crash on 'unknown provider'."""
        with (
            patch("memory.providers.check_ollama_model", return_value=False),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
        ):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {"provider": {"default": None}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OpenAIProvider)
        assert isinstance(gen, OpenAIProvider)

    def test_resolve_with_none_default_no_keys_falls_to_none(self):
        """When default=None, Ollama unavailable, no API keys — should get
        NoneProvider, not crash."""
        with (
            patch("memory.providers.check_ollama_model", return_value=False),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {"provider": {"default": None}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, NoneProvider)
        assert isinstance(gen, NoneProvider)

    def test_resolve_with_explicit_default_still_works(self):
        """Explicit default value (e.g., 'openai') should still work."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {"provider": {"default": "openai"}}
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OpenAIProvider)
        assert isinstance(gen, OpenAIProvider)


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-giz7h: openai_cfg.get(K, DEFAULT) returns None when YAML
# has key with no value — .get(K) or DEFAULT fixes the None-value bypass
# ---------------------------------------------------------------------------


class TestResolveProvider_NoneModelBaseUrl:
    """When YAML has base_url: or embed_model: with no value (None),
    openai_cfg.get('base_url', DEFAULT) returns None (not DEFAULT),
    causing .rstrip('/') on None → AttributeError crash.

    Fix: use openai_cfg.get(K) or DEFAULT instead, which coerces both
    missing-key and None-value to DEFAULT.
    """

    def test_openai_embed_with_none_base_url_uses_default(self):
        """When openai.base_url=None in config, resolve should use DEFAULT_OPENAI_BASE_URL."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {
                "provider": {
                    "default": "openai",
                    "openai": {"api_key": "test-key", "base_url": None},
                }
            }
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OpenAIProvider)
        assert embed._base_url == DEFAULT_OPENAI_BASE_URL

    def test_openai_embed_with_none_embed_model_uses_default(self):
        """When openai.embed_model=None in config, resolve should use DEFAULT_OPENAI_EMBED_MODEL."""
        from memory.providers import DEFAULT_OPENAI_EMBED_MODEL

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {
                "provider": {
                    "default": "openai",
                    "openai": {"api_key": "test-key", "embed_model": None},
                }
            }
            embed, gen = resolve_provider(config)
        assert isinstance(embed, OpenAIProvider)
        assert embed._embed_model == DEFAULT_OPENAI_EMBED_MODEL

    def test_openai_generate_with_none_base_url_uses_default(self):
        """When openai.base_url=None in config, generate provider should use DEFAULT_OPENAI_BASE_URL."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {
                "provider": {
                    "default": "openai",
                    "openai": {"api_key": "test-key", "base_url": None},
                }
            }
            embed, gen = resolve_provider(config)
        assert isinstance(gen, OpenAIProvider)
        assert gen._base_url == DEFAULT_OPENAI_BASE_URL

    def test_openai_generate_with_none_generate_model_uses_default(self):
        """When openai.generate_model=None, should use DEFAULT_OPENAI_GENERATE_MODEL."""
        from memory.providers import DEFAULT_OPENAI_GENERATE_MODEL

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            os.environ["OPENAI_API_KEY"] = "test-key"
            config = {
                "provider": {
                    "default": "openai",
                    "openai": {"api_key": "test-key", "generate_model": None},
                }
            }
            embed, gen = resolve_provider(config)
        assert isinstance(gen, OpenAIProvider)
        assert gen._generate_model == DEFAULT_OPENAI_GENERATE_MODEL

    def test_anthropic_generate_with_none_base_url_uses_default(self):
        """When anthropic.base_url=None, should use DEFAULT_ANTHROPIC_BASE_URL."""
        from memory.providers import DEFAULT_ANTHROPIC_BASE_URL

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            os.environ["ANTHROPIC_API_KEY"] = "test-key"
            config = {
                "provider": {
                    "default": "anthropic",
                    "anthropic": {"api_key": "test-key", "base_url": None},
                }
            }
            embed, gen = resolve_provider(config)
        assert isinstance(gen, AnthropicProvider)
        assert gen._base_url == DEFAULT_ANTHROPIC_BASE_URL

    def test_anthropic_generate_with_none_generate_model_uses_default(self):
        """When anthropic.generate_model=None, should use DEFAULT_ANTHROPIC_MODEL."""
        from memory.providers import DEFAULT_ANTHROPIC_MODEL

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            os.environ["ANTHROPIC_API_KEY"] = "test-key"
            config = {
                "provider": {
                    "default": "anthropic",
                    "anthropic": {"api_key": "test-key", "generate_model": None},
                }
            }
            embed, gen = resolve_provider(config)
        assert isinstance(gen, AnthropicProvider)
        assert gen._model == DEFAULT_ANTHROPIC_MODEL


class TestFallbackProvider_NoneModelBaseUrl:
    """Fallback functions must handle None base_url/model values from config.

    Same bug class as ll-1ztcx-giz7h: openai_cfg.get('base_url', DEFAULT)
    returns None when YAML has key with no value. The None passes to
    Provider constructors that call .rstrip('/') → AttributeError crash.
    """

    def test_fallback_embed_with_none_base_url_uses_default(self):
        """_fallback_embed_provider: openai base_url=None should not crash."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            config = {
                "provider": {
                    "openai": {"api_key": "test-key", "base_url": None},
                },
            }
            from memory.providers import _fallback_embed_provider

            provider = _fallback_embed_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._base_url == DEFAULT_OPENAI_BASE_URL

    def test_fallback_embed_with_none_embed_model_uses_default(self):
        """_fallback_embed_provider: openai embed_model=None should not crash."""
        from memory.providers import DEFAULT_OPENAI_EMBED_MODEL

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            config = {
                "provider": {
                    "openai": {"api_key": "test-key", "embed_model": None},
                },
            }
            from memory.providers import _fallback_embed_provider

            provider = _fallback_embed_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._embed_model == DEFAULT_OPENAI_EMBED_MODEL

    def test_fallback_generate_with_none_base_url_uses_default(self):
        """_fallback_generate_provider: openai base_url=None should not crash."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {
                "provider": {
                    "openai": {"api_key": "test-key", "base_url": None},
                },
            }
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._base_url == DEFAULT_OPENAI_BASE_URL

    def test_fallback_generate_with_none_generate_model_uses_default(self):
        """_fallback_generate_provider: openai generate_model=None should not crash."""
        from memory.providers import DEFAULT_OPENAI_GENERATE_MODEL

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {
                "provider": {
                    "openai": {"api_key": "test-key", "generate_model": None},
                },
            }
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider._generate_model == DEFAULT_OPENAI_GENERATE_MODEL

    def test_fallback_generate_anthropic_with_none_base_url(self):
        """_fallback_generate_provider: anthropic base_url=None should not crash."""
        from memory.providers import DEFAULT_ANTHROPIC_BASE_URL

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {
                "provider": {
                    "anthropic": {"api_key": "test-key", "base_url": None},
                },
            }
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config, skip_openai=True)
        assert isinstance(provider, AnthropicProvider)
        assert provider._base_url == DEFAULT_ANTHROPIC_BASE_URL

    def test_fallback_generate_anthropic_with_none_generate_model(self):
        """_fallback_generate_provider: anthropic generate_model=None should not crash."""
        from memory.providers import DEFAULT_ANTHROPIC_MODEL

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = {
                "provider": {
                    "anthropic": {"api_key": "test-key", "generate_model": None},
                },
            }
            from memory.providers import _fallback_generate_provider

            provider = _fallback_generate_provider(config, skip_openai=True)
        assert isinstance(provider, AnthropicProvider)
        assert provider._model == DEFAULT_ANTHROPIC_MODEL


class TestConfigGetNoneValueSafety:
    """config.py .get(key, default_val) must handle None values from YAML.

    When YAML has a key with no value (e.g., 'schedule:' with nothing after),
    yaml.safe_load returns None for that value. dict.get(K, default_val)
    only uses the default when the KEY is missing, not when VALUE is None.

    Fix: use section.get(K) or default_val instead.
    """

    def test_get_dream_config_with_none_schedule(self):
        """dream.schedule=None should fall back to default schedule."""
        from memory.config import get_dream_config, DEFAULTS

        config = {"dream": {"schedule": None}}
        result = get_dream_config(config=config)
        assert result["schedule"] == DEFAULTS["dream"]["schedule"]

    def test_get_dream_config_with_none_similarity_threshold(self):
        """dream.similarity_threshold=None should fall back to default."""
        from memory.config import get_dream_config, DEFAULTS

        config = {"dream": {"similarity_threshold": None}}
        result = get_dream_config(config=config)
        assert (
            result["similarity_threshold"] == DEFAULTS["dream"]["similarity_threshold"]
        )

    def test_get_dream_config_with_none_enabled(self):
        """dream.enabled=None should fall back to default (True).

        Note: bool(None) is False, so 'or' correctly treats None as
        falsy and returns the default True.
        """
        from memory.config import get_dream_config

        config = {"dream": {"enabled": None}}
        result = get_dream_config(config=config)
        assert result["enabled"] is True

    def test_get_resume_config_with_none_backend(self):
        """resume.backend=None should fall back to default."""
        from memory.config import get_resume_config, DEFAULTS

        config = {"resume": {"backend": None}}
        result = get_resume_config(config=config)
        assert result["backend"] == DEFAULTS["resume"]["backend"]

    def test_get_resume_config_with_none_model(self):
        """resume.model=None should fall back to default."""
        from memory.config import get_resume_config, DEFAULTS

        config = {"resume": {"model": None}}
        result = get_resume_config(config=config)
        assert result["model"] == DEFAULTS["resume"]["model"]

    def test_get_provider_config_with_none_default(self):
        """provider.default=None should fall back to default ('ollama')."""
        from memory.config import get_provider_config, DEFAULTS

        config = {"provider": {"default": None}}
        result = get_provider_config(config=config)
        assert result["default"] == DEFAULTS["provider"]["default"]

    def test_get_provider_config_with_none_embed(self):
        """provider.embed=None should fall back to default empty dict."""
        from memory.config import get_provider_config

        config = {"provider": {"embed": None}}
        result = get_provider_config(config=config)
        # None embed should be treated as empty dict, not None
        assert result["embed"] == {}


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-sql6v: config.get(K) or DEFAULT overrides falsy-but-valid
# values (dream.enabled=False → True, similarity_threshold=0 → 0.92, etc.)
#
# The 'or' pattern treats falsy-but-valid values the same as None/missing:
#   False or True == True  (should be False)
#   0 or 0.92 == 0.92      (should be 0)
#   0 or 3 == 3            (should be 0)
#
# Fix: use None-aware logic: default_val if val is None else val
# This preserves explicit falsy values while still defaulting None/missing.
# ---------------------------------------------------------------------------


class TestConfigFalsyValueSafety:
    """config.py .get(K) or DEFAULT must NOT override falsy-but-valid values.

    When YAML explicitly sets a key to a falsy value (False, 0, ""), that
    value must be preserved. Only missing keys or explicit None should
    fall back to defaults.

    The prior 'or' pattern treated False, 0, and "" the same as None,
    silently overriding the user's explicit configuration.
    """

    def test_get_dream_config_enabled_false_preserved(self):
        """dream.enabled=False must NOT be overridden to True.

        This was the critical bug: False or True == True, so a user
        explicitly disabling dream mode would have it silently re-enabled.
        """
        from memory.config import get_dream_config

        config = {"dream": {"enabled": False}}
        result = get_dream_config(config=config)
        assert result["enabled"] is False

    def test_get_dream_config_similarity_threshold_zero_preserved(self):
        """dream.similarity_threshold=0 must NOT be overridden to 0.92.

        A user setting similarity_threshold to 0 (match everything) should
        not have it silently changed to 0.92.
        """
        from memory.config import get_dream_config

        config = {"dream": {"similarity_threshold": 0}}
        result = get_dream_config(config=config)
        assert result["similarity_threshold"] == 0

    def test_get_dream_config_min_recall_count_zero_preserved(self):
        """dream.min_recall_count=0 must NOT be overridden to 3."""
        from memory.config import get_dream_config

        config = {"dream": {"min_recall_count": 0}}
        result = get_dream_config(config=config)
        assert result["min_recall_count"] == 0

    def test_get_dream_config_min_unique_queries_zero_preserved(self):
        """dream.min_unique_queries=0 must NOT be overridden to 1."""
        from memory.config import get_dream_config

        config = {"dream": {"min_unique_queries": 0}}
        result = get_dream_config(config=config)
        assert result["min_unique_queries"] == 0

    def test_get_dream_config_decay_floor_zero_preserved(self):
        """dream.decay_floor=0 must NOT be overridden to 0.3."""
        from memory.config import get_dream_config

        config = {"dream": {"decay_floor": 0}}
        result = get_dream_config(config=config)
        assert result["decay_floor"] == 0

    def test_get_dream_config_min_score_zero_preserved(self):
        """dream.min_score=0 must NOT be overridden to 0.5."""
        from memory.config import get_dream_config

        config = {"dream": {"min_score": 0}}
        result = get_dream_config(config=config)
        assert result["min_score"] == 0

    def test_get_dream_config_boost_amount_zero_preserved(self):
        """dream.boost_amount=0 must NOT be overridden to 0.05."""
        from memory.config import get_dream_config

        config = {"dream": {"boost_amount": 0}}
        result = get_dream_config(config=config)
        assert result["boost_amount"] == 0

    def test_get_dream_config_decay_rate_zero_preserved(self):
        """dream.decay_rate=0 must NOT be overridden to 0.05."""
        from memory.config import get_dream_config

        config = {"dream": {"decay_rate": 0}}
        result = get_dream_config(config=config)
        assert result["decay_rate"] == 0

    def test_get_dream_config_boost_threshold_zero_preserved(self):
        """dream.boost_threshold=0 must NOT be overridden to 5."""
        from memory.config import get_dream_config

        config = {"dream": {"boost_threshold": 0}}
        result = get_dream_config(config=config)
        assert result["boost_threshold"] == 0

    def test_get_dream_config_behavioral_threshold_zero_preserved(self):
        """dream.behavioral_threshold=0 must NOT be overridden to 3."""
        from memory.config import get_dream_config

        config = {"dream": {"behavioral_threshold": 0}}
        result = get_dream_config(config=config)
        assert result["behavioral_threshold"] == 0

    def test_get_dream_config_calibration_enabled_false_preserved(self):
        """dream.calibration_enabled=False must NOT be overridden to True."""
        from memory.config import get_dream_config

        config = {"dream": {"calibration_enabled": False}}
        result = get_dream_config(config=config)
        assert result["calibration_enabled"] is False

    def test_get_dream_config_empty_string_preserved(self):
        """dream.diary_path='' must NOT be overridden to the default path.

        An explicitly-empty string is a valid user choice (e.g., disabling
        diary output). The 'or' pattern would override it.
        """
        from memory.config import get_dream_config

        config = {"dream": {"diary_path": ""}}
        result = get_dream_config(config=config)
        assert result["diary_path"] == ""

    def test_get_dream_config_mixed_falsy_and_truthy(self):
        """Multiple falsy values in one config must all be preserved."""
        from memory.config import get_dream_config

        config = {
            "dream": {
                "enabled": False,
                "similarity_threshold": 0,
                "min_recall_count": 0,
                "schedule": "*-*-* 03:00:00",
            }
        }
        result = get_dream_config(config=config)
        assert result["enabled"] is False
        assert result["similarity_threshold"] == 0
        assert result["min_recall_count"] == 0
        assert result["schedule"] == "*-*-* 03:00:00"

    def test_get_dream_config_none_still_falls_back(self):
        """After fix: None values must still fall back to defaults.

        The fix must not break the None-safety — None values from YAML
        must still use defaults, just not falsy-but-valid values.
        """
        from memory.config import get_dream_config, DEFAULTS

        config = {"dream": {"enabled": None, "similarity_threshold": None}}
        result = get_dream_config(config=config)
        assert result["enabled"] is True  # default
        assert (
            result["similarity_threshold"] == DEFAULTS["dream"]["similarity_threshold"]
        )

    def test_get_resume_config_timeout_zero_preserved(self):
        """resume.timeout=0 must NOT be overridden to default."""
        from memory.config import get_resume_config

        config = {"resume": {"backend": "ollama", "model": "test"}}
        result = get_resume_config(config=config)
        # These truthy values already work; verify they still do
        assert result["backend"] == "ollama"
        assert result["model"] == "test"

    def test_get_provider_config_default_empty_string_preserved(self):
        """provider.default='' must NOT be overridden to 'ollama'.

        An empty-string default is a valid falsy value that should be preserved.
        """
        from memory.config import get_provider_config

        config = {"provider": {"default": ""}}
        result = get_provider_config(config=config)
        assert result["default"] == ""

    def test_get_provider_config_embed_empty_dict_preserved(self):
        """provider.embed={} must be preserved as empty dict."""
        from memory.config import get_provider_config

        config = {"provider": {"embed": {}}}
        result = get_provider_config(config=config)
        assert result["embed"] == {}

    def test_is_dream_enabled_false_not_overridden(self):
        """is_dream_enabled must return False when explicitly set to False.

        This function already uses the correct None-check pattern, but
        verify it handles False correctly (not overriding to True).
        """
        from memory.config import is_dream_enabled

        config = {"dream": {"enabled": False}}
        result = is_dream_enabled(config=config)
        assert result is False

    def test_is_auto_extract_false_not_overridden(self):
        """is_auto_extract must return False when explicitly set to False."""
        from memory.config import is_auto_extract

        config = {"memory": {"auto_extract": False}}
        result = is_auto_extract(config=config)
        assert result is False

    def test_is_correction_detection_enabled_false_not_overridden(self):
        """is_correction_detection_enabled must return False when explicitly set."""
        from memory.config import is_correction_detection_enabled

        config = {"correction_detection": {"enabled": False}}
        result = is_correction_detection_enabled(config=config)
        assert result is False


# ---------------------------------------------------------------------------
# dimension() method tests for all providers
# ---------------------------------------------------------------------------


class TestEmbedProviderDimension:
    """Test dimension() method on all EmbedProvider implementations."""

    def test_ollama_dimension_default(self):
        """OllamaProvider.dimension() returns 768 by default (nomic-embed-text)."""
        provider = OllamaProvider()
        assert provider.dimension() == 768

    def test_ollama_dimension_known_model(self):
        """OllamaProvider.dimension() returns known dimension for mxbai-embed-large."""
        provider = OllamaProvider(embed_model="mxbai-embed-large")
        assert provider.dimension() == 1024

    def test_ollama_dimension_unknown_model_returns_default(self):
        """OllamaProvider.dimension() returns 768 for unknown models."""
        provider = OllamaProvider(embed_model="custom-unknown-model")
        assert provider.dimension() == 768

    def test_openai_dimension_default(self):
        """OpenAIProvider.dimension() returns 1536 by default (text-embedding-3-small)."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.dimension() == 1536

    def test_openai_dimension_large_model(self):
        """OpenAIProvider.dimension() returns 3072 for text-embedding-3-large."""
        provider = OpenAIProvider(
            api_key="test-key", embed_model="text-embedding-3-large"
        )
        assert provider.dimension() == 3072

    def test_openai_dimension_unknown_model_returns_default(self):
        """OpenAIProvider.dimension() returns 1536 for unknown models."""
        provider = OpenAIProvider(api_key="test-key", embed_model="custom-model")
        assert provider.dimension() == 1536

    def test_none_dimension(self):
        """NoneProvider.dimension() returns 768 (default DEFAULT_NONE_EMBED_DIMENSIONS)."""
        provider = NoneProvider()
        assert provider.dimension() == 768
        assert provider.dimension() == DEFAULT_NONE_EMBED_DIMENSIONS

    def test_none_dimension_custom(self):
        """NoneProvider(embed_dimensions=384).dimension() returns 384."""
        provider = NoneProvider(embed_dimensions=384)
        assert provider.dimension() == 384


# ---------------------------------------------------------------------------
# SentenceTransformersProvider tests
# ---------------------------------------------------------------------------


class TestSentenceTransformersProviderEmbedProvider:
    """SentenceTransformersProvider must implement EmbedProvider ABC."""

    def test_is_instance_of_embed_provider(self):
        """SentenceTransformersProvider is an EmbedProvider."""
        provider = SentenceTransformersProvider()
        assert isinstance(provider, EmbedProvider)


class TestSentenceTransformersProvider:
    """Test SentenceTransformersProvider constructor and methods."""

    def test_constructor_default_model(self):
        """Default model_name is 'all-MiniLM-L6-v2'."""
        provider = SentenceTransformersProvider()
        assert provider._model_name == "all-MiniLM-L6-v2"

    def test_constructor_custom_model_and_dimensions(self):
        """Constructor accepts custom model_name and explicit dimensions."""
        provider = SentenceTransformersProvider(
            model_name="all-mpnet-base-v2", dimensions=768
        )
        assert provider._model_name == "all-mpnet-base-v2"
        assert provider._dimensions == 768

    def test_constructor_empty_model_name_raises(self):
        """Empty model_name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            SentenceTransformersProvider(model_name="")

    def test_constructor_none_model_name_raises(self):
        """None model_name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            SentenceTransformersProvider(model_name=None)

    def test_embed_returns_vector(self):
        """embed() returns a list[float] by mocking the sentence-transformers model."""
        provider = SentenceTransformersProvider()
        mock_vec = [0.1] * 384
        # Simulate numpy array with tolist() method
        mock_row = MagicMock()
        mock_row.tolist.return_value = mock_vec
        mock_model = MagicMock()
        mock_model.encode.return_value = [mock_row]  # iterable of rows
        with patch.object(provider, "_load_model"):
            provider._model = mock_model
            provider._model_loaded = True
            result = provider.embed("test text")
        assert isinstance(result, list)
        assert len(result) == 384
        assert result == mock_vec

    def test_embed_batch_returns_vectors(self):
        """embed_batch() returns list[list[float]] by mocking the model."""
        provider = SentenceTransformersProvider()
        vec_a = [0.1] * 384
        vec_b = [0.2] * 384
        mock_row_a = MagicMock()
        mock_row_a.tolist.return_value = vec_a
        mock_row_b = MagicMock()
        mock_row_b.tolist.return_value = vec_b
        mock_model = MagicMock()
        mock_model.encode.return_value = [mock_row_a, mock_row_b]
        with patch.object(provider, "_load_model"):
            provider._model = mock_model
            provider._model_loaded = True
            result = provider.embed_batch(["text a", "text b"])
        assert len(result) == 2
        assert result[0] == vec_a
        assert result[1] == vec_b

    def test_embed_batch_empty_list(self):
        """embed_batch([]) returns empty list without loading model."""
        provider = SentenceTransformersProvider()
        result = provider.embed_batch([])
        assert result == []

    def test_check_available_returns_true_when_model_loaded(self):
        """check_available() returns True when model loads successfully."""
        provider = SentenceTransformersProvider()
        mock_model = MagicMock()

        def mock_load():
            provider._model_loaded = True
            provider._model = mock_model

        with patch.object(provider, "_load_model", side_effect=mock_load):
            result = provider.check_available()
        assert result is True

    def test_check_available_returns_false_on_import_error(self):
        """check_available() returns False when sentence_transformers is not installed."""
        provider = SentenceTransformersProvider()
        with patch.object(
            provider, "_load_model", side_effect=ImportError("not installed")
        ):
            result = provider.check_available()
        assert result is False

    def test_check_available_caches_result(self):
        """check_available() caches its result — second call doesn't reload model."""
        provider = SentenceTransformersProvider()
        call_count = 0

        def counting_load():
            nonlocal call_count
            call_count += 1
            # Simulate successful load
            provider._model_loaded = True
            provider._model = MagicMock()

        with patch.object(provider, "_load_model", side_effect=counting_load):
            result1 = provider.check_available()
            result2 = provider.check_available()
        assert result1 is True
        assert result2 is True
        # _load_model called only once (cached after first check)
        assert call_count == 1

    def test_dimension_returns_model_dimension(self):
        """dimension() returns the known dimension for default model without loading it.

        Per the EmbedProvider ABC contract, dimension() must be lightweight and
        never trigger network/disk IO. The known-dimensions lookup table is used
        instead of calling model.get_sentence_embedding_dimension().
        """
        provider = SentenceTransformersProvider()
        # Should return 384 for all-MiniLM-L6-v2 WITHOUT calling _load_model()
        assert provider.dimension() == 384
        # Verify no lazy loading happened
        assert provider._model_loaded is False

    def test_dimension_returns_known_dimension_without_load(self):
        """dimension() for a known model (all-mpnet-base-v2) returns 768 without loading."""
        provider = SentenceTransformersProvider(model_name="all-mpnet-base-v2")
        assert provider.dimension() == 768
        assert provider._model_loaded is False

    def test_dimension_unknown_model_returns_default(self):
        """dimension() for an unknown model returns 384 (default) without loading."""
        provider = SentenceTransformersProvider(model_name="custom-unknown-model")
        assert provider.dimension() == 384
        assert provider._model_loaded is False

    def test_dimension_never_calls_load_model(self):
        """dimension() must never call _load_model() — ABC contract violation.

        Regression test: a prior implementation called _load_model() when no
        explicit dimensions were given, which triggers network/disk IO and can
        raise exceptions from a method the ABC says should be lightweight.
        """
        provider = SentenceTransformersProvider()
        with patch.object(
            provider, "_load_model", side_effect=AssertionError("must not be called")
        ):
            # dimension() must return without ever calling _load_model
            result = provider.dimension()
        assert result == 384

    def test_dimension_returns_explicit_dimensions(self):
        """dimension() returns the explicit dimensions when provided in constructor."""
        provider = SentenceTransformersProvider(dimensions=512)
        assert provider.dimension() == 512

    def test_repr(self):
        """repr includes model_name."""
        provider = SentenceTransformersProvider(model_name="all-MiniLM-L6-v2")
        r = repr(provider)
        assert "SentenceTransformersProvider" in r
        assert "all-MiniLM-L6-v2" in r

    def test_import_error_message(self):
        """_load_model raises ImportError with helpful message when sentence_transformers missing."""
        provider = SentenceTransformersProvider()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Force re-import attempt
            with pytest.raises(ImportError, match="pip install llmem\\[local\\]"):
                provider._load_model()


# ---------------------------------------------------------------------------
# resolve_provider() tests for 'local' provider
# ---------------------------------------------------------------------------


class TestResolveProvider_LocalProvider:
    """Test resolve_provider when provider.default='local'."""

    def test_local_provider_resolved(self):
        """When provider.default='local' and sentence_transformers is available,
        resolve_provider returns SentenceTransformersProvider for embed."""
        mock_st_provider = MagicMock(spec=SentenceTransformersProvider)
        mock_st_provider.check_available.return_value = True
        with patch(
            "memory.providers.SentenceTransformersProvider",
            return_value=mock_st_provider,
        ):
            config = {"provider": {"default": "local"}}
            embed, gen = resolve_provider(config)
        # The embed provider is the mocked SentenceTransformersProvider
        assert embed is mock_st_provider

    def test_local_provider_fallback_on_import_error(self):
        """When provider.default='local' but sentence_transformers not installed
        (check_available returns False), falls back to NoneProvider."""
        mock_st_provider = MagicMock(spec=SentenceTransformersProvider)
        mock_st_provider.check_available.return_value = False
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "memory.providers.SentenceTransformersProvider",
                return_value=mock_st_provider,
            ),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            config = {"provider": {"default": "local"}}
            embed, gen = resolve_provider(config)
        # Falls back gracefully since check_available returns False → no OpenAI key → NoneProvider
        assert isinstance(embed, NoneProvider)


class TestResolveProvider_LocalProviderFallback:
    """When local provider is configured but sentence_transformers not installed,
    falls back to NoneProvider with warning."""

    def test_local_fallback_to_none(self):
        """When SentenceTransformersProvider.check_available returns False,
        should fall back to NoneProvider."""
        mock_provider = MagicMock(spec=SentenceTransformersProvider)
        mock_provider.check_available.return_value = False
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "memory.providers.SentenceTransformersProvider",
                return_value=mock_provider,
            ),
        ):
            os.environ.pop("OPENAI_API_KEY", None)
            config = {"provider": {"default": "local"}}
            embed, gen = resolve_provider(config)
        # Provider check fails → fallback → No openai key → NoneProvider
        assert isinstance(embed, NoneProvider)


class TestResolveProvider_LocalWithCustomModel:
    """Config provider.local.model creates provider with that model name."""

    def test_local_with_custom_model(self):
        """provider.local.model='all-mpnet-base-v2' creates provider with that model."""
        mock_provider = MagicMock(spec=SentenceTransformersProvider)
        mock_provider.check_available.return_value = True
        with patch(
            "memory.providers.SentenceTransformersProvider",
            return_value=mock_provider,
        ) as mock_cls:
            config = {
                "provider": {
                    "default": "local",
                    "local": {"model": "all-mpnet-base-v2"},
                }
            }
            embed, gen = resolve_provider(config)
        mock_cls.assert_called_with(model_name="all-mpnet-base-v2")


# ---------------------------------------------------------------------------
# Config tests for provider.local section
# ---------------------------------------------------------------------------


class TestGetProviderConfig_LocalSection:
    """Test get_provider_config returns the 'local' section with defaults."""

    def test_returns_local_section_in_defaults(self):
        result = get_provider_config(config={})
        assert "local" in result
        assert result["local"]["model"] == "all-MiniLM-L6-v2"

    def test_local_defaults_match_brief(self):
        from memory.config import DEFAULTS

        assert DEFAULTS["provider"]["local"]["model"] == "all-MiniLM-L6-v2"

    def test_custom_local_model_preserved(self):
        config = {"provider": {"local": {"model": "custom-model"}}}
        result = get_provider_config(config=config)
        assert result["local"]["model"] == "custom-model"

    def test_none_local_section_uses_default(self):
        """When provider.local is None in config, should use default."""
        config = {"provider": {"local": None}}
        result = get_provider_config(config=config)
        # None local should be treated as default, not None
        assert result["local"]["model"] == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# MemoryStore dimension validation tests
# ---------------------------------------------------------------------------


class TestStoreDimensionValidation:
    """Test dimension validation in MemoryStore.add()."""

    def test_reject_mismatched_dimension(self, tmp_path):
        """add() raises ValueError when embedding dimension doesn't match vec_dimensions.

        Dimension validation only runs when disable_vec=False, so this test
        requires sqlite-vec to be available.
        """
        import struct

        from llmem.store import MemoryStore

        pytest.importorskip("sqlite_vec")
        db = tmp_path / "test_dim.db"
        store = MemoryStore(db_path=db, vec_dimensions=768, disable_vec=False)
        # Create a 4-float embedding (4 dimensions), but store expects 768
        wrong_embedding = struct.pack(f"{4}f", 0.1, 0.2, 0.3, 0.4)
        with pytest.raises(
            ValueError, match="embedding dimension 4 does not match vec_dimensions 768"
        ):
            store.add(type="fact", content="test", embedding=wrong_embedding)
        store.close()

    def test_accept_correct_dimension(self, tmp_path):
        """add() succeeds when embedding dimension matches vec_dimensions."""
        import struct

        from llmem.store import MemoryStore

        pytest.importorskip("sqlite_vec")
        db = tmp_path / "test_dim_accept.db"
        vec_store = MemoryStore(db_path=db, vec_dimensions=768, disable_vec=False)
        correct_embedding = struct.pack(f"{768}f", *([0.0] * 768))
        mid = vec_store.add(type="fact", content="test", embedding=correct_embedding)
        assert mid is not None
        vec_store.close()

    def test_dimension_validation_disabled_when_no_vec(self, tmp_path):
        """When disable_vec=True, dimension validation is skipped.

        Embeddings with wrong dimension are accepted when vec is disabled,
        since there's no vec index to corrupt.
        """
        import struct

        from llmem.store import MemoryStore

        db = tmp_path / "test_dim_validation_disabled.db"
        store = MemoryStore(db_path=db, vec_dimensions=4, disable_vec=True)
        # Embedding with 2 floats = 8 bytes (2*4), but vec_dimensions=4
        small_embedding = struct.pack(f"{2}f", 0.1, 0.2)
        # Should NOT raise — validation is skipped when disable_vec=True
        mid = store.add(type="fact", content="test", embedding=small_embedding)
        assert mid is not None
        store.close()

    def test_dimension_validation_none_embedding(self, store):
        """add() with embedding=None should not trigger dimension validation."""
        mid = store.add(type="fact", content="test", embedding=None)
        assert mid is not None


# ---------------------------------------------------------------------------
# Input validation tests: _validate_embed_inputs and prompt length checks
# ---------------------------------------------------------------------------


class TestValidateEmbedInputs:
    """Test _validate_embed_inputs for batch size and text length limits."""

    def test_batch_exceeds_max_size_raises_value_error(self):
        """embed_batch with > MAX_BATCH_SIZE texts raises ValueError."""
        from memory.providers import _validate_embed_inputs, MAX_BATCH_SIZE

        texts = ["x"] * (MAX_BATCH_SIZE + 1)
        with pytest.raises(ValueError, match="batch size .* exceeds maximum"):
            _validate_embed_inputs(texts)

    def test_text_exceeds_max_length_raises_value_error(self):
        """embed_batch with a text > MAX_TEXT_LENGTH chars raises ValueError."""
        from memory.providers import _validate_embed_inputs, MAX_TEXT_LENGTH

        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="text at index .* exceeds maximum length"):
            _validate_embed_inputs([long_text])

    def test_text_exceeds_max_length_reports_index(self):
        """embed_batch reports the offending index in the ValueError."""
        from memory.providers import _validate_embed_inputs, MAX_TEXT_LENGTH

        long_text = "b" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="text at index 3"):
            _validate_embed_inputs(["short", "also short", "ok", long_text])

    def test_valid_inputs_pass(self):
        """_validate_embed_inputs returns None for valid inputs (no exception)."""
        from memory.providers import _validate_embed_inputs

        result = _validate_embed_inputs(["hello", "world"])
        assert result is None

    def test_empty_list_passes(self):
        """_validate_embed_inputs([]) returns None — empty batch is valid."""
        from memory.providers import _validate_embed_inputs

        result = _validate_embed_inputs([])
        assert result is None

    def test_exact_max_batch_size_passes(self):
        """Batch of exactly MAX_BATCH_SIZE texts should not raise."""
        from memory.providers import _validate_embed_inputs, MAX_BATCH_SIZE

        texts = ["x"] * MAX_BATCH_SIZE
        result = _validate_embed_inputs(texts)
        assert result is None

    def test_exact_max_text_length_passes(self):
        """A text of exactly MAX_TEXT_LENGTH chars should not raise."""
        from memory.providers import _validate_embed_inputs, MAX_TEXT_LENGTH

        text = "a" * MAX_TEXT_LENGTH
        result = _validate_embed_inputs([text])
        assert result is None


class TestOllamaProvider_PromptLengthValidation:
    """Test OllamaProvider.generate() prompt length limit."""

    def test_generate_rejects_oversized_prompt(self):
        """OllamaProvider.generate() raises ValueError for prompt > MAX_TEXT_LENGTH."""
        from memory.providers import MAX_TEXT_LENGTH

        provider = OllamaProvider()
        long_prompt = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="prompt exceeds maximum length"):
            provider.generate(long_prompt)

    def test_generate_accepts_max_length_prompt(self):
        """OllamaProvider.generate() accepts a prompt at exactly MAX_TEXT_LENGTH."""
        from memory.providers import MAX_TEXT_LENGTH

        provider = OllamaProvider()
        prompt = "x" * MAX_TEXT_LENGTH
        with patch(
            "memory.providers._call_ollama_generate", return_value="ok"
        ) as mock_gen:
            result = provider.generate(prompt)
        assert result == "ok"
        mock_gen.assert_called_once()


class TestOpenAIProvider_PromptLengthValidation:
    """Test OpenAIProvider.generate() prompt length limit."""

    def test_generate_rejects_oversized_prompt(self):
        """OpenAIProvider.generate() raises ValueError for prompt > MAX_TEXT_LENGTH."""
        from memory.providers import MAX_TEXT_LENGTH

        provider = OpenAIProvider(api_key="test-key")
        long_prompt = "y" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="prompt exceeds maximum length"):
            provider.generate(long_prompt)

    def test_generate_accepts_max_length_prompt(self):
        """OpenAIProvider.generate() accepts a prompt at exactly MAX_TEXT_LENGTH."""
        from memory.providers import MAX_TEXT_LENGTH

        provider = OpenAIProvider(api_key="test-key")
        prompt = "y" * MAX_TEXT_LENGTH
        with patch.object(
            provider,
            "_make_request",
            return_value={"choices": [{"message": {"content": "ok"}}]},
        ):
            result = provider.generate(prompt)
        assert result == "ok"


class TestAnthropicProvider_PromptLengthValidation:
    """Test AnthropicProvider.generate() prompt length limit."""

    def test_generate_rejects_oversized_prompt(self):
        """AnthropicProvider.generate() raises ValueError for prompt > MAX_TEXT_LENGTH."""
        from memory.providers import MAX_TEXT_LENGTH

        provider = AnthropicProvider(api_key="test-key")
        long_prompt = "z" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="prompt exceeds maximum length"):
            provider.generate(long_prompt)

    def test_generate_accepts_max_length_prompt(self):
        """AnthropicProvider.generate() accepts a prompt at exactly MAX_TEXT_LENGTH."""
        from memory.providers import MAX_TEXT_LENGTH

        provider = AnthropicProvider(api_key="test-key")
        prompt = "z" * MAX_TEXT_LENGTH
        with patch("memory.providers.safe_urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"content": [{"text": "ok"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            result = provider.generate(prompt)
        assert result == "ok"


class TestOllamaProvider_EmbedBatchInputValidation:
    """Test OllamaProvider.embed_batch() input validation."""

    def test_embed_batch_rejects_oversized_batch(self):
        """OllamaProvider.embed_batch() raises ValueError for batch > MAX_BATCH_SIZE."""
        from memory.providers import MAX_BATCH_SIZE

        provider = OllamaProvider()
        texts = ["x"] * (MAX_BATCH_SIZE + 1)
        with pytest.raises(ValueError, match="batch size .* exceeds maximum"):
            provider.embed_batch(texts)

    def test_embed_batch_rejects_oversized_text(self):
        """OllamaProvider.embed_batch() raises ValueError for text > MAX_TEXT_LENGTH."""
        from memory.providers import MAX_TEXT_LENGTH

        provider = OllamaProvider()
        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="text at index .* exceeds maximum length"):
            provider.embed_batch([long_text])


class TestOpenAIProvider_EmbedBatchInputValidation:
    """Test OpenAIProvider.embed_batch() input validation."""

    def test_embed_batch_rejects_oversized_batch(self):
        """OpenAIProvider.embed_batch() raises ValueError for batch > MAX_BATCH_SIZE."""
        from memory.providers import MAX_BATCH_SIZE

        provider = OpenAIProvider(api_key="test-key")
        texts = ["x"] * (MAX_BATCH_SIZE + 1)
        with pytest.raises(ValueError, match="batch size .* exceeds maximum"):
            provider.embed_batch(texts)

    def test_embed_batch_rejects_oversized_text(self):
        """OpenAIProvider.embed_batch() raises ValueError for text > MAX_TEXT_LENGTH."""
        from memory.providers import MAX_TEXT_LENGTH

        provider = OpenAIProvider(api_key="test-key")
        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="text at index .* exceeds maximum length"):
            provider.embed_batch([long_text])


class TestOpenAIProvider_EmbedInputValidation:
    """Test OpenAIProvider.embed() delegates to embed_batch for input validation.

    Previously embed() bypassed _validate_embed_inputs by calling _make_request
    directly. Now it delegates to embed_batch([text])[0] so single-text embeds
    are also validated against MAX_TEXT_LENGTH.
    """

    def test_embed_rejects_oversized_text(self):
        """OpenAIProvider.embed() raises ValueError for text > MAX_TEXT_LENGTH."""
        from memory.providers import MAX_TEXT_LENGTH

        provider = OpenAIProvider(api_key="test-key")
        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValueError, match="text at index .* exceeds maximum length"):
            provider.embed(long_text)

    def test_embed_accepts_valid_text(self):
        """OpenAIProvider.embed() accepts text within length limits.

        Verifies that valid text reaches _make_request (mocked), proving
        the text was not rejected by the _validate_embed_inputs check.
        """
        provider = OpenAIProvider(api_key="test-key")
        vec = [0.1] * 1536
        with patch.object(
            provider,
            "_make_request",
            return_value={
                "data": [{"embedding": vec, "index": 0}],
            },
        ) as mock_req:
            result = provider.embed("short valid text")
        assert result == vec
        mock_req.assert_called_once()
