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
    DEFAULT_NONE_EMBED_DIMENSIONS,
    DEFAULT_OPENAI_BASE_URL,
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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

        with patch("memory.providers.urllib.request.urlopen", side_effect=mock_urlopen):
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
        assert call_args[0][1]["input"] == "test text"

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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
            "memory.providers.urllib.request.urlopen",
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
            "memory.providers.urllib.request.urlopen",
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
        assert call_kwargs.get("timeout") == 60, (
            f"Expected default timeout=60 passed to _make_request, got kwargs={call_kwargs}"
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
        with patch("memory.providers.urllib.request.urlopen") as mock_urlopen:
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
