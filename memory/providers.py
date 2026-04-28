"""Provider abstraction layer for LLM embeddings and generation.

Defines EmbedProvider and GenerateProvider abstract base classes,
and implements them for Ollama, OpenAI, Anthropic, and a no-op NoneProvider.
Provides resolve_provider for graceful degradation across providers.
"""

import abc
import json
import logging
import os
import urllib.request
import urllib.error

from .ollama import check_ollama_model, _call_ollama_generate
from .url_validate import is_safe_url, safe_urlopen

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class EmbedProvider(abc.ABC):
    """Abstract base class for embedding providers.

    Concrete providers must implement embed, embed_batch, and check_available.
    """

    @abc.abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a float vector.

        Args:
            text: The string to embed.

        Returns:
            A list of floats representing the embedding vector.
        """

    @abc.abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings into float vectors.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """

    @abc.abstractmethod
    def check_available(self) -> bool:
        """Check whether the provider endpoint is reachable and the model exists.

        Returns:
            True if available, False on any error. Never raises.
        """


class GenerateProvider(abc.ABC):
    """Abstract base class for text generation providers.

    Concrete providers must implement generate and check_available.
    """

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int | None = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt text.
            temperature: Sampling temperature. Defaults to 0.1.
            max_tokens: Maximum tokens to generate. Defaults to 2048.
            timeout: Request timeout in seconds. Defaults to None, which
                uses the constructor-configured timeout.

        Returns:
            The generated text string.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
            ValueError: On URL validation failures.
        """

    @abc.abstractmethod
    def check_available(self) -> bool:
        """Check whether the provider endpoint is reachable and the model exists.

        Returns:
            True if available, False on any error. Never raises.
        """


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_GENERATE_MODEL = "qwen2.5:1.5b"


class OllamaProvider(EmbedProvider, GenerateProvider):
    """Ollama provider implementing both embedding and generation.

    Delegates embed calls to the Ollama /api/embeddings endpoint,
    and generate calls to _call_ollama_generate from memory.ollama.
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_OLLAMA_EMBED_MODEL,
        generate_model: str = DEFAULT_OLLAMA_GENERATE_MODEL,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: int = 60,
    ):
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("providers: Ollama URL must be http/https")
        if not is_safe_url(base_url, allow_remote=True):
            raise ValueError("providers: Ollama URL blocked (unsafe address)")
        self._embed_model = embed_model
        self._generate_model = generate_model
        self._base_url = base_url
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"OllamaProvider(base_url={self._base_url!r})"

    def embed(self, text: str) -> list[float]:
        """Embed a single text string via Ollama /api/embeddings.

        Args:
            text: The string to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
            ValueError: On URL validation failures or unexpected response format.
        """
        return self._embed_batch_internal([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings via Ollama /api/embeddings.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
            ValueError: On URL validation failures or unexpected response format.
        """
        return self._embed_batch_internal(texts)

    def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            payload = json.dumps(
                {
                    "model": self._embed_model,
                    "prompt": text,
                }
            ).encode()
            req = urllib.request.Request(
                f"{self._base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with safe_urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                raise RuntimeError(
                    f"providers: Ollama embedding API returned HTTP {e.code}: {e.reason}"
                ) from e
            except (urllib.error.URLError, OSError) as e:
                raise RuntimeError(
                    f"providers: Ollama embedding request failed: {e.reason if hasattr(e, 'reason') else e}"
                ) from e
            if "embedding" not in data:
                raise ValueError(
                    f"providers: unexpected Ollama embedding response: {list(data.keys())}"
                )
            results.append(data["embedding"])
        return results

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int | None = None,
    ) -> str:
        """Generate text from a prompt via Ollama /api/generate.

        Delegates to _call_ollama_generate from memory.ollama.

        Args:
            prompt: The prompt text.
            temperature: Sampling temperature. Defaults to 0.1.
            max_tokens: Maximum tokens to generate. Defaults to 2048.
            timeout: Request timeout in seconds. Defaults to None, which
                uses the constructor-configured timeout.

        Returns:
            The generated text string.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
            ValueError: On URL validation failures.
        """
        effective_timeout = timeout if timeout is not None else self._timeout
        return _call_ollama_generate(
            model=self._generate_model,
            base_url=self._base_url,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=effective_timeout,
        )

    def check_available(self) -> bool:
        """Check if both the embed and generate models are available at the Ollama endpoint.

        Returns:
            True if both models are found, False on any error. Never raises.
        """
        return check_ollama_model(
            self._embed_model, self._base_url
        ) and check_ollama_model(self._generate_model, self._base_url)


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_GENERATE_MODEL = "gpt-4o-mini"


class OpenAIProvider(EmbedProvider, GenerateProvider):
    """OpenAI provider implementing both embedding and generation.

    Uses the OpenAI /v1/embeddings and /v1/chat/completions endpoints.
    API key is sourced from the constructor parameter or the OPENAI_API_KEY
    environment variable.
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_OPENAI_EMBED_MODEL,
        generate_model: str = DEFAULT_OPENAI_GENERATE_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        timeout: int = 60,
    ):
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "providers: OpenAI API key required — pass api_key or set OPENAI_API_KEY env var"
            )
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("providers: OpenAI URL must be http/https")
        if not is_safe_url(base_url, allow_remote=True):
            raise ValueError("providers: OpenAI URL blocked (unsafe address)")
        self._embed_model = embed_model
        self._generate_model = generate_model
        self._base_url = base_url
        self._api_key = resolved_key
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"OpenAIProvider(base_url={self._base_url!r}, api_key='***masked***')"

    def _make_request(
        self,
        endpoint: str,
        payload: dict,
        timeout: int | None = None,
    ) -> dict:
        """Send a JSON POST request to an OpenAI endpoint.

        Args:
            endpoint: The API endpoint path (e.g., '/v1/embeddings').
            payload: The JSON-serializable request body.
            timeout: Request timeout in seconds. Defaults to self._timeout
                if None.

        Returns:
            The parsed JSON response as a dict.

        Raises:
            RuntimeError: On HTTP errors (with status code) or connection failures.
        """
        effective_timeout = timeout if timeout is not None else self._timeout
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._base_url}{endpoint}",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with safe_urlopen(req, timeout=effective_timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"providers: OpenAI API returned HTTP {e.code}: {e.reason}"
            ) from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(
                f"providers: OpenAI request failed: {e.reason if hasattr(e, 'reason') else e}"
            ) from e

    def embed(self, text: str) -> list[float]:
        """Embed a single text string via OpenAI /v1/embeddings.

        Args:
            text: The string to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
        """
        result = self._make_request(
            "/v1/embeddings",
            {
                "model": self._embed_model,
                "input": text,
            },
            timeout=self._timeout,
        )
        return result["data"][0]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings via OpenAI /v1/embeddings.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
        """
        result = self._make_request(
            "/v1/embeddings",
            {
                "model": self._embed_model,
                "input": texts,
            },
            timeout=self._timeout,
        )
        # OpenAI returns embeddings in the same order as input
        sorted_data = sorted(result["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in sorted_data]

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int | None = None,
    ) -> str:
        """Generate text from a prompt via OpenAI /v1/chat/completions.

        Args:
            prompt: The prompt text.
            temperature: Sampling temperature. Defaults to 0.1.
            max_tokens: Maximum tokens to generate. Defaults to 2048.
            timeout: Request timeout in seconds. Defaults to None, which
                uses the constructor-configured timeout.

        Returns:
            The generated text string.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
        """
        result = self._make_request(
            "/v1/chat/completions",
            {
                "model": self._generate_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        return result["choices"][0]["message"]["content"]

    def check_available(self) -> bool:
        """Check if the OpenAI API key is valid by calling /v1/models.

        Returns:
            True if the models endpoint returns 200, False on any error. Never raises.
        """
        try:
            req = urllib.request.Request(
                f"{self._base_url}/v1/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                method="GET",
            )
            with safe_urlopen(req, timeout=self._timeout) as resp:
                json.loads(resp.read())
            return True
        except Exception:
            log.debug("providers: OpenAI check_available failed", exc_info=True)
            return False


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(GenerateProvider):
    """Anthropic provider implementing text generation only.

    Anthropic does not provide an embedding API, so this provider
    implements GenerateProvider only. Use OpenAIProvider for embeddings.
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
        api_key: str | None = None,
        timeout: int = 60,
    ):
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "providers: Anthropic API key required — pass api_key or set ANTHROPIC_API_KEY env var"
            )
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("providers: Anthropic URL must be http/https")
        if not is_safe_url(base_url, allow_remote=True):
            raise ValueError("providers: Anthropic URL blocked (unsafe address)")
        self._model = model
        self._base_url = base_url
        self._api_key = resolved_key
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"AnthropicProvider(base_url={self._base_url!r}, api_key='***masked***')"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int | None = None,
    ) -> str:
        """Generate text from a prompt via Anthropic /v1/messages.

        Args:
            prompt: The prompt text.
            temperature: Sampling temperature. Defaults to 0.1.
            max_tokens: Maximum tokens to generate. Defaults to 2048.
            timeout: Request timeout in seconds. Defaults to None, which
                uses the constructor-configured timeout.

        Returns:
            The generated text string.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
        """
        effective_timeout = timeout if timeout is not None else self._timeout
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            method="POST",
        )
        try:
            with safe_urlopen(req, timeout=effective_timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"providers: Anthropic API returned HTTP {e.code}: {e.reason}"
            ) from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(
                f"providers: Anthropic request failed: {e.reason if hasattr(e, 'reason') else e}"
            ) from e
        return data["content"][0]["text"]

    def check_available(self) -> bool:
        """Check if the Anthropic API key is valid by sending a minimal request.

        Returns:
            True if a minimal request succeeds (2xx), False on any error. Never raises.
        """
        try:
            payload = json.dumps(
                {
                    "model": self._model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                }
            ).encode()
            req = urllib.request.Request(
                f"{self._base_url}/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                method="POST",
            )
            with safe_urlopen(req, timeout=self._timeout) as resp:
                json.loads(resp.read())
            return True
        except Exception:
            log.debug("providers: Anthropic check_available failed", exc_info=True)
            return False


# ---------------------------------------------------------------------------
# NoneProvider
# ---------------------------------------------------------------------------

DEFAULT_NONE_EMBED_DIMENSIONS = 768


class NoneProvider(EmbedProvider, GenerateProvider):
    """No-op fallback provider implementing both protocols.

    Used when no LLM provider is available (FTS5-only mode).
    """

    def __init__(self, embed_dimensions: int = DEFAULT_NONE_EMBED_DIMENSIONS):
        self._embed_dimensions = embed_dimensions

    def __repr__(self) -> str:
        return f"NoneProvider(embed_dimensions={self._embed_dimensions})"

    def embed(self, text: str) -> list[float]:
        """Return a zero vector of the configured dimension.

        Args:
            text: Ignored.

        Returns:
            A list of 0.0 floats with length equal to embed_dimensions.
        """
        return [0.0] * self._embed_dimensions

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return zero vectors for each input text.

        Args:
            texts: A list of strings (ignored).

        Returns:
            A list of zero vectors, one per input text.
        """
        return [self.embed(t) for t in texts]

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: int | None = None,
    ) -> str:
        """Return an empty string.

        Args:
            prompt: Ignored.
            temperature: Ignored.
            max_tokens: Ignored.
            timeout: Ignored.

        Returns:
            An empty string.
        """
        return ""

    def check_available(self) -> bool:
        """Always returns False.

        Returns:
            False — NoneProvider is never available.
        """
        return False


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def resolve_provider(
    config: dict | None = None,
) -> tuple[EmbedProvider, GenerateProvider]:
    """Resolve the best available provider from config.

    Reads ``provider.default`` from config for explicit provider name.
    If not set, tries Ollama (by calling check_available), then OpenAI
    (if OPENAI_API_KEY in env), then Anthropic (if ANTHROPIC_API_KEY in env).
    Falls back to NoneProvider.

    Per-operation overrides: provider.embed and provider.generate config
    sections can specify different provider names and models.

    Args:
        config: Optional pre-loaded config dict. If None, an empty dict is used.

    Returns:
        A tuple of (embed_provider, generate_provider). Providers may be
        different types (e.g., OpenAIProvider for embeddings, AnthropicProvider
        for generation). Never returns None — NoneProvider is the floor.
    """
    if config is None:
        config = {}

    provider_cfg = config.get("provider", {})
    default_name = provider_cfg.get("default", "ollama")

    # Also support the legacy memory.ollama_url for backward compat
    legacy_ollama_url = config.get("memory", {}).get("ollama_url")

    embed_cfg = provider_cfg.get("embed", {})
    generate_cfg = provider_cfg.get("generate", {})

    embed_provider_name = embed_cfg.get("provider") or default_name
    generate_provider_name = generate_cfg.get("provider") or default_name

    ollama_cfg = provider_cfg.get("ollama", {})
    ollama_base_url = (
        ollama_cfg.get("base_url") or legacy_ollama_url or DEFAULT_OLLAMA_BASE_URL
    )

    # Resolve embed provider
    embed_provider = _resolve_embed_provider(
        embed_provider_name,
        embed_cfg,
        ollama_base_url,
        config,
    )

    # Resolve generate provider
    generate_provider = _resolve_generate_provider(
        generate_provider_name,
        generate_cfg,
        ollama_base_url,
        config,
    )

    return embed_provider, generate_provider


def _resolve_embed_provider(
    name: str,
    embed_cfg: dict,
    ollama_base_url: str,
    config: dict,
) -> EmbedProvider:
    """Resolve the embed provider based on name and config.

    Args:
        name: Provider name ('ollama', 'openai', 'anthropic', 'none').
        embed_cfg: The provider.embed config section.
        ollama_base_url: The Ollama base URL.
        config: The full config dict.

    Returns:
        An EmbedProvider instance.
    """
    model_override = embed_cfg.get("model")
    base_url_override = embed_cfg.get("base_url")

    if name == "ollama":
        try:
            provider = OllamaProvider(
                embed_model=model_override or DEFAULT_OLLAMA_EMBED_MODEL,
                base_url=base_url_override or ollama_base_url,
            )
        except ValueError:
            log.info(
                "providers: ollama embed config invalid (bad URL), trying fallback"
            )
            return _fallback_embed_provider(config)
        if provider.check_available():
            return provider
        log.info("providers: ollama embed not available, trying fallback")
        return _fallback_embed_provider(config)
    elif name == "openai":
        provider_cfg = config.get("provider", {})
        openai_cfg = provider_cfg.get("openai", {})
        api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.info(
                "providers: openai embed not available (no API key), trying fallback"
            )
            return _fallback_embed_provider(config, skip_openai=True)
        try:
            return OpenAIProvider(
                embed_model=model_override
                or openai_cfg.get("embed_model", DEFAULT_OPENAI_EMBED_MODEL),
                base_url=base_url_override
                or openai_cfg.get("base_url", DEFAULT_OPENAI_BASE_URL),
                api_key=api_key,
            )
        except ValueError:
            log.info(
                "providers: openai embed config invalid (bad URL), trying fallback"
            )
            return _fallback_embed_provider(config, skip_openai=True)
    elif name == "anthropic":
        log.info("providers: anthropic has no embedding API, trying fallback")
        return _fallback_embed_provider(config)
    elif name == "none":
        return NoneProvider()
    else:
        log.warning("providers: unknown embed provider name %r, falling back", name)
        return _fallback_embed_provider(config)


def _resolve_generate_provider(
    name: str,
    generate_cfg: dict,
    ollama_base_url: str,
    config: dict,
) -> GenerateProvider:
    """Resolve the generate provider based on name and config.

    Args:
        name: Provider name ('ollama', 'openai', 'anthropic', 'none').
        generate_cfg: The provider.generate config section.
        ollama_base_url: The Ollama base URL.
        config: The full config dict.

    Returns:
        A GenerateProvider instance.
    """
    model_override = generate_cfg.get("model")
    base_url_override = generate_cfg.get("base_url")

    if name == "ollama":
        try:
            provider = OllamaProvider(
                generate_model=model_override or DEFAULT_OLLAMA_GENERATE_MODEL,
                base_url=base_url_override or ollama_base_url,
            )
        except ValueError:
            log.info(
                "providers: ollama generate config invalid (bad URL), trying fallback"
            )
            return _fallback_generate_provider(config)
        if provider.check_available():
            return provider
        log.info("providers: ollama generate not available, trying fallback")
        return _fallback_generate_provider(config)
    elif name == "openai":
        provider_cfg = config.get("provider", {})
        openai_cfg = provider_cfg.get("openai", {})
        api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.info(
                "providers: openai generate not available (no API key), trying fallback"
            )
            return _fallback_generate_provider(config, skip_openai=True)
        try:
            return OpenAIProvider(
                generate_model=model_override
                or openai_cfg.get("generate_model", DEFAULT_OPENAI_GENERATE_MODEL),
                base_url=base_url_override
                or openai_cfg.get("base_url", DEFAULT_OPENAI_BASE_URL),
                api_key=api_key,
            )
        except ValueError:
            log.info(
                "providers: openai generate config invalid (bad URL), trying fallback"
            )
            return _fallback_generate_provider(config, skip_openai=True)
    elif name == "anthropic":
        provider_cfg = config.get("provider", {})
        anthropic_cfg = provider_cfg.get("anthropic", {})
        api_key = anthropic_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.info(
                "providers: anthropic generate not available (no API key), trying fallback"
            )
            return _fallback_generate_provider(config, skip_anthropic=True)
        try:
            return AnthropicProvider(
                model=model_override
                or anthropic_cfg.get("generate_model", DEFAULT_ANTHROPIC_MODEL),
                base_url=base_url_override
                or anthropic_cfg.get("base_url", DEFAULT_ANTHROPIC_BASE_URL),
                api_key=api_key,
            )
        except ValueError:
            log.info(
                "providers: anthropic generate config invalid (bad URL), trying fallback"
            )
            return _fallback_generate_provider(config, skip_anthropic=True)
    elif name == "none":
        return NoneProvider()
    else:
        log.warning("providers: unknown generate provider name %r, falling back", name)
        return _fallback_generate_provider(config)


def _fallback_embed_provider(
    config: dict,
    skip_openai: bool = False,
) -> EmbedProvider:
    """Try fallback embed providers: openai, then none.

    Args:
        config: The full config dict.
        skip_openai: If True, skip OpenAI fallback.

    Returns:
        An EmbedProvider instance (at worst, NoneProvider).
    """
    if not skip_openai:
        provider_cfg = config.get("provider", {})
        openai_cfg = provider_cfg.get("openai", {})
        api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                return OpenAIProvider(
                    embed_model=openai_cfg.get(
                        "embed_model", DEFAULT_OPENAI_EMBED_MODEL
                    ),
                    base_url=openai_cfg.get("base_url", DEFAULT_OPENAI_BASE_URL),
                    api_key=api_key,
                )
            except ValueError:
                log.info("providers: openai embed fallback config invalid (bad URL)")
    log.info("providers: no embed provider available, using NoneProvider")
    return NoneProvider()


def _fallback_generate_provider(
    config: dict,
    skip_openai: bool = False,
    skip_anthropic: bool = False,
) -> GenerateProvider:
    """Try fallback generate providers: openai, anthropic, then none.

    Args:
        config: The full config dict.
        skip_openai: If True, skip OpenAI fallback.
        skip_anthropic: If True, skip Anthropic fallback.

    Returns:
        A GenerateProvider instance (at worst, NoneProvider).
    """
    provider_cfg = config.get("provider", {})
    if not skip_openai:
        openai_cfg = provider_cfg.get("openai", {})
        openai_api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            try:
                return OpenAIProvider(
                    generate_model=openai_cfg.get(
                        "generate_model", DEFAULT_OPENAI_GENERATE_MODEL
                    ),
                    base_url=openai_cfg.get("base_url", DEFAULT_OPENAI_BASE_URL),
                    api_key=openai_api_key,
                )
            except ValueError:
                log.info("providers: openai generate fallback config invalid (bad URL)")
    if not skip_anthropic:
        anthropic_cfg = provider_cfg.get("anthropic", {})
        anthropic_api_key = anthropic_cfg.get("api_key") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if anthropic_api_key:
            try:
                return AnthropicProvider(
                    model=anthropic_cfg.get("generate_model", DEFAULT_ANTHROPIC_MODEL),
                    base_url=anthropic_cfg.get("base_url", DEFAULT_ANTHROPIC_BASE_URL),
                    api_key=anthropic_api_key,
                )
            except ValueError:
                log.info(
                    "providers: anthropic generate fallback config invalid (bad URL)"
                )
    log.info("providers: no generate provider available, using NoneProvider")
    return NoneProvider()
