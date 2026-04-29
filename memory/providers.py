"""Provider abstraction layer for LLM embeddings and generation.

Defines EmbedProvider and GenerateProvider abstract base classes,
and implements them for Ollama, OpenAI, Anthropic, and a no-op NoneProvider.
Provides resolve_provider for graceful degradation across providers.
"""

import abc
import ipaddress
import json
import logging
import os
import urllib.request
import urllib.error
from urllib.parse import urlparse

from .ollama import check_ollama_model, _call_ollama_generate
from .url_validate import is_safe_url, safe_urlopen, _strip_credentials

log = logging.getLogger(__name__)

# Input size limits to prevent abuse (OOM, resource exhaustion).
# Individual text inputs longer than MAX_TEXT_LENGTH are rejected.
# Batch requests larger than MAX_BATCH_SIZE are rejected.
MAX_TEXT_LENGTH = 100_000  # 100K characters per text
MAX_BATCH_SIZE = 2048  # max texts per batch request


def _validate_embed_inputs(texts: list[str]) -> None:
    """Validate embed/embed_batch inputs against size limits.

    Args:
        texts: List of texts to validate.

    Raises:
        ValueError: If batch size exceeds MAX_BATCH_SIZE or any text
            exceeds MAX_TEXT_LENGTH.
    """
    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(
            f"providers: batch size {len(texts)} exceeds maximum {MAX_BATCH_SIZE}"
        )
    for i, text in enumerate(texts):
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"providers: text at index {i} exceeds maximum length "
                f"{MAX_TEXT_LENGTH} (got {len(text)} characters)"
            )


# Loopback hostnames that are safe for HTTP (non-HTTPS) API key delivery.
# These are checked via exact hostname match (not substring) to prevent
# bypass via URLs like http://localhost.evil.com or http://127.0.0.1.evil.com
_LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_loopback_hostname(url: str) -> bool:
    """Check whether a URL's hostname is a loopback address.

    Uses urlparse to extract the hostname and checks it against known
    loopback identifiers (string match) and also validates via ipaddress
    for IPv6-mapped IPv4 addresses (e.g. ::ffff:127.0.0.1).

    This prevents substring-matching bypasses where a URL like
    http://localhost.evil.com contains 'localhost' as a substring but
    actually resolves to a remote host.

    Args:
        url: The URL to check.

    Returns:
        True if the hostname is a loopback address.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return False
    if hostname in _LOOPBACK_HOSTNAMES:
        return True
    # Also check via ipaddress for IPv6-mapped loopback addresses
    # (e.g. ::ffff:127.0.0.1, ::ffff:7f00:1) that urlparse extracts
    # without brackets.
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_loopback:
            return True
    except ValueError:
        pass
    return False


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class EmbedProvider(abc.ABC):
    """Abstract base class for embedding providers.

    Concrete providers must implement embed, embed_batch, check_available,
    and dimension.
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

    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the output vector dimensionality of this provider.

        Must be implementable without making API calls (known from model
        metadata or constructor config).

        Returns:
            The number of floats in each embedding vector produced by
            this provider.
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
        _validate_embed_inputs(texts)
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
        if len(prompt) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"providers: prompt exceeds maximum length {MAX_TEXT_LENGTH} "
                f"(got {len(prompt)} characters)"
            )
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

    def dimension(self) -> int:
        """Return the embedding dimension for the configured Ollama model.

        Defaults to 768 (nomic-embed-text dimension). The dimension can
        be overridden via the embed_model constructor parameter — known
        model dimensions are looked up, others return the default.

        Returns:
            The number of floats in each embedding vector.
        """
        _KNOWN_OLLAMA_DIMENSIONS: dict[str, int] = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
            "snowflake-arctic-embed": 1024,
        }
        return _KNOWN_OLLAMA_DIMENSIONS.get(self._embed_model, 768)


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

    Security: API keys are only sent to HTTPS URLs or loopback HTTP URLs.
    When the base_url differs from the official OpenAI endpoint, a warning
    is logged to alert the user of potential credential exfiltration risk.
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
        # Block credential exfiltration: refuse to send API keys over
        # non-HTTPS to non-loopback hosts. Uses exact hostname matching
        # (not substring) to prevent bypass via localhost.evil.com.
        if base_url.startswith("http://") and not _is_loopback_hostname(base_url):
            raise ValueError(
                "providers: OpenAI API key cannot be sent over non-HTTPS to non-loopback URL "
                f"— use HTTPS or a localhost base URL, got {_strip_credentials(base_url)!r}"
            )
        if base_url != DEFAULT_OPENAI_BASE_URL:
            log.warning(
                "providers: OpenAI API key sent to non-default base_url %r "
                "— verify this is not a credential exfiltration attack",
                _strip_credentials(base_url),
            )
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

        Delegates to embed_batch to ensure input validation is applied
        consistently (MAX_TEXT_LENGTH, MAX_BATCH_SIZE checks).

        Args:
            text: The string to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
            ValueError: If text length exceeds limits.
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings via OpenAI /v1/embeddings.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
            ValueError: If batch size or text length exceeds limits.
        """
        _validate_embed_inputs(texts)
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
            ValueError: If prompt length exceeds limit.
        """
        if len(prompt) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"providers: prompt exceeds maximum length {MAX_TEXT_LENGTH} "
                f"(got {len(prompt)} characters)"
            )
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
        return result["choices"][0]["message"]["content"] or ""

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

    def dimension(self) -> int:
        """Return the embedding dimension for the configured OpenAI model.

        Defaults to 1536 (text-embedding-3-small dimension). The dimension
        can be overridden via the embed_model constructor parameter — known
        model dimensions are looked up, others return the default.

        Returns:
            The number of floats in each embedding vector.
        """
        _KNOWN_OPENAI_DIMENSIONS: dict[str, int] = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return _KNOWN_OPENAI_DIMENSIONS.get(self._embed_model, 1536)


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

    Security: API keys are only sent to HTTPS URLs or loopback HTTP URLs.
    When the base_url differs from the official Anthropic endpoint, a warning
    is logged to alert the user of potential credential exfiltration risk.
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
        # Block credential exfiltration: refuse to send API keys over
        # non-HTTPS to non-loopback hosts. Uses exact hostname matching
        # (not substring) to prevent bypass via localhost.evil.com.
        if base_url.startswith("http://") and not _is_loopback_hostname(base_url):
            raise ValueError(
                "providers: Anthropic API key cannot be sent over non-HTTPS to non-loopback URL "
                f"— use HTTPS or a localhost base URL, got {_strip_credentials(base_url)!r}"
            )
        if base_url != DEFAULT_ANTHROPIC_BASE_URL:
            log.warning(
                "providers: Anthropic API key sent to non-default base_url %r "
                "— verify this is not a credential exfiltration attack",
                _strip_credentials(base_url),
            )
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
            ValueError: If prompt length exceeds limit.
        """
        if len(prompt) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"providers: prompt exceeds maximum length {MAX_TEXT_LENGTH} "
                f"(got {len(prompt)} characters)"
            )
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
        return data["content"][0]["text"] or ""

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

    def dimension(self) -> int:
        """Return the configured zero-vector dimension.

        Returns:
            The number of floats in each embedding vector (defaults to 768).
        """
        return self._embed_dimensions


# ---------------------------------------------------------------------------
# SentenceTransformersProvider
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"

# Known sentence-transformers model dimensions.  Used by
# SentenceTransformersProvider.dimension() so it never needs
# to load the model (honouring the EmbedProvider ABC contract).
_KNOWN_LOCAL_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "paraphrase-mpnet-base-v2": 768,
    "all-roberta-large-v1": 1024,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "multi-qa-mpnet-base-dot-v1": 768,
}
_DEFAULT_LOCAL_DIMENSION = 384


class SentenceTransformersProvider(EmbedProvider):
    """Local embedding provider using sentence-transformers.

    Runs embeddings locally without any server dependency. Requires the
    ``sentence-transformers`` package (install via ``pip install llmem[local]``).

    The constructor does NOT make network calls — model download happens
    lazily on first ``embed()`` call or ``check_available()``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        dimensions: int | None = None,
    ):
        if not model_name or not isinstance(model_name, str):
            raise ValueError("providers: local: model_name must be a non-empty string")
        self._model_name = model_name
        self._dimensions = dimensions
        self._model = None
        self._model_loaded = False
        self._availability_checked = False
        self._available: bool | None = None

    def __repr__(self) -> str:
        return f"SentenceTransformersProvider(model_name={self._model_name!r})"

    def _load_model(self):
        """Lazily load the sentence-transformers model.

        Raises:
            ImportError: If sentence_transformers is not installed.
            RuntimeError: If the model fails to load.
        """
        if self._model_loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "providers: local: sentence-transformers is not installed. "
                "Install it with: pip install llmem[local]"
            )
        try:
            self._model = SentenceTransformer(self._model_name)
            self._model_loaded = True
        except Exception as e:
            raise RuntimeError(
                f"providers: local: failed to load model {self._model_name!r}: {e}"
            ) from e

    def embed(self, text: str) -> list[float]:
        """Embed a single text string using sentence-transformers.

        Args:
            text: The string to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RuntimeError: If the model failed to load.
        """
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings using sentence-transformers.

        Uses the SentenceTransformer.encode() batch API for efficiency.

        Args:
            texts: A list of strings to embed. Empty input returns empty list.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            RuntimeError: If the model failed to load.
            ValueError: If batch size or text length exceeds limits.
        """
        if not texts:
            return []
        _validate_embed_inputs(texts)
        try:
            self._load_model()
        except Exception:
            log.error("providers: local: embed failed: model not loaded", exc_info=True)
            raise
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return [row.tolist() for row in embeddings]
        except Exception as e:
            log.error("providers: local: embed failed: %s", e, exc_info=True)
            raise RuntimeError(f"providers: local: embed failed: {e}") from e

    def check_available(self) -> bool:
        """Check if the sentence-transformers model can be loaded.

        Returns:
            True if the model loads successfully, False on any error.
            Never raises. Caches the result so repeated calls don't
            re-load the model.
        """
        if self._availability_checked:
            return bool(self._available)
        try:
            self._load_model()
            self._available = True
        except Exception:
            log.debug("providers: local: check_available failed", exc_info=True)
            self._available = False
        self._availability_checked = True
        return bool(self._available)

    def dimension(self) -> int:
        """Return the embedding dimension for the model.

        If ``dimensions`` was provided in the constructor, returns that
        value. Otherwise, looks up the model name in the known-dimensions
        table. Unknown models return the default (384).

        This method is lightweight — it never loads the model or makes
        network calls, honouring the EmbedProvider ABC contract.

        Returns:
            The number of floats in each embedding vector.
        """
        if self._dimensions is not None:
            return self._dimensions
        return _KNOWN_LOCAL_DIMENSIONS.get(self._model_name, _DEFAULT_LOCAL_DIMENSION)


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

    provider_cfg = config.get("provider") or {}
    default_name = provider_cfg.get("default") or "ollama"

    # Also support the legacy memory.ollama_url for backward compat
    legacy_ollama_url = (config.get("memory") or {}).get("ollama_url")

    embed_cfg = provider_cfg.get("embed") or {}
    generate_cfg = provider_cfg.get("generate") or {}

    embed_provider_name = embed_cfg.get("provider") or default_name
    generate_provider_name = generate_cfg.get("provider") or default_name

    ollama_cfg = provider_cfg.get("ollama") or {}
    ollama_base_url = (
        ollama_cfg.get("base_url") or legacy_ollama_url or DEFAULT_OLLAMA_BASE_URL
    )

    local_cfg = provider_cfg.get("local") or {}

    # Resolve embed provider
    embed_provider = _resolve_embed_provider(
        embed_provider_name,
        embed_cfg,
        ollama_base_url,
        local_cfg,
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
    local_cfg: dict,
    config: dict,
) -> EmbedProvider:
    """Resolve the embed provider based on name and config.

    Args:
        name: Provider name ('ollama', 'openai', 'anthropic', 'local', 'none').
        embed_cfg: The provider.embed config section.
        ollama_base_url: The Ollama base URL.
        local_cfg: The provider.local config section.
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
        provider_cfg = config.get("provider") or {}
        openai_cfg = provider_cfg.get("openai") or {}
        api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.info(
                "providers: openai embed not available (no API key), trying fallback"
            )
            return _fallback_embed_provider(config, skip_openai=True)
        try:
            return OpenAIProvider(
                embed_model=model_override
                or (openai_cfg.get("embed_model") or DEFAULT_OPENAI_EMBED_MODEL),
                base_url=base_url_override
                or (openai_cfg.get("base_url") or DEFAULT_OPENAI_BASE_URL),
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
    elif name == "local":
        return _resolve_local_embed_provider(local_cfg, model_override, config)
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
        provider_cfg = config.get("provider") or {}
        openai_cfg = provider_cfg.get("openai") or {}
        api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log.info(
                "providers: openai generate not available (no API key), trying fallback"
            )
            return _fallback_generate_provider(config, skip_openai=True)
        try:
            return OpenAIProvider(
                generate_model=model_override
                or (openai_cfg.get("generate_model") or DEFAULT_OPENAI_GENERATE_MODEL),
                base_url=base_url_override
                or (openai_cfg.get("base_url") or DEFAULT_OPENAI_BASE_URL),
                api_key=api_key,
            )
        except ValueError:
            log.info(
                "providers: openai generate config invalid (bad URL), trying fallback"
            )
            return _fallback_generate_provider(config, skip_openai=True)
    elif name == "anthropic":
        provider_cfg = config.get("provider") or {}
        anthropic_cfg = provider_cfg.get("anthropic") or {}
        api_key = anthropic_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.info(
                "providers: anthropic generate not available (no API key), trying fallback"
            )
            return _fallback_generate_provider(config, skip_anthropic=True)
        try:
            return AnthropicProvider(
                model=model_override
                or (anthropic_cfg.get("generate_model") or DEFAULT_ANTHROPIC_MODEL),
                base_url=base_url_override
                or (anthropic_cfg.get("base_url") or DEFAULT_ANTHROPIC_BASE_URL),
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


def _resolve_local_embed_provider(
    local_cfg: dict,
    model_override: str | None,
    config: dict,
) -> EmbedProvider:
    """Resolve the local (sentence-transformers) embed provider.

    Args:
        local_cfg: The provider.local config section.
        model_override: Model name override from provider.embed.model.
        config: The full config dict.

    Returns:
        A SentenceTransformersProvider if available, or NoneProvider with warning.
    """
    model_name = model_override or local_cfg.get("model") or DEFAULT_LOCAL_MODEL
    try:
        provider = SentenceTransformersProvider(model_name=model_name)
    except ValueError:
        log.warning(
            "providers: local embed config invalid, falling back to NoneProvider"
        )
        return _fallback_embed_provider(config, skip_openai=False, skip_local=True)
    if provider.check_available():
        return provider
    log.warning(
        "providers: local embed provider (sentence-transformers) not available, "
        "falling back to NoneProvider. Install sentence-transformers with: "
        "pip install llmem[local]"
    )
    return _fallback_embed_provider(config, skip_openai=False, skip_local=True)


def _fallback_embed_provider(
    config: dict,
    skip_openai: bool = False,
    skip_local: bool = False,
) -> EmbedProvider:
    """Try fallback embed providers: local, openai, then none.

    Args:
        config: The full config dict.
        skip_openai: If True, skip OpenAI fallback.
        skip_local: If True, skip local (sentence-transformers) fallback.

    Returns:
        An EmbedProvider instance (at worst, NoneProvider).
    """
    provider_cfg = config.get("provider") or {}
    if not skip_local:
        local_cfg = provider_cfg.get("local") or {}
        local_model = local_cfg.get("model") or DEFAULT_LOCAL_MODEL
        try:
            provider = SentenceTransformersProvider(model_name=local_model)
            if provider.check_available():
                return provider
        except ValueError:
            log.info("providers: local embed not available in fallback, skipping")
    if not skip_openai:
        openai_cfg = provider_cfg.get("openai") or {}
        api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                return OpenAIProvider(
                    embed_model=openai_cfg.get("embed_model")
                    or DEFAULT_OPENAI_EMBED_MODEL,
                    base_url=openai_cfg.get("base_url") or DEFAULT_OPENAI_BASE_URL,
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
    provider_cfg = config.get("provider") or {}
    if not skip_openai:
        openai_cfg = provider_cfg.get("openai") or {}
        openai_api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            try:
                return OpenAIProvider(
                    generate_model=openai_cfg.get("generate_model")
                    or DEFAULT_OPENAI_GENERATE_MODEL,
                    base_url=openai_cfg.get("base_url") or DEFAULT_OPENAI_BASE_URL,
                    api_key=openai_api_key,
                )
            except ValueError:
                log.info("providers: openai generate fallback config invalid (bad URL)")
    if not skip_anthropic:
        anthropic_cfg = provider_cfg.get("anthropic") or {}
        anthropic_api_key = anthropic_cfg.get("api_key") or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if anthropic_api_key:
            try:
                return AnthropicProvider(
                    model=anthropic_cfg.get("generate_model")
                    or DEFAULT_ANTHROPIC_MODEL,
                    base_url=anthropic_cfg.get("base_url")
                    or DEFAULT_ANTHROPIC_BASE_URL,
                    api_key=anthropic_api_key,
                )
            except ValueError:
                log.info(
                    "providers: anthropic generate fallback config invalid (bad URL)"
                )
    log.info("providers: no generate provider available, using NoneProvider")
    return NoneProvider()
