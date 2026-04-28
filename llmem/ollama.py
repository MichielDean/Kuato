"""Ollama API helper for llmem."""

import json
import logging
import os
import urllib.request
import urllib.error

from .url_validate import is_safe_url, _strip_credentials

log = logging.getLogger(__name__)

DEFAULT_BASE = "http://localhost:11434"


def is_ollama_running(base_url: str) -> bool:
    """Check whether Ollama is reachable and responding at the given base URL.

    Calls Ollama's ``/api/tags`` endpoint. Returns ``True`` if Ollama
    responds with a parseable JSON body containing a ``"models"`` key.
    Returns ``False`` on any error (connection refused, timeout, invalid
    JSON, HTTP error).

    Args:
        base_url: The Ollama base URL (e.g. ``"http://localhost:11434"``).
            Must be a non-empty string starting with ``http://`` or ``https://``.

    Returns:
        True if Ollama is reachable and responding, False otherwise.

    Raises:
        ValueError: If *base_url* is not a valid/safe URL.
    """
    if not base_url or not base_url.startswith(("http://", "https://")):
        raise ValueError(
            f"llmem: ollama: invalid base_url (must start with http:// or https://): "
            f"{base_url!r}"
        )
    if not is_safe_url(base_url, allow_remote=True):
        raise ValueError(f"llmem: ollama: unsafe URL: {_strip_credentials(base_url)!r}")

    from .url_validate import safe_urlopen

    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with safe_urlopen(url) as resp:
            data = json.loads(resp.read())
            return isinstance(data, dict) and "models" in data
    except Exception as e:
        log.debug("llmem: ollama: is_ollama_running probe failed: %s", e)
        return False


class ProviderDetector:
    """Detect available LLM providers from environment variables and HTTP probes.

    Checks (in order of precedence):
    1. Whether Ollama is reachable at the given base URL
    2. ``OPENAI_API_KEY`` environment variable
    3. ``ANTHROPIC_API_KEY`` environment variable

    Ollama takes precedence over API keys when running (local inference
    is the default preference). OpenAI takes precedence over Anthropic
    when both keys are set (more commonly used for embeddings).

    The detector is stateless — all detection is performed in ``detect()``
    via read-only env access and HTTP probes. No constructor parameters
    are required.
    """

    def __init__(self) -> None:
        pass

    def detect(self, ollama_url: str = DEFAULT_BASE) -> dict[str, str]:
        """Detect available LLM providers and return a results dict.

        Args:
            ollama_url: The Ollama base URL to probe. Defaults to
                ``"http://localhost:11434"``.

        Returns:
            A dict with keys:
            - ``"provider"``: One of ``"ollama"``, ``"openai"``,
              ``"anthropic"``, or ``"none"``.
            - ``"ollama_url"``: The detected or default Ollama URL.
            - ``"openai_key_found"``: ``"true"`` or ``"false"``.
            - ``"anthropic_key_found"``: ``"true"`` or ``"false"``.
        """
        ollama_running = is_ollama_running(ollama_url)
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

        provider = "none"
        if ollama_running:
            provider = "ollama"
        elif openai_key:
            provider = "openai"
        elif anthropic_key:
            provider = "anthropic"

        return {
            "provider": provider,
            "ollama_url": ollama_url,
            "openai_key_found": "true" if openai_key else "false",
            "anthropic_key_found": "true" if anthropic_key else "false",
        }


def _call_ollama_generate(
    model: str,
    prompt: str,
    base_url: str = DEFAULT_BASE,
    timeout: int = 120,
) -> str | None:
    """Call Ollama generate API and return the response text.

    Args:
        model: Model name (e.g., 'qwen2.5:1.5b').
        prompt: The prompt to send.
        base_url: Ollama base URL.
        timeout: Request timeout in seconds.

    Returns:
        The generated text, or None on error.
    """
    from .url_validate import safe_urlopen

    if not is_safe_url(base_url, allow_remote=True):
        raise ValueError(f"llmem: ollama: unsafe URL: {_strip_credentials(base_url)!r}")

    url = f"{base_url.rstrip('/')}/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with safe_urlopen(req, allow_remote=True) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        log.error("llmem: ollama: generate call failed: %s", e)
        return None
