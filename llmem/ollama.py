"""Ollama API helper for llmem."""

import json
import logging
import urllib.request
import urllib.error

from .url_validate import is_safe_url

log = logging.getLogger(__name__)

DEFAULT_BASE = "http://localhost:11434"


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
    if not is_safe_url(base_url, allow_remote=True):
        raise ValueError(f"llmem: ollama: unsafe URL: {base_url!r}")

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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        log.error("llmem: ollama: generate call failed: %s", e)
        return None
