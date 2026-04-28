"""Shared Ollama API utilities for LLM feature modules.

Provides a DRY helper for calling the Ollama /api/generate endpoint,
extracted from the repeated pattern in ExtractionEngine, ProspectiveIndexer,
and IntrospectionAnalyzer.
"""

import json
import logging
import urllib.request
import urllib.error

from .url_validate import is_safe_url

log = logging.getLogger(__name__)


def check_ollama_model(model: str, base_url: str) -> bool:
    """Check if the specified model is available at the Ollama endpoint.

    Returns True if the model name appears in the /api/tags response,
    False on any error (network, parse, model not found, SSRF validation). Never raises.

    Note: Matches both full model names (e.g. 'qwen2.5:1.5b') and
    base model names split on ':' (e.g. 'qwen2.5' matches 'qwen2.5:1.5b').
    This differs from EmbeddingEngine.check_available which uses exact
    equality on the full model tag.
    """
    if not is_safe_url(base_url, allow_remote=True):
        log.warning("ollama: model check blocked — unsafe URL: %s", base_url)
        return False
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        model_base = model.split(":")[0]
        return any(m == model or m.split(":")[0] == model_base for m in models)
    except Exception:
        return False


def _call_ollama_generate(
    model: str,
    base_url: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: int = 60,
) -> str:
    """Send a prompt to Ollama /api/generate and return the response text.

    Raises RuntimeError with HTTP error details for HTTP errors.
    Raises ValueError if base_url fails SSRF validation.
    Returns the response text string for successful calls; never returns None.
    """
    if not is_safe_url(base_url, allow_remote=True):
        raise ValueError(
            f"Unsafe Ollama URL in _call_ollama_generate (must be http/https to permitted address): {base_url!r}"
        )
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama API returned HTTP {e.code}: {e.reason}") from e
    return data.get("response", "")
