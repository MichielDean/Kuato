"""Memory extraction using local Ollama (qwen2.5:1.5b)."""

import json
import logging
import re

import urllib.request
import urllib.error

from .taxonomy import ERROR_TAXONOMY, ERROR_TAXONOMY_KEYS, SELF_ASSESSMENT_FIELDS
from .url_validate import is_safe_url, _strip_credentials, validate_base_url

log = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE = "http://localhost:11434"

_SA_FIELD_LINES = "\n".join(f"{name}: {desc}" for name, desc in SELF_ASSESSMENT_FIELDS)
_TAXONOMY_LINES = "\n".join(f"- {k}: {v}" for k, v in ERROR_TAXONOMY.items())
_CATEGORY_CHOICES = ", ".join(ERROR_TAXONOMY_KEYS)

EXTRACTION_PROMPT = f"""You are a memory extraction system. Extract key memories from the text below.

Return a JSON array of objects with these fields:
- type: one of "fact", "decision", "preference", "event", "project_state", "conversation", "procedure", "self_assessment"
- content: a clear, specific statement (not vague)
- confidence: 0.0 to 1.0 (how certain this is a lasting memory)
- category: (self_assessment only) one of: {_CATEGORY_CHOICES}

If no memories are worth extracting, return an empty array [].

Text:
"""


class ExtractionEngine:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE):
        base_url = validate_base_url(base_url, module="extract")
        self._model = model
        self._base_url = base_url

    def extract(self, text: str) -> list[dict]:
        """Extract memories from text using Ollama."""
        from .url_validate import safe_urlopen

        prompt = EXTRACTION_PROMPT + text
        url = f"{self._base_url}/api/generate"
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
        ).encode()

        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with safe_urlopen(req, allow_remote=True) as resp:
                data = json.loads(resp.read())
                response_text = data.get("response", "").strip()
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
        ) as e:
            log.error("llmem: extract: Ollama extraction failed: %s", e)
            return []

        # Try to parse the JSON array from the response
        try:
            # Try direct parse first
            memories = json.loads(response_text)
            if isinstance(memories, list):
                return memories
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array from within markdown code blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
        if match:
            try:
                memories = json.loads(match.group(1).strip())
                if isinstance(memories, list):
                    return memories
            except json.JSONDecodeError:
                pass

        # Try to find a JSON array anywhere in the response
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if match:
            try:
                memories = json.loads(match.group(0))
                if isinstance(memories, list):
                    return memories
            except json.JSONDecodeError:
                pass

        log.warning("llmem: extract: could not parse extraction response as JSON array")
        return []

    def check_available(self) -> bool:
        """Check if the extraction model is available in Ollama."""
        from .url_validate import safe_urlopen

        url = f"{self._base_url}/api/tags"
        try:
            with safe_urlopen(url, allow_remote=True) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                return any(m.startswith(self._model) for m in models)
        except Exception:
            return False

    def pull_model(self) -> bool:
        """Pull the extraction model from Ollama."""
        from .url_validate import safe_urlopen

        url = f"{self._base_url}/api/pull"
        payload = json.dumps({"name": self._model, "stream": False}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with safe_urlopen(req, allow_remote=True) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "success"
        except Exception as e:
            log.error("llmem: extract: failed to pull model %s: %s", self._model, e)
            return False
