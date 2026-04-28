"""Embedding engine using local Ollama (nomic-embed-text)."""

import json
import logging
import struct
import urllib.request
import urllib.error

from .url_validate import is_safe_url

import math

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIMENSIONS = 768
OLLAMA_BASE = "http://localhost:11434"


class EmbeddingEngine:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE,
        max_cache_size: int = 2048,
    ):
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"Unsafe Ollama URL (must be http/https): {base_url!r}")
        if not is_safe_url(base_url, allow_remote=True):
            raise ValueError(f"Unsafe Ollama URL (blocked address): {base_url!r}")
        self._model = model
        self._base_url = base_url
        self._max_cache_size = max_cache_size
        self._cache: dict[str, list[float]] = {}

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        if text in self._cache:
            return self._cache[text]

        url = f"{self._base_url}/api/embeddings"
        payload = json.dumps({"model": self._model, "prompt": text}).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                vec = data.get("embedding", [])
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
        ) as e:
            log.error("llmem: embed: Ollama embedding request failed: %s", e)
            raise

        if len(self._cache) >= self._max_cache_size:
            self._cache.clear()
        self._cache[text] = vec
        return vec

    @staticmethod
    def vec_to_bytes(vec: list[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    def check_available(self) -> bool:
        """Check if the embedding model is available in Ollama."""
        url = f"{self._base_url}/api/tags"
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                return any(m.startswith(self._model) for m in models)
        except Exception:
            return False
