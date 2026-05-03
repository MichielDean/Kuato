"""Tests for _call_ollama_generate.

Tests cover happy path, error paths, SSRF validation, and HTTP error handling.
"""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from llmem.ollama import _call_ollama_generate, is_ollama_running, ProviderDetector


class TestCallOllamaGenerate:
    """Tests for _call_ollama_generate: happy path, error paths, SSRF."""

    def test_happy_path(self):
        with patch("llmem.url_validate.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"response": "Hello, world!"}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = _call_ollama_generate(
                model="qwen2.5:1.5b",
                base_url="http://localhost:11434",
                prompt="Say hello",
            )
        assert result == "Hello, world!"

    def test_returns_none_on_http_error(self):
        """HTTP errors should return None, not raise."""
        with patch(
            "llmem.url_validate.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:11434/api/generate",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=None,
            ),
        ):
            result = _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
            )
        assert result is None

    def test_unsafe_url_raises_value_error(self):
        """Unsafe URLs should raise ValueError."""
        with patch("llmem.ollama.is_safe_url", return_value=False):
            with pytest.raises(ValueError):
                _call_ollama_generate(
                    model="test",
                    base_url="http://169.254.169.254:11434",
                    prompt="test",
                )

    def test_uses_correct_endpoint(self):
        """Should call /api/generate endpoint."""
        with patch("llmem.url_validate.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": "ok"}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
            )
        req = mock_open.call_args[0][0]
        assert req.full_url == "http://localhost:11434/api/generate"

    def test_sends_data_payload(self):
        """The generate request should include data (POST)."""
        with patch("llmem.url_validate.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": "ok"}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
            )
        req = mock_open.call_args[0][0]
        assert req.data is not None
        body = json.loads(req.data.decode())
        assert body["model"] == "test"
        assert body["prompt"] == "test"
        assert body["stream"] is False

    def test_sends_json_content_type(self):
        """The request should have Content-Type: application/json."""
        with patch("llmem.url_validate.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": "ok"}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
            )
        req = mock_open.call_args[0][0]
        headers = dict(req.header_items())
        assert headers.get("Content-type") == "application/json"

    def test_timeout_passed_to_safe_urlopen(self):
        """The timeout parameter should be passed to safe_urlopen."""
        with patch("llmem.url_validate.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": "ok"}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
                timeout=120,
            )
        call_kwargs = mock_open.call_args[1]
        assert call_kwargs.get("timeout") == 120

    def test_returns_none_on_url_error(self):
        """URLError should return None, not raise."""
        with patch(
            "llmem.url_validate.safe_urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
            )
        assert result is None

    def test_returns_empty_string_when_response_key_missing(self):
        """When API returns {} (no 'response' key), must return '' not None."""
        with patch("llmem.url_validate.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
            )
        assert result == ""

    def test_returns_actual_response_when_present(self):
        """When API returns a real response string, it should be returned."""
        with patch("llmem.url_validate.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"response": "Hello, world!"}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
            )
        assert result == "Hello, world!"

    def test_empty_base_url_raises_value_error(self):
        """Empty base_url should raise ValueError."""
        with pytest.raises(ValueError):
            _call_ollama_generate(model="test", base_url="", prompt="test")


class TestIsOllamaRunning:
    """Tests for is_ollama_running."""

    def test_returns_true_when_ollama_responds(self):
        with patch("llmem.url_validate.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"models": [{"name": "qwen2.5:1.5b"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            assert is_ollama_running("http://localhost:11434") is True

    def test_returns_false_on_network_error(self):
        with patch(
            "llmem.url_validate.safe_urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            assert is_ollama_running("http://localhost:11434") is False

    def test_returns_false_on_bad_json(self):
        with patch("llmem.url_validate.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"not json"
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            assert is_ollama_running("http://localhost:11434") is False

    def test_empty_base_url_raises_value_error(self):
        with pytest.raises(ValueError):
            is_ollama_running("")


class TestProviderDetector:
    """Tests for ProviderDetector."""

    def test_detect_ollama(self):
        with patch("llmem.ollama.is_ollama_running", return_value=True):
            detector = ProviderDetector()
            result = detector.detect()
        assert result["provider"] == "ollama"

    def test_detect_no_provider(self):
        with patch("llmem.ollama.is_ollama_running", return_value=False):
            detector = ProviderDetector()
            result = detector.detect()
        assert result["provider"] == "none"