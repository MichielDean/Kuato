"""Tests for Ollama API utilities.

Tests check_ollama_model and _call_ollama_generate directly,
not via mocks in test_providers.py. Covers base-model matching,
error paths, SSRF validation, and HTTP error handling.
"""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from memory.ollama import check_ollama_model, _call_ollama_generate


# ---------------------------------------------------------------------------
# check_ollama_model tests
# ---------------------------------------------------------------------------


class TestCheckOllamaModel:
    """Tests for check_ollama_model: model matching, error handling, SSRF."""

    def test_exact_model_match(self):
        """An exact model name match (e.g. 'qwen2.5:1.5b') should return True."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"models": [{"name": "qwen2.5:1.5b"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = check_ollama_model("qwen2.5:1.5b", "http://localhost:11434")
        assert result is True

    def test_base_model_match(self):
        """Splitting on ':' should match base names.
        'qwen2.5' should match 'qwen2.5:1.5b'.
        """
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"models": [{"name": "qwen2.5:1.5b"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = check_ollama_model("qwen2.5", "http://localhost:11434")
        assert result is True

    def test_model_not_found(self):
        """If the model isn't in the response, return False."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"models": [{"name": "llama3:8b"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = check_ollama_model("qwen2.5:1.5b", "http://localhost:11434")
        assert result is False

    def test_base_model_no_match(self):
        """If no model's base name matches, return False."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"models": [{"name": "llama3:8b"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = check_ollama_model("qwen2.5", "http://localhost:11434")
        assert result is False

    def test_empty_models_list(self):
        """Empty models list should return False."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"models": []}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = check_ollama_model("qwen2.5:1.5b", "http://localhost:11434")
        assert result is False

    def test_network_error_returns_false(self):
        """Network errors should return False, never raise."""
        with patch(
            "memory.ollama.safe_urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = check_ollama_model("qwen2.5:1.5b", "http://localhost:11434")
        assert result is False

    def test_http_error_returns_false(self):
        """HTTP errors should return False, never raise."""
        with patch(
            "memory.ollama.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:11434/api/tags",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=None,
            ),
        ):
            result = check_ollama_model("qwen2.5:1.5b", "http://localhost:11434")
        assert result is False

    def test_json_parse_error_returns_false(self):
        """Malformed JSON response should return False, never raise."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"not json"
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = check_ollama_model("qwen2.5:1.5b", "http://localhost:11434")
        assert result is False

    def test_unsafe_url_returns_false(self):
        """Unsafe URLs should be blocked and return False."""
        with patch("memory.ollama.is_safe_url", return_value=False):
            result = check_ollama_model("qwen2.5:1.5b", "http://169.254.169.254:11434")
        assert result is False

    def test_base_model_match_with_colon_tag(self):
        """Model 'myrepo/mymodel' should match 'myrepo/mymodel:latest'."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                {"models": [{"name": "myrepo/mymodel:latest"}]}
            ).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = check_ollama_model("myrepo/mymodel", "http://localhost:11434")
        assert result is True

    def test_uses_correct_endpoint(self):
        """Should call /api/tags endpoint."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"models": []}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            check_ollama_model("test", "http://localhost:11434")
        req = mock_open.call_args[0][0]
        assert req.full_url == "http://localhost:11434/api/tags"

    def test_uses_get_method(self):
        """The /api/tags request should use GET method."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"models": []}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            check_ollama_model("test", "http://localhost:11434")
        req = mock_open.call_args[0][0]
        assert req.method == "GET"


# ---------------------------------------------------------------------------
# _call_ollama_generate tests
# ---------------------------------------------------------------------------


class TestCallOllamaGenerate:
    """Tests for _call_ollama_generate: happy path, error paths, SSRF."""

    def test_happy_path(self):
        with patch("memory.ollama.safe_urlopen") as mock_open:
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

    def test_sends_correct_payload(self):
        """Verify the request payload contains model, prompt, stream, and options."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": "ok"}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            _call_ollama_generate(
                model="qwen2.5:1.5b",
                base_url="http://localhost:11434",
                prompt="test prompt",
                temperature=0.5,
                max_tokens=1024,
            )
        req = mock_open.call_args[0][0]
        body = json.loads(req.data.decode())
        assert body["model"] == "qwen2.5:1.5b"
        assert body["prompt"] == "test prompt"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.5
        assert body["options"]["num_predict"] == 1024

    def test_http_error_raises_runtime_error(self):
        """HTTP errors should raise RuntimeError with status code."""
        with patch(
            "memory.ollama.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:11434/api/generate",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=None,
            ),
        ):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                _call_ollama_generate(
                    model="test",
                    base_url="http://localhost:11434",
                    prompt="test",
                )

    def test_http_404_raises_runtime_error(self):
        """404 should raise RuntimeError (model not found)."""
        with patch(
            "memory.ollama.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:11434/api/generate",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=None,
            ),
        ):
            with pytest.raises(RuntimeError, match="HTTP 404"):
                _call_ollama_generate(
                    model="nonexistent",
                    base_url="http://localhost:11434",
                    prompt="test",
                )

    def test_unsafe_url_raises_value_error(self):
        """Unsafe URLs should raise ValueError."""
        with patch("memory.ollama.is_safe_url", return_value=False):
            with pytest.raises(ValueError, match="ollama.*URL must be http"):
                _call_ollama_generate(
                    model="test",
                    base_url="http://169.254.169.254:11434",
                    prompt="test",
                )

    def test_uses_correct_endpoint(self):
        """Should call /api/generate endpoint."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
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

    def test_uses_post_method(self):
        """The generate request should use POST method."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
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
        assert req.method == "POST"

    def test_sends_json_content_type(self):
        """The request should have Content-Type: application/json."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
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
        with patch("memory.ollama.safe_urlopen") as mock_open:
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

    def test_missing_response_key_returns_empty_string(self):
        """If the response lacks a 'response' key, data.get('response', '') returns ''."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
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

    def test_default_parameters(self):
        """Verify default temperature, max_tokens, and timeout."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
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
        body = json.loads(req.data.decode())
        assert body["options"]["temperature"] == 0.1
        assert body["options"]["num_predict"] == 2048
        call_kwargs = mock_open.call_args[1]
        assert call_kwargs.get("timeout") == 60


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-tiy84: check_ollama_model logs on exception
# ---------------------------------------------------------------------------


class TestCheckOllamaModel_LogsOnException:
    """check_ollama_model must log at debug level when exceptions occur,
    not silently swallow them. Operators need diagnostic info when health checks fail.
    """

    def test_logs_on_network_error(self):
        """Network errors should be logged, not silently swallowed."""
        with (
            patch(
                "memory.ollama.safe_urlopen",
                side_effect=urllib.error.URLError("Connection refused"),
            ),
            patch("memory.ollama.log") as mock_log,
        ):
            result = check_ollama_model("test", "http://localhost:11434")
        assert result is False
        mock_log.debug.assert_called_once_with(
            "ollama: model check failed", exc_info=True
        )

    def test_logs_on_http_error(self):
        """HTTP errors should be logged, not silently swallowed."""
        with (
            patch(
                "memory.ollama.safe_urlopen",
                side_effect=urllib.error.HTTPError(
                    url="http://localhost:11434/api/tags",
                    code=500,
                    msg="Internal Server Error",
                    hdrs=None,
                    fp=None,
                ),
            ),
            patch("memory.ollama.log") as mock_log,
        ):
            result = check_ollama_model("test", "http://localhost:11434")
        assert result is False
        mock_log.debug.assert_called_once_with(
            "ollama: model check failed", exc_info=True
        )

    def test_logs_on_json_parse_error(self):
        """Malformed JSON responses should be logged."""
        with (
            patch("memory.ollama.safe_urlopen") as mock_open,
            patch("memory.ollama.log") as mock_log,
        ):
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"not json"
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = check_ollama_model("test", "http://localhost:11434")
        assert result is False
        mock_log.debug.assert_called_once_with(
            "ollama: model check failed", exc_info=True
        )


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-odcyv: _call_ollama_generate contract — never returns None
# ---------------------------------------------------------------------------


class TestCallOllamaGenerate_NeverReturnsNone:
    """_call_ollama_generate docstring promises 'never returns None'.
    data.get('response', '') can return None when the key exists but has value None.
    Fix: use data.get('response') or '' which coerces both missing-key and
    None-value to empty string.
    """

    def test_returns_empty_string_when_response_key_is_null(self):
        """When API returns {'response': null}, must return '' not None."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": None}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
            )
        assert result == ""
        assert result is not None

    def test_returns_empty_string_when_response_key_missing(self):
        """When API returns {} (no 'response' key), must return '' not None."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
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
        assert result is not None

    def test_returns_actual_response_when_present(self):
        """When API returns a real response string, it should be returned."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
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

    def test_returns_empty_string_when_response_is_empty_string(self):
        """When API returns {'response': ''}, must return '' (not None)."""
        with patch("memory.ollama.safe_urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"response": ""}).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_resp
            result = _call_ollama_generate(
                model="test",
                base_url="http://localhost:11434",
                prompt="test",
            )
        assert result == ""
        assert result is not None


# ---------------------------------------------------------------------------
# Issue ll-1ztcx-6eq2d: _call_ollama_generate wraps URLError/OSError into RuntimeError
# ---------------------------------------------------------------------------


class TestCallOllamaGenerate_ConnectionErrors:
    """_call_ollama_generate must wrap URLError and OSError into RuntimeError.

    The documented contract says 'Raises RuntimeError'. Raw URLError/OSError
    would violate that contract. Connection failures must be wrapped.
    """

    def test_url_error_wrapped_as_runtime_error(self):
        """URLError (connection refused) must be wrapped in RuntimeError."""
        with patch(
            "memory.ollama.safe_urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(RuntimeError, match="generate request failed"):
                _call_ollama_generate(
                    model="test",
                    base_url="http://localhost:11434",
                    prompt="test",
                )

    def test_os_error_wrapped_as_runtime_error(self):
        """OSError (network unreachable) must be wrapped in RuntimeError."""
        with patch(
            "memory.ollama.safe_urlopen",
            side_effect=OSError("Network is unreachable"),
        ):
            with pytest.raises(RuntimeError, match="generate request failed"):
                _call_ollama_generate(
                    model="test",
                    base_url="http://localhost:11434",
                    prompt="test",
                )

    def test_http_error_still_raises_runtime_error(self):
        """HTTPError must still raise RuntimeError (existing behavior, unchanged)."""
        with patch(
            "memory.ollama.safe_urlopen",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:11434/api/generate",
                code=500,
                msg="Internal Server Error",
                hdrs=None,
                fp=None,
            ),
        ):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                _call_ollama_generate(
                    model="test",
                    base_url="http://localhost:11434",
                    prompt="test",
                )
