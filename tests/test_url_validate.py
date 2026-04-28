"""Tests for URL validation — SSRF protection and scheme enforcement.

Verifies is_safe_url and validate_url logic directly, not via mocks.
Covers: scheme blocking, private IP blocking, link-local blocking,
loopback+port logic, allow_remote=True/False behavior, DNS resolution,
and validate_url raising ValueError.
"""

import ipaddress
from unittest.mock import patch

import pytest

from memory.url_validate import (
    OLLAMA_DEFAULT_PORT,
    _ip_is_blocked,
    _get_effective_port,
    is_safe_url,
    validate_url,
)


# ---------------------------------------------------------------------------
# _ip_is_blocked tests
# ---------------------------------------------------------------------------


class TestIPBlocked:
    """Verify _ip_is_blocked correctly classifies IP addresses."""

    @pytest.mark.parametrize(
        "ip_str",
        [
            "10.0.0.1",  # private class A
            "172.16.0.1",  # private class B
            "192.168.1.1",  # private class C
            "169.254.1.1",  # link-local
            "224.0.0.1",  # multicast
            "240.0.0.1",  # reserved
        ],
    )
    def test_blocked_ips(self, ip_str):
        ip = ipaddress.ip_address(ip_str)
        assert _ip_is_blocked(ip) is True

    @pytest.mark.parametrize(
        "ip_str",
        [
            "1.2.3.4",  # public
            "8.8.8.8",  # public
        ],
    )
    def test_allowed_ips_when_remote(self, ip_str):
        """Public IPs are NOT blocked (they are reachable)."""
        ip = ipaddress.ip_address(ip_str)
        assert _ip_is_blocked(ip) is False

    def test_loopback_is_not_blocked(self):
        """Loopback (127.0.0.1) is explicitly NOT blocked by _ip_is_blocked.
        Loopback access control happens at the is_safe_url level based on
        allow_remote and port.
        """
        ip = ipaddress.ip_address("127.0.0.1")
        assert _ip_is_blocked(ip) is False


# ---------------------------------------------------------------------------
# _get_effective_port tests
# ---------------------------------------------------------------------------


class TestGetEffectivePort:
    def test_explicit_port(self):
        from urllib.parse import urlparse

        parsed = urlparse("http://localhost:8080/path")
        assert _get_effective_port(parsed) == 8080

    def test_default_http_port(self):
        from urllib.parse import urlparse

        parsed = urlparse("http://localhost/path")
        assert _get_effective_port(parsed) == 80

    def test_default_https_port(self):
        from urllib.parse import urlparse

        parsed = urlparse("https://example.com/path")
        assert _get_effective_port(parsed) == 443


# ---------------------------------------------------------------------------
# is_safe_url scheme tests
# ---------------------------------------------------------------------------


class TestIsSafeURL_SchemeBlocking:
    """Only http and https schemes are allowed."""

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/file",
            "data:text/html,<script>alert(1)</script>",
            "gopher://example.com",
            "javascript:alert(1)",
            "ssh://user@host",
        ],
    )
    def test_rejects_non_http_schemes(self, url):
        assert is_safe_url(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434",
            "https://api.openai.com",
        ],
    )
    def test_accepts_http_and_https(self, url):
        # These may still fail for other reasons (e.g. no hostname),
        # but they should not fail on scheme alone
        # We test specifically that scheme check passes by using valid URLs
        # with mocked DNS resolution to avoid network dependency
        with patch("memory.url_validate._resolve_hostname", return_value=None):
            # allow_remote=True so hostname resolution failure is allowed
            result = is_safe_url(url, allow_remote=True)
            assert result is True


# ---------------------------------------------------------------------------
# is_safe_url hostname tests
# ---------------------------------------------------------------------------


class TestIsSafeURL_Hostname:
    def test_rejects_empty_hostname(self):
        assert is_safe_url("http:///path") is False

    def test_rejects_no_hostname(self):
        assert is_safe_url("http://") is False


# ---------------------------------------------------------------------------
# is_safe_url with allow_remote=False (default) — loopback + port
# ---------------------------------------------------------------------------


class TestIsSafeURL_LoopbackPortLogic:
    """When allow_remote=False, only loopback on the Ollama default port is allowed."""

    def test_loopback_on_default_port_allowed(self):
        assert is_safe_url(f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}") is True

    def test_loopback_on_non_default_port_blocked(self):
        assert is_safe_url("http://127.0.0.1:8080") is False

    def test_localhost_on_default_port_allowed(self):
        """'localhost' resolves to 127.0.0.1 — should be allowed on default port."""
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["127.0.0.1"],
        ):
            assert is_safe_url(f"http://localhost:{OLLAMA_DEFAULT_PORT}") is True

    def test_localhost_on_non_default_port_blocked(self):
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["127.0.0.1"],
        ):
            assert is_safe_url("http://localhost:8080") is False

    def test_loopback_https_on_non_default_port_blocked(self):
        """HTTPS on loopback defaults to port 443, not 11434, so blocked."""
        assert is_safe_url("https://127.0.0.1/") is False

    def test_loopback_https_on_ollama_port_allowed(self):
        """HTTPS on loopback with explicit Ollama port is allowed."""
        assert is_safe_url(f"https://127.0.0.1:{OLLAMA_DEFAULT_PORT}") is True


# ---------------------------------------------------------------------------
# is_safe_url private/link-local/reserved/multicast IP blocking
# ---------------------------------------------------------------------------


class TestIsSafeURL_PrivateIPBlocking:
    """Private, link-local, reserved, and multicast IPs are always blocked,
    regardless of allow_remote setting.
    """

    @pytest.mark.parametrize(
        "ip,url",
        [
            ("10.0.0.1", "http://10.0.0.1:11434"),
            ("172.16.0.1", "http://172.16.0.1:11434"),
            ("192.168.1.1", "http://192.168.1.1:11434"),
        ],
    )
    def test_private_ips_blocked_even_with_allow_remote(self, ip, url):
        assert is_safe_url(url, allow_remote=True) is False

    def test_link_local_blocked(self):
        assert is_safe_url("http://169.254.1.1:11434", allow_remote=True) is False

    def test_multicast_blocked(self):
        assert is_safe_url("http://224.0.0.1:11434", allow_remote=True) is False

    def test_reserved_blocked(self):
        assert is_safe_url("http://240.0.0.1:11434", allow_remote=True) is False


# ---------------------------------------------------------------------------
# is_safe_url DNS rebinding protection
# ---------------------------------------------------------------------------


class TestIsSafeURL_DNSRebinding:
    """When a hostname resolves to a blocked IP, the URL should be rejected."""

    def test_dns_resolving_to_private_ip_blocked(self):
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["10.0.0.1"],
        ):
            assert (
                is_safe_url("http://evil.example.com:11434", allow_remote=True) is False
            )

    def test_dns_resolving_to_loopback_blocked_when_not_allow_remote_and_non_default_port(
        self,
    ):
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["127.0.0.1"],
        ):
            assert is_safe_url("http://evil.example.com:8080") is False

    def test_dns_resolving_to_loopback_allowed_when_default_port(self):
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["127.0.0.1"],
        ):
            assert is_safe_url(f"http://evil.example.com:{OLLAMA_DEFAULT_PORT}") is True

    def test_dns_resolving_to_public_ip_allowed(self):
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["1.2.3.4"],
        ):
            assert (
                is_safe_url("http://api.example.com:11434", allow_remote=True) is True
            )

    def test_dns_resolution_failure_blocked_when_not_allow_remote(self):
        """If DNS fails and allow_remote=False, the URL is rejected (no way to verify safety)."""
        with patch("memory.url_validate._resolve_hostname", return_value=None):
            assert is_safe_url("http://unknown.host:11434") is False

    def test_dns_resolution_failure_allowed_when_allow_remote(self):
        """If DNS fails and allow_remote=True, the URL is allowed (host may be temporarily down)."""
        with patch("memory.url_validate._resolve_hostname", return_value=None):
            assert is_safe_url("http://unknown.host:11434", allow_remote=True) is True

    def test_dns_resolving_to_unparseable_addr_blocked(self):
        """If getaddrinfo returns a non-IP address string, it should be treated as blocked."""
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["not-an-ip"],
        ):
            assert (
                is_safe_url("http://evil.example.com:11434", allow_remote=True) is False
            )

    def test_dns_with_mixed_results_one_bad_blocks(self):
        """If DNS returns multiple IPs and one is blocked, the URL is rejected."""
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["1.2.3.4", "10.0.0.1"],
        ):
            assert (
                is_safe_url("http://evil.example.com:11434", allow_remote=True) is False
            )


# ---------------------------------------------------------------------------
# is_safe_url allow_remote=True behavior
# ---------------------------------------------------------------------------


class TestIsSafeURL_AllowRemote:
    """allow_remote=True allows any reachable hostname (still blocks blocked IPs)."""

    def test_public_ip_allowed_when_allow_remote(self):
        assert is_safe_url("http://1.2.3.4:443", allow_remote=True) is True

    def test_public_hostname_with_dns_allowed(self):
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["1.2.3.4"],
        ):
            assert is_safe_url("http://api.openai.com:443", allow_remote=True) is True

    def test_private_ip_still_blocked_with_allow_remote(self):
        assert is_safe_url("http://10.0.0.1:443", allow_remote=True) is False


# ---------------------------------------------------------------------------
# validate_url tests
# ---------------------------------------------------------------------------


class TestValidateURL:
    """validate_url returns the URL on success, raises ValueError on failure."""

    def test_returns_url_on_valid(self):
        url = f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}"
        assert validate_url(url) == url

    def test_raises_value_error_on_invalid(self):
        with pytest.raises(ValueError, match="URL rejected"):
            validate_url("file:///etc/passwd")

    def test_raises_value_error_on_blocked_ip(self):
        with pytest.raises(ValueError, match="URL rejected"):
            validate_url("http://10.0.0.1:11434")

    def test_raises_value_error_on_non_default_port_loopback(self):
        with pytest.raises(ValueError, match="URL rejected"):
            validate_url("http://127.0.0.1:8080")

    def test_validate_url_with_allow_remote(self):
        url = "http://1.2.3.4:443"
        assert validate_url(url, allow_remote=True) == url

    def test_validate_url_blocked_with_allow_remote(self):
        with pytest.raises(ValueError, match="URL rejected"):
            validate_url("http://10.0.0.1:443", allow_remote=True)
