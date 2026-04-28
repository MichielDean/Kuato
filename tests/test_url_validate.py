"""Tests for URL validation — SSRF protection and scheme enforcement.

Verifies is_safe_url and validate_url logic directly, not via mocks.
Covers: scheme blocking, private IP blocking, link-local blocking,
loopback+port logic, allow_remote=True/False behavior, DNS resolution,
credential rejection, default timeout, safe_urlopen validation,
and validate_url raising ValueError.
"""

import ipaddress
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from memory.url_validate import (
    DEFAULT_URLOPEN_TIMEOUT,
    OLLAMA_DEFAULT_PORT,
    _check_ip_access,
    _ip_is_blocked,
    _get_effective_port,
    _strip_credentials,
    is_safe_url,
    validate_url,
    sanitize_url_for_log,
    safe_urlopen,
    SafeRedirectHandler,
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
# _check_ip_access tests (loopback port restriction)
# ---------------------------------------------------------------------------


class TestCheckIpAccess:
    """Verify _check_ip_access restricts loopback to Ollama port only."""

    def test_loopback_on_ollama_port_allowed(self):
        """Loopback on the Ollama default port is always allowed."""
        ip = ipaddress.ip_address("127.0.0.1")
        assert (
            _check_ip_access(ip, allow_remote=False, port=OLLAMA_DEFAULT_PORT) is True
        )
        assert _check_ip_access(ip, allow_remote=True, port=OLLAMA_DEFAULT_PORT) is True

    def test_loopback_on_non_ollama_port_blocked_even_with_allow_remote(self):
        """Loopback on a non-Ollama port is blocked even when allow_remote=True."""
        ip = ipaddress.ip_address("127.0.0.1")
        assert _check_ip_access(ip, allow_remote=True, port=8080) is False
        assert _check_ip_access(ip, allow_remote=False, port=8080) is False

    def test_public_ip_blocked_when_allow_remote_false(self):
        """Public IPs are blocked when allow_remote=False."""
        ip = ipaddress.ip_address("1.2.3.4")
        assert _check_ip_access(ip, allow_remote=False, port=443) is False

    def test_public_ip_allowed_when_allow_remote_true(self):
        """Public IPs are allowed when allow_remote=True."""
        ip = ipaddress.ip_address("1.2.3.4")
        assert _check_ip_access(ip, allow_remote=True, port=443) is True

    def test_private_ip_always_blocked(self):
        """Private IPs are blocked regardless of allow_remote."""
        ip = ipaddress.ip_address("10.0.0.1")
        assert _check_ip_access(ip, allow_remote=False, port=11434) is False
        assert _check_ip_access(ip, allow_remote=True, port=11434) is False


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
# is_safe_url credential rejection
# ---------------------------------------------------------------------------


class TestIsSafeURL_CredentialRejection:
    """URLs with embedded credentials (user:password@) must be rejected."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://user:pass@127.0.0.1:11434/api",
            "http://admin@127.0.0.1:11434/api",
            "http://user:pass@10.0.0.1:11434/api",
            "https://key:secret@api.openai.com/v1/models",
        ],
    )
    def test_rejects_urls_with_credentials(self, url):
        """URLs containing user:password@ or user@ must be rejected."""
        assert is_safe_url(url) is False
        assert is_safe_url(url, allow_remote=True) is False

    def test_rejects_credential_url_even_on_safe_host(self):
        """Even a safe loopback URL is rejected if it contains credentials."""
        assert (
            is_safe_url(f"http://user:pass@127.0.0.1:{OLLAMA_DEFAULT_PORT}/api")
            is False
        )


# ---------------------------------------------------------------------------
# is_safe_url with allow_remote=False (default) — loopback + port
# ---------------------------------------------------------------------------


class TestIsSafeURL_LoopbackPortLogic:
    """Loopback addresses are only permitted on the Ollama default port,
    regardless of allow_remote setting. This prevents SSRF attacks that
    would access other loopback services (Redis on 6379, etc.).
    """

    def test_loopback_on_default_port_allowed(self):
        assert is_safe_url(f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}") is True

    def test_loopback_on_non_default_port_blocked(self):
        assert is_safe_url("http://127.0.0.1:8080") is False

    def test_loopback_on_non_default_port_blocked_even_with_allow_remote(self):
        """allow_remote=True does NOT open up arbitrary loopback ports."""
        assert is_safe_url("http://127.0.0.1:8080", allow_remote=True) is False

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

    def test_dns_resolving_to_loopback_blocked_on_non_default_port(self):
        """DNS resolving to loopback on non-Ollama port is blocked even with allow_remote=True."""
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["127.0.0.1"],
        ):
            assert is_safe_url("http://evil.example.com:8080") is False
            assert (
                is_safe_url("http://evil.example.com:8080", allow_remote=True) is False
            )

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
    """allow_remote=True allows any reachable hostname (still blocks blocked IPs).
    Loopback is still restricted to the Ollama port only.
    """

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

    def test_loopback_on_arbitrary_port_blocked_with_allow_remote(self):
        """allow_remote=True does NOT permit loopback on arbitrary ports."""
        assert is_safe_url("http://127.0.0.1:6379", allow_remote=True) is False


class TestIsSafeURL_PublicIPBlockedByDefault:
    """SSRF bypass fix: when allow_remote=False (default), public IPs must be blocked.

    Previously, _ip_is_blocked returned False for public IPs, and the code
    only checked port restrictions for loopback — meaning public IPs like
    93.184.216.34 would be allowed through even with allow_remote=False.
    The documented contract says allow_remote=False restricts to loopback only.
    """

    def test_public_ip_blocked_when_allow_remote_false(self):
        """Public IP addresses must be blocked when allow_remote=False."""
        assert is_safe_url("http://93.184.216.34:11434") is False

    def test_public_ip_on_port_80_blocked_when_allow_remote_false(self):
        """Public IPs on any port are blocked when allow_remote=False."""
        assert is_safe_url("http://93.184.216.34:80") is False

    def test_public_ip_on_port_443_blocked_when_allow_remote_false(self):
        """Public IPs on port 443 are blocked when allow_remote=False."""
        assert is_safe_url("https://93.184.216.34") is False

    def test_public_hostname_dns_blocked_when_allow_remote_false(self):
        """Hostnames resolving to public IPs are blocked when allow_remote=False."""
        with patch(
            "memory.url_validate._resolve_hostname",
            return_value=["93.184.216.34"],
        ):
            assert is_safe_url("http://example.com:11434") is False

    def test_loopback_on_default_port_still_allowed_when_allow_remote_false(self):
        """Loopback on Ollama default port remains allowed (regression check)."""
        assert is_safe_url(f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}") is True


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

    def test_validate_url_rejects_credentials(self):
        """validate_url raises ValueError for URLs with embedded credentials."""
        with pytest.raises(ValueError, match="URL rejected"):
            validate_url(f"http://user:pass@127.0.0.1:{OLLAMA_DEFAULT_PORT}/api")


# ---------------------------------------------------------------------------
# sanitize_url_for_log tests
# ---------------------------------------------------------------------------


class TestSanitizeUrlForLog:
    """Verify that sanitize_url_for_log strips credentials from URLs."""

    def test_strips_user_password(self):
        """URLs with user:password@ should have credentials removed."""
        result = sanitize_url_for_log("https://user:secret@api.openai.com/v1/models")
        assert result == "https://api.openai.com/v1/models"

    def test_strips_user_only(self):
        """URLs with just user@ should have credentials removed."""
        result = sanitize_url_for_log("https://admin@api.openai.com/v1/embeddings")
        assert result == "https://api.openai.com/v1/embeddings"

    def test_url_without_credentials_unchanged(self):
        """URLs without credentials should pass through unchanged."""
        result = sanitize_url_for_log("https://api.openai.com/v1/models")
        assert result == "https://api.openai.com/v1/models"

    def test_preserves_port(self):
        """Port numbers should be preserved after sanitization."""
        result = sanitize_url_for_log("http://user:pass@localhost:11434/api/tags")
        assert result == "http://localhost:11434/api/tags"

    def test_preserves_path(self):
        """Path components should be preserved after sanitization."""
        result = sanitize_url_for_log(
            "https://key:secret@host.example.com/path/to/api?query=1"
        )
        assert result == "https://host.example.com/path/to/api?query=1"

    def test_validates_url_not_leaking_credentials(self):
        """validate_url error messages should not contain credentials."""
        url_with_creds = "https://admin:secretpass@10.0.0.1:11434/api"
        with pytest.raises(ValueError) as exc_info:
            validate_url(url_with_creds, allow_remote=True)
        # The error message must NOT contain the password
        assert "secretpass" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# _strip_credentials tests
# ---------------------------------------------------------------------------


class TestStripCredentials:
    """Verify _strip_credentials removes userinfo from URLs."""

    def test_strips_user_password(self):
        result = _strip_credentials("http://user:pass@host/path")
        assert "user" not in result
        assert "pass" not in result
        assert "host/path" in result

    def test_preserves_port(self):
        result = _strip_credentials("http://user:pass@host:8080/path")
        assert ":8080" in result
        assert "pass" not in result

    def test_no_credentials_unchanged(self):
        url = "http://host:8080/path"
        assert _strip_credentials(url) == url


# ---------------------------------------------------------------------------
# DEFAULT_URLOPEN_TIMEOUT tests
# ---------------------------------------------------------------------------


class TestDefaultUrlopenTimeout:
    """Verify that DEFAULT_URLOPEN_TIMEOUT is a positive integer."""

    def test_default_timeout_is_positive(self):
        assert DEFAULT_URLOPEN_TIMEOUT > 0
        assert isinstance(DEFAULT_URLOPEN_TIMEOUT, int)

    def test_default_timeout_is_30(self):
        """The default timeout is 30 seconds."""
        assert DEFAULT_URLOPEN_TIMEOUT == 30


# ---------------------------------------------------------------------------
# safe_urlopen tests
# ---------------------------------------------------------------------------


class TestSafeUrlopen:
    """Verify safe_urlopen validates URLs, sets timeout, and blocks redirects."""

    def test_rejects_unsafe_url(self):
        """safe_urlopen raises ValueError for URLs that fail is_safe_url."""
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen("http://10.0.0.1:11434/api")

    def test_rejects_url_with_credentials(self):
        """safe_urlopen raises ValueError for URLs with credentials."""
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen(f"http://user:pass@127.0.0.1:{OLLAMA_DEFAULT_PORT}/api")

    def test_validates_url_internally(self):
        """safe_urlopen calls is_safe_url internally — no need for external validation."""
        # A URL that would pass is_safe_url but safe_urlopen adds extra
        # re-resolution protection. Test that a private IP is rejected.
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen("http://10.0.0.1:11434/api", allow_remote=True)

    def test_has_default_timeout(self):
        """safe_urlopen has a default timeout parameter (not None)."""
        import inspect

        sig = inspect.signature(safe_urlopen)
        timeout_param = sig.parameters["timeout"]
        assert timeout_param.default == DEFAULT_URLOPEN_TIMEOUT

    def test_rejects_file_scheme(self):
        """safe_urlopen rejects file:// scheme URLs."""
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen("file:///etc/passwd")

    def test_re_dns_rebinding_on_open(self):
        """safe_urlopen re-resolves DNS before opening and blocks changed IPs.

        If DNS rebinding occurs between is_safe_url check and the actual request,
        the re-resolve catches it. We simulate this by having _resolve_hostname
        return a safe IP first (for is_safe_url) then a blocked IP (for re-resolve).
        """
        call_count = 0

        def mock_resolve(hostname):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                # First call from is_safe_url: return safe loopback
                return ["127.0.0.1"]
            else:
                # Second call from safe_urlopen re-resolve: return blocked IP
                return ["10.0.0.1"]

        with patch(
            "memory.url_validate._resolve_hostname",
            side_effect=mock_resolve,
        ):
            with pytest.raises(ValueError, match="re-resolve"):
                safe_urlopen(
                    f"http://localhost:{OLLAMA_DEFAULT_PORT}/api",
                    allow_remote=True,
                )


# ---------------------------------------------------------------------------
# SafeRedirectHandler tests
# ---------------------------------------------------------------------------


class TestSafeRedirectHandler:
    """Verify that SSRF via HTTP redirects is blocked.

    SafeRedirectHandler validates redirect target URLs against is_safe_url
    before following them. This prevents SSRF via redirects from safe-looking
    URLs to private/internal IP addresses.
    """

    def test_allows_redirect_to_public_url(self):
        """Redirects to safe public URLs should be followed."""
        handler = SafeRedirectHandler(allow_remote=True)
        req = MagicMock()
        req.full_url = "https://safe.example.com/start"
        req.get_method.return_value = "GET"
        # Redirect to a public URL should be allowed
        result = handler.redirect_request(
            req,
            MagicMock(),
            302,
            "Found",
            MagicMock(),
            "https://api.openai.com/v1/models",
        )
        assert result is not None

    def test_blocks_redirect_to_private_ip(self):
        """Redirects to private IPs must be blocked even if the initial URL is safe."""
        handler = SafeRedirectHandler(allow_remote=True)
        req = MagicMock()
        req.full_url = "https://safe.example.com/start"
        result = handler.redirect_request(
            req, MagicMock(), 302, "Found", MagicMock(), "http://10.0.0.1/"
        )
        assert result is None

    def test_blocks_redirect_to_non_ollama_loopback_port(self):
        """Redirects to loopback on non-Ollama port are blocked even with allow_remote=True."""
        handler = SafeRedirectHandler(allow_remote=True)
        req = MagicMock()
        req.full_url = f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}/start"
        result = handler.redirect_request(
            req, MagicMock(), 302, "Found", MagicMock(), "http://127.0.0.1:6379/"
        )
        assert result is None  # Should be blocked — Redis port

    def test_blocks_redirect_to_metadata_endpoint(self):
        """SSRF via redirect to cloud metadata (169.254.169.254) must be blocked."""
        handler = SafeRedirectHandler(allow_remote=True)
        req = MagicMock()
        req.full_url = f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}/start"
        result = handler.redirect_request(
            req,
            MagicMock(),
            302,
            "Found",
            MagicMock(),
            "http://169.254.169.254/latest/meta-data/",
        )
        assert result is None  # Should be blocked

    def test_redirect_handler_returns_none_for_blocked_url(self):
        """redirect_request should return None when is_safe_url returns False
        for the redirect target, preventing the redirect from being followed."""
        handler = SafeRedirectHandler(allow_remote=True)
        req = MagicMock()
        req.full_url = "https://safe.example.com/start"
        # A redirect to a private IP should be blocked
        result = handler.redirect_request(
            req, MagicMock(), 302, "Found", MagicMock(), "http://10.0.0.1/"
        )
        assert result is None

    def test_redirect_handler_delegates_for_safe_url(self):
        """redirect_request should delegate to the parent for safe redirect targets.
        The parent class should handle GET redirects to safe URLs."""
        handler = SafeRedirectHandler(allow_remote=True)
        req = MagicMock()
        req.full_url = "https://safe.example.com/start"
        req.get_method.return_value = "GET"
        # A redirect to another public URL should be allowed
        result = handler.redirect_request(
            req,
            MagicMock(),
            302,
            "Found",
            MagicMock(),
            "https://api.openai.com/v1/models",
        )
        # The parent class returns a Request object for valid GET redirects
        assert result is not None  # Should not be blocked by SSRF check

    def test_allow_remote_false_blocks_redirect_to_public(self):
        """When allow_remote=False, redirects to public hosts should be blocked."""
        handler = SafeRedirectHandler(allow_remote=False)
        req = MagicMock()
        req.full_url = f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}/start"
        # is_safe_url("https://api.openai.com", allow_remote=False) would
        # block non-loopback-on-default-port
        result = handler.redirect_request(
            req, MagicMock(), 302, "Found", MagicMock(), "http://192.168.1.1:11434/"
        )
        assert result is None  # Should be blocked


class TestIsSafeURL_PercentEncodedSSRF:
    """Percent-encoded IP hostnames must be decoded and checked to prevent SSRF bypass.

    An attacker can encode a private IP address (e.g. 127.0.0.1) as
    percent-encoded octets (%31%32%37%2e%30%2e%30%2e%31). urlparse
    preserves the encoding in .hostname, so ip_address() on the
    raw hostname would fail, DNS resolution would fail, and
    allow_remote=True would return True — but urllib normalizes the
    hostname and connects to the decoded private IP.
    """

    def test_percent_encoded_loopback_ollama_port_allowed_allow_remote(self):
        """Percent-encoded 127.0.0.1 on Ollama port is allowed with allow_remote=True.

        This is correct behavior — loopback addresses are trusted.
        The SSRF bypass specifically targets private/link-local IPs.
        """
        # %31%32%37 = "127", %2e = ".", %30 = "0", %2e = ".", %30 = "0", %2e = ".", %31 = "1"
        url = "http://%31%32%37%2e%30%2e%30%2e%31:11434/"
        assert is_safe_url(url, allow_remote=True) is True

    def test_percent_encoded_private_ip_blocked(self):
        """Percent-encoded 10.0.0.1 must be blocked even with allow_remote=True.
        Decodes to a private class A address.
        """
        # %31%30 = "10", %2e = ".", %30 = "0", %2e = ".", %30 = "0", %2e = ".", %31 = "1"
        url = "http://%31%30%2e%30%2e%30%2e%31:11434/"
        assert is_safe_url(url, allow_remote=True) is False

    def test_percent_encoded_metadata_ip_blocked(self):
        """Percent-encoded 169.254.169.254 (cloud metadata) must be blocked."""
        # 169.254.169.254 encoded: %31%36%39%2e%32%35%34%2e%31%36%39%2e%32%35%34
        url = "http://%31%36%39%2e%32%35%34%2e%31%36%39%2e%32%35%34:80/"
        assert is_safe_url(url, allow_remote=True) is False

    def test_percent_encoded_192_168_blocked(self):
        """Percent-encoded 192.168.1.1 must be blocked even with allow_remote=True."""
        url = "http://%31%39%32%2e%31%36%38%2e%31%2e%31:11434/"
        assert is_safe_url(url, allow_remote=True) is False

    def test_hostname_with_partial_encoding_still_checked(self):
        """Mixed encoded/plain hostnames are also decoded and checked."""
        # "127.0.0.1" with "127" percent-encoded but rest plain
        url = "http://%31%32%37.0.0.1:11434/"
        # This should be allowed because 127.0.0.1 on 11434 is OK
        assert is_safe_url(url, allow_remote=True) is True
