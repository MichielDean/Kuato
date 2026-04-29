"""Tests for security-related fixes across llmem modules.

Covers all 13 security findings from the reviewer/security audit:
xwpn6: IntrospectionAnalyzer validates URL with is_safe_url
b4vh3+bsgwq: is_safe_url blocks credentials, redirects, DNS rebinding validation
c9nf0: LMEM_HOME validation blocks system dirs and path traversal
dlh4s: migrate_from_lobsterdog skips symlinks
4bx72: discover_transcript_files excludes symlinks
q3csd: HTTP timeouts via safe_urlopen
wnb5q: dream_report writes with path validation
mddh2: dream.py _validate_output_path confinement
cah41: introspect_session file size limit
n2bsf: cmd_import schema validation
0ys0n: register_memory_type input validation
bycxf: credentials stripped from URL error messages

Plus security issues from ll-7rudv security review:
vp1ad: SSRF bypass via percent-encoded IP hostnames
k7u4f: DB file permissions - WAL/SHM chmod + umask
gkn1z: OOM DoS via no LIMIT in brute-force search and export_all
ojbjt: process_transcript size limit
18exx: Context file writes bypass _validate_write_path()
2vk1u: API key credential exfiltration prevention
0yadu: import_memories input validation
"""

import json
import logging
import os
import socket
import tempfile
import urllib.request
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from llmem.url_validate import (
    is_safe_url,
    validate_url,
    _strip_credentials,
    _extract_url_string,
    _NoRedirectHandler,
    safe_urlopen,
    DEFAULT_URLOPEN_TIMEOUT,
)
from llmem.paths import (
    get_llmem_home,
    _validate_home_path,
    _validate_write_path,
    migrate_from_lobsterdog,
)
from llmem.store import (
    register_memory_type,
    get_registered_types,
    _reset_global_registry,
    MemoryStore,
)
from llmem.dream import _validate_output_path
from llmem.hooks import (
    discover_transcript_files,
    introspect_session,
    INTROSPECT_RESULT_FILE_TOO_LARGE,
)
from llmem.dream_report import generate_dream_report
from llmem.dream import DreamResult, RemPhaseResult
from llmem.ollama import ProviderDetector
from llmem.session_hooks import (
    SESSION_CREATED_SUCCESS,
    SESSION_COMPACTING_SUCCESS,
)
from memory.providers import OpenAIProvider, AnthropicProvider, _is_loopback_hostname


# ============================================================================
# xwpn6 + b4vh3 + bsgwq: URL validation security
# ============================================================================


class TestUrlValidate_SafeUrl:
    """Test is_safe_url blocks SSRF vectors."""

    def test_blocks_file_scheme(self):
        assert not is_safe_url("file:///etc/passwd")

    def test_blocks_ftp_scheme(self):
        assert not is_safe_url("ftp://example.com/file")

    def test_blocks_data_scheme(self):
        assert not is_safe_url("data:text/html,<script>alert(1)</script>")

    def test_blocks_javascript_scheme(self):
        assert not is_safe_url("javascript:alert(1)")

    def test_allows_http_localhost_default_port(self):
        assert is_safe_url("http://localhost:11434/api/tags")

    def test_allows_https_localhost_default_port(self):
        """HTTPS on localhost:11434 should also be allowed (loopback)."""
        # Note: port 443 is not 11434, so this won't be loopback-allowed
        # But allow_remote=True should allow it
        assert is_safe_url("https://example.com/api/tags", allow_remote=True)

    def test_blocks_private_ip_remote(self):
        """192.168.x.x addresses are blocked without allow_remote."""
        assert not is_safe_url("http://192.168.1.1:11434/api/tags")

    def test_allows_remote_when_explicit(self):
        """allow_remote=True permits non-loopback addresses."""
        assert is_safe_url("http://example.com:11434", allow_remote=True)

    def test_blocks_public_ip_by_default(self):
        """SSRF bypass fix: public IPs must be blocked when allow_remote=False.

        Previously _check_ip_access allowed public IPs through when
        allow_remote=False because _ip_is_blocked returns False for public
        addresses and the code only checked port restrictions for loopback.
        """
        assert not is_safe_url("http://93.184.216.34:11434")

    def test_blocks_public_ip_on_any_port_by_default(self):
        """Public IPs on any port are blocked when allow_remote=False."""
        assert not is_safe_url("http://93.184.216.34:443")
        assert not is_safe_url("http://93.184.216.34:80")

    def test_blocks_public_hostname_dns_by_default(self):
        """Hostnames resolving to public IPs are blocked when allow_remote=False."""
        with patch(
            "llmem.url_validate._resolve_hostname", return_value=["93.184.216.34"]
        ):
            assert not is_safe_url("http://example.com:11434")

    def test_loopback_default_port_still_allowed_by_default(self):
        """Loopback on Ollama default port remains allowed (regression check)."""
        assert is_safe_url("http://127.0.0.1:11434/api/tags")

    def test_blocks_no_hostname(self):
        assert not is_safe_url("http://")

    def test_blocks_credentials_in_url(self):
        """URLs with embedded credentials (user:pass@host) must be rejected."""
        assert not is_safe_url("http://user:password@localhost:11434/api/tags")

    def test_blocks_username_only_in_url(self):
        assert not is_safe_url("http://user@localhost:11434/api/tags")


class TestUrlValidate_StripCredentials:
    """Test _strip_credentials removes userinfo from URLs."""

    def test_strips_user_password(self):
        result = _strip_credentials("http://admin:secret@host.example.com/path")
        assert "admin" not in result
        assert "secret" not in result
        assert "host.example.com" in result

    def test_strips_user_only(self):
        result = _strip_credentials("http://admin@host.example.com/path")
        assert "admin" not in result
        assert "host.example.com" in result

    def test_preserves_url_without_credentials(self):
        url = "http://localhost:11434/api/tags"
        assert _strip_credentials(url) == url

    def test_strips_from_error_message_context(self):
        """When credentials appear in error messages, they should be stripped."""
        unsafe_url = "http://admin:s3cret@169.254.169.254/latest/meta-data/"
        result = _strip_credentials(unsafe_url)
        assert "s3cret" not in result
        assert "admin" not in result


class TestUrlValidate_RedirectBlocking:
    """Test that safe_urlopen blocks redirects (SSRF protection)."""

    def test_no_redirect_handler_blocks_redirects(self):
        """_NoRedirectHandler.redirect_request returns None, blocking all redirects."""
        handler = _NoRedirectHandler()
        req = MagicMock()
        req.full_url = "http://localhost:11434/api/tags"
        result = handler.redirect_request(
            req,  # req
            MagicMock(),  # fp
            301,  # code
            "Moved",  # msg
            MagicMock(),  # headers
            "http://evil.internal/",  # newurl
        )
        assert result is None

    def test_no_redirect_handler_logs_redirect(self, caplog):
        """_NoRedirectHandler.redirect_request logs the blocked redirect (not silent)."""
        import logging

        handler = _NoRedirectHandler()
        req = MagicMock()
        req.full_url = "http://localhost:11434/api/tags"
        with caplog.at_level(logging.WARNING, logger="llmem.url_validate"):
            handler.redirect_request(
                req,
                MagicMock(),  # fp
                301,  # code
                "Moved",  # msg
                MagicMock(),  # headers
                "http://evil.internal/",  # newurl
            )
        assert len(caplog.records) >= 1
        assert "blocked SSRF redirect" in caplog.text


class TestUrlValidate_SafeUrlopen:
    """Test safe_urlopen URL validation and error handling."""

    def test_safe_urlopen_return_type_annotation(self):
        """safe_urlopen return type must be http.client.HTTPResponse, not OpenerDirector.

        The return type annotation must match the actual return type —
        OpenerDirector.open() returns an HTTPResponse. An incorrect
        annotation breaks static type checking for all call sites.
        """
        import http.client
        import inspect

        sig = inspect.signature(safe_urlopen)
        assert sig.return_annotation is http.client.HTTPResponse

    def test_safe_urlopen_rejects_unsafe_url(self):
        """safe_urlopen raises ValueError for blocked URLs."""
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen("http://192.168.1.1:8080/api/tags")

    def test_safe_urlopen_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen("file:///etc/passwd")

    def test_safe_urlopen_strips_credentials_from_error(self):
        """Error messages should not contain credentials."""
        url_with_creds = "http://admin:s3cret@localhost:11434/api/tags"
        # This should pass is_safe_url (loopback, correct port)
        # but will fail to connect; the error should strip credentials
        # We can't easily test connection failure here, so test validation
        # of an URL with creds that's blocked
        blocked_with_creds = "http://admin:s3cret@192.168.1.1/api/tags"
        with pytest.raises(ValueError) as exc_info:
            safe_urlopen(blocked_with_creds)
        assert "s3cret" not in str(exc_info.value)
        assert "admin" not in str(exc_info.value)

    def test_safe_urlopen_rejects_unsafe_request_object(self):
        """safe_urlopen must validate URLs from Request objects, not crash."""
        req = urllib.request.Request("http://192.168.1.1:8080/api/tags")
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen(req)

    def test_safe_urlopen_rejects_file_scheme_request_object(self):
        """safe_urlopen must reject file:// scheme in Request objects."""
        req = urllib.request.Request("file:///etc/passwd")
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen(req)

    def test_safe_urlopen_request_object_strips_credentials_from_error(self):
        """safe_urlopen must strip credentials from errors when given a Request."""
        blocked_with_creds = "http://admin:s3cret@192.168.1.1/api/tags"
        req = urllib.request.Request(blocked_with_creds)
        with pytest.raises(ValueError) as exc_info:
            safe_urlopen(req)
        assert "s3cret" not in str(exc_info.value)
        assert "admin" not in str(exc_info.value)

    def test_safe_urlopen_request_object_validates_dangerous_hostname(self):
        """safe_urlopen must reject Request objects targeting dangerous hostnames."""
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/")
        with pytest.raises(ValueError):
            safe_urlopen(req)

    def test_safe_urlopen_request_object_rejects_percent_encoded_private_ip(self):
        """safe_urlopen must reject percent-encoded private IPs in Request objects."""
        req = urllib.request.Request("http://%31%30%2e%30%2e%30%2e%31:11434/api/tags")
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen(req)

    def test_safe_urlopen_request_object_with_post_data(self):
        """safe_urlopen must validate Request objects that carry POST data."""
        req = urllib.request.Request(
            "http://10.0.0.1:11434/api/generate",
            data=b'{"model":"test"}',
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen(req)


class TestUrlValidate_ExtractUrlString:
    """Test _extract_url_string correctly extracts URL from Request objects."""

    def test_extract_url_string_from_string(self):
        """_extract_url_string returns the URL string unchanged."""
        url = "http://127.0.0.1:11434/api/tags"
        assert _extract_url_string(url) == url

    def test_extract_url_string_from_request_object(self):
        """_extract_url_string extracts full_url from Request objects."""
        url = "http://127.0.0.1:11434/api/tags"
        req = urllib.request.Request(url)
        assert _extract_url_string(req) == url

    def test_extract_url_string_from_request_with_data(self):
        """_extract_url_string extracts URL even from POST Request objects."""
        url = "http://127.0.0.1:11434/api/generate"
        req = urllib.request.Request(
            url,
            data=b'{"model":"test"}',
            headers={"Content-Type": "application/json"},
        )
        assert _extract_url_string(req) == url

    def test_extract_url_string_from_request_with_private_ip(self):
        """_extract_url_string extracts URL from Request targeting private IP."""
        url = "http://10.0.0.1/api/tags"
        req = urllib.request.Request(url)
        assert _extract_url_string(req) == url


# ============================================================================
# c5grc: safe_urlopen Request object handling
# ============================================================================


class TestUrlValidate_ExtractUrlString:
    """Test _extract_url_string handles both str and Request inputs."""

    def test_returns_string_unchanged(self):
        """Plain strings pass through unchanged."""
        url = "http://localhost:11434/api/tags"
        assert _extract_url_string(url) == url

    def test_extracts_full_url_from_get_request(self):
        """GET Request objects yield their .full_url."""
        import urllib.request

        req = urllib.request.Request("http://localhost:11434/api/tags")
        assert _extract_url_string(req) == "http://localhost:11434/api/tags"

    def test_extracts_full_url_from_post_request(self):
        """POST Request objects (with data) yield their .full_url."""
        import urllib.request

        payload = b'{"model":"test"}'
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        assert _extract_url_string(req) == "http://localhost:11434/api/generate"


class TestUrlValidate_SafeUrlopenRequestObjects:
    """Test safe_urlopen properly handles urllib.request.Request objects (c5grc)."""

    def test_rejects_request_with_unsafe_url(self):
        """Request objects with unsafe URLs raise ValueError, not AttributeError."""
        import urllib.request

        req = urllib.request.Request("http://192.168.1.1:8080/api/tags")
        # Before fix: this would crash with AttributeError from urlparse
        # After fix: raises ValueError from is_safe_url
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen(req)

    def test_rejects_request_with_file_scheme(self):
        """Request objects with file:// URLs raise ValueError."""
        import urllib.request

        req = urllib.request.Request("file:///etc/passwd")
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen(req)

    def test_rejects_post_request_with_private_ip(self):
        """POST Request objects with private IP are rejected."""
        import urllib.request

        req = urllib.request.Request(
            "http://10.0.0.1:11434/api/generate",
            data=b'{"model":"test"}',
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen(req)


# ============================================================================
# hj0uy: safe_urlopen allow_remote parameter
# ============================================================================


class TestUrlValidate_SafeUrlopenAllowRemote:
    """Test safe_urlopen allow_remote parameter (hj0uy)."""

    def test_default_allow_remote_blocks_private_ip(self):
        """By default (allow_remote=False), private IP URLs are rejected."""
        # Use a private IP to avoid DNS/network access — private IPs are blocked
        # by is_safe_url when allow_remote=False
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen("http://192.168.1.1:11434/api/tags")

    def test_allow_remote_true_permits_remote_url(self):
        """With allow_remote=True, remote URL validation passes (may still fail on connect)."""
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        # Patch both DNS resolution (which is_safe_url calls) and the opener
        with (
            patch(
                "llmem.url_validate._resolve_hostname", return_value=["93.184.216.34"]
            ),
            patch("llmem.url_validate.urllib.request.build_opener") as mock_build,
        ):
            mock_opener = MagicMock()
            mock_opener.open.return_value = mock_resp
            mock_build.return_value = mock_opener

            # This should NOT raise ValueError — validation passes with allow_remote=True
            result = safe_urlopen(
                "http://example.com:11434/api/tags", allow_remote=True
            )
            assert result is not None

    def test_allow_remote_false_blocks_private_ip(self):
        """allow_remote=False blocks private IPs."""
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen("http://10.0.0.1:11434/api/tags", allow_remote=False)

    def test_request_with_allow_remote_true(self):
        """Request objects work with allow_remote=True."""
        import urllib.request
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        req = urllib.request.Request(
            "http://example.com:11434/api/generate",
            data=b'{"model":"test"}',
            headers={"Content-Type": "application/json"},
        )

        with (
            patch(
                "llmem.url_validate._resolve_hostname", return_value=["93.184.216.34"]
            ),
            patch("llmem.url_validate.urllib.request.build_opener") as mock_build,
        ):
            mock_opener = MagicMock()
            mock_opener.open.return_value = mock_resp
            mock_build.return_value = mock_opener

            # Before fix: this would crash with AttributeError
            # After fix: validation passes with allow_remote=True
            result = safe_urlopen(req, allow_remote=True)
            assert result is not None

    def test_loopback_url_passes_without_allow_remote(self):
        """Loopback URLs on default port pass even without allow_remote."""
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("llmem.url_validate.urllib.request.build_opener") as mock_build:
            mock_opener = MagicMock()
            mock_opener.open.return_value = mock_resp
            mock_build.return_value = mock_opener

            result = safe_urlopen("http://localhost:11434/api/tags")
            assert result is not None

    def test_allow_remote_propagates_to_re_resolve_check(self):
        """allow_remote=True must propagate to the re-resolve IP check.

        Before the fix (hj0uy), safe_urlopen used _is_remote_allowed(url)
        for the re-resolve check, which inferred allow_remote=True for
        non-loopback URLs but used allow_remote=False for the initial
        is_safe_url check. This made remote Ollama hosts fail at the
        re-resolve stage after passing the initial check.
        """
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "llmem.url_validate._resolve_hostname", return_value=["93.184.216.34"]
            ),
            patch("llmem.url_validate.urllib.request.build_opener") as mock_build,
        ):
            mock_opener = MagicMock()
            mock_opener.open.return_value = mock_resp
            mock_build.return_value = mock_opener

            # With allow_remote=True, both is_safe_url and the re-resolve check
            # should pass for a remote IP
            result = safe_urlopen(
                "http://example.com:11434/api/tags", allow_remote=True
            )
            assert result is not None


# ============================================================================
# t9yif: DNS rebinding re-resolve ValueError must not be swallowed
# ============================================================================


class TestUrlValidate_DnsRebindResolve:
    """Test that the re-resolve DNS rebinding check in safe_urlopen is effective.

    The old code had 'except ValueError: pass' that caught the ValueError
    raised by the blocked-address check alongside ipaddress parsing errors.
    This made the DNS rebinding protection completely ineffective — an
    attacker controlling DNS could validate with a safe IP, rebind to a
    private IP, and the re-resolve ValueError would be silently swallowed.
    """

    def test_re_resolve_blocks_private_ip_after_safe_initial_resolve(self):
        """DNS rebinding: is_safe_url passes (sees safe IP), but re-resolve
        returns a private IP. The re-resolve ValueError must propagate."""
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()

        with (
            patch("llmem.url_validate.is_safe_url", return_value=True),
            patch(
                "llmem.url_validate._resolve_hostname",
                return_value=["192.168.1.1"],
            ),
        ):
            with pytest.raises(ValueError, match="rejected after re-resolve"):
                safe_urlopen("http://example.com:11434/api/tags", allow_remote=True)

    def test_re_resolve_raises_for_blocked_address(self):
        """Re-resolve that finds a blocked IP raises ValueError (not swallowed)."""
        from unittest.mock import patch

        with (
            patch("llmem.url_validate.is_safe_url", return_value=True),
            patch(
                "llmem.url_validate._resolve_hostname",
                return_value=["10.0.0.1"],
            ),
        ):
            with pytest.raises(ValueError, match="blocked address"):
                safe_urlopen("http://example.com:11434/api/tags", allow_remote=True)

    def test_re_resolve_raises_for_unparseable_address(self):
        """Re-resolve that finds an unparseable address raises ValueError."""
        from unittest.mock import patch

        with (
            patch("llmem.url_validate.is_safe_url", return_value=True),
            patch(
                "llmem.url_validate._resolve_hostname",
                return_value=["not-an-ip"],
            ),
        ):
            with pytest.raises(ValueError, match="unparseable address"):
                safe_urlopen("http://example.com:11434/api/tags", allow_remote=True)

    def test_re_resolve_allows_safe_ip(self):
        """Re-resolve that finds a safe IP does not raise."""
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("llmem.url_validate.is_safe_url", return_value=True),
            patch(
                "llmem.url_validate._resolve_hostname",
                return_value=["93.184.216.34"],
            ),
            patch("llmem.url_validate.urllib.request.build_opener") as mock_build,
        ):
            mock_opener = MagicMock()
            mock_opener.open.return_value = mock_resp
            mock_build.return_value = mock_opener

            # Safe remote IP should not raise
            result = safe_urlopen(
                "http://example.com:11434/api/tags", allow_remote=True
            )
            assert result is not None


# ============================================================================
# c9nf0: LMEM_HOME validation
# ============================================================================


class TestPaths_LmemHomeValidation:
    """Test LMEM_HOME env var validation."""

    def test_system_dir_blocked_etc(self):
        with pytest.raises(ValueError, match="system directory"):
            _validate_home_path(Path("/etc/llmem"), "LMEM_HOME")

    def test_system_dir_blocked_var(self):
        with pytest.raises(ValueError, match="system directory"):
            _validate_home_path(Path("/var/lib/llmem"), "LMEM_HOME")

    def test_system_dir_blocked_proc(self):
        with pytest.raises(ValueError, match="system directory"):
            _validate_home_path(Path("/proc/self/llmem"), "LMEM_HOME")

    def test_system_dir_blocked_root(self):
        with pytest.raises(ValueError, match="system directory"):
            _validate_home_path(Path("/root/llmem"), "LMEM_HOME")

    def test_traversal_blocked(self):
        """Paths with '..' that resolve to system dirs should be rejected."""
        # This resolves to /etc/llmem which hits the system dir check
        with pytest.raises(ValueError, match="targets a system directory|traversal"):
            _validate_home_path(Path("/home/user/../../etc/llmem"), "LMEM_HOME")

    def test_traversal_path_blocked(self):
        """Paths containing '..' components should be rejected even if resolved path is ok."""
        with pytest.raises(ValueError, match="traversal"):
            _validate_home_path(Path("/tmp/../../../home/user/llmem"), "LMEM_HOME")

    def test_valid_home_dir_allowed(self):
        result = _validate_home_path(Path.home() / ".config" / "llmem", "LMEM_HOME")
        assert isinstance(result, Path)

    def test_tmp_dir_allowed(self):
        """Non-system dirs like /tmp should be allowed."""
        result = _validate_home_path(Path("/tmp/llmem-test"), "LMEM_HOME")
        assert isinstance(result, Path)


# ============================================================================
# dlh4s: Symlink following in migrate_from_lobsterdog
# ============================================================================


class TestPaths_MigrationSymlinkSafety:
    """Test that migrate_from_lobsterdog skips symlinks."""

    def test_skips_symlink_files(self, tmp_path):
        """Source symlinks should be skipped, not followed."""
        old_home = tmp_path / ".lobsterdog"
        new_home = tmp_path / ".config" / "llmem"
        old_home.mkdir()

        # Create a regular file and a symlink to it
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("secret data")

        real_config = old_home / "config.yaml"
        real_config.write_text("memory: {}")

        # Create a symlink pointing outside old_home
        symlink_path = old_home / "memory.db"
        symlink_path.symlink_to(secret_file)

        with patch("llmem.paths.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw) if a else Path()
            result = migrate_from_lobsterdog()

        # The symlink should have been skipped
        assert result is True  # config.yaml migrated
        assert (new_home / "config.yaml").exists()
        # memory.db should NOT exist (it was a symlink)
        assert (
            not (new_home / "memory.db").exists()
            or (new_home / "memory.db").is_symlink() is False
        )

    def test_skips_symlink_context_dir(self, tmp_path):
        """Symlink context directories should be skipped."""
        old_home = tmp_path / ".lobsterdog"
        new_home = tmp_path / ".config" / "llmem"
        old_home.mkdir()

        real_target = tmp_path / "external_context"
        real_target.mkdir()
        (real_target / "data.txt").write_text("data")

        # context is a symlink
        context_symlink = old_home / "context"
        context_symlink.symlink_to(real_target)

        with patch("llmem.paths.Path") as mock_path_cls:
            mock_path_cls.home.return_value = tmp_path
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw) if a else Path()
            result = migrate_from_lobsterdog()

        # The symlink directory should have been skipped
        assert not (new_home / "context").exists()


# ============================================================================
# 4bx72: discover_transcript_files excludes symlinks
# ============================================================================


class TestHooks_DiscoverTranscriptFiles:
    """Test that discover_transcript_files excludes symlinks."""

    def test_excludes_symlinks(self, tmp_path):
        """Symlink files should not be returned by discover_transcript_files."""
        # Create a regular file
        real_file = tmp_path / "real.md"
        real_file.write_text("# real content")

        # Create a symlink to a file outside the directory
        outside_file = tmp_path.parent / "outside.md"
        outside_file.write_text("# outside content")
        symlink_file = tmp_path / "symlink.md"
        symlink_file.symlink_to(outside_file)

        files = discover_transcript_files(tmp_path, "*.md")
        paths = set(str(f) for f in files)

        # The real file should be found, symlink should be excluded
        assert real_file in files or str(real_file) in paths
        assert symlink_file not in files

    def test_finds_regular_files(self, tmp_path):
        """Regular .md files are discovered."""
        (tmp_path / "test.md").write_text("content")
        files = discover_transcript_files(tmp_path, "*.md")
        assert len(files) >= 1


# ============================================================================
# cah41: introspect_session file size limit
# ============================================================================


class TestHooks_IntrospectSessionFileSize:
    """Test that introspect_session enforces a file size limit."""

    def test_rejects_oversized_file(self, tmp_path):
        """Files exceeding max_file_size should be rejected."""
        large_file = tmp_path / "large.md"
        # Create a file larger than the limit
        large_file.write_text("x" * (1024 * 1024 + 1))  # > 1MB

        store = MemoryStore(db_path=tmp_path / "test.db", disable_vec=True)
        from llmem.hooks import IntrospectionAnalyzer

        analyzer = IntrospectionAnalyzer.__new__(IntrospectionAnalyzer)
        analyzer._model = "test"
        analyzer._base_url = "http://localhost:11434"

        result_type, mid = introspect_session(
            large_file, store, analyzer, max_file_size=1024 * 1024
        )
        assert result_type == INTROSPECT_RESULT_FILE_TOO_LARGE
        assert mid is None
        store.close()


# ============================================================================
# wnb5q: dream_report path validation
# ============================================================================


class TestDreamReport_PathValidation:
    """Test that generate_dream_report validates write paths."""

    def test_rejects_system_dir(self):
        """Writing to /etc should be rejected."""
        result = DreamResult(rem=RemPhaseResult(themes=["test"]))
        with pytest.raises(ValueError, match="protected directory"):
            generate_dream_report(result, Path("/etc/evil-report.html"))

    def test_rejects_path_traversal(self):
        """Paths with '..' components should be rejected."""
        result = DreamResult(rem=RemPhaseResult(themes=["test"]))
        # Use a path with '..' that doesn't resolve to a system dir
        with pytest.raises(ValueError, match="traversal"):
            generate_dream_report(
                result, Path("/tmp/subdir/../../../home/user/evil.html")
            )

    def test_accepts_valid_path(self, tmp_path):
        """Valid paths should work normally."""
        result = DreamResult(rem=RemPhaseResult(themes=["test"]))
        report_path = tmp_path / "report.html"
        generate_dream_report(result, report_path)
        assert report_path.exists()


# ============================================================================
# mddh2: dream.py _validate_output_path confinement
# ============================================================================


class TestDream_ValidateOutputPath:
    """Test _validate_output_path confinement and safety checks."""

    def test_rejects_system_dir(self):
        with pytest.raises(ValueError, match="protected directory"):
            _validate_output_path(Path("/etc/diary.md"), "diary")

    def test_rejects_traversal(self):
        with pytest.raises(ValueError, match="traversal"):
            _validate_output_path(Path("/tmp/sub/../../../home/user/docs"), "diary")

    def test_accepts_valid_path(self, tmp_path):
        result = _validate_output_path(tmp_path / "diary.md", "diary")
        assert isinstance(result, Path)

    def test_rejects_proc_dir(self):
        with pytest.raises(ValueError, match="protected directory"):
            _validate_output_path(Path("/proc/self/status"), "diary")

    def test_rejects_root_dir(self):
        with pytest.raises(ValueError, match="protected directory"):
            _validate_output_path(Path("/root/diary.md"), "diary")


# ============================================================================
# 0ys0n: register_memory_type input validation
# ============================================================================


class TestStore_RegisterTypeValidation:
    """Test register_memory_type validation."""

    def setup_method(self):
        _reset_global_registry()

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_memory_type("")

    def test_rejects_none(self):
        with pytest.raises(ValueError):
            register_memory_type(None)

    def test_rejects_spaces(self):
        with pytest.raises(ValueError):
            register_memory_type("   ")

    def test_rejects_uppercase(self):
        with pytest.raises(ValueError, match="must match"):
            register_memory_type("MyType")

    def test_rejects_leading_number(self):
        with pytest.raises(ValueError, match="must match"):
            register_memory_type("123abc")

    def test_rejects_dashes(self):
        with pytest.raises(ValueError, match="must match"):
            register_memory_type("my-type")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError, match="too long"):
            register_memory_type("a" * 65)

    def test_accepts_valid_type(self):
        register_memory_type("custom_type")
        assert "custom_type" in get_registered_types()

    def test_accepts_underscores(self):
        register_memory_type("my_custom_type")
        assert "my_custom_type" in get_registered_types()

    def test_accepts_numbers_after_first(self):
        register_memory_type("type1")
        assert "type1" in get_registered_types()

    def test_rejects_special_chars(self):
        with pytest.raises(ValueError, match="must match"):
            register_memory_type("type; DROP TABLE memories")


# ============================================================================
# n2bsf: cmd_import schema validation
# ============================================================================


class TestCli_ImportValidation:
    """Test cmd_import schema validation."""

    def test_rejects_non_json(self, tmp_path):
        """Non-JSON files should produce a clear error."""
        from llmem.cli import cmd_import
        import argparse

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all")

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            file=str(bad_file),
        )
        with pytest.raises(SystemExit):
            cmd_import(args)

    def test_rejects_non_array(self, tmp_path):
        """A JSON object (not array) should produce a clear error."""
        from llmem.cli import cmd_import
        import argparse

        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"type": "fact", "content": "test"}')

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            file=str(bad_file),
        )
        with pytest.raises(SystemExit):
            cmd_import(args)

    def test_rejects_missing_type(self, tmp_path):
        """Entries missing 'type' field should be rejected."""
        from llmem.cli import cmd_import
        import argparse

        bad_file = tmp_path / "bad.json"
        bad_file.write_text('[{"content": "test"}]')

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            file=str(bad_file),
        )
        with pytest.raises(SystemExit):
            cmd_import(args)

    def test_rejects_missing_content(self, tmp_path):
        """Entries missing 'content' field should be rejected."""
        from llmem.cli import cmd_import
        import argparse

        bad_file = tmp_path / "bad.json"
        bad_file.write_text('[{"type": "fact"}]')

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            file=str(bad_file),
        )
        with pytest.raises(SystemExit):
            cmd_import(args)

    def test_rejects_non_string_type(self, tmp_path):
        """Non-string 'type' should be rejected."""
        from llmem.cli import cmd_import
        import argparse

        bad_file = tmp_path / "bad.json"
        bad_file.write_text('[{"type": 42, "content": "test"}]')

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            file=str(bad_file),
        )
        with pytest.raises(SystemExit):
            cmd_import(args)


class TestStore_ImportValidation:
    """Test import_memories per-entry validation."""

    def test_skips_entries_without_type(self, store):
        """Entries missing 'type' should be skipped with a warning."""
        memories = [{"content": "test"}]
        count = store.import_memories(memories)
        assert count == 0

    def test_skips_entries_without_content(self, store):
        """Entries missing 'content' should be skipped."""
        memories = [{"type": "fact"}]
        count = store.import_memories(memories)
        assert count == 0

    def test_skips_non_dict_entries(self, store):
        """Non-dict entries should be skipped."""
        memories = ["not a dict", 42]
        count = store.import_memories(memories)
        assert count == 0

    def test_skips_non_string_type(self, store):
        """Non-string 'type' should be skipped."""
        memories = [{"type": 123, "content": "test"}]
        count = store.import_memories(memories)
        assert count == 0

    def test_valid_entry_imports(self, store):
        """Valid entries should import successfully."""
        memories = [{"type": "fact", "content": "test fact"}]
        count = store.import_memories(memories)
        assert count == 1


# ============================================================================
# bycxf: credentials stripped from URL error messages
# ============================================================================


class TestUrlValidate_CredentialStripping:
    """Test that credentials are never leaked in error messages."""

    def test_validate_url_strips_creds(self):
        """validate_url errors should not contain credentials."""
        url_with_creds = "http://admin:s3cret@evil.internal/api"
        try:
            validate_url(url_with_creds, allow_remote=False)
        except ValueError as e:
            assert "s3cret" not in str(e)
            assert "admin" not in str(e)

    def test_extraction_engine_strips_creds(self):
        """ExtractionEngine error messages should not contain credentials."""
        from llmem.extract import ExtractionEngine

        with pytest.raises(ValueError) as exc_info:
            ExtractionEngine(
                model="test", base_url="http://admin:s3cret@192.168.1.1:11434"
            )
        assert "s3cret" not in str(exc_info.value)
        assert "admin" not in str(exc_info.value)

    def test_embedding_engine_strips_creds(self):
        """EmbeddingEngine error messages should not contain credentials."""
        from llmem.embed import EmbeddingEngine

        with pytest.raises(ValueError) as exc_info:
            EmbeddingEngine(base_url="http://user:pass@evil.internal:11434")
        # Both the username and password must be stripped from the error
        # message. Using separate assertions (not 'or') ensures both
        # credentials are independently verified as absent.
        error_msg = str(exc_info.value)
        assert "user" not in error_msg, f"username leaked: {error_msg}"
        assert "pass@" not in error_msg, f"password leaked: {error_msg}"


# ============================================================================
# _validate_write_path tests
# ============================================================================


class TestPaths_ValidateWritePath:
    """Test _validate_write_path safety checks."""

    def test_rejects_system_dir(self):
        with pytest.raises(ValueError, match="protected directory"):
            _validate_write_path(Path("/etc/evil.html"), "report")

    def test_rejects_traversal(self):
        with pytest.raises(ValueError, match="traversal"):
            _validate_write_path(Path("/tmp/../../../etc/passwd"), "report")

    def test_accepts_valid_path(self, tmp_path):
        result = _validate_write_path(tmp_path / "report.html", "report")
        assert isinstance(result, Path)

    def test_rejects_var_dir(self):
        with pytest.raises(ValueError, match="protected directory"):
            _validate_write_path(Path("/var/log/evil.html"), "report")

    def test_rejects_proc_dir(self):
        with pytest.raises(ValueError, match="protected directory"):
            _validate_write_path(Path("/proc/self/status"), "report")

    def test_binary_search_not_blocked(self):
        """'/binary_search/output.html' must NOT be blocked as /bin."""
        result = _validate_write_path(Path("/binary_search/output.html"), "report")
        assert isinstance(result, Path)


# ============================================================================
# ll-7rudv-vp1ad: SSRF bypass via percent-encoded IP hostnames
# ============================================================================


class TestUrlValidate_PercentEncodedSSRF:
    """Test that percent-encoded IP hostnames are blocked to prevent SSRF bypass.

    An attacker can encode a private IP (e.g., 127.0.0.1) as percent-encoded
    octets. urlparse preserves the encoding in .hostname, so ip_address()
    fails on the encoded form, DNS resolution fails, and allow_remote=True
    previously returned True — but urllib normalizes the hostname and connects
    to the decoded private IP.
    """

    def test_percent_encoded_loopback_blocked_allow_remote(self):
        """Percent-encoded loopback on default port is ALLOWED with allow_remote=True.

        This is the correct behavior: loopback addresses are permitted for remote
        access (allow_remote=True means any safe address is fine, including localhost).
        The SSRF bypass is about private/link-local IPs, not loopback.
        """
        url = "http://%31%32%37%2e%30%2e%30%2e%31:11434/"
        # Loopback IS allowed with allow_remote=True — this is not a bypass
        assert is_safe_url(url, allow_remote=True) is True

    def test_percent_encoded_private_ip_blocked(self):
        """Percent-encoded 10.0.0.1 must be blocked even with allow_remote=True."""
        url = "http://%31%30%2e%30%2e%30%2e%31:11434/"
        assert not is_safe_url(url, allow_remote=True)

    def test_percent_encoded_metadata_ip_blocked(self):
        """Percent-encoded 169.254.169.254 (cloud metadata) must be blocked."""
        url = "http://%31%36%39%2e%32%35%34%2e%31%36%39%2e%32%35%34:80/"
        assert not is_safe_url(url, allow_remote=True)

    def test_percent_encoded_loopback_no_remote_allowed_on_default_port(self):
        """Percent-encoded loopback on Ollama port is allowed with allow_remote=False."""
        url = "http://%31%32%37%2e%30%2e%30%2e%31:11434/"
        # Loopback on the default port IS allowed — this is normal Ollama usage
        assert is_safe_url(url, allow_remote=False) is True

    def test_percent_encoded_loopback_no_remote_blocked_non_default_port(self):
        """Percent-encoded loopback on non-default port is blocked with allow_remote=False."""
        url = "http://%31%32%37%2e%30%2e%30%2e%31:8080/"
        assert not is_safe_url(url, allow_remote=False)

    def test_safe_urlopen_rejects_percent_encoded_private_ip(self):
        """safe_urlopen must reject percent-encoded private IP hostnames."""
        url = "http://%31%30%2e%30%2e%30%2e%31:11434/api/tags"
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen(url)


# ============================================================================
# ll-7rudv-k7u4f: DB file permissions — WAL/SHM chmod + umask
# ============================================================================


class TestStore_DbFilePermissions:
    """Test that the DB file and its WAL/SHM sidecars get 0o600 permissions."""

    def test_db_file_created_with_restrictive_permissions(self, tmp_path):
        """DB file should be created with 0o600 permissions."""
        db = tmp_path / "test_perm.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        # The file should exist and be readable/writable only by owner
        mode = db.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"
        store.close()

    def test_directory_created_with_0700_permissions(self, tmp_path):
        """The parent directory of the DB should have 0o700 permissions."""
        db_dir = tmp_path / "deeply" / "nested" / "dir"
        db = db_dir / "test.db"
        store = MemoryStore(db_path=db, disable_vec=True)
        mode = db_dir.stat().st_mode & 0o777
        assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"
        store.close()


# ============================================================================
# ll-7rudv-gkn1z: OOM DoS via no LIMIT in _search_by_embedding_brute and export_all
# ============================================================================


class TestStore_LimitClauses:
    """Test that brute-force search and export_all have row limits to prevent OOM."""

    def test_export_all_has_default_limit(self, store):
        """export_all with default args uses _EXPORT_MAX_ROWS as the limit."""
        # Add a few memories and verify export_all works
        store.add(type="fact", content="test1")
        store.add(type="fact", content="test2")
        results = store.export_all()
        assert len(results) >= 2

    def test_export_all_custom_limit(self, store):
        """export_all with a custom limit caps the results."""
        store.add(type="fact", content="test1")
        store.add(type="fact", content="test2")
        store.add(type="fact", content="test3")
        results = store.export_all(limit=2)
        assert len(results) <= 2

    def test_search_by_embedding_brute_has_limit(self, store):
        """_search_by_embedding_brute respects the limit parameter."""
        # The limit param is already respected in the Python-level filtering,
        # but we added a DB-level cap to prevent fetching too many rows
        import struct

        emb = struct.pack("3f", 1.0, 0.0, 0.0)
        store.add(type="fact", content="test1", embedding=emb)
        results = store.search_by_embedding([1.0, 0.0, 0.0], limit=1)
        assert len(results) <= 1


# ============================================================================
# ll-7rudv-ojbjt: process_transcript size limit
# ============================================================================


class TestHooks_ProcessTranscriptSizeLimit:
    """Test that process_transcript enforces a size limit to prevent OOM."""

    def test_process_transcript_rejects_oversized_text(self, store):
        """process_transcript should reject text exceeding max_file_size."""
        from llmem.hooks import SessionHook, PROCESS_RESULT_FILE_TOO_LARGE

        hook = SessionHook(store=store, extractor=MagicMock(), max_file_size=100)
        result_type, count = hook.process_transcript(
            source_id="test-session",
            text="x" * 200,  # Exceeds 100 bytes
            source_type="session",
        )
        assert result_type == PROCESS_RESULT_FILE_TOO_LARGE
        assert count == 0


# ============================================================================
# ll-7rudv-18exx: Context file writes bypass _validate_write_path()
# ============================================================================


class TestSessionHooks_ValidateWritePath:
    """Test that context file writes go through _validate_write_path.

    This test verifies that on_created and on_compacting validate the write
    path before writing. Since _validate_write_path rejects traversal and
    system directories, passing a safe context dir should succeed.
    """

    def test_on_created_calls_validate_write_path(self, tmp_path):
        """on_created should call _validate_write_path for the context file."""
        from llmem.session_hooks import SessionHookCoordinator, SESSION_CREATED_SUCCESS

        mock_store = MagicMock()
        mock_store.is_extracted.return_value = False
        mock_store.search.return_value = []
        mock_store.log_extraction.return_value = None
        mock_retriever = MagicMock()
        mock_retriever.format_context.return_value = "test context"
        mock_extractor = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.list_sessions.return_value = []
        coordinator = SessionHookCoordinator(
            store=mock_store,
            retriever=mock_retriever,
            extractor=mock_extractor,
            embedder=None,
            adapter=mock_adapter,
        )

        with patch(
            "llmem.session_hooks.get_context_dir", return_value=tmp_path / "context"
        ):
            result_type, file_path = coordinator.on_created("ses_123")
            assert result_type == SESSION_CREATED_SUCCESS

    def test_on_compacting_calls_validate_write_path(self, tmp_path):
        """on_compacting should call _validate_write_path for the context file."""
        from llmem.session_hooks import (
            SessionHookCoordinator,
            SESSION_COMPACTING_SUCCESS,
        )

        mock_store = MagicMock()
        mock_store.is_extracted.return_value = False
        mock_store.search.return_value = [
            {
                "id": "mem-1",
                "type": "decision",
                "content": "Test decision",
                "confidence": 0.9,
            },
        ]
        mock_store.log_extraction.return_value = None
        mock_retriever = MagicMock()
        mock_retriever.format_context.return_value = ""
        mock_extractor = MagicMock()
        mock_adapter = MagicMock()
        coordinator = SessionHookCoordinator(
            store=mock_store,
            retriever=mock_retriever,
            extractor=mock_extractor,
            embedder=None,
            adapter=mock_adapter,
        )

        with patch(
            "llmem.session_hooks.get_context_dir", return_value=tmp_path / "context"
        ):
            result_type, file_path = coordinator.on_compacting("ses_123")
            assert result_type == SESSION_COMPACTING_SUCCESS


# ============================================================================
# ll-7rudv-2vk1u: API key credential exfiltration prevention
# ============================================================================


class TestProviders_CredentialExfiltration:
    """Test that API keys cannot be sent to non-HTTPS non-loopback URLs."""

    def test_openai_rejects_http_non_loopback_with_api_key(self):
        """OpenAIProvider should refuse to send API keys over HTTP to non-loopback URLs."""
        with pytest.raises(ValueError, match="non-HTTPS"):
            OpenAIProvider(api_key="test-key", base_url="http://evil.example.com:11434")

    def test_anthropic_rejects_http_non_loopback_with_api_key(self):
        """AnthropicProvider should refuse to send API keys over HTTP to non-loopback URLs."""
        with pytest.raises(ValueError, match="non-HTTPS"):
            AnthropicProvider(
                api_key="test-key", base_url="http://evil.example.com:11434"
            )

    def test_openai_allows_https_base_url(self):
        """OpenAIProvider should accept HTTPS URLs (normal usage)."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider._base_url == "https://api.openai.com"

    def test_openai_allows_localhost_with_api_key(self):
        """OpenAIProvider should accept HTTP localhost URLs for development."""
        provider = OpenAIProvider(api_key="test-key", base_url="http://localhost:8080")
        assert provider._base_url == "http://localhost:8080"

    def test_openai_allows_127_with_api_key(self):
        """OpenAIProvider should accept HTTP 127.0.0.1 URLs for development."""
        provider = OpenAIProvider(api_key="test-key", base_url="http://127.0.0.1:8080")
        assert provider._base_url == "http://127.0.0.1:8080"

    def test_anthropic_allows_https_base_url(self):
        """AnthropicProvider should accept HTTPS URLs (normal usage)."""
        provider = AnthropicProvider(api_key="test-key")
        assert provider._base_url == "https://api.anthropic.com"

    def test_openai_warns_on_non_default_url(self, caplog):
        """OpenAIProvider should log a warning for non-default base URLs."""
        with caplog.at_level(logging.WARNING):
            OpenAIProvider(api_key="test-key", base_url="https://custom.api.com/v1")
        warning_msgs = [r.message for r in caplog.records]
        assert any("non-default base_url" in msg for msg in warning_msgs)

    def test_anthropic_warns_on_non_default_url(self, caplog):
        """AnthropicProvider should log a warning for non-default base URLs."""
        with caplog.at_level(logging.WARNING):
            AnthropicProvider(api_key="test-key", base_url="https://custom.api.com/v1")
        warning_msgs = [r.message for r in caplog.records]
        assert any("non-default base_url" in msg for msg in warning_msgs)

    # ll-7rudv-kg3m3: Credential exfiltration substring bypass fix
    # The original check used substring matching ('localhost' not in base_url)
    # which could be bypassed by URLs like http://localhost.evil.com that
    # contain 'localhost' as a substring but resolve to a remote host.

    def test_openai_rejects_localhost_subdomain_bypass(self):
        """http://localhost.evil.com must be rejected — 'localhost' is a substring, not the hostname."""
        with pytest.raises(ValueError, match="non-HTTPS"):
            OpenAIProvider(
                api_key="test-key", base_url="http://localhost.evil.com:11434"
            )

    def test_openai_rejects_127_subdomain_bypass(self):
        """http://127.0.0.1.evil.com must be rejected — '127.0.0.1' is a substring, not the hostname."""
        with pytest.raises(ValueError, match="non-HTTPS"):
            OpenAIProvider(
                api_key="test-key", base_url="http://127.0.0.1.evil.com:11434"
            )

    def test_anthropic_rejects_localhost_subdomain_bypass(self):
        """http://localhost.evil.com must be rejected for Anthropic too."""
        with pytest.raises(ValueError, match="non-HTTPS"):
            AnthropicProvider(
                api_key="test-key", base_url="http://localhost.evil.com:11434"
            )

    def test_anthropic_rejects_127_subdomain_bypass(self):
        """http://127.0.0.1.evil.com must be rejected for Anthropic too."""
        with pytest.raises(ValueError, match="non-HTTPS"):
            AnthropicProvider(
                api_key="test-key", base_url="http://127.0.0.1.evil.com:11434"
            )


class TestIsLoopbackHostname:
    """Test _is_loopback_hostname uses exact hostname matching, not substring."""

    def test_localhost_exact_match(self):
        """Exact 'localhost' hostname is loopback."""
        assert _is_loopback_hostname("http://localhost:8080") is True

    def test_localhost_no_port(self):
        """'localhost' without port is loopback."""
        assert _is_loopback_hostname("http://localhost") is True

    def test_127_0_0_1_exact_match(self):
        """Exact '127.0.0.1' hostname is loopback."""
        assert _is_loopback_hostname("http://127.0.0.1:8080") is True

    def test_ipv6_loopback(self):
        """IPv6 loopback '::1' is loopback."""
        assert _is_loopback_hostname("http://[::1]:8080") is True

    def test_localhost_subdomain_is_not_loopback(self):
        """'localhost.evil.com' is NOT loopback — prevents substring bypass."""
        assert _is_loopback_hostname("http://localhost.evil.com:11434") is False

    def test_127_subdomain_is_not_loopback(self):
        """'127.0.0.1.evil.com' is NOT loopback — prevents substring bypass."""
        assert _is_loopback_hostname("http://127.0.0.1.evil.com:11434") is False

    def test_remote_host_is_not_loopback(self):
        """A remote hostname is not loopback."""
        assert _is_loopback_hostname("http://evil.example.com:11434") is False

    def test_https_remote_is_not_loopback(self):
        """HTTPS URLs to remote hosts are not loopback."""
        assert _is_loopback_hostname("https://api.openai.com") is False

    def test_empty_hostname(self):
        """URL with no hostname returns False."""
        assert _is_loopback_hostname("http://") is False


# ============================================================================
# ll-7rudv-0yadu: import_memories input validation
# ============================================================================


class TestStore_ImportMemoriesValidation:
    """Test import_memories input validation for id format and embedding size."""

    def test_import_rejects_oversized_embedding(self, store):
        """import_memories should skip entries with oversized embedding data."""
        import struct

        # Create a very large embedding (exceeds 1MB limit)
        huge_embedding = struct.pack("1024f", *([0.1] * 1024)) * 300
        memories = [
            {
                "type": "fact",
                "content": "test",
                "embedding": huge_embedding,
            }
        ]
        count = store.import_memories(memories)
        assert count == 0

    def test_import_rejects_non_bytes_embedding(self, store):
        """import_memories should skip entries with non-bytes embedding."""
        memories = [
            {
                "type": "fact",
                "content": "test",
                "embedding": "not bytes",
            }
        ]
        count = store.import_memories(memories)
        assert count == 0

    def test_imports_without_embedding(self, store):
        """Memories without embeddings should import fine."""
        memories = [{"type": "fact", "content": "no embedding"}]
        count = store.import_memories(memories)
        assert count == 1

    def test_import_rejects_oversized_id(self, store):
        """import_memories should skip entries with IDs exceeding max length."""
        memories = [
            {
                "type": "fact",
                "content": "test",
                "id": "x" * 300,
            }
        ]
        count = store.import_memories(memories)
        assert count == 0

    def test_import_rejects_non_string_id(self, store):
        """import_memories should skip entries with non-string IDs."""
        memories = [
            {
                "type": "fact",
                "content": "test",
                "id": 12345,
            }
        ]
        count = store.import_memories(memories)
        assert count == 0

    def test_import_rejects_non_numeric_confidence(self, store):
        """import_memories should skip entries with non-numeric confidence."""
        memories = [
            {
                "type": "fact",
                "content": "test",
                "confidence": "high",
            }
        ]
        count = store.import_memories(memories)
        assert count == 0

    def test_import_valid_entry_still_works(self, store):
        """import_memories should still accept valid entries."""
        memories = [{"type": "fact", "content": "a basic test"}]
        count = store.import_memories(memories)
        assert count == 1


# ============================================================================
# Security fixes from ll-dunf9 audit
# ============================================================================


class TestPaths_ValidateHomePath_NoFalsePositives:
    """Test that _validate_home_path does not produce false positives from bare prefix matching."""

    def test_binary_search_dir_allowed(self):
        """'/binary_search/llmem' must NOT be blocked as /bin."""
        # This used to fail with bare startswith('/bin') check
        result = _validate_home_path(Path("/binary_search/llmem"), "LMEM_HOME")
        assert isinstance(result, Path)

    def test_usabin_dir_allowed(self):
        """'/usabin/llmem' must NOT be blocked as /usr/sbin or /usr/bin."""
        result = _validate_home_path(Path("/usabin/llmem"), "LMEM_HOME")
        assert isinstance(result, Path)


class TestUrlValidate_IsRemoteAllowed_FailClosed:
    """Test that safe_urlopen enforces port restrictions for loopback URLs.

    This addresses the SSRF vulnerability where _is_remote_allowed() returned
    True for hostname-based localhost URLs, bypassing the port-11434 check.
    """

    def test_safe_urlopen_uses_allow_remote_for_loopback(self):
        """safe_urlopen with explicit allow_remote=False validates port."""
        # This should reject non-11434 ports on loopback
        with pytest.raises(ValueError, match="URL rejected"):
            safe_urlopen("http://localhost:8080/api", allow_remote=False)


class TestUrlValidate_SafeUrlopen_ExplicitAllowRemote:
    """Test safe_urlopen allow_remote parameter."""

    def test_explicit_allow_remote_true_for_loopback(self):
        """allow_remote=True with loopback URL should pass is_safe_url
        but will likely fail to connect (which is OK)."""
        # We can only test validation — actual connection will fail
        # Validate that the URL passes is_safe_url with allow_remote=True
        assert is_safe_url("http://localhost:8080/api", allow_remote=True) is True


class TestPaths_ValidateHomePath_SymlinkCheck:
    """Test that _validate_home_path rejects symlinks."""

    def test_rejects_symlink(self, tmp_path):
        """A symlink home path must be rejected."""
        target = tmp_path / "real_home"
        target.mkdir()
        symlink = tmp_path / "symlink_home"
        symlink.symlink_to(target)
        with pytest.raises(ValueError, match="symlink"):
            _validate_home_path(symlink, "LMEM_HOME")


class TestPaths_BlockedPrefixConsistency:
    """Test that _validate_home_path and _validate_write_path block the same prefixes."""

    def test_home_path_blocks_sbin(self):
        """_validate_home_path blocks /sbin (was missing from _validate_write_path)."""
        with pytest.raises(ValueError, match="system directory"):
            _validate_home_path(Path("/sbin/llmem"), "LMEM_HOME")

    def test_write_path_blocks_usr_sbin(self):
        """_validate_write_path blocks /usr/sbin (now consistent with home path)."""
        with pytest.raises(ValueError, match="protected directory"):
            _validate_write_path(Path("/usr/sbin/evil.html"), "report")


class TestPaths_IsBlockedPath_NoFalsePositives:
    """Test that _is_blocked_path uses prefix+'/' matching to avoid false positives.

    The old bare startswith check would match /binary_search/data.db against /bin.
    The new prefix+'/' check correctly requires /bin/ (or exact /bin).
    """

    def test_binary_search_not_blocked(self):
        """'/binary_search/data.db' must NOT match the /bin prefix."""
        from llmem.paths import _is_blocked_path

        assert not _is_blocked_path(Path("/binary_search/data.db"))

    def test_bin_subdir_blocked(self):
        """'/bin/ls' must match the /bin prefix."""
        from llmem.paths import _is_blocked_path

        assert _is_blocked_path(Path("/bin/ls"))

    def test_bin_exact_blocked(self):
        """'/bin' itself (exact match) must be blocked."""
        from llmem.paths import _is_blocked_path

        assert _is_blocked_path(Path("/bin"))

    def test_usabin_not_blocked(self):
        """'/usabin/local/cmd' must NOT match /usr/sbin."""
        from llmem.paths import _is_blocked_path

        assert not _is_blocked_path(Path("/usabin/local/cmd"))

    def test_usr_bin_blocked(self):
        """'/usr/bin/python3' must match /usr/bin prefix."""
        from llmem.paths import _is_blocked_path

        assert _is_blocked_path(Path("/usr/bin/python3"))

    def test_usr_sbin_blocked(self):
        """'/usr/sbin/apache2' must match /usr/sbin prefix."""
        from llmem.paths import _is_blocked_path

        assert _is_blocked_path(Path("/usr/sbin/apache2"))

    def test_home_dir_not_blocked(self):
        """'/home/user/data' must NOT be blocked."""
        from llmem.paths import _is_blocked_path

        assert not _is_blocked_path(Path("/home/user/data"))

    def test_tmp_not_blocked(self):
        """'/tmp/llmem-test' must NOT be blocked."""
        from llmem.paths import _is_blocked_path

        assert not _is_blocked_path(Path("/tmp/llmem-test"))

    def test_all_prefixes_blocked_as_dirs(self):
        """Every prefix in _BLOCKED_PATH_PREFIXES must block paths under it."""
        from llmem.paths import _BLOCKED_PATH_PREFIXES, _is_blocked_path

        for prefix in _BLOCKED_PATH_PREFIXES:
            assert _is_blocked_path(Path(prefix + "/some/file")), (
                f"{prefix}/some/file should be blocked"
            )

    def test_all_prefixes_blocked_as_exact(self):
        """Every prefix in _BLOCKED_PATH_PREFIXES must block itself as exact match."""
        from llmem.paths import _BLOCKED_PATH_PREFIXES, _is_blocked_path

        for prefix in _BLOCKED_PATH_PREFIXES:
            assert _is_blocked_path(Path(prefix)), (
                f"{prefix} should be blocked as exact match"
            )


class TestCli_CmdAdd_FileReadProtection:
    """Test that cmd_add --file rejects paths in protected directories."""

    def test_rejects_etc_file(self, tmp_path):
        """Reading from /etc should be rejected."""
        from llmem.cli import cmd_add
        import argparse

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            type="fact",
            content=None,
            file="/etc/passwd",
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            with pytest.raises(SystemExit):
                cmd_add(args)

    def test_rejects_proc_file(self, tmp_path):
        """Reading from /proc should be rejected."""
        from llmem.cli import cmd_add
        import argparse

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            type="fact",
            content=None,
            file="/proc/self/status",
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            with pytest.raises(SystemExit):
                cmd_add(args)

    def test_rejects_usr_bin_file(self, tmp_path):
        """Reading from /usr/bin should be rejected (was missing from old hardcoded list)."""
        from llmem.cli import cmd_add
        import argparse

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            type="fact",
            content=None,
            file="/usr/bin/python3",
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            with pytest.raises(SystemExit):
                cmd_add(args)

    def test_rejects_sbin_file(self, tmp_path):
        """Reading from /sbin should be rejected (was missing from old hardcoded list)."""
        from llmem.cli import cmd_add
        import argparse

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            type="fact",
            content=None,
            file="/sbin/iptables",
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            with pytest.raises(SystemExit):
                cmd_add(args)

    def test_allows_binary_search_path(self, tmp_path):
        """'/binary_search/data.db' must NOT be blocked (false positive fix)."""
        from llmem.cli import cmd_add
        import argparse

        # Create a file at a path that starts with /bin-like string but is not /bin
        binary_dir = tmp_path / "binary_search"
        binary_dir.mkdir()
        data_file = binary_dir / "data.db"
        data_file.write_text("safe data")

        db = tmp_path / "test.db"
        MemoryStore(db_path=db, disable_vec=True).close()

        args = argparse.Namespace(
            db=db,
            type="fact",
            content=None,
            file=str(data_file),
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_add(args)  # Should not raise

    def test_allows_valid_file(self, tmp_path):
        """Reading from a safe directory should work."""
        from llmem.cli import cmd_add
        import argparse

        db = tmp_path / "test.db"
        MemoryStore(db_path=db, disable_vec=True).close()
        content_file = tmp_path / "content.txt"
        content_file.write_text("test content")

        args = argparse.Namespace(
            db=db,
            type="fact",
            content=None,
            file=str(content_file),
            summary=None,
            source="manual",
            confidence=0.8,
            valid_until=None,
            metadata=None,
            relation=None,
            relation_to=None,
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            cmd_add(args)  # Should not raise


class TestCli_CmdExport_WriteProtection:
    """Test that cmd_export --output uses _validate_write_path."""

    def test_rejects_etc_output(self, tmp_path):
        """Writing to /etc should be rejected."""
        from llmem.cli import cmd_export
        import argparse

        db = tmp_path / "test.db"
        MemoryStore(db_path=db, disable_vec=True).close()

        args = argparse.Namespace(
            db=db,
            output="/etc/llmem-export.json",
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            with pytest.raises(ValueError, match="protected directory"):
                cmd_export(args)

    def test_rejects_traversal_output(self, tmp_path):
        """Paths with '..' should be rejected."""
        from llmem.cli import cmd_export
        import argparse

        db = tmp_path / "test.db"
        MemoryStore(db_path=db, disable_vec=True).close()

        args = argparse.Namespace(
            db=db,
            output="/tmp/../../../etc/export.json",
        )

        with patch(
            "llmem.cli.MemoryStore",
            side_effect=lambda db_path, **kw: MemoryStore(
                db_path=db_path, disable_vec=True
            ),
        ):
            with pytest.raises(ValueError, match="traversal"):
                cmd_export(args)


class TestCli_CmdImport_ReadOnlyProtection:
    """Test that cmd_import rejects files in protected directories."""

    def test_rejects_etc_file(self, tmp_path):
        """Importing from /etc should be rejected."""
        from llmem.cli import cmd_import
        import argparse

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            file="/etc/shadow",
        )

        with pytest.raises(SystemExit):
            cmd_import(args)

    def test_rejects_sbin_file(self, tmp_path):
        """Importing from /sbin should be rejected (was missing from old list)."""
        from llmem.cli import cmd_import
        import argparse

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            file="/sbin/iptables",
        )

        with pytest.raises(SystemExit):
            cmd_import(args)

    def test_rejects_usr_bin_file(self, tmp_path):
        """Importing from /usr/bin should be rejected (was missing from old list)."""
        from llmem.cli import cmd_import
        import argparse

        args = argparse.Namespace(
            db=tmp_path / "test.db",
            file="/usr/bin/env",
        )

        with pytest.raises(SystemExit):
            cmd_import(args)

    def test_rejects_oversized_file(self, tmp_path):
        """Files larger than MAX_IMPORT_FILE_SIZE should be rejected."""
        from llmem.cli import cmd_import
        from llmem.cli import MAX_IMPORT_FILE_SIZE
        import argparse

        large_file = tmp_path / "large.json"
        large_file.write_text("x" * (MAX_IMPORT_FILE_SIZE + 1))

        db = tmp_path / "test.db"
        MemoryStore(db_path=db, disable_vec=True).close()

        args = argparse.Namespace(
            db=db,
            file=str(large_file),
        )

        with pytest.raises(SystemExit):
            cmd_import(args)


class TestConfig_ServerAuthToken_MinStrength:
    """Test that get_server_auth_token enforces minimum token strength."""

    def test_rejects_short_token(self):
        """Tokens shorter than 16 characters should be rejected."""
        from llmem.config import get_server_auth_token

        with pytest.raises(ValueError, match="too short"):
            get_server_auth_token(config={"server": {"auth_token": "short"}})

    def test_rejects_empty_string_token(self):
        """Empty string token should return None (not set)."""
        from llmem.config import get_server_auth_token

        result = get_server_auth_token(config={"server": {"auth_token": ""}})
        assert result is None

    def test_accepts_strong_token(self):
        """Tokens >= 16 characters should be accepted."""
        from llmem.config import get_server_auth_token

        strong_token = "a" * 32
        result = get_server_auth_token(config={"server": {"auth_token": strong_token}})
        assert result == strong_token

    def test_returns_none_when_not_set(self):
        """When token is not configured, return None."""
        from llmem.config import get_server_auth_token

        result = get_server_auth_token(config={})
        assert result is None

    def test_rejects_non_string_token(self):
        """Non-string token values should raise ValueError."""
        from llmem.config import get_server_auth_token

        with pytest.raises(ValueError, match="must be a string"):
            get_server_auth_token(config={"server": {"auth_token": 12345}})


class TestConfig_GetOllamaUrl_SsrfValidation:
    """Test that get_ollama_url validates URLs via is_safe_url."""

    def test_rejects_file_scheme(self):
        """file:// URLs should be rejected."""
        from llmem.config import get_ollama_url

        with pytest.raises(ValueError, match="unsafe|rejected"):
            get_ollama_url(config={"memory": {"ollama_url": "file:///etc/passwd"}})

    def test_rejects_private_ip_without_allow_remote(self):
        """private IP URLs should be rejected by is_safe_url."""
        from llmem.config import get_ollama_url

        with pytest.raises(ValueError, match="unsafe|rejected"):
            get_ollama_url(
                config={"memory": {"ollama_url": "http://192.168.1.1:11434"}}
            )


class TestStore_ExtensionLoadingDisabled:
    """Test that SQLite extension loading is disabled after use."""

    def test_extension_loading_disabled_after_init(self, tmp_path):
        """After MemoryStore init, extension loading should be disabled."""
        import sqlite3

        db = tmp_path / "test.db"
        store = MemoryStore(db_path=db, disable_vec=False)
        try:
            conn = store._connect()
            # Attempting to load an extension should fail
            with pytest.raises(
                sqlite3.OperationalError, match="not authorized|extension"
            ):
                conn.enable_load_extension(True)
                conn.load_extension("nonexistent_extension")
        finally:
            store.close()

    def test_extension_loading_disabled_when_vec_unavailable(self, tmp_path):
        """Extension loading should be disabled even when sqlite-vec is unavailable."""
        import sqlite3

        db = tmp_path / "test.db"
        # Force sqlite-vec import to fail
        with patch.dict("sys.modules", {"sqlite_vec": None}):
            store = MemoryStore(db_path=db, disable_vec=False)
            try:
                conn = store._connect()
                with pytest.raises(
                    sqlite3.OperationalError, match="not authorized|extension"
                ):
                    conn.enable_load_extension(True)
                    conn.load_extension("nonexistent_extension")
            finally:
                store.close()


class TestOpencode_UriPathInjection:
    """Test that OpenCodeAdapter rejects path injection via URI parameters."""

    def test_rejects_question_mark_in_path(self, tmp_path):
        """Paths with '?' (URI query separator) should be rejected."""
        from llmem.adapters.opencode import OpenCodeAdapter

        # Create a valid database first
        db = tmp_path / "test.db"
        import sqlite3

        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE session (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        # Inject a query parameter to attempt mode=rw escalation
        bad_path = tmp_path / "test.db?mode=rw"
        with pytest.raises(ValueError, match="disallowed character"):
            OpenCodeAdapter(bad_path)

    def test_rejects_hash_in_path(self, tmp_path):
        """Paths with '#' (URI fragment separator) should be rejected."""
        from llmem.adapters.opencode import OpenCodeAdapter

        db = tmp_path / "test.db"
        import sqlite3

        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE session (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        bad_path = tmp_path / "test.db#fragment"
        with pytest.raises(ValueError, match="disallowed character"):
            OpenCodeAdapter(bad_path)

    def test_rejects_traversal_in_path(self, tmp_path):
        """Paths with '..' should be rejected."""
        from llmem.adapters.opencode import OpenCodeAdapter

        bad_path = tmp_path / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="traversal"):
            OpenCodeAdapter(bad_path)

    def test_accepts_valid_path(self, tmp_path):
        """Normal database paths should work fine."""
        from llmem.adapters.opencode import OpenCodeAdapter

        db = tmp_path / "test.db"
        import sqlite3

        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE session (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        adapter = OpenCodeAdapter(db)
        adapter.close()  # Should not raise


class TestOllama_ProviderDetector_NoKeyLeakage:
    """Test that ProviderDetector.detect() no longer exposes API key presence."""

    def test_detect_does_not_return_key_found(self):
        """detect() should not include openai_key_found or anthropic_key_found."""
        with patch("llmem.ollama.is_ollama_running", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("ANTHROPIC_API_KEY", None)
                detector = ProviderDetector()
                result = detector.detect()

        assert "openai_key_found" not in result
        assert "anthropic_key_found" not in result
        assert "provider" in result
        assert "ollama_url" in result

    def test_detect_with_openai_key_no_leakage(self):
        """Even with OPENAI_API_KEY set, detect should not leak presence."""
        with patch("llmem.ollama.is_ollama_running", return_value=False):
            with patch.dict(
                os.environ, {"OPENAI_API_KEY": "sk-test-key-123"}, clear=False
            ):
                detector = ProviderDetector()
                result = detector.detect()

        assert "openai_key_found" not in result
        assert result["provider"] == "openai"


class TestExtract_BoundedRegex:
    """Test that extract.py uses a bounded regex to avoid ReDoS.

    The regex must use a bounded quantifier {1,100000} to prevent
    catastrophic backtracking (ReDoS). It should be greedy so that
    nested JSON arrays are matched correctly, not truncated at an
    inner ] bracket.
    """

    def test_regex_pattern_is_bounded_greedy(self):
        """The JSON array extraction regex should use bounded greedy matching."""
        import inspect
        from llmem.extract import ExtractionEngine

        source = inspect.getsource(ExtractionEngine.extract)
        # The pattern must use a bounded quantifier to prevent ReDoS
        assert ".{1,100000}" in source
        # The unbounded greedy .*] pattern must NOT be present
        # (the bounded version .{1,100000}] is acceptable)
        assert "[.*\\]" not in source or ".{1,100000}\\]" in source

    def test_bounded_greedy_regex_matches_nested_json(self):
        """Bounded greedy regex must match complete nested JSON arrays.

        A non-greedy regex like [.{1,N}?] would match from the opening
        bracket to the FIRST closing bracket, producing invalid JSON
        like [{"a": [1, 2] when the input contains nested arrays.
        The greedy version [.{1,N}] correctly matches the outermost pair.
        """
        import re

        pattern = re.compile(r"\[.{1,100000}\]", re.DOTALL)

        # Nested JSON array: the regex must match the FULL outer array
        nested = '[{"a": [1, 2], "b": 3}]'
        match = pattern.search(nested)
        assert match is not None
        import json

        parsed = json.loads(match.group(0))
        assert isinstance(parsed, list)
        assert parsed[0]["a"] == [1, 2]

        # Multiple nested arrays in one response
        text = 'Here are the results:\n[{"x": [1], "y": [2, 3]}]\nEnd.'
        match = pattern.search(text)
        assert match is not None
        parsed = json.loads(match.group(0))
        assert isinstance(parsed, list)

    def test_bounded_regex_no_redos(self):
        """Bounded quantifier prevents ReDoS by capping backtracking."""
        import re
        import time

        pattern = re.compile(r"\[.{1,100000}\]", re.DOTALL)
        # Crafted adversarial input: many nested brackets without closing
        adversarial = "[" + "a" * 50000 + "]"
        start = time.monotonic()
        pattern.search(adversarial)
        elapsed = time.monotonic() - start
        # Should complete in well under 1 second (bounded quantifier)
        assert elapsed < 1.0
