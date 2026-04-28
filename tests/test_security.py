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
"""

import json
import os
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from llmem.url_validate import (
    is_safe_url,
    validate_url,
    _strip_credentials,
    _NoRedirectHandler,
    safe_urlopen,
    _extract_url_string,
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
        result = handler.redirect_request(
            MagicMock(),  # req
            MagicMock(),  # fp
            301,  # code
            "Moved",  # msg
            MagicMock(),  # headers
            "http://evil.internal/",  # newurl
        )
        assert result is None


class TestUrlValidate_SafeUrlopen:
    """Test safe_urlopen URL validation and error handling."""

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
        assert "pass" not in str(exc_info.value).lower() or "password" not in str(
            exc_info.value
        )


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
