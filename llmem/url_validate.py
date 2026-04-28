"""URL validation for urllib calls — prevent SSRF and file-scheme attacks."""

import http.client
import ipaddress
import socket
import urllib.request
import urllib.error
from urllib.parse import urlparse, urlunparse, unquote

OLLAMA_DEFAULT_PORT = 11434

# Default timeout for all urllib calls (seconds).
# Prevents infinite hangs on unresponsive hosts.
DEFAULT_URLOPEN_TIMEOUT = 30


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP address should be blocked (private/link-local/etc).

    Loopback addresses are NOT blocked — Ollama typically runs on localhost.
    """
    if ip.is_loopback:
        return False
    return bool(ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast)


def _resolve_hostname(hostname: str) -> list[str] | None:
    """Resolve a hostname to a list of IP address strings.

    Returns None on resolution failure.
    """
    try:
        addrs = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return [a[4][0] for a in addrs]
    except Exception:
        return None


def _check_ip_access(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    allow_remote: bool,
    port: int,
) -> bool:
    """Check whether a resolved IP is permitted for access.

    When allow_remote is False (the default), only loopback addresses on the
    Ollama default port are permitted. All other IPs — including public IPs
    that are not blocked by _ip_is_blocked — are rejected.

    When allow_remote is True, any IP that is not blocked (private, link-local,
    reserved, multicast) is permitted.

    Returns True if allowed, False if blocked.

    Loopback addresses are only permitted on the Ollama default port,
    regardless of allow_remote. The allow_remote flag controls whether
    non-loopback public addresses are allowed — it does NOT open up
    all loopback ports.
    """
    if _ip_is_blocked(ip):
        return False
    if ip.is_loopback:
        if port != OLLAMA_DEFAULT_PORT:
            return False
    elif not allow_remote:
        # Non-loopback addresses require allow_remote=True
        return False
    return True


def _get_effective_port(parsed) -> int:
    """Return the port for a parsed URL, using scheme defaults if missing."""
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    return 80


def _strip_credentials(url: str) -> str:
    """Remove userinfo (credentials) from a URL for safe error display.

    Turns 'http://user:pass@host/path' into 'http://host/path'.
    """
    parsed = urlparse(url)
    # Rebuild URL without userinfo
    safe = urlunparse(
        (
            parsed.scheme,
            parsed.hostname or "",
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )
    port_str = f":{parsed.port}" if parsed.port else ""
    # Reconstruct with hostname + port but no credentials
    netloc = f"{parsed.hostname or ''}{port_str}"
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def is_safe_url(url: str, allow_remote: bool = False) -> bool:
    """Validate that a URL is safe to pass to urllib.

    Checks:
    - Scheme must be http or https (blocks file://, ftp://, data://, etc.)
    - Must have a hostname
    - Percent-encoded hostnames are decoded before IP checks to prevent
      SSRF bypass (e.g. %31%32%37%2e%30%2e%30%2e%31 → 127.0.0.1).
    - If allow_remote is False (default), only loopback addresses on the
      Ollama default port are allowed.
    - If allow_remote is True, any reachable hostname is allowed (still
      blocks non-http schemes and obviously invalid hostnames).
    - Validates resolved IP to prevent DNS-rebinding TOCTOU: the hostname
      must resolve to permitted addresses immediately before the request.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    # Reject URLs with credentials embedded — these should never reach urllib
    if parsed.username or parsed.password:
        return False
    port = _get_effective_port(parsed)
    # Percent-decode the hostname before IP checks to prevent SSRF bypass
    # via encoded IP addresses (e.g. %31%32%37%2e%30%2e%30%2e%31 → 127.0.0.1).
    # urllib normalizes percent-encoded hostnames, so we must check the
    # decoded form to match what urllib will actually connect to.
    decoded_hostname = unquote(hostname)
    try:
        ip = ipaddress.ip_address(decoded_hostname)
        return _check_ip_access(ip, allow_remote, port)
    except ValueError:
        resolved = _resolve_hostname(decoded_hostname)
        if resolved is None:
            if not allow_remote:
                return False
            # If DNS fails but the decoded hostname looks like a raw IP,
            # block it — DNS failure on a percent-encoded IP means urllib
            # may still connect to the decoded private IP.
            try:
                ip = ipaddress.ip_address(decoded_hostname)
                return _check_ip_access(ip, allow_remote, port)
            except ValueError:
                pass
        else:
            for addr in resolved:
                try:
                    ip = ipaddress.ip_address(addr)
                except ValueError:
                    return False
                if not _check_ip_access(ip, allow_remote, port):
                    return False
    return True


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that blocks all HTTP redirects.

    Prevents SSRF via redirect: an attacker-controlled URL may redirect
    to an internal service (e.g. http://169.254.169.254/latest/meta-data/).
    By blocking all redirects, we ensure the URL resolved at validation
    time is the same one actually fetched.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Block all redirects


def _extract_url_string(url: str | urllib.request.Request) -> str:
    """Extract the URL string from a Request object or return the string as-is.

    Callers may pass either a URL string or a urllib.request.Request object
    to safe_urlopen(). This helper normalises the input so that downstream
    validation functions (which expect str) always receive a string.

    Args:
        url: A URL string or a urllib.request.Request object.

    Returns:
        The URL string.

    Raises:
        ValueError: If url is a Request with no full_url attribute.
    """
    if isinstance(url, urllib.request.Request):
        return url.full_url
    return url


def safe_urlopen(
    url: str | urllib.request.Request,
    timeout: int = DEFAULT_URLOPEN_TIMEOUT,
    allow_remote: bool = False,
    **kwargs,
) -> http.client.HTTPResponse:
    """Open a URL with SSRF protections: validates URL, blocks redirects, sets timeout.

    This is the safe replacement for urllib.request.urlopen(). It:
    1. Validates the URL via is_safe_url()
    2. Builds an opener that blocks HTTP redirects (prevents redirect-based SSRF)
    3. Re-resolves the hostname immediately before the request (mitigates DNS rebinding TOCTOU)
    4. Enforces a timeout (prevents infinite hangs)
    5. Strips credentials from error messages

    Args:
        url: The URL to open — either a string or a urllib.request.Request
            object. Must pass is_safe_url() validation.
        timeout: Request timeout in seconds. Defaults to 30.
        allow_remote: If True, allow non-loopback addresses. Must match the
            policy used during URL construction (e.g. ExtractionEngine passes
            allow_remote=True because Ollama can run on remote hosts). Defaults
            to False for safety.
        **kwargs: Additional arguments passed to the opener's open() method.

    Returns:
        An http.client.HTTPResponse object (context manager).

    Raises:
        ValueError: If the URL fails is_safe_url() validation.
        urllib.error.HTTPError: If the server returns an HTTP error (credentials stripped).
        urllib.error.URLError: If the connection fails (credentials stripped).
    """
    url_str = _extract_url_string(url)

    if not is_safe_url(url_str, allow_remote=allow_remote):
        raise ValueError(
            f"llmem: url_validate: URL rejected: must be http(s) to a permitted "
            f"address — got {_strip_credentials(url_str)!r}"
        )

    # Re-resolve the hostname immediately before the request to mitigate
    # DNS rebinding TOCTOU: an attacker might change DNS after validation.
    # Structured to match is_safe_url()'s pattern: ValueError from
    # ipaddress.ip_address() is caught separately so it does not swallow
    # the deliberate ValueError raised for blocked re-resolved addresses.
    parsed = urlparse(url_str)
    hostname = parsed.hostname
    if hostname:
        # Use percent-decoded hostname for re-resolution (matches is_safe_url behavior)
        decoded_hostname = unquote(hostname)
        port = _get_effective_port(parsed)
        resolved = _resolve_hostname(decoded_hostname)
        if resolved:
            for addr in resolved:
                try:
                    ip = ipaddress.ip_address(addr)
                except ValueError:
                    # Unparseable address — treat as blocked.
                    raise ValueError(
                        f"llmem: url_validate: URL rejected after re-resolve: "
                        f"hostname {hostname!r} resolved to unparseable address "
                        f"{addr!r} — got {_strip_credentials(url_str)!r}"
                    )
                if not _check_ip_access(ip, allow_remote, port):
                    raise ValueError(
                        f"llmem: url_validate: URL rejected after re-resolve: "
                        f"hostname {hostname!r} resolved to blocked address {addr} "
                        f"— got {_strip_credentials(url_str)!r}"
                    )

    no_redirect_opener = urllib.request.build_opener(_NoRedirectHandler)

    try:
        return no_redirect_opener.open(url, timeout=timeout, **kwargs)
    except urllib.error.HTTPError as e:
        raise urllib.error.HTTPError(
            _strip_credentials(url_str), e.code, e.reason, e.headers, e.fp
        ) from e
    except urllib.error.URLError as e:
        safe_url = _strip_credentials(url_str)
        raise urllib.error.URLError(
            f"request to {safe_url!r} failed: {e.reason}"
        ) from e


def validate_url(url: str, allow_remote: bool = False) -> str:
    """Validate URL and return it, or raise ValueError."""
    if not is_safe_url(url, allow_remote=allow_remote):
        raise ValueError(
            f"URL rejected: must be http(s) to a permitted address — got {_strip_credentials(url)!r}"
        )
    return url


def validate_base_url(base_url: str, module: str = "url_validate") -> str:
    """Validate and normalize an Ollama base URL.

    Performs three checks shared by EmbeddingEngine, ExtractionEngine,
    and IntrospectionAnalyzer constructors:
    1. Strips trailing slash
    2. Validates http:// or https:// prefix
    3. Validates via is_safe_url with allow_remote=True

    Args:
        base_url: The Ollama base URL to validate.
        module: Module name for error messages (e.g. 'embed', 'extract', 'introspection').

    Returns:
        The stripped, validated URL string.

    Raises:
        ValueError: If the URL is not http(s) or fails is_safe_url().
    """
    base_url = base_url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(
            f"llmem: {module}: unsafe Ollama URL (must be http/https): {_strip_credentials(base_url)!r}"
        )
    if not is_safe_url(base_url, allow_remote=True):
        raise ValueError(
            f"llmem: {module}: unsafe Ollama URL (blocked address): {_strip_credentials(base_url)!r}"
        )
    return base_url
