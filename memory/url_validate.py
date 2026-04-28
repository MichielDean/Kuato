"""URL validation for urllib calls — prevent SSRF and file-scheme attacks."""

import ipaddress
import logging
import socket
import urllib.request
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler

log = logging.getLogger(__name__)

OLLAMA_DEFAULT_PORT = 11434


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_loopback:
        return False
    return bool(ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast)


def _resolve_hostname(hostname: str) -> list[str] | None:
    try:
        addrs = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        return [a[4][0] for a in addrs]
    except Exception:
        return None


def _get_effective_port(parsed) -> int:
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    return 80


def is_safe_url(url: str, allow_remote: bool = False) -> bool:
    """Validate that a URL is safe to pass to urllib.

    Checks:
    - Scheme must be http or https (blocks file://, ftp://, data://, etc.)
    - Must have a hostname
    - If allow_remote is False (default), only loopback addresses on the
      Ollama default port are allowed. Private/link-local/reserved/multicast
      IPs are blocked regardless.
    - If allow_remote is True, any reachable hostname is allowed (still
      blocks non-http schemes and obviously invalid hostnames).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = parsed.hostname
    if not hostname:
        return False
    try:
        ip = ipaddress.ip_address(hostname)
        if _ip_is_blocked(ip):
            return False
        if not allow_remote and ip.is_loopback:
            if _get_effective_port(parsed) != OLLAMA_DEFAULT_PORT:
                return False
    except ValueError:
        resolved = _resolve_hostname(hostname)
        if resolved is None:
            if not allow_remote:
                return False
        else:
            for addr in resolved:
                try:
                    ip = ipaddress.ip_address(addr)
                except ValueError:
                    return False
                if _ip_is_blocked(ip):
                    return False
                if not allow_remote and ip.is_loopback:
                    if _get_effective_port(parsed) != OLLAMA_DEFAULT_PORT:
                        return False
    return True


def validate_url(url: str, allow_remote: bool = False) -> str:
    """Validate URL and return it, or raise ValueError."""
    if not is_safe_url(url, allow_remote=allow_remote):
        raise ValueError(f"URL rejected: must be http(s) to a permitted address")
    return url


def sanitize_url_for_log(url: str) -> str:
    """Strip credentials from a URL for safe inclusion in logs and error messages.

    Removes userinfo (user:password@) from URLs like
    'https://user:pass@host/path' → 'https://host/path'.
    Returns the scheme and host/path portion only.
    """
    parsed = urlparse(url)
    # Rebuild without netloc credentials
    safe_netloc = parsed.hostname or ""
    if parsed.port:
        safe_netloc = f"{safe_netloc}:{parsed.port}"
    return parsed._replace(netloc=safe_netloc).geturl()


class SafeRedirectHandler(HTTPRedirectHandler):
    """HTTP redirect handler that validates each redirect target URL against
    is_safe_url before following it.

    Prevents SSRF via HTTP redirects: an attacker could point a safe-looking URL
    (e.g. https://attacker.com/) to redirect to a private IP (e.g. http://169.254.169.254/).
    Without this handler, urllib would follow the redirect without re-checking the target.
    """

    def __init__(self, allow_remote: bool = False):
        self.allow_remote = allow_remote
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not is_safe_url(newurl, allow_remote=self.allow_remote):
            log.warning(
                "url_validate: blocked SSRF redirect from %s to %s",
                sanitize_url_for_log(req.full_url),
                sanitize_url_for_log(newurl),
            )
            return None  # abort the redirect
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# Module-level safe opener — uses SafeRedirectHandler to validate
# redirect targets and prevent SSRF via HTTP redirects.
_safe_opener = urllib.request.build_opener(SafeRedirectHandler(allow_remote=True))


def safe_urlopen(request, timeout=None):
    """Open a URL using the SSRF-safe opener that validates redirect targets.

    This is a thin wrapper around the module-level opener's .open() method.
    Use this instead of urllib.request.urlopen to get SSRF redirect protection.

    Args:
        request: A urllib.request.Request object or URL string.
        timeout: Optional timeout in seconds.

    Returns:
        An http.client.HTTPResponse object (context manager).
    """
    if timeout is not None:
        return _safe_opener.open(request, timeout=timeout)
    return _safe_opener.open(request)
