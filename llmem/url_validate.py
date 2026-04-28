"""URL validation for urllib calls — prevent SSRF and file-scheme attacks."""

import ipaddress
import socket
from urllib.parse import urlparse

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
      Ollama default port are allowed.
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
        raise ValueError(
            f"URL rejected: must be http(s) to a permitted address — got {url!r}"
        )
    return url
