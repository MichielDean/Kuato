// Package urlvalidate provides SSRF-protected URL validation and safe HTTP access.
// It blocks private/link-local IPs, percent-encoded SSRF bypasses, and redirect-based
// attacks, while allowing loopback on the Ollama default port for local development.
package urlvalidate

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// DefaultOllamaPort is the default port for Ollama API.
const DefaultOllamaPort = 11434

// DefaultURLTimeout is the default timeout for HTTP requests.
const DefaultURLTimeout = 30 * time.Second

// fmtErr wraps an error with the "llmem: urlvalidate:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: urlvalidate: "+format, args...)
}

// isPrivateIP checks if an IP is a private, link-local, reserved, or multicast address.
// Loopback addresses return false — they are handled separately.
func isPrivateIP(ip net.IP) bool {
	if ip.IsLoopback() {
		return false
	}
	return ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() ||
		ip.IsUnspecified() || ip.IsMulticast()
}

// checkIPAccess checks whether a resolved IP is permitted for access.
//
// When allowRemote is false (the default), only loopback addresses on the
// Ollama default port are permitted. All other IPs — including public IPs
// that are not blocked by isPrivateIP — are rejected.
//
// When allowRemote is true, any IP that is not private/link-local/reserved/multicast
// is permitted. Loopback addresses are still restricted to the Ollama default port
// regardless of allowRemote.
func checkIPAccess(ip net.IP, allowRemote bool, port int) bool {
	if isPrivateIP(ip) {
		return false
	}
	if ip.IsLoopback() {
		if port != DefaultOllamaPort {
			return false
		}
	} else if !allowRemote {
		return false
	}
	return true
}

// resolveHostname resolves a hostname to a list of IP addresses.
// Returns nil if resolution fails.
func resolveHostname(hostname string) []net.IP {
	addrs, err := net.LookupIP(hostname)
	if err != nil {
		return nil
	}
	return addrs
}

// getEffectivePort returns the port for a parsed URL, using scheme defaults if missing.
func getEffectivePort(u *url.URL) int {
	if u.Port() != "" {
		port, err := net.LookupPort("tcp", u.Port())
		if err == nil {
			return port
		}
	}
	if u.Scheme == "https" {
		return 443
	}
	return 80
}

// IsSafeURL validates that a URL is safe to access.
//
// It checks: scheme is http or https, hostname is present, no embedded credentials,
// resolved IPs are not blocked. Loopback is allowed on the Ollama default port
// when allowRemote is false. Percent-encoded hostnames are decoded before IP checks
// to prevent SSRF bypass.
//
// Never panics. Returns false for malformed URLs.
func IsSafeURL(urlStr string, allowRemote bool) bool {
	u, err := url.Parse(urlStr)
	if err != nil {
		return false
	}

	if u.Scheme != "http" && u.Scheme != "https" {
		return false
	}

	hostname := u.Hostname()
	if hostname == "" {
		return false
	}

	// Reject URLs with embedded credentials
	if u.User != nil {
		return false
	}

	port := getEffectivePort(u)

	// Percent-decode the hostname before IP checks to prevent SSRF bypass
	// via encoded IP addresses (e.g. %31%32%37%2e%30%2e%30%2e%31 → 127.0.0.1).
	decodedHostname := decodeHostname(hostname)

	// Try parsing as an IP address first
	if ip := net.ParseIP(decodedHostname); ip != nil {
		return checkIPAccess(ip, allowRemote, port)
	}

	// Resolve hostname and check all IPs
	addrs := resolveHostname(decodedHostname)
	if addrs == nil {
		if !allowRemote {
			return false
		}
		// DNS failed but allowRemote — check decoded hostname as raw IP again
		if ip := net.ParseIP(decodedHostname); ip != nil {
			return checkIPAccess(ip, allowRemote, port)
		}
		// DNS failed and not an IP — allow if allowRemote (could be a domain)
		return true
	}

	for _, addr := range addrs {
		if !checkIPAccess(addr, allowRemote, port) {
			return false
		}
	}
	return true
}

// decodeHostname percent-decodes a hostname for SSRF bypass prevention.
// Handles percent-encoded octets like %31%32%37 → "127".
func decodeHostname(hostname string) string {
	// Use url.PathUnescape which decodes percent-encoded sequences
	decoded, err := url.PathUnescape(hostname)
	if err != nil {
		// If decoding fails, return the hostname as-is
		return hostname
	}
	return decoded
}

// noRedirectTransport is an unexported http.RoundTripper that blocks redirects
// for SSRF prevention. An attacker-controlled URL may redirect to an internal
// service (e.g. http://169.254.169.254/latest/meta-data/).
type noRedirectTransport struct {
	base http.RoundTripper
}

func (t noRedirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err != nil {
		return nil, err
	}

	// Check if the response is a redirect
	if resp.StatusCode >= 300 && resp.StatusCode < 400 {
		location := resp.Header.Get("Location")
		safeURL := stripCredentials(req.URL.String())
		slog.Warn("llmem: urlvalidate: blocked SSRF redirect", "from", safeURL, "to", location)
		// Close the response body to avoid resource leaks
		resp.Body.Close()
		return nil, fmtErr("blocked redirect from %s to %s", safeURL, location)
	}
	return resp, nil
}

// SafeURLOpen opens a URL with SSRF protections.
//
// It validates the URL via IsSafeURL, blocks redirects, re-resolves the
// hostname before the request (DNS rebinding protection), and enforces a
// timeout. Returns *http.Response on success — the caller MUST close the
// response body.
//
// If the URL fails IsSafeURL, returns an error with the domain prefix.
// Context cancellation is respected.
func SafeURLOpen(ctx context.Context, urlStr string, timeout time.Duration, allowRemote bool) (*http.Response, error) {
	if !IsSafeURL(urlStr, allowRemote) {
		return nil, fmtErr("URL rejected: must be http(s) to a permitted address — got %s", stripCredentials(urlStr))
	}

	if timeout <= 0 {
		timeout = DefaultURLTimeout
	}

	u, err := url.Parse(urlStr)
	if err != nil {
		return nil, fmtErr("parse URL: %w", err)
	}

	// Re-resolve the hostname immediately before the request to mitigate
	// DNS rebinding TOCTOU: an attacker might change DNS after validation.
	hostname := u.Hostname()
	if hostname != "" {
		decodedHostname := decodeHostname(hostname)
		port := getEffectivePort(u)
		addrs := resolveHostname(decodedHostname)
		if addrs != nil {
			for _, addr := range addrs {
				if !checkIPAccess(addr, allowRemote, port) {
					return nil, fmtErr("URL rejected after re-resolve: hostname %q resolved to blocked address %s — got %s", hostname, addr, stripCredentials(urlStr))
				}
			}
		}
	}

	client := &http.Client{
		Transport: noRedirectTransport{base: http.DefaultTransport},
		Timeout:   timeout,
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, urlStr, nil)
	if err != nil {
		return nil, fmtErr("create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmtErr("request to %s failed: %w", stripCredentials(urlStr), err)
	}

	return resp, nil
}

// ValidateBaseURL validates and normalizes an Ollama base URL.
//
// It strips the trailing slash, validates the http/https scheme, and validates
// the URL via IsSafeURL with allowRemote=true. Returns the validated URL string.
// Error messages include the module parameter for context.
func ValidateBaseURL(baseURL, module string) (string, error) {
	baseURL = strings.TrimRight(baseURL, "/")
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		return "", fmtErr("%s: unsafe Ollama URL (must be http/https): %s", module, stripCredentials(baseURL))
	}
	// Validate with allowRemote=true since Ollama can run on remote hosts
	if !IsSafeURL(baseURL, true) {
		return "", fmtErr("%s: unsafe Ollama URL (blocked address): %s", module, stripCredentials(baseURL))
	}
	return baseURL, nil
}

// IsRemoteAllowed infers allowRemote from a URL.
// Returns false for loopback addresses, private/link-local IPs, and local hostnames.
// Returns true for public IPs and hostnames that resolve to public IPs.
// Returns false (fail-closed) for hostnames that fail DNS resolution.
func IsRemoteAllowed(urlStr string) bool {
	u, err := url.Parse(urlStr)
	if err != nil {
		return false
	}
	hostname := u.Hostname()
	if hostname == "" {
		return false
	}

	// Check if it's an IP address
	if ip := net.ParseIP(hostname); ip != nil {
		return !ip.IsLoopback() && !isPrivateIP(ip)
	}

	// Check for local hostnames
	localHostnames := map[string]bool{
		"localhost":             true,
		"localhost.localdomain": true,
		"localhost6":            true,
	}
	if localHostnames[strings.ToLower(hostname)] {
		return false
	}

	// Resolve the hostname and check all resulting IPs.
	// If DNS resolution fails, fail-closed (return false).
	addrs := resolveHostname(hostname)
	if addrs == nil {
		return false
	}
	for _, addr := range addrs {
		if addr.IsLoopback() || isPrivateIP(addr) {
			return false
		}
	}
	return true
}

// stripCredentials removes userinfo (credentials) from a URL for safe error display.
// Turns 'http://user:pass@host/path' into 'http://host/path'.
func stripCredentials(urlStr string) string {
	u, err := url.Parse(urlStr)
	if err != nil {
		return urlStr // Return as-is if parsing fails
	}
	// Reconstruct with hostname + port but no credentials
	return u.Scheme + "://" + u.Host + u.Path
	// Note: u.Host already excludes userinfo
}