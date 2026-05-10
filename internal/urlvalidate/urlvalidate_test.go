package urlvalidate

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestIsSafeURL_HTTPS(t *testing.T) {
	// HTTPS URLs to external hosts should pass
	if !IsSafeURL("https://example.com/path", true) {
		t.Error("expected HTTPS URL to external host to be safe")
	}
}

func TestIsSafeURL_FileSchemeRejected(t *testing.T) {
	if IsSafeURL("file:///etc/passwd", true) {
		t.Error("expected file:// URL to be rejected")
	}
}

func TestIsSafeURL_PrivateIPRejected(t *testing.T) {
	cases := []struct {
		name string
		url  string
	}{
		{"private_10", "http://10.0.0.1/path"},
		{"private_172", "http://172.16.0.1/path"},
		{"private_192", "http://192.168.1.1/path"},
		{"link_local", "http://169.254.169.254/latest/meta-data/"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if IsSafeURL(tc.url, true) {
				t.Errorf("expected private IP URL %q to be rejected", tc.url)
			}
		})
	}
}

func TestIsSafeURL_LinkLocalRejected(t *testing.T) {
	if IsSafeURL("http://169.254.169.254/latest/meta-data/", true) {
		t.Error("expected link-local IP to be rejected even with allowRemote=true")
	}
}

func TestIsSafeURL_LoopbackAllowedLocalhost(t *testing.T) {
	// localhost on default Ollama port should be allowed when allowRemote=false
	if !IsSafeURL("http://localhost:11434/api/embeddings", false) {
		t.Error("expected localhost on default Ollama port to be allowed")
	}
}

func TestIsSafeURL_LoopbackNonDefaultPortRejected(t *testing.T) {
	// localhost on a non-default port should be rejected when allowRemote=false
	if IsSafeURL("http://localhost:8080/api/test", false) {
		t.Error("expected localhost on non-default port to be rejected when allowRemote=false")
	}
}

func TestIsSafeURL_LoopbackNonDefaultPortAllowedWithRemote(t *testing.T) {
	// localhost on non-default port should be rejected even with allowRemote=true
	// because loopback is restricted to DefaultOllamaPort regardless
	if IsSafeURL("http://localhost:8080/api/test", true) {
		t.Error("expected localhost on non-default port to be rejected even with allowRemote=true")
	}
}

func TestIsSafeURL_PercentEncodedSSRFBypass(t *testing.T) {
	// Percent-encoded 127.0.0.1 should be caught
	if IsSafeURL("http://%31%32%37%2e%30%2e%30%2e%31:11434/api/embeddings", false) {
		t.Error("expected percent-encoded loopback IP to be caught")
	}
}

func TestIsSafeURL_EmptyHostname(t *testing.T) {
	if IsSafeURL("http://", false) {
		t.Error("expected empty hostname to be rejected")
	}
}

func TestIsSafeURL_CredentialsRejected(t *testing.T) {
	if IsSafeURL("http://user:pass@example.com/path", true) {
		t.Error("expected URL with credentials to be rejected")
	}
}

func TestIsSafeURL_InvalidScheme(t *testing.T) {
	cases := []string{
		"ftp://example.com/path",
		"data:text/plain,hello",
		"javascript:alert(1)",
	}
	for _, u := range cases {
		if IsSafeURL(u, true) {
			t.Errorf("expected URL %q to be rejected (invalid scheme)", u)
		}
	}
}

func TestIsRemoteAllowed(t *testing.T) {
	tests := []struct {
		name     string
		url      string
		expected bool
	}{
		{"loopback_ip", "http://127.0.0.1:11434/", false},
		{"localhost", "http://localhost:11434/", false},
		{"localhost_localdomain", "http://localhost.localdomain:11434/", false},
		{"localhost6", "http://localhost6:11434/", false},
		{"invalid", "not-a-url", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := IsRemoteAllowed(tc.url)
			if got != tc.expected {
				t.Errorf("IsRemoteAllowed(%q) = %v, want %v", tc.url, got, tc.expected)
			}
		})
	}
}

func TestIsRemoteAllowed_PublicIP(t *testing.T) {
	// Test with a literal public IP (no DNS needed)
	if !IsRemoteAllowed("http://8.8.8.8/path") {
		t.Error("expected 8.8.8.8 to be remote (not loopback, not private)")
	}
}

func TestIsRemoteAllowed_PrivateIP(t *testing.T) {
	if IsRemoteAllowed("http://192.168.1.1/path") {
		t.Error("expected private IP to not be remote")
	}
}

func TestValidateBaseURL_StripsTrailingSlash(t *testing.T) {
	result, err := ValidateBaseURL("http://localhost:11434/", "embed")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.HasSuffix(result, "/") {
		t.Errorf("expected trailing slash stripped, got %q", result)
	}
}

func TestValidateBaseURL_NonHTTPRejected(t *testing.T) {
	cases := []string{
		"ftp://example.com",
		"file:///tmp/data",
		"data:text/plain,hello",
	}
	for _, u := range cases {
		_, err := ValidateBaseURL(u, "embed")
		if err == nil {
			t.Errorf("expected error for non-http URL %q", u)
		}
	}
}

func TestValidateBaseURL_BlockPrivateIP(t *testing.T) {
	_, err := ValidateBaseURL("http://192.168.1.1:11434", "embed")
	if err == nil {
		t.Error("expected private IP to be rejected")
	}
}

func TestSafeURLOpen_BlocksRedirects(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/redirect" {
			w.Header().Set("Location", "/target")
			w.WriteHeader(http.StatusFound)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	ctx := context.Background()
	_, err := SafeURLOpen(ctx, server.URL+"/redirect", 5*time.Second, true)
	if err == nil {
		t.Error("expected redirect to be blocked")
	}
	if !strings.Contains(err.Error(), "redirect") {
		t.Errorf("expected redirect-related error, got: %v", err)
	}
}

func TestSafeURLOpen_InvalidURL(t *testing.T) {
	ctx := context.Background()
	_, err := SafeURLOpen(ctx, "http://10.0.0.1/path", 5*time.Second, true)
	if err == nil {
		t.Error("expected unsafe URL to be rejected")
	}
}

func TestSafeURLOpen_ContextCancellation(t *testing.T) {
	// Verify that context cancellation is propagated through SafeURLOpen.
	// We use a server that binds to loopback:11434 (Ollama default port)
	// so IsSafeURL allows the request, then cancel context to observe timeout.
	//
	// Since httptest.NewServer uses a random port (not 11434), we cannot
	// directly test SafeURLOpen with a slow server through SSRF protection.
	// Instead, test that context cancellation returns an error promptly,
	// which validates that SafeURLOpen respects ctx.Done().
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := SafeURLOpen(ctx, "https://example.com/test", 5*time.Second, true)
	if err == nil {
		t.Error("expected error from cancelled context")
	}
}

func TestCheckIPAccess(t *testing.T) {
	ip := net.ParseIP("127.0.0.1")
	if !checkIPAccess(ip, false, DefaultOllamaPort) {
		t.Error("expected loopback on default port to be allowed")
	}
	if checkIPAccess(ip, false, 8080) {
		t.Error("expected loopback on non-default port to be rejected (allowRemote=false)")
	}

	// Private IPs should be blocked even with allowRemote=true
	privateIP := net.ParseIP("192.168.1.1")
	if checkIPAccess(privateIP, true, 80) {
		t.Error("expected private IP to be blocked even with allowRemote=true")
	}

	// Public IPs should be allowed with allowRemote=true
	publicIP := net.ParseIP("8.8.8.8")
	if !checkIPAccess(publicIP, true, 80) {
		t.Error("expected public IP to be allowed with allowRemote=true")
	}

	// Public IPs should be blocked with allowRemote=false
	if checkIPAccess(publicIP, false, 80) {
		t.Error("expected public IP to be blocked with allowRemote=false")
	}
}

func TestIsSafeURL_AllowRemotePublicIP(t *testing.T) {
	tests := []struct {
		name        string
		url         string
		allowRemote bool
		want        bool
	}{
		{
			name:        "public_ip_allow_remote",
			url:         "http://8.8.8.8/path",
			allowRemote: true,
			want:        true,
		},
		{
			name:        "public_ip_deny_no_remote",
			url:         "http://8.8.8.8/path",
			allowRemote: false,
			want:        false,
		},
		{
			name:        "private_ip_deny_even_with_remote",
			url:         "http://192.168.1.1/path",
			allowRemote: true,
			want:        false,
		},
		{
			name:        "loopback_deny_even_with_remote_non_ollama_port",
			url:         "http://127.0.0.1:8080/path",
			allowRemote: true,
			want:        false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := IsSafeURL(tc.url, tc.allowRemote)
			if got != tc.want {
				t.Errorf("IsSafeURL(%q, %v) = %v, want %v", tc.url, tc.allowRemote, got, tc.want)
			}
		})
	}
}

func TestStripCredentials(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"with_credentials", "http://user:pass@example.com/path", "http://example.com/path"},
		{"without_credentials", "http://example.com/path", "http://example.com/path"},
		{"with_port", "http://user:pass@example.com:8080/path", "http://example.com:8080/path"},
		{"with_query_string", "http://user:pass@example.com/path?q=1&r=2", "http://example.com/path?q=1&r=2"},
		{"with_fragment", "http://user:pass@example.com/path#section", "http://example.com/path#section"},
		{"with_query_and_fragment", "http://user:pass@example.com/path?q=1#section", "http://example.com/path?q=1#section"},
		{"query_and_fragment_no_credentials", "http://example.com/path?key=val#frag", "http://example.com/path?key=val#frag"},
		{"path_only_no_credentials", "http://example.com/path", "http://example.com/path"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := stripCredentials(tc.input)
			if result != tc.expected {
				t.Errorf("stripCredentials(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestIsRemoteAllowed_PercentEncodedSSRF(t *testing.T) {
	// Percent-encoded loopback IP should be caught after decoding,
	// consistent with IsSafeURL's SSRF defense.
	if IsRemoteAllowed("http://%31%32%37%2e%30%2e%30%2e%31:11434/") {
		t.Error("expected percent-encoded loopback IP to be caught by IsRemoteAllowed")
	}
}

func TestIsRemoteAllowed_PercentEncodedHostname(t *testing.T) {
	// Percent-encoded "localhost" should be caught after decoding.
	if IsRemoteAllowed("http://%6c%6f%63%61%6c%68%6f%73%74:11434/") {
		t.Error("expected percent-encoded 'localhost' to be caught by IsRemoteAllowed")
	}
}