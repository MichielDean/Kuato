package store

import (
	"encoding/binary"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"
)

// nowUTC returns the current UTC time formatted as ISO 8601 (RFC 3339).
// This matches Python's datetime.now(timezone.utc).isoformat().
func nowUTC() string {
	return time.Now().UTC().Format(time.RFC3339)
}

// placeholders returns a comma-separated string of n parameter placeholders.
// Example: placeholders(3) -> "?,?,?"
func placeholders(n int) string {
	if n <= 0 {
		return ""
	}
	parts := make([]string, n)
	for i := range parts {
		parts[i] = "?"
	}
	return strings.Join(parts, ",")
}

// sanitizeFTSQuery sanitizes a query string for FTS5 MATCH queries.
// It strips unsafe FTS5 operators, splits on whitespace, and joins with OR.
// Returns a quoted empty string if no safe tokens remain.
func sanitizeFTSQuery(query string) string {
	ftsOperators := map[string]bool{
		"AND": true, "OR": true, "NOT": true, "NEAR": true,
	}
	tokens := strings.Fields(query)
	var safeTokens []string
	for _, t := range tokens {
		if ftsOperators[strings.ToUpper(t)] {
			continue
		}
		// Remove all non-word characters
		clean := regexp.MustCompile(`[^\w]+`).ReplaceAllString(t, " ")
		parts := strings.Fields(clean)
		safeTokens = append(safeTokens, parts...)
	}
	if len(safeTokens) == 0 {
		return `""`
	}
	return strings.Join(safeTokens, " OR ")
}

// escapeLike escapes special LIKE pattern characters.
func escapeLike(query string) string {
	s := strings.ReplaceAll(query, `\`, `\\`)
	s = strings.ReplaceAll(s, `%`, `\%`)
	s = strings.ReplaceAll(s, `_`, `\_`)
	return s
}

// bytesToVec decodes a packed float32 byte slice into a []float32.
// Matches Python's struct.unpack(f"{dim}f", data).
func bytesToVec(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	dim := len(data) / 4
	result := make([]float32, dim)
	for i := 0; i < dim; i++ {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return result
}

// vecToBytes encodes a []float32 into packed little-endian bytes.
// Matches Python's struct.pack(f"{dim}f", *vec).
func vecToBytes(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// cosineSimilarity computes the cosine similarity between two float32 vectors.
// Returns 0.0 when either vector has zero magnitude.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}
	var dot, magA, magB float64
	for i := range a {
		fa := float64(a[i])
		fb := float64(b[i])
		dot += fa * fb
		magA += fa * fa
		magB += fb * fb
	}
	if magA == 0 || magB == 0 {
		return 0.0
	}
	return dot / (math.Sqrt(magA) * math.Sqrt(magB))
}

// isValidTypeName checks whether a type name matches ^[a-z][a-z0-9_]*$ and is <=64 chars.
func isValidTypeName(name string) bool {
	if len(name) == 0 || len(name) > 64 {
		return false
	}
	valid, _ := regexp.MatchString(`^[a-z][a-z0-9_]*$`, name)
	return valid
}

// fmtErr wraps an error with the "llmem: store:" domain prefix.
func fmtErr(format string, args ...any) error {
	return fmt.Errorf("llmem: store: "+format, args...)
}