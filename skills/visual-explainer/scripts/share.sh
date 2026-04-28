#!/bin/bash

# Share Visual Explainer HTML via Vercel
# Usage: ./share.sh <html-file>
# Returns: Live URL instantly (no auth required)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

HTML_FILE="${1}"

if [ -z "$HTML_FILE" ]; then
    echo -e "${RED}Error: Please provide an HTML file to share${NC}" >&2
    echo "Usage: $0 <html-file>" >&2
    exit 1
fi

if [ ! -f "$HTML_FILE" ]; then
    echo -e "${RED}Error: File not found: $HTML_FILE${NC}" >&2
    exit 1
fi

# Validate file is a plausible HTML file (not scripts, binaries, etc.)
MIME_TYPE=$(file -b --mime-type "$HTML_FILE" 2>/dev/null || echo "")
if [ -n "$MIME_TYPE" ]; then
    case "$MIME_TYPE" in
        text/html|text/xml|application/xml|text/plain) ;;
        *)
            echo -e "${RED}Error: File is not HTML (detected: $MIME_TYPE)${NC}" >&2
            exit 1
            ;;
    esac
fi

# Basic content check: must contain an HTML tag
if ! grep -qiE '<html|<!doctype html' "$HTML_FILE"; then
    echo -e "${RED}Error: File does not appear to be valid HTML${NC}" >&2
    exit 1
fi

# Scan for embedded secrets/credentials before deploying to a public URL.
# This prevents accidental exposure of API keys, tokens, passwords, and
# private keys in files published to the internet.
SECRET_PATTERNS=(
    # Private keys (PEM, OpenSSH, etc.)
    '-----BEGIN [A-Z ]*PRIVATE KEY-----'
    # Generic API key patterns (<service>_API_KEY, <service>_TOKEN, etc.)
    '[A-Za-z0-9_]*API_KEY[A-Za-z0-9_]*\s*[:=]\s*['\''"][A-Za-z0-9_\-]{16,}'
    '[A-Za-z0-9_]*SECRET[A-Za-z0-9_]*\s*[:=]\s*['\''"][A-Za-z0-9_\-]{16,}'
    '[A-Za-z0-9_]*TOKEN[A-Za-z0-9_]*\s*[:=]\s*['\''"][A-Za-z0-9_\-]{16,}'
    '[A-Za-z0-9_]*PASSWORD[A-Za-z0-9_]*\s*[:=]\s*['\''"][A-Za-z0-9_\-]{8,}'
    # Common cloud provider key prefixes
    'AKIA[0-9A-Z]{16}'              # AWS access key IDs
    'sk-[a-zA-Z0-9]{32,}'           # OpenAI-style API keys
    'ghp_[A-Za-z0-9_]{36,}'        # GitHub personal access tokens
    'gho_[A-Za-z0-9_]{36,}'        # GitHub OAuth tokens
    # Hardcoded password in HTML attributes/inputs
    'type\s*=\s*['\''"]password['\''"][^>]*value\s*=\s*['\''"][^'\''"]{8,}'
)

SECRETS_FOUND=()
for pattern in "${SECRET_PATTERNS[@]}"; do
    if grep -qE -- "$pattern" "$HTML_FILE"; then
        SECRETS_FOUND+=("$pattern")
    fi
done

if [ ${#SECRETS_FOUND[@]} -gt 0 ]; then
    echo -e "${RED}Error: Potential secrets or credentials detected in HTML file${NC}" >&2
    echo -e "${RED}This file would be deployed to a PUBLIC URL accessible by anyone.${NC}" >&2
    echo -e "${RED}Refusing to deploy. Remove secrets before sharing.${NC}" >&2
    echo "" >&2
    echo -e "${RED}Matched patterns:${NC}" >&2
    for p in "${SECRETS_FOUND[@]}"; do
        echo -e "${RED}  - $p${NC}" >&2
    done
    exit 1
fi

# Find vercel-deploy skill
VERCEL_SCRIPT=""
for dir in ~/.agents/skills/vercel-deploy/scripts /mnt/skills/user/vercel-deploy/scripts; do
    if [ -f "$dir/deploy.sh" ]; then
        VERCEL_SCRIPT="$dir/deploy.sh"
        break
    fi
done

if [ -z "$VERCEL_SCRIPT" ]; then
    echo -e "${RED}Error: vercel-deploy skill not found${NC}" >&2
    echo "Install it with: npm install -g vercel-deploy" >&2
    exit 1
fi

# Create temp directory with index.html
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Copy file as index.html (Vercel serves index.html at root)
cp "$HTML_FILE" "$TEMP_DIR/index.html"

echo -e "${CYAN}Sharing $(basename "$HTML_FILE")...${NC}" >&2

# Deploy via vercel-deploy skill
# Temporarily disable errexit to capture deployment errors
set +e
RESULT=$(bash "$VERCEL_SCRIPT" "$TEMP_DIR" 2>&1)
DEPLOY_EXIT=$?
set -e

if [ $DEPLOY_EXIT -ne 0 ]; then
    echo -e "${RED}Error: Deployment failed${NC}" >&2
    echo "$RESULT" >&2
    exit 1
fi

# Extract preview URL
PREVIEW_URL=$(echo "$RESULT" | grep -oE 'https://[^"]+\.vercel\.app' | head -1)
CLAIM_URL=$(echo "$RESULT" | grep -oE 'https://vercel\.com/claim-deployment[^"]+' | head -1)

if [ -z "$PREVIEW_URL" ]; then
    echo -e "${RED}Error: Deployment failed${NC}" >&2
    echo "$RESULT" >&2
    exit 1
fi

echo "" >&2
echo -e "${GREEN}✓ Shared successfully!${NC}" >&2
echo "" >&2
echo -e "${GREEN}Live URL:  ${PREVIEW_URL}${NC}" >&2
echo -e "${CYAN}Claim URL: ${CLAIM_URL}${NC}" >&2
echo "" >&2

# Output JSON for programmatic use (extract from vercel-deploy output)
echo "$RESULT" | grep -E '^\{' | head -1
