#!/usr/bin/env bash
#
# LLMem setup script — clone, install, init, verify.
#
# Usage:
#   ./setup.sh
#   ./setup.sh --plugin opencode
#   ./setup.sh --plugin copilot
#   ./setup.sh --plugin none
#   ./setup.sh --extras vec,local --plugin both
#   ./setup.sh --repo /path/to/existing/LLMem
#
# Idempotent — safe to re-run if something fails.
# Provider detection is automatic: llmem init detects Ollama, OpenAI,
# Anthropic, or local providers and writes config accordingly.

set -euo pipefail

REPO_URL="https://github.com/MichielDean/LLMem.git"
REPO_DIR="LLMem"

EXTRAS=""
PLUGIN="both"
REPO_ALREADY_CLONED=0

usage() {
    cat <<'EOF'
LLMem setup script — install everything from source.

Usage:
  setup.sh [OPTIONS]

Options:
  --extras EXTRAS      Comma-separated pip extras (vec, local, dev). Default: none
  --plugin PLUGIN      Which plugin to install: opencode, copilot, both, none.
                       Default: both
  --repo PATH          Use an existing repo directory instead of cloning
  -h, --help           Show this help

Examples:
  setup.sh                                    # CLI + both plugins (default)
  setup.sh --plugin opencode                  # CLI + OpenCode plugin only
  setup.sh --plugin copilot                   # CLI + Copilot CLI plugin only
  setup.sh --plugin none                       # CLI only, no plugins
  setup.sh --extras vec,local --plugin opencode
  setup.sh --repo ./LLMem                     # use existing clone

After setup, configure an embedding provider:
  - Ollama:     install from https://ollama.com, then: ollama pull nomic-embed-text
  - OpenAI:     set OPENAI_API_KEY environment variable
  - Anthropic:  set ANTHROPIC_API_KEY environment variable
  - Local:      rerun with --extras local (uses sentence-transformers, no server needed)
  - None:       llmem works in FTS5-only mode without any provider
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --extras)
            EXTRAS="$2"
            shift 2
            ;;
        --plugin)
            PLUGIN="$2"
            shift 2
            ;;
        --repo)
            REPO_DIR="$2"
            REPO_ALREADY_CLONED=1
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

case "$PLUGIN" in
    opencode|copilot|both|none) ;;
    *)
        echo "ERROR: --plugin must be opencode, copilot, both, or none (got: $PLUGIN)" >&2
        exit 1
        ;;
esac

INSTALL_OPENCODE=0
INSTALL_COPILOT=0
if [[ "$PLUGIN" == "opencode" || "$PLUGIN" == "both" ]]; then
    INSTALL_OPENCODE=1
fi
if [[ "$PLUGIN" == "copilot" || "$PLUGIN" == "both" ]]; then
    INSTALL_COPILOT=1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() {
    echo -e "\n${GREEN}==> $1${NC}"
}

warn() {
    echo -e "${YELLOW}WARNING: $1${NC}" >&2
}

fail() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        fail "$1 is required but not found on PATH. Install it first."
    fi
}

# ── Prerequisites ──────────────────────────────────────────────

step "Checking prerequisites"
check_cmd python3
check_cmd git

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: ${PYTHON_VERSION}"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo "Python version: OK (>= 3.11)"
else
    fail "Python 3.11+ required, found ${PYTHON_VERSION}"
fi

NEED_NPM=0
if [[ $INSTALL_OPENCODE -eq 1 || $INSTALL_COPILOT -eq 1 ]]; then
    NEED_NPM=1
fi

if [[ $NEED_NPM -eq 1 ]]; then
    check_cmd node
    check_cmd npm
    echo "Node.js: $(node --version)"
fi

# ── Clone ──────────────────────────────────────────────────────

if [[ $REPO_ALREADY_CLONED -eq 0 ]]; then
    step "Cloning LLMem repository"
    if [[ -d "$REPO_DIR" ]]; then
        echo "Directory ${REPO_DIR} already exists — skipping clone."
    else
        git clone "$REPO_URL" "$REPO_DIR"
    fi
fi

cd "$REPO_DIR"

# ── Install Python package ────────────────────────────────────

step "Installing llmem Python package"

PIP_EXTRAS=""
if [[ -n "$EXTRAS" ]]; then
    PIP_EXTRAS="[${EXTRAS}]"
fi

if pip install ".${PIP_EXTRAS}" 2>&1 | tail -5; then
    echo "Python package installed."
else
    fail "pip install failed. Try: pip install --break-system-packages .${PIP_EXTRAS}"
fi

# ── Initialize ─────────────────────────────────────────────────

step "Initializing llmem (config + database)"
if llmem init --non-interactive 2>&1; then
    echo "Initialization complete."
else
    # llmem init may fail if config already exists without --force
    # Try with --force as fallback
    warn "llmem init failed — retrying with --force"
    if llmem init --non-interactive --force 2>&1; then
        echo "Initialization complete (forced)."
    else
        fail "llmem init failed. Check the errors above."
    fi
fi

# ── Install plugins ────────────────────────────────────────────

if [[ $INSTALL_OPENCODE -eq 1 ]]; then
    step "Installing OpenCode plugin (opencode-llmem)"
    if [[ -d "opencode-llmem" ]]; then
        (cd opencode-llmem && npm install 2>&1 | tail -3)
        echo "OpenCode plugin installed to ~/.agents/plugins/llmem/"
    else
        warn "opencode-llmem directory not found — skipping."
    fi
fi

if [[ $INSTALL_COPILOT -eq 1 ]]; then
    step "Installing Copilot CLI plugin (copilot-llmem)"
    if command -v copilot &>/dev/null; then
        echo "Installing via copilot plugin install..."
        if copilot plugin install MichielDean/LLMem:copilot-llmem 2>&1; then
            echo "Copilot CLI plugin installed via copilot plugin install."
        else
            warn "copilot plugin install failed — falling back to npm install"
            if [[ -d "copilot-llmem" ]]; then
                (cd copilot-llmem && npm install 2>&1 | tail -3)
                echo "Copilot CLI plugin installed via npm."
            else
                warn "copilot-llmem directory not found — skipping."
            fi
        fi
    else
        echo "copilot CLI not found — installing via npm."
        if [[ -d "copilot-llmem" ]]; then
            (cd copilot-llmem && npm install 2>&1 | tail -3)
            echo "Copilot CLI plugin installed via npm."
        else
            warn "copilot-llmem directory not found — skipping."
        fi
    fi
fi

# Root npm install copies all skills to ~/.agents/skills/
if [[ $NEED_NPM -eq 1 ]]; then
    step "Installing skill files"
    if [[ -f "package.json" ]]; then
        npm install 2>&1 | tail -3
        echo "Skills installed to ~/.agents/skills/"
    fi
fi

# ── Verify ─────────────────────────────────────────────────────

step "Verifying installation"

echo ""
echo "CLI:"
if llmem --help >/dev/null 2>&1; then
    echo "  llmem CLI: OK"
else
    warn "llmem CLI not found on PATH"
fi

echo ""
echo "Database:"
if llmem stats 2>&1; then
    echo "  Database: OK"
else
    warn "llmem stats failed — database may not be initialized"
fi

echo ""
echo "Python library:"
if python3 -c "from llmem import MemoryStore; print('  MemoryStore: OK')" 2>/dev/null; then
    :
else
    warn "Cannot import llmem Python library"
fi

if [[ $INSTALL_OPENCODE -eq 1 ]]; then
    echo ""
    echo "OpenCode plugin:"
    if [[ -d "$HOME/.agents/plugins/llmem" ]]; then
        echo "  ~/.agents/plugins/llmem: OK"
    else
        warn "  ~/.agents/plugins/llmem: NOT FOUND"
    fi
fi

if [[ $INSTALL_COPILOT -eq 1 ]]; then
    echo ""
    echo "Copilot CLI plugin:"
    if [[ -d "$HOME/.agents/skills/llmem" ]]; then
        echo "  ~/.agents/skills/llmem: OK"
    else
        warn "  ~/.agents/skills/llmem: NOT FOUND"
    fi
fi

if [[ $NEED_NPM -eq 1 ]]; then
    echo ""
    echo "Skills:"
    SKILLS_DIR="$HOME/.agents/skills"
    for skill in llmem introspection git-sync task-intake test-and-verify branch-strategy critical-code-reviewer pre-pr-review visual-explainer introspection-review-tracker; do
        if [[ -d "${SKILLS_DIR}/${skill}" ]]; then
            echo "  ${skill}: OK"
        else
            warn "  ${skill}: NOT FOUND in ${SKILLS_DIR}"
        fi
    done
fi

# ── Summary ─────────────────────────────────────────────────────

step "Setup complete"
echo ""
echo "Quick start:"
echo "  llmem add --type fact --content 'Hello, LLMem'"
echo "  llmem search 'hello'"
echo "  llmem stats"
echo ""
echo "Configuration: ~/.config/llmem/config.yaml"
echo "Database:      ~/.config/llmem/memory.db"
if [[ $INSTALL_OPENCODE -eq 1 ]]; then
    echo "OpenCode:      ~/.agents/plugins/llmem/"
fi
if [[ $INSTALL_COPILOT -eq 1 ]]; then
    echo "Copilot:      ~/.agents/skills/llmem/ (skills) + hooks.json"
fi
echo ""
echo "Embedding providers (pick one):"
echo "  - Ollama:     install from https://ollama.com, then: ollama pull nomic-embed-text"
echo "  - OpenAI:     set OPENAI_API_KEY"
echo "  - Anthropic:  set ANTHROPIC_API_KEY"
echo "  - Local:      pip install \".[local]\" (sentence-transformers, no server)"
echo "  - None:       FTS5-only mode (works without any provider)"