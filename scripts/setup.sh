#!/usr/bin/env bash
# Green Vision – One-time setup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "═══ Green Vision Setup ═══"

# 1. Virtual environment
if [[ ! -f .venv/bin/activate ]]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Python dependencies
echo "[2/4] Installing Python packages..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. System tools (OCR, input)
echo "[3/4] Checking system dependencies..."
MISSING=()
for cmd in tesseract xdotool wmctrl; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING+=("$cmd")
    fi
done
if (( ${#MISSING[@]} > 0 )); then
    echo "  Missing: ${MISSING[*]}"
    echo "  Install with: sudo apt install ${MISSING[*]}"
    # Try to install if we have sudo
    if command -v sudo &>/dev/null; then
        read -p "  Install now? [y/N] " REPLY
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            sudo apt-get update -qq && sudo apt-get install -y -qq "${MISSING[@]}"
        fi
    fi
else
    echo "  All system tools present ✓"
fi

# 4. Directories
echo "[4/4] Creating data directories..."
mkdir -p logs data/models data/knowledge data/replays

chmod +x scripts/run.sh main.py

echo ""
echo "Setup complete! Run with:"
echo "  ./scripts/run.sh              # full autonomous mode"
echo "  ./scripts/run.sh --dashboard  # monitor only"
echo "  ./scripts/run.sh --no-launch  # attach to running game"
