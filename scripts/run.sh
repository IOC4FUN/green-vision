#!/usr/bin/env bash
# Green Vision – Launch script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

# Ensure log dir
mkdir -p logs data/models data/knowledge

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader 2>/dev/null | head -1)"
else
    echo "WARNING: nvidia-smi not found – will run on CPU (slow)"
fi

# Run the agent
exec python3 main.py "$@"
