#!/usr/bin/env bash
# Green Vision — formal launcher wrapper (venv, guards, optional game auto-start).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

WITH_GAME=0
DASHBOARD=1
PORT="${GREEN_VISION_PORT:-8090}"
for arg in "$@"; do
  case "$arg" in
    --with-game) WITH_GAME=1 ;;
    --no-dashboard) DASHBOARD=0 ;;
    --port=*) PORT="${arg#--port=}" ;;
  esac
done

[[ -f .venv/bin/activate ]] && source .venv/bin/activate
mkdir -p logs data/models data/knowledge

# Disk guard: abort if < 5 GB free.
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if (( FREE_GB < 5 )); then echo "ABORT: only ${FREE_GB}GB free (<5GB)"; exit 1; fi

# GPU check.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.used,power.draw --format=csv,noheader | head -1)"
else
  echo "WARNING: no nvidia-smi (pretrain requires CUDA)"
fi

# Optional game auto-launch (Proton/Wine, Steam AppID 940230).
if (( WITH_GAME )); then
  if command -v steam >/dev/null 2>&1; then
    echo "Launching Geoplanetical via Steam..."
    steam -app 940230 &
    sleep 8
  else
    echo "WARNING: steam CLI not found; skipping game launch"
  fi
fi

ARGS=(main.py)
(( DASHBOARD )) && ARGS+=(--web-dashboard --port "$PORT")
echo "Starting agent: python3 ${ARGS[*]}"
exec python3 "${ARGS[@]}"
