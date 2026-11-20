#!/usr/bin/env bash
# gpu_handoff.sh — pause llama-server during training, resume after.
# Usage: ./scripts/gpu_handoff.sh <command...>
#   e.g. ./scripts/gpu_handoff.sh .venv/bin/python scripts/pretrain_from_video.py --stage both
set -euo pipefail

LLAMA_PIDS=$(pgrep -f 'llama-server' || true)
PAUSED=()

cleanup() {
  if [[ ${#PAUSED[@]} -gt 0 ]]; then
    echo "[gpu-handoff] resuming llama-server PIDs: ${PAUSED[*]}" >&2
    for pid in "${PAUSED[@]}"; do
      kill -CONT "$pid" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT INT TERM

if [[ -n "$LLAMA_PIDS" ]]; then
  echo "[gpu-handoff] pausing llama-server PIDs: $LLAMA_PIDS" >&2
  for pid in $LLAMA_PIDS; do
    kill -STOP "$pid"
    PAUSED+=("$pid")
  done
  sleep 1  # let the kernel reclaim VRAM pages
else
  echo "[gpu-handoff] no llama-server found, nothing to pause" >&2
fi

echo "[gpu-handoff] GPU state:" >&2
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >&2

echo "[gpu-handoff] running: $*" >&2
exec "$@"
