#!/usr/bin/env python3
r"""
Auto-stage B launcher — watches for Stage A completion, then starts Stage B.
Runs in background until Stage A checkpoint exists with sufficient size.
Usage: <venv-python> scripts/auto_stage_b.py
MAX VRAM UTILIZATION MODE (RTX 4090 @ 24GB)
"""
import os
import sys
import time
from pathlib import Path
PROJECT_ROOT = Path(r"E:\Green Vision")
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "models" / "cnn_recon.pt"
MIN_CHECKPOINT_SIZE = 50 * 1024 * 1024  # 50 MB minimum
# MAX VRAM settings for RTX 4090 (24 GB) - push batch size to limit
def check_stage_a_complete():
    if not CHECKPOINT_PATH.exists():
        return False
    size = CHECKPOINT_PATH.stat().st_size
    print(f"[CHECK] cnn_recon.pt size: {size:,} bytes ({size/1024:.1f} KB)")
    return size >= MIN_CHECKPOINT_SIZE
def run_stage_b_max_vram():
    # Prefer the project venv; fall back to system Python.
    candidates = [
        str(PROJECT_ROOT / '.venv-cuda' / 'Scripts' / 'python.exe'),
        r'C:\Users\Benny\AppData\Local\Programs\Python\Python312\python.exe',
        sys.executable,
    ]
    python_exe = next((p for p in candidates if Path(p).exists()), sys.executable)
    script_path = str(PROJECT_ROOT / 'scripts' / 'pretrain_from_video.py')
    cmd = [
        python_exe,
        script_path,
        '--stage', 'b',
        '--epochs-b', '50',
        '--batch-size', '64',
        '--lr', '1e-4'
    ]
    print("="*70)
    print("STAGE A COMPLETE — LAUNCHING STAGE B (MAX VRAM MODE)")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)  # replaces process; works on Windows via CreateProcess
def main():
    print("Auto-stage B launcher started.")
    print("Waiting for Stage A checkpoint (min 50 MB)...")
    while True:
        if check_stage_a_complete():
            try:
                run_stage_b_max_vram()
            except Exception as e:
                print(f"ERROR launching Stage B: {e}")
            break
        time.sleep(5)
if __name__ == "__main__":
    main()