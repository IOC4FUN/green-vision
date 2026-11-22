#!/usr/bin/env python3
"""
Auto-stage B launcher — watches for Stage A completion, then starts Stage B.
Runs in background until Stage A checkpoint exists with sufficient size.
Usage: C:\Users\Benny\AppData\Local\Programs\Python\Python312\python.exe scripts/auto_stage_b.py
"""
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(r"E:\Green Vision")
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "models" / "cnn_recon.pt"
MIN_CHECKPOINT_SIZE = 50 * 1024 * 1024  # 50 MB minimum (Stage A should be ~162MB+)

def check_stage_a_complete():
    """Check if Stage A CNN reconstruction is complete."""
    if not CHECKPOINT_PATH.exists():
        return False
    size = CHECKPOINT_PATH.stat().st_size
    print(f"[CHECK] cnn_recon.pt size: {size:,} bytes ({size/1024:.1f} KB)")
    return size >= MIN_CHECKPOINT_SIZE

def run_stage_b():
    """Launch Stage B action SFT training."""
    cmd = [
        r'C:\Users\Benny\AppData\Local\Programs\Python\Python312\python.exe',
        str(PROJECT_ROOT / 'scripts' / 'pretrain_from_video.py'),
        '--stage', 'b',
        '--epochs-b', '5',
        '--batch-size', '8',
        '--lr', '1e-4'
    ]
    print("="*60)
    print("STAGE A COMPLETE — LAUNCHING STAGE B")
    print("="*60)
    print("Running: pretrain_from_video.py --stage b")
    os.execv(cmd[0], cmd)

def main():
    print("Auto-stage B launcher started.")
    print("Waiting for Stage A checkpoint (min 50 MB)...")
    
    while True:
        if check_stage_a_complete():
            try:
                run_stage_b()
            except Exception as e:
                print(f"ERROR launching Stage B: {e}")
            break
        time.sleep(5)

if __name__ == "__main__":
    main()
