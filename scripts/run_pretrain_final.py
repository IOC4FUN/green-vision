#!/usr/bin/env python3
"""
PRETRAINING RUNNER — FINAL VERSION.
Uses SYSTEM PYTHON (torch verified).
ASCII ONLY output. No unicode crashes.
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, desc="Running"):
    print("\n[{}]".format(desc))
    cmd_str = ' '.join(str(c) for c in cmd)
    print("CMD: {}".format(cmd_str))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.stdout:
        for line in result.stdout.split(chr(10))[:50]:
            print(line)
    if result.stderr and 'ERROR' in result.stderr.upper():
        print("ERR: {}".format(result.stderr[:500]))
    return result.returncode == 0

def find_torch_python():
    candidates = [
        r'C:\Users\Benny\AppData\Local\Programs\Python\Python312\python.exe',
        'C:\Users\Benny\.venv-cuda\Scripts\python.exe',
        '.venv-cuda\Scripts\python.exe',
        'python.exe',
    ]
    for pth in candidates:
        full_path = Path(pth)
        if not full_path.is_absolute():
            full_path = Path.cwd() / full_path
        if full_path.exists() and full_path.is_file():
            try:
                result = subprocess.run([str(full_path), '-c', 'import torch; print(torch.__version__)'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return str(full_path).strip()
            except Exception as e:
                pass
    return None

def main():
    project_root = r'E:\Green Vision'
    script_path = Path(project_root) / 'scripts' / 'pretrain_from_video.py'
    python_exe = find_torch_python()
    data_dir = Path(project_root) / 'data' / 'videos'
    
    if not python_exe:
        print('FAIL: No Python with torch found')
        sys.exit(1)
    
    print('OK Using:', python_exe)
    
    # Verify torch/CUDA
    try:
        import torch
        print('OK PyTorch:', torch.__version__)
        print('OK CUDA:', torch.cuda.is_available())
        print('OK Devices:', torch.cuda.device_count())
        if not torch.cuda.is_available():
            print('WARN: CUDA not detected!')
            sys.exit(1)
    except ImportError as e:
        print('FAIL:', str(e))
        sys.exit(1)
    
    # Check frames (recursive search)
    frame_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.jpeg'):
                frame_files.append(Path(root) / f)
    print('OK Frames:', len(frame_files))
    
    if len(frame_files) == 0:
        print('WARN: No frames found')
    else:
        print('OK Data dir exists:', data_dir.exists())
    
    # Run training
    cmd = [
        python_exe,
        str(script_path),
        '--stage', 'a',
        '--epochs-a', '2',
        '--batch-size', '8'
    ]
    success = run_command(cmd, "Starting pretraining")
    
    if success:
        print("\n" + "="*70)
        print("PRETRAINING COMPLETE!")
        print("Checkpoint saved to: data\\models\\")
        print("="*70)
    else:
        print("\nFAIL: Pretraining failed or timed out")
    return success

if __name__ == '__main__':
    main()