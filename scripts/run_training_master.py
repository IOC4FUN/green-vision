#!/usr/bin/env python3
"""
MASTER PRETRAINING LAUNCHER — AUTONOMOUS EXECUTION.
Single entry point. Auto-detects Python. ASCII ONLY output.
Production ready for Green Vision Stage A pretraining.
"""
import os
import subprocess
import sys
from pathlib import Path

def find_torch_python():
    """Auto-detect Python with torch installed."""
    candidates = [
        r'C:\Users\Benny\AppData\Local\Programs\Python\Python312\python.exe',
        r'E:\Green Vision\.venv-cuda\Scripts\python.exe',
        r'.venv-cuda\Scripts\python.exe',
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
            except Exception:
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
    
    print('='*70)
    print('GREEN VISION - STAGE A PRETRAINING (AUTONOMOUS)')
    print('='*70)
    print('Using:', python_exe)
    
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA:', torch.cuda.is_available())
    print('Devices:', torch.cuda.device_count())
    
    if not torch.cuda.is_available():
        print('WARN: CUDA not detected!')
        sys.exit(1)
    
    frame_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.jpeg'):
                frame_files.append(Path(root) / f)
    print('Frames:', len(frame_files))
    
    if len(frame_files) == 0:
        print('WARN: No frames found')
    else:
        print('Data dir exists:', data_dir.exists())
    
    cmd = [
        python_exe,
        str(script_path),
        '--stage', 'a',
        '--epochs-a', '2',
        '--batch-size', '8'
    ]
    
    print('\nStarting pretraining...')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    
    if result.stdout:
        for line in result.stdout.split(chr(10))[:50]:
            print(line)
    
    if result.stderr and 'ERROR' in result.stderr.upper():
        print('ERR:', result.stderr[:500])
    
    return result.returncode == 0

if __name__ == '__main__':
    main()