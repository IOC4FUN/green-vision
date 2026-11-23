#!/usr/bin/env python3
# GREEN VISION - AI TEST SUITE v4
# Run with the CUDA venv python:
#   E:\Green Vision\.venv-cuda\Scripts\python.exe E:/Green Vision/scripts/test_ai.py
# ASCII ONLY output.
import sys
from pathlib import Path

project_root = r'E:\Green Vision'
script_dir = Path(project_root) / 'scripts'
data_dir = Path(project_root) / 'data'
models_dir = data_dir / 'models'
videos_dir = data_dir / 'videos'

def check_torch():
    """Verify torch is importable in the CURRENT interpreter."""
    try:
        import torch
        print('OK PyTorch:', torch.__version__)
        cuda_ok = torch.cuda.is_available()
        print('OK CUDA Available:', cuda_ok)
        if cuda_ok:
            print('OK GPU Devices:', torch.cuda.device_count())
            print('OK GPU Name:', torch.cuda.get_device_name(0))
            print('OK CUDA Version (runtime):', torch.version.cuda)
        else:
            print('WARN: CUDA not detected!')
            print('HINT: Install CUDA torch:')
            print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124')
        return cuda_ok
    except ImportError as e:
        print('FAIL: Cannot import torch:', str(e)[:200])
        return False

def test_environment():
    print('=' * 60)
    print('GREEN VISION - ENVIRONMENT TEST')
    print('=' * 60)
    ok = check_torch()
    print('OK Project Root Exists:', Path(project_root).exists())
    print('OK Scripts Dir Exists:', script_dir.exists())
    print('OK Data Dir Exists:', data_dir.exists())
    frames = list(videos_dir.rglob('*.jpg')) + list(videos_dir.rglob('*.jpeg'))
    print('OK Frames Found (recursive):', len(frames))
    if len(frames) > 0:
        print('OK Sample frame path:', frames[0])
    # Check for saved model checkpoint
    ckpt = models_dir / 'cnn_recon.pt'
    if ckpt.exists():
        print('OK Model checkpoint exists:', ckpt, '({} bytes)'.format(ckpt.stat().st_size))
    else:
        print('WARN: No model checkpoint at', ckpt)
    return ok and len(frames) > 0

def test_curiosity_module():
    print('\n' + '=' * 60)
    print('GREEN VISION - CURIOSITY MODULE TEST')
    print('=' * 60)
    try:
        sys.path.insert(0, str(script_dir))
        import torch
        from curiosity import CuriosityModule, CuriosityScheduler
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cm = CuriosityModule(hidden_dim=512, device=device)
        print('OK Module Loaded on', device)
        obs = torch.randn(4, 32).to(device)
        pred = torch.randn(4, 32).to(device)
        reward = cm(obs, pred)
        print('OK Forward Works, Reward Shape:', tuple(reward.shape))
        sched = CuriosityScheduler(initial_weight=0.5, decay_rate=0.999)
        w = sched.get_weight(step=100)
        print('OK Scheduler Weight at step 100:', round(w, 4))
        return True
    except Exception as e:
        print('FAIL:', type(e).__name__, '-', str(e)[:300])
        return False

def test_curriculum_manager():
    print('\n' + '=' * 60)
    print('GREEN VISION - CURRICULUM MANAGER TEST')
    print('=' * 60)
    try:
        sys.path.insert(0, str(script_dir))
        from curriculum import CurriculumManager
        cm = CurriculumManager(num_stages=5)
        print('OK Module Loaded')
        mult = cm.get_difficulty_multiplier()
        print('OK Difficulty Multiplier (stage 0):', mult)
        ready = cm.evaluate_performance(0.75)
        print('OK Performance Eval Ready:', ready)
        if ready:
            print('OK Advanced to Stage:', cm.current_stage)
            print('OK New Multiplier:', cm.get_difficulty_multiplier())
        import numpy as np
        state_vec = np.random.randn(64).astype(np.float32)
        cw = cm.get_state_complexity_weight(state_vec)
        print('OK State Complexity Weight:', round(cw, 4))
        return True
    except Exception as e:
        print('FAIL:', type(e).__name__, '-', str(e)[:300])
        return False

def test_inference():
    print('\n' + '=' * 60)
    print('GREEN VISION - INFERENCE TEST')
    print('=' * 60)
    frames = list(videos_dir.rglob('*.jpg'))[:3]
    if len(frames) == 0:
        print('SKIP: No frames found for inference test')
        return None
    print('OK Testing on', len(frames), 'frames')
    try:
        sys.path.insert(0, str(project_root))
        import torch
        from src.brain.network import CNNEncoder
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cnn = CNNEncoder(out_dim=512).to(device)
        print('OK CNNEncoder Created on', device)
        # Forward pass with a real frame
        from PIL import Image
        import numpy as np
        img = Image.open(str(frames[0])).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = cnn(tensor)
        print('OK Forward Pass OK, Feature Shape:', tuple(feat.shape))
        # Check saved checkpoint
        ckpt_path = models_dir / 'cnn_recon.pt'
        if ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
            keys = list(sd.keys())[:5]
            print('OK Checkpoint loaded, top keys:', keys)
        return True
    except Exception as e:
        print('FAIL:', type(e).__name__, '-', str(e)[:300])
        return False

def main():
    tests = [
        ('Environment', test_environment),
        ('Curiosity Module', test_curiosity_module),
        ('Curriculum Manager', test_curriculum_manager),
        ('Inference', test_inference),
    ]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            r = fn()
            if r is None:
                print('\n[SKIP]', name); skipped += 1
            elif r:
                print('\n[PASS]', name); passed += 1
            else:
                print('\n[FAIL]', name); failed += 1
        except Exception as e:
            print('\n[ERROR]', name, ':', str(e)[:200]); failed += 1
    print('\n' + '=' * 60)
    print('TEST SUMMARY')
    print('=' * 60)
    print('Passed: {}  Failed: {}  Skipped: {}'.format(passed, failed, skipped))
    if failed == 0 and passed > 0:
        print('\nALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!')
    return failed == 0

if __name__ == '__main__':
    sys.exit(0 if main() else 1)