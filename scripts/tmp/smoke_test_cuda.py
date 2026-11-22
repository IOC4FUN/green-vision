#!/usr/bin/env python3
r"""
Quick CUDA + model smoke test for Green Vision pretraining.
Verifies: torch/CUDA, CNNEncoder forward/backward, ReconDataset loads real frames.
Run with system Python (has torch+cu121).
"""
import sys, time
from pathlib import Path

PROJECT_ROOT = Path(r"E:\Green Vision")
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("GREEN VISION — SMOKE TEST")
print("=" * 60)

# 1. Torch + CUDA
import torch
print(f"[1/4] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE - ABORT"
device = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
print(f"      GPU: {props.name} ({props.total_memory // 1024**2} MB VRAM)")

# 2. Model architecture (forward + backward)
from src.brain.network import CNNEncoder, STATE_DIM, NUM_ACTIONS, SEQ_LEN
import torch.nn as nn
import torch.nn.functional as F

print(f"[2/4] Building CNNEncoder(out_dim=512) + recon head...")
cnn = CNNEncoder(out_dim=512).to(device)
recon_head = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 3 * 224 * 224),
).to(device)
params = list(cnn.parameters()) + list(recon_head.parameters())
opt = torch.optim.AdamW(params, lr=1e-4)

# Dummy batch
B = 2
x = torch.randn(B, 3, 224, 224, device=device)
target = torch.randn(B, 3, 224, 224, device=device)

t0 = time.perf_counter()
feat = cnn(x)
recon = recon_head(feat).view(-1, 3, 224, 224)
loss = F.mse_loss(recon, target)
loss.backward()
opt.step()
fwd_bwd_ms = (time.perf_counter() - t0) * 1000
print(f"      Forward+backward OK | loss={loss.item():.4f} | {fwd_bwd_ms:.1f} ms/batch")
print(f"      Params: {sum(p.numel() for p in params):,}")

# 3. Dataset — load real frames from one clip
from scripts.pretrain_from_video import ReconDataset
clip_dir = PROJECT_ROOT / "data" / "videos" / "Geoplanetical_S1Ep005"
assert (clip_dir / "manifest.json").exists(), f"No manifest at {clip_dir}"
ds = ReconDataset([clip_dir], stride=10)
print(f"[3/4] ReconDataset: {len(ds)} pairs from {clip_dir.name}")
if len(ds) > 0:
    t0 = time.perf_counter()
    frame_t, frame_t1 = ds[0]
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"      Loaded sample 0: shape={tuple(frame_t.shape)} dtype={frame_t.dtype} | {load_ms:.1f} ms")
    assert frame_t.shape == (3, 224, 224), f"Bad shape: {frame_t.shape}"
else:
    print("      WARNING: no samples!")

# 4. DataLoader with CUDA pinning
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
it = iter(dl)
t0 = time.perf_counter()
batches_loaded = 0
for frames_t, frames_t1 in dl:
    frames_t = frames_t.to(device, non_blocking=True)
    frames_t1 = frames_t1.to(device, non_blocking=True)
    batches_loaded += 1
    if batches_loaded >= 3:
        break
dl_ms = (time.perf_counter() - t0) * 1000
print(f"[4/4] DataLoader: loaded {batches_loaded} batches in {dl_ms:.1f} ms ({dl_ms/batches_loaded:.1f} ms/batch)")

print()
print("=" * 60)
print("SMOKE TEST PASSED — all systems go for Stage A")
print("=" * 60)
