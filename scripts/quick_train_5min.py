#!/usr/bin/env python3
"""
Quick 5-minute training session with LLM swap.
Swap heavy→light, train Stage A (CNN recon) for 1 epoch, restore heavy.
Usage:
    .venv/bin/python scripts/quick_train_5min.py
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Step 1: Swap to light model ─────────────────────────────────────────────
print("=" * 60)
print("[SWAP] Heavy → Light")
print("=" * 60)
t_swap_start = time.perf_counter()
from src.utils.llm_manager import LLMManager
mgr = LLMManager.from_config(PROJECT_ROOT / "config" / "settings.yaml")
ok = mgr.swap_to_light()
swap_time = time.perf_counter() - t_swap_start
if not ok:
    print("❌ Swap failed. Aborting.")
    sys.exit(1)
print(f"✅ Swapped to light in {swap_time:.1f}s\n")

# ── Step 2: Run training (Stage A only, 1 epoch) ────────────────────────────
print("=" * 60)
print("[TRAIN] Stage A – CNN Reconstruction (1 epoch)")
print("=" * 60)
import torch
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quick_train")

from scripts.pretrain_from_video import stage_a_recon, ReconDataset

device = "cuda"
props = torch.cuda.get_device_properties(0)
free_mb = props.total_mem // 1024**2
log.info("GPU: %s (%d MB total)", props.name, free_mb)

# Discover clips
data_dir = PROJECT_ROOT / "data" / "videos"
clip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()])
if not clip_dirs:
    log.error("No video clips found.")
    sys.exit(1)
total_frames = sum(len(list(d.glob("frame_*.jpg"))) for d in clip_dirs)
log.info("Found %d clips, ~%d frames total", len(clip_dirs), total_frames)

save_dir = PROJECT_ROOT / "data" / "models"
save_dir.mkdir(parents=True, exist_ok=True)
cnn_path = save_dir / "cnn_recon_quick.pt"

t_train_start = time.perf_counter()
cnn = stage_a_recon(
    clip_dirs=clip_dirs,
    epochs=1,
    batch_size=16,   # larger batch since we have full VRAM now
    lr=1e-4,
    device=device,
    save_path=cnn_path,
)
train_time = time.perf_counter() - t_train_start
print(f"\n✅ Training done in {train_time:.1f}s")

# ── Step 3: Restore heavy model ─────────────────────────────────────────────
print("=" * 60)
print("[SWAP] Light → Heavy (restore)")
print("=" * 60)
t_restore_start = time.perf_counter()
ok = mgr.restore_heavy()
restore_time = time.perf_counter() - t_restore_start
if not ok:
    print("❌ Restore failed!")
else:
    print(f"✅ Restored heavy in {restore_time:.1f}s")

# ── Summary ─────────────────────────────────────────────────────────────────
total_time = swap_time + train_time + restore_time
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Swap to light : {swap_time:.1f}s")
print(f"  Training      : {train_time:.1f}s ({total_frames} frames, batch=16)")
print(f"  Restore heavy : {restore_time:.1f}s")
print(f"  Total         : {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Model saved   : {cnn_path}")
print("=" * 60)
sys.exit(0 if ok else 1)
