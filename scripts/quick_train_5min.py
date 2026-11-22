#!/usr/bin/env python3
"""
Quick 5-minute training session.
Runs Stage A (CNN reconstruction) for 1 epoch on available video data.
Usage:
    .venv/Scripts/python scripts/quick_train_5min.py
"""
import sys
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Run training (Stage A only, 1 epoch) ────────────────────────────────────
print("=" * 60)
print("[TRAIN] Stage A – CNN Reconstruction (1 epoch)")
print("=" * 60)
import torch
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quick_train")
from scripts.pretrain_from_video import stage_a_recon, ReconDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    free_mb = props.total_memory // 1024**2
    log.info("GPU: %s (%d MB total)", props.name, free_mb)
else:
    log.warning("CUDA not available — running on CPU (will be slow)")

# Discover clips
data_dir = PROJECT_ROOT / "data" / "videos"
clip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()]) if data_dir.exists() else []
if not clip_dirs:
    log.error("No video clips found in %s. Run gather_videos.py first.", data_dir)
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
    batch_size=16,
    lr=1e-4,
    device=device,
    save_path=cnn_path,
)
train_time = time.perf_counter() - t_train_start
print(f"\n✅ Training done in {train_time:.1f}s")
# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Training      : {train_time:.1f}s ({total_frames} frames, batch=16)")
print(f"  Total         : {train_time:.1f}s ({train_time/60:.1f} min)")
print(f"  Model saved   : {cnn_path}")
print("=" * 60)
sys.exit(0)
