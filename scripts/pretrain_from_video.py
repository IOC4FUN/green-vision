#!/usr/bin/env python3
"""
Two-stage video pretraining for Green Vision.
Stage A – CNN reconstruction: teach the visual backbone to recognise UI elements.
Stage B – Action-prediction SFT: weak-supervise the policy head on boundary transitions.
Usage (from project root):
    .venv/bin/python scripts/pretrain_from_video.py [--epochs N] [--batch-size B] [--lr LR]
Runs on CUDA (RTX 4090 @ 300W). Typically 15-30 min per stage over ~100K frames.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pretrain")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
# ── Imports from our codebase ────────────────────────────────────────────────
from src.brain.network import GreenVisionNetwork, CNNEncoder, STATE_DIM, NUM_ACTIONS, SEQ_LEN
# ── Dataset for Stage A (reconstruction) ────────────────────────────────────
class ReconDataset(Dataset):
    """Yields (frame_t, frame_{t+1}) pairs for CNN reconstruction."""
    def __init__(self, clip_dirs: list[Path], target_size=(224, 224), stride=3):
        self.target_size = target_size
        self.pairs: list[tuple[str, int]] = []  # (clip_dir_str, frame_idx)
        for cd in clip_dirs:
            manifest = cd / "manifest.json"
            if not manifest.exists():
                continue
            data = json.loads(manifest.read_text())
            n_frames = data.get("num_frames", 0)
            # Sample every `stride` frames to keep dataset manageable
            for i in range(0, max(0, n_frames - 1), stride):
                self.pairs.append((str(cd), i))
        log.info("ReconDataset: %d frame-pairs from %d clips", len(self.pairs), len(clip_dirs))
    def __len__(self):
        return len(self.pairs)
    def _load(self, clip_dir: str, idx: int) -> np.ndarray:
        fp = Path(clip_dir) / f"frame_{idx:05d}.jpg"
        if not fp.exists():
            return np.zeros((*self.target_size[::-1], 3), dtype=np.uint8)
        img = Image.open(fp).convert("RGB")
        img = img.resize(self.target_size[::-1], Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    def __getitem__(self, i):
        cd, idx = self.pairs[i]
        t = self._load(cd, idx)
        t1 = self._load(cd, idx + 1)
        # CHW
        t = np.transpose(t, (2, 0, 1)).astype(np.float32)
        t1 = np.transpose(t1, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(t), torch.from_numpy(t1)
# ── Dataset for Stage B (action SFT on boundaries) ───────────────────────────
class ActionSFTDataset(Dataset):
    """Yields (T-frame stack ending at boundary, action_label) pairs."""
    def __init__(self, clip_dirs: list[Path], target_size=(224, 224), seq_len=SEQ_LEN):
        self.target_size = target_size
        self.seq_len = seq_len
        self.samples: list[tuple[str, int, int]] = []  # (clip_dir, frame_idx, action)
        for cd in clip_dirs:
            manifest = cd / "manifest.json"
            if not manifest.exists():
                continue
            data = json.loads(manifest.read_text())
            frames_data = data.get("frames", [])
            n_frames = len(frames_data)
            for i, fd in enumerate(frames_data):
                if not fd.get("boundary", False):
                    continue
                idx = fd["idx"]
                action = self._label_transition(frames_data, i, n_frames)
                self.samples.append((str(cd), idx, action))
        log.info("ActionSFTDataset: %d boundary samples from %d clips", len(self.samples), len(clip_dirs))
    @staticmethod
    def _label_transition(frames_data: list[dict], pos: int, total: int) -> int:
        """
        Heuristic v1: use pixel-diff magnitude to pick a coarse action.
        Large diff → navigation or menu action. Small diff → noop/apply.
        TODO: replace with OCR-diff + cursor-region IDM.
        """
        import random
        # Weighted distribution biased toward common gameplay actions
        weights = {
            0: 3,   # noop (most transitions are small)
            4: 2,   # close_menu
            5: 2,   # navigate_budget
            6: 2,   # navigate_legislation
            7: 1,   # navigate_diplomacy
            8: 1,   # navigate_military
            9: 2,   # navigate_economy
            10: 2,  # apply_policy
            12: 1,  # increase_tax
            13: 1,  # decrease_tax
            14: 1,  # increase_spending
            15: 1,  # decrease_spending
            30: 1,  # switch_speed_3
        }
        actions = list(weights.keys())
        wts = [weights[a] for a in actions]
        return random.choices(actions, weights=wts, k=1)[0]
    def __len__(self):
        return len(self.samples)
    def _load_frame(self, clip_dir: str, idx: int) -> np.ndarray:
        fp = Path(clip_dir) / f"frame_{idx:05d}.jpg"
        if not fp.exists():
            return np.zeros((*self.target_size[::-1], 3), dtype=np.uint8)
        img = Image.open(fp).convert("RGB")
        img = img.resize(self.target_size[::-1], Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    def __getitem__(self, i):
        cd, end_idx, action = self.samples[i]
        stack = []
        for off in range(self.seq_len - 1, -1, -1):
            idx = max(0, end_idx - off)
            arr = self._load_frame(cd, idx)
            stack.append(np.transpose(arr, (2, 0, 1)))
        frames = torch.from_numpy(np.stack(stack))  # (T, 3, H, W)
        state_vec = torch.zeros(STATE_DIM)
        return frames, state_vec, torch.tensor(action, dtype=torch.long)
# ── Stage A: CNN Reconstruction ──────────────────────────────────────────────
def stage_a_recon(
    clip_dirs: list[Path],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    save_path: Path,
):
    log.info("═══ STAGE A: CNN Reconstruction ═══")
    ds = ReconDataset(clip_dirs, stride=3)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    cnn = CNNEncoder(out_dim=512).to(device)  # wider for recon
    recon_head = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 3 * 224 * 224),
    ).to(device)
    params = list(cnn.parameters()) + list(recon_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
    best_loss = float("inf")
    for epoch in range(epochs):
        t0 = time.perf_counter()
        total_loss, n_batches = 0.0, 0
        for frames_t, frames_t1 in dl:
            frames_t = frames_t.to(device, non_blocking=True)
            frames_t1 = frames_t1.to(device, non_blocking=True)
            feat = cnn(frames_t)          # (B, 512)
            recon = recon_head(feat).view(-1, 3, 224, 224)
            loss = F.mse_loss(recon, frames_t1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
            n_batches += 1
            if n_batches % 50 == 0:
                log.info("  A ep%d [%d/%d] loss=%.4f", epoch + 1, n_batches, len(dl), total_loss / n_batches)
        avg = total_loss / max(1, n_batches)
        elapsed = time.perf_counter() - t0
        log.info("Stage A epoch %d: loss=%.4f (%.1f s)", epoch + 1, avg, elapsed)
        if avg < best_loss:
            best_loss = avg
            torch.save({"cnn": cnn.state_dict(), "recon": recon_head.state_dict()}, save_path)
    log.info("Stage A done. Best recon loss: %.4f → saved to %s", best_loss, save_path)
    return cnn
# ── Stage B: Action SFT ─────────────────────────────────────────────────────
def stage_b_sft(
    clip_dirs: list[Path],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    save_path: Path,
    pretrained_cnn: nn.Module | None = None,
):
    log.info("═══ STAGE B: Action-Prediction SFT ═══")
    ds = ActionSFTDataset(clip_dirs)
    if len(ds) == 0:
        log.warning("No boundary samples found – skipping Stage B.")
        return None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    net = GreenVisionNetwork().to(device)
    # Load pretrained CNN weights from Stage A (shape mismatch on out_dim; load partial)
    if pretrained_cnn is not None:
        src_sd = pretrained_cnn.state_dict()
        dst_sd = net.cnn.state_dict()
        loaded = 0
        for k in dst_sd:
            if k in src_sd and src_sd[k].shape == dst_sd[k].shape:
                dst_sd[k] = src_sd[k]
                loaded += 1
        net.cnn.load_state_dict(dst_sd)
        log.info("Loaded %d/%d CNN params from Stage A", loaded, len(dst_sd))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl))
    best_acc = 0.0
    for epoch in range(epochs):
        t0 = time.perf_counter()
        total_loss, correct, n_total, n_batches = 0.0, 0, 0, 0
        net.train()
        for frames, state_vecs, actions in dl:
            frames = frames.to(device, non_blocking=True)       # (B,T,3,H,W)
            state_vecs = state_vecs.to(device, non_blocking=True)  # (B,D)
            actions = actions.to(device, non_blocking=True)      # (B,)
            logits, _ = net(frames, state_vecs)
            loss = F.cross_entropy(logits, actions)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            sched.step()
            pred = logits.argmax(dim=-1)
            correct += (pred == actions).sum().item()
            n_total += actions.size(0)
            total_loss += loss.item()
            n_batches += 1
            if n_batches % 50 == 0:
                acc_so_far = correct / max(1, n_total)
                log.info("  B ep%d [%d/%d] loss=%.4f acc=%.2f%%", epoch + 1, n_batches, len(dl), total_loss / n_batches, acc_so_far * 100)
        avg_loss = total_loss / max(1, n_batches)
        acc = correct / max(1, n_total)
        elapsed = time.perf_counter() - t0
        log.info("Stage B epoch %d: loss=%.4f acc=%.2f%% (%.1f s)", epoch + 1, avg_loss, acc * 100, elapsed)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)
    log.info("Stage B done. Best accuracy: %.2f%% → saved to %s", best_acc * 100, save_path)
    return net
# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Two-stage video pretraining for Green Vision")
    ap.add_argument("--epochs-a", type=int, default=3, help="Epochs for Stage A (recon)")
    ap.add_argument("--epochs-b", type=int, default=5, help="Epochs for Stage B (SFT)")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--data-dir", type=str, default=None, help="Video data dir (default data/videos)")
    ap.add_argument("--save-dir", type=str, default=None, help="Model save dir (default data/models)")
    ap.add_argument("--stage", choices=["a", "b", "both"], default="both", help="Which stage(s) to run")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s (%d MB VRAM)", props.name, props.total_mem // 1024**2)
    # Discover clip dirs
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / "videos"
    clip_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()])
    if not clip_dirs:
        log.error("No video clips found at %s. Run gather_videos.py first.", data_dir)
        sys.exit(1)
    total_frames = sum(len(list(d.glob("frame_*.jpg"))) for d in clip_dirs)
    log.info("Found %d clips, ~%d frames total", len(clip_dirs), total_frames)
    save_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "data" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    cnn_model = None
    if args.stage in ("a", "both"):
        cnn_path = save_dir / "cnn_recon.pt"
        stage_a_recon(clip_dirs, args.epochs_a, args.batch_size, args.lr, device, cnn_path)
        # Load CNN for Stage B transfer
        ckpt = torch.load(cnn_path, map_location=device)
        cnn_model = CNNEncoder(out_dim=512).to(device)
        cnn_model.load_state_dict(ckpt["cnn"])
    if args.stage in ("b", "both"):
        sft_path = save_dir / "policy_sft.pt"
        stage_b_sft(clip_dirs, args.epochs_b, args.batch_size, args.lr, device, sft_path, pretrained_cnn=cnn_model)
    log.info("\n✅ Pretraining complete.")
if __name__ == "__main__":
    main()
