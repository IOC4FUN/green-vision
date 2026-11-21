#!/usr/bin/env python3
"""
Autonomous training loop for Green Vision.
Cycles through: gather more data → pretrain → PPO fine-tune → evaluate → repeat.
Designed to run unattended (nohup / tmux) and self-manage disk/GPU resources.
Usage:
    .venv/bin/python scripts/train_autonomous.py [--cycles N] [--max-hours H]
Each cycle:
  1. Check GPU + disk; abort gracefully if resources insufficient.
  2. Gather new videos (unlimited mode, stops when no new results).
  3. Run Stage A+B pretraining on all accumulated frames.
  4. Load SFT weights into full network; run short PPO warm-up on synthetic env.
  5. Save checkpoint + update .convolog.md with metrics.
  6. Sleep briefly, then next cycle.
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_auto")
# ── Resource guards ──────────────────────────────────────────────────────────
def check_resources(min_free_gb=20.0, min_gpu_mem_mb=8192) -> bool:
    """
    Return True if enough disk + GPU memory for another training cycle.
    Thresholds are generous so the loop can run full-batch pretraining
    (Stage A recon at 64×224² needs ~6-7 GB VRAM; Stage B with transformer
    + optimizer states peaks around 10-12 GB on the 4090's 24 GB).
    Disk guard at 20 GB leaves room for unlimited video gathering + checkpoints.
    """
    usage = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = usage.free / 1024**3
    if free_gb < min_free_gb:
        log.error("Disk low: %.1f GB free (< %.1f GB). Aborting cycle.", free_gb, min_free_gb)
        return False
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,power.draw,power.limit", "--format=csv,noheader,nounits"], text=True).strip()
        parts = [p.strip() for p in out.split(",")]
        used_mb, total_mb = int(parts[0]), int(parts[1])
        power_draw = float(parts[2]) if len(parts) > 2 else 0.0
        power_limit = float(parts[3]) if len(parts) > 3 else 0.0
        free_mb = total_mb - used_mb
        if free_mb < min_gpu_mem_mb:
            log.error("GPU VRAM low: %d MB free (< %d MB). Aborting cycle.", free_mb, min_gpu_mem_mb)
            return False
        # Warn (don't block) if GPU is thermally throttling or over power budget
        if power_limit > 0 and power_draw > power_limit * 0.95:
            log.warning("GPU near power limit: %.0f/%.0f W – may throttle during training", power_draw, power_limit)
        log.info("Resources OK: disk=%.1f GB free, GPU=%d/%d MB free (%.0f/%.0f W)", free_gb, free_mb, total_mb, power_draw, power_limit)
    except Exception as e:
        log.warning("Could not check GPU (continuing anyway): %s", e)
    return True


# ── Step 1: Gather videos ────────────────────────────────────────────────────
def gather_videos_step() -> dict:
    """Run the video gatherer; returns summary."""
    log.info("── Step 1/5: Gathering new videos ──")
    script = PROJECT_ROOT / "scripts" / "gather_videos.py"
    venv_py = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    cmd = [venv_py, str(script), "--max-videos", "0"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # up to 2h
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        log.warning("Gatherer exited non-zero (%d): %s", proc.returncode, proc.stderr[-300:])
    # Count manifests before/after would be ideal; for now just report
    clip_dirs = list((PROJECT_ROOT / "data" / "videos").glob("*/manifest.json"))
    total_frames = sum(len(list(d.parent.glob("frame_*.jpg"))) for d in clip_dirs)
    result = {"clips": len(clip_dirs), "frames": total_frames, "time_s": round(elapsed, 1)}
    log.info("Gather done: %d clips, ~%d frames (%.0f s)", result["clips"], result["frames"], elapsed)
    return result
# ── Step 2: Pretrain (Stage A + B) ───────────────────────────────────────────
def pretrain_step(epochs_a=2, epochs_b=4, batch_size=64) -> dict:
    log.info("── Step 2/5: Pretraining (A+B) ──")
    script = PROJECT_ROOT / "scripts" / "pretrain_from_video.py"
    venv_py = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    cmd = [venv_py, str(script),
           "--stage", "both",
           "--epochs-a", str(epochs_a),
           "--epochs-b", str(epochs_b),
           "--batch-size", str(batch_size),
           ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)  # up to 4h
    elapsed = time.perf_counter() - t0
    ok = proc.returncode == 0
    if not ok:
        log.error("Pretrain failed: %s", proc.stderr[-500:])
    else:
        log.info("Pretrain done in %.0f s", elapsed)
    return {"ok": ok, "time_s": round(elapsed, 1)}
# ── Step 3: PPO warm-up on synthetic env ─────────────────────────────────────
def ppo_warmup_steps(num_updates=200) -> dict:
    """
    Load SFT checkpoint and run a short PPO warm-up.
    In production this would connect to the live game; for now we use
    random-frame synthetic transitions to validate the training loop end-to-end.
    TODO: wire to real Geoplanetical process via agent.py when game is available.
    """
    log.info("── Step 3/5: PPO warm-up (%d updates) ──", num_updates)
    import torch
    from src.brain.network import GreenVisionNetwork, STATE_DIM, NUM_ACTIONS, SEQ_LEN
    from src.brain.ppo_agent import PPOAgent, PPOConfig, Transition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = GreenVisionNetwork().to(device)
    # Try loading SFT weights
    sft_path = PROJECT_ROOT / "data" / "models" / "policy_sft.pt"
    if sft_path.exists():
        sd = torch.load(sft_path, map_location=device)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        log.info("Loaded SFT weights (missing=%d, unexpected=%d)", len(missing), len(unexpected))
    else:
        log.warning("No SFT checkpoint found – using random init")
    cfg = PPOConfig(update_every_steps=4, batch_size=32, rollout_capacity=128, replay_capacity=10_000)
    agent = PPOAgent(net, cfg)
    t0 = time.perf_counter()
    total_reward = 0.0
    for step in range(num_updates * 4):  # 4 env steps per update
        frame = np.random.rand(3, 224, 224).astype(np.float32)  # synthetic
        state = np.random.randn(STATE_DIM).astype(np.float32) * 0.1
        logits, value = agent.observe(frame, state)
        action = agent.select_action(logits)
        reward = float(np.random.randn() * 0.5)  # synthetic reward
        done = False
        agent.store_transition(frame, state, action, reward, done, logits[action].item(), value.item())
        if agent.should_update():
            metrics = agent.update()
            total_reward += reward
    elapsed = time.perf_counter() - t0
    log.info("PPO warm-up done: %d updates in %.0f s (avg reward=%.3f)", num_updates, elapsed, total_reward / max(1, num_updates))
    return {"ok": True, "updates": num_updates, "time_s": round(elapsed, 1), "avg_reward": round(total_reward / max(1, num_updates), 4)}
# ── Step 4: Save checkpoint + log ────────────────────────────────────────────
def save_checkpoint(cycle_idx: int, gather_res: dict, pretrain_res: dict, ppo_res: dict):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = PROJECT_ROOT / "data" / "models" / f"cycle_{cycle_idx}_{ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": ts,
        "gather": gather_res,
        "pretrain": pretrain_res,
        "ppo_warmup": ppo_res,
    }
    (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # Also copy the latest SFT weights
    sft_src = PROJECT_ROOT / "data" / "models" / "policy_sft.pt"
    if sft_src.exists():
        shutil.copy2(sft_src, ckpt_dir / "policy_sft.pt")
    log.info("Checkpoint saved to %s", ckpt_dir)
    return ckpt_dir
# ── Step 5: Update .convolog.md ──────────────────────────────────────────────
def update_convolog(cycle_idx: int, gather_res: dict, pretrain_res: dict, ppo_res: dict):
    convolog = PROJECT_ROOT / ".convolog.md"
    entry = (
        f"\n### Cycle {cycle_idx} — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"- Gather: {gather_res.get('clips', '?')} clips, ~{gather_res.get('frames', '?')} frames ({gather_res.get('time_s', '?')}s)\n"
        f"- Pretrain: {'OK' if pretrain_res.get('ok') else 'FAILED'} ({pretrain_res.get('time_s', '?')}s)\n"
        f"- PPO warmup: {ppo_res.get('updates', '?')} updates, avg_reward={ppo_res.get('avg_reward', '?')} ({ppo_res.get('time_s', '?')}s)\n"
    )
    try:
        with open(convolog, "a") as f:
            f.write(entry)
    except Exception as e:
        log.warning("Could not update .convolog.md: %s", e)
# ── Main loop ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Autonomous training loop for Green Vision")
    ap.add_argument("--cycles", type=int, default=0, help="Number of cycles (0 = run until stopped)")
    ap.add_argument("--max-hours", type=float, default=24.0, help="Max wall-clock hours before stopping")
    ap.add_argument("--epochs-a", type=int, default=2, help="Stage A epochs per cycle")
    ap.add_argument("--epochs-b", type=int, default=4, help="Stage B epochs per cycle")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size")
    ap.add_argument("--ppo-updates", type=int, default=200, help="PPO warm-up updates per cycle")
    args = ap.parse_args()
    start_time = time.perf_counter()
    max_seconds = args.max_hours * 3600
    log.info("═══ Autonomous Training Loop started ═══")
    log.info("Config: cycles=%s, max_hours=%.1f, batch=%d", args.cycles or "∞", args.max_hours, args.batch_size)
    import numpy as np  # ensure available for PPO step
    cycle = 0
    while True:
        cycle += 1
        elapsed_total = time.perf_counter() - start_time
        if elapsed_total > max_seconds:
            log.info("Max wall-clock reached (%.1f h). Stopping.", elapsed_total / 3600)
            break
        if args.cycles > 0 and cycle > args.cycles:
            log.info("Completed %d cycles. Done.", args.cycles)
            break
        log.info("\n══════ CYCLE %d (t+%.1fh) ══════", cycle, elapsed_total / 3600)
        # Resource check
        if not check_resources():
            log.error("Insufficient resources. Waiting 30 min…")
            time.sleep(1800)
            continue
        # Step 1: Gather
        gather_res = gather_videos_step()
        # Step 2: Pretrain
        pretrain_res = pretrain_step(args.epochs_a, args.epochs_b, args.batch_size)
        # Step 3: PPO warm-up
        ppo_res = ppo_warmup_steps(args.ppo_updates)
        # Step 4: Checkpoint
        save_checkpoint(cycle, gather_res, pretrain_res, ppo_res)
        # Step 5: Log
        update_convolog(cycle, gather_res, pretrain_res, ppo_res)
        # Brief cooldown between cycles (shortened to keep throughput high)
        log.info("Cycle %d complete. Cooling down 20 s...", cycle)
        time.sleep(20)
    log.info("\n═══ Autonomous Training Loop finished after %d cycles ═══", cycle)
if __name__ == "__main__":
    main()
