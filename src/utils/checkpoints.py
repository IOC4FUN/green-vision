"""
Checkpoint manager — periodic model snapshots with best-reward tracking.

Saves the PPO network every N steps and keeps a `best.pt` copy whenever
the rolling-average reward exceeds the previous best. A crash mid-session
no longer loses all progress since last shutdown.

> Keep in sync with code — update this file whenever the module API changes.
"""
from __future__ import annotations

import logging
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("utils.checkpoints")


class CheckpointManager:
    def __init__(
        self,
        dir_path: str | Path = "data/models/checkpoints",
        save_every_steps: int = 500,
        window: int = 200,
    ):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every_steps
        self.window = window
        self._rewards: deque[float] = deque(maxlen=window)
        self._best_reward: float = -float("inf")
        self._last_save_step: int = 0

    # ── Public API ────────────────────────────────────────────────
    def record_reward(self, reward: float) -> None:
        self._rewards.append(reward)

    def maybe_save(
        self,
        step: int,
        model: Any,          # torch.nn.Module (PPO net)
        optimizer: Any,      # torch.optim.Optimizer
    ) -> Optional[str]:
        """
        Save a checkpoint if it's time. Returns the path if saved, else None.
        Also promotes to best.pt when rolling avg beats previous best.
        """
        if step - self._last_save_step < self.save_every:
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.dir / f"ckpt_{step:07d}_{ts}.pt"
        try:
            import torch
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "best_reward": self._best_reward,
            }, path)
            logger.info("Checkpoint saved → %s", path.name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Checkpoint save failed: %s", e)
            return None

        self._last_save_step = step

        # Promote to best?
        if len(self._rewards) >= min(50, self.window):
            avg = sum(self._rewards) / len(self._rewards)
            if avg > self._best_reward:
                self._best_reward = avg
                best_path = self.dir / "best.pt"
                shutil.copy2(path, best_path)
                logger.info("New best checkpoint (avg reward %.4f) → best.pt", avg)

        return str(path)

    @property
    def best_reward(self) -> float:
        return self._best_reward

    def list_checkpoints(self) -> list[str]:
        return sorted(p.name for p in self.dir.glob("ckpt_*.pt"))
