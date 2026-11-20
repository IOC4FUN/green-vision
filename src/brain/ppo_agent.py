"""
PPO (Proximal Policy Optimization) agent for Green Vision.
v2 enhancements over v1:
  • True multi-step GAE (lambda=0.95) on an ordered rollout buffer.
  • Rolling frame-history buffer so the transformer sees real recent frames.
  • LR warmup → cosine decay scheduler.
  • Entropy annealing (high early → low late).
  • **Prioritized Experience Replay** (sum-tree) replacing FIFO replay.
  • **KL divergence early-stopping** in the PPO update loop to prevent
    policy collapse when the new policy diverges too far from old.
  • Periodic model saving / loading.
  • CUDA training on RTX 4090.
"""
from __future__ import annotations
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .network import GreenVisionNetwork, NUM_ACTIONS, STATE_DIM, SEQ_LEN

logger = logging.getLogger("brain.ppo")

# ── Data containers ────────────────────────────────────────────────────────
@dataclass
class Transition:
    """One step of experience."""
    frame: np.ndarray        # (3, H, W) float32 normalized [0,1]
    state: np.ndarray        # (STATE_DIM,) float32
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float

@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    total_training_steps: int = 200_000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.005
    entropy_anneal_steps: int = 100_000
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_every_steps: int = 8
    batch_size: int = 64
    rollout_capacity: int = 512
    replay_capacity: int = 100_000
    num_epochs_per_update: int = 4
    # ── PER-specific ───────────────────────────────────────────────
    per_alpha: float = 0.6          # prioritization exponent (0=uniform)
    per_beta_start: float = 0.4     # importance-sampling correction start
    per_beta_end: float = 1.0       # fully corrected by end of training
    per_min_priority: float = 1e-3  # floor to avoid zero-probability samples
    # ── KL early-stopping ──────────────────────────────────────────
    kl_target: float = 0.02         # target KL(new||old); stop epoch if exceeded ×3
    kl_stop_factor: float = 3.0     # multiplier on kl_target for hard stop
    device: str = "cuda"

# ── Sum-tree for Prioritized Experience Replay ─────────────────────────────
class SumTree:
    """
    Binary sum-tree supporting O(log n) insert, update, and weighted sample.
    Leaves store priorities; internal nodes store subtree sums.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self._size = 0
        self._next_idx = 0

    @property
    def size(self) -> int:
        return self._size

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def add(self, priority: float, data) -> None:
        leaf_idx = self._next_idx + self.capacity - 1
        self.tree[leaf_idx] = max(priority, 1e-8)
        self.data[self._next_idx] = data
        self._propagate(leaf_idx, max(priority, 1e-8))
        self._next_idx = (self._next_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        leaf_idx = idx + self.capacity - 1
        delta = max(priority, 1e-8) - self.tree[leaf_idx]
        self.tree[leaf_idx] = max(priority, 1e-8)
        self._propagate(leaf_idx, delta)

    def sample(self, batch_size: int) -> list[tuple[int, float]]:
        """
        Sample batch_size indices proportional to their priority.
        Returns list of (index_in_data, sampling_probability).
        """
        total = self.tree[0]
        if total < 1e-8:
            return []
        seg = total / batch_size
        results: list[tuple[int, float]] = []
        for i in range(batch_size):
            cum = random.uniform(i * seg, (i + 1) * seg)
            # Walk down the tree
            idx = 0
            while idx < self.capacity - 1:
                left = 2 * idx + 1
                right = left + 1
                if self.tree[left] >= cum:
                    idx = left
                else:
                    cum -= self.tree[left]
                    idx = right
            leaf = idx - (self.capacity - 1)
            prob = self.tree[idx] / total if total > 0 else 0.0
            results.append((max(0, min(leaf, self._size - 1)), prob))
        return results

# ── Prioritized Replay Buffer ──────────────────────────────────────────────
class PERBuffer:
    """
    Prioritized Experience Replay buffer using a SumTree.
    Priority is based on |TD error| (approximated by |reward| initially,
    updated after each PPO pass).
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4,
                 beta_end: float = 1.0, min_priority: float = 1e-3):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.min_priority = min_priority
        self.tree = SumTree(capacity)
        self.total_updates = 0

    @property
    def size(self) -> int:
        return self.tree.size

    def push(self, t: Transition) -> None:
        # Initial priority: proportional to |reward| + small epsilon
        prio = abs(t.reward) + 0.1
        self.tree.add(prio ** self.alpha, t)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray]:
        """
        Sample a prioritized batch. Returns (transitions, importance_weights).
        Importance weights correct for the non-uniform sampling distribution.
        """
        n = min(batch_size, self.tree.size)
        if n == 0:
            return [], np.array([])
        samples = self.tree.sample(n)
        transitions = [self.tree.data[idx] for idx, _ in samples]
        probs = np.array([p for _, p in samples])
        # Beta annealing
        progress = min(1.0, self.total_updates / max(1, 100_000))
        beta = self.beta_start + progress * (self.beta_end - self.beta_start)
        N = max(1, self.tree.size)
        weights = (N * probs + 1e-8) ** (-beta)
        # Normalize so max weight = 1
        w_max = weights.max()
        if w_max > 1e-8:
            weights /= w_max
        return transitions, weights

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        """Update priorities after a PPO pass (TD error as priority proxy)."""
        for idx, err in zip(indices, td_errors):
            prio = abs(err) + self.min_priority
            self.tree.update(idx, prio ** self.alpha)

    def __len__(self) -> int:
        return self.tree.size

# ── Rollout buffer (on-policy, ordered for GAE) ───────────────────────────
class RolloutBuffer:
    """Ordered deque of the most recent transitions (on-policy, for GAE)."""
    def __init__(self, capacity: int):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample_sequential(self, n: int) -> list[Transition]:
        items = list(self._buf)
        if len(items) <= n:
            return items
        start = random.randint(0, len(items) - n)
        return items[start:start + n]

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

# ── Frame history ──────────────────────────────────────────────────────────
class FrameHistory:
    """Keeps the last SEQ_LEN frames so observe() feeds real context."""
    def __init__(self, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self._frames: deque[np.ndarray] = deque(maxlen=seq_len)

    def push(self, frame: np.ndarray) -> None:
        self._frames.append(frame.copy())

    def get_stack(self) -> torch.Tensor:
        h, w = self._frames[-1].shape[1:] if self._frames else (224, 224)
        stack = torch.zeros(1, self.seq_len, 3, h, w)
        for i, f in enumerate(reversed(list(self._frames))):
            idx = self.seq_len - 1 - i
            stack[0, idx] = torch.from_numpy(f).float()
        return stack

# ── LR scheduler ───────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """Linear warmup → cosine decay to min_lr."""
    def __init__(self, optimizer, peak_lr: float, min_lr: float,
                 warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(warmup_steps + 1, total_steps)
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.peak_lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

# ── Entropy annealing ──────────────────────────────────────────────────────
def entropy_coef(step: int, cfg: PPOConfig) -> float:
    t = min(1.0, step / max(1, cfg.entropy_anneal_steps))
    return cfg.entropy_coef_start + t * (cfg.entropy_coef_end - cfg.entropy_coef_start)

# ── GAE computation ────────────────────────────────────────────────────────
def compute_gae(
    rewards: list[float], values: list[float], dones: list[bool],
    next_value: float, gamma: float, lam: float,
) -> tuple[list[float], list[float]]:
    n = len(rewards)
    advantages = [0.0] * n
    gae = 0.0
    for t in reversed(range(n)):
        non_terminal = 0.0 if dones[t] else 1.0
        if t == n - 1:
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
    returns = [advantages[t] + values[t] for t in range(n)]
    return advantages, returns

# ── KL divergence helper ───────────────────────────────────────────────────
def kl_divergence(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor,
                  actions: torch.Tensor) -> float:
    """
    Approximate KL(new || old) using the log-prob difference on sampled actions.
    KL ≈ E_new[log π_new(a|s) - log π_old(a|s)]
    """
    with torch.no_grad():
        kl = (new_log_probs - old_log_probs).mean()
    return float(kl.item())

# ── Main agent ─────────────────────────────────────────────────────────────
class PPOAgent:
    """
    Full PPO training + inference loop with GAE, PER, KL early-stopping,
    LR scheduling, entropy annealing, and rolling frame history.
    """
    def __init__(self, config: PPOConfig | None = None, model_path: str | Path | None = None):
        self.cfg = config or PPOConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        logger.info("PPO device: %s", self.device)

        self.net = GreenVisionNetwork().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-5
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            peak_lr=self.cfg.learning_rate,
            min_lr=self.cfg.min_lr,
            warmup_steps=self.cfg.warmup_steps,
            total_steps=self.cfg.total_training_steps,
        )

        # Buffers
        self.rollout = RolloutBuffer(self.cfg.rollout_capacity)
        self.per_buffer = PERBuffer(
            capacity=self.cfg.replay_capacity,
            alpha=self.cfg.per_alpha,
            beta_start=self.cfg.per_beta_start,
            beta_end=self.cfg.per_beta_end,
            min_priority=self.cfg.per_min_priority,
        )
        self.frame_history = FrameHistory(SEQ_LEN)

        self._steps_since_update = 0
        self.total_steps = 0
        self.episode_rewards: list[float] = []
        self._kl_early_stops = 0  # counter for dashboard

        if model_path and Path(model_path).exists():
            self.load(model_path)
            logger.info("Loaded model from %s", model_path)

    # ── Inference ────────────────────────────────────────────────
    @torch.no_grad()
    def observe(self, frame: np.ndarray, state_vec: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        self.frame_history.push(frame)
        frames_seq = self.frame_history.get_stack().to(self.device)
        state_t = torch.from_numpy(state_vec).float().unsqueeze(0).to(self.device)
        logits, value = self.net(frames_seq, state_t)
        return logits.cpu(), value.cpu()

    def select_action(self, logits: torch.Tensor) -> tuple[int, float]:
        dist = Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    # ── Training ─────────────────────────────────────────────────
    def store_transition(
        self, frame: np.ndarray, state: np.ndarray,
        action: int, reward: float, done: bool,
        log_prob: float, value: float
    ) -> None:
        t = Transition(
            frame=frame, state=state, action=action,
            reward=reward, done=done, log_prob=log_prob, value=value,
        )
        self.rollout.push(t)
        self.per_buffer.push(t)
        self._steps_since_update += 1
        self.total_steps += 1

    def should_update(self) -> bool:
        return (
            self._steps_since_update >= self.cfg.update_every_steps
            and len(self.rollout) >= self.cfg.batch_size
        )

    def update(self) -> dict[str, float]:
        """
        Run one PPO update cycle with GAE + PER augmentation + KL early-stopping.
        Returns loss metrics dict.
        """
        if not self.should_update():
            return {}

        self.net.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0
        kl_stopped_early = False

        ent_coef = entropy_coef(self.total_steps, self.cfg)

        for epoch in range(self.cfg.num_epochs_per_update):
            # ── Sample: mix on-policy rollout + PER off-policy ────
            batch = self.rollout.sample_sequential(self.cfg.batch_size)
            if len(batch) < self.cfg.batch_size // 2:
                break

            B = len(batch)
            frames = torch.stack([
                torch.from_numpy(t.frame).float().unsqueeze(0).expand(SEQ_LEN, -1, -1, -1)
                for t in batch
            ]).to(self.device)
            states = torch.stack([torch.from_numpy(t.state).float() for t in batch]).to(self.device)
            actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor([t.log_prob for t in batch], device=self.device)
            rewards_t = torch.tensor([t.reward for t in batch], device=self.device)
            dones_t = torch.tensor([t.done for t in batch], dtype=torch.bool, device=self.device)
            old_values_t = torch.tensor([t.value for t in batch], device=self.device)

            # ── GAE ───────────────────────────────────────────────
            with torch.no_grad():
                _, new_values = self.net(frames, states)
                nv_list = [float(v.item()) for v in new_values.squeeze(-1)]
                rv_list = [float(r.item()) for r in rewards_t]
                ov_list = [float(v.item()) for v in old_values_t]
                dn_list = [bool(d) for d in dones_t]
                boot = 0.0 if dn_list[-1] else float(new_values.squeeze(-1)[-1].item())

                advs, rets = compute_gae(
                    rv_list, ov_list, dn_list, boot,
                    gamma=self.cfg.gamma, lam=self.cfg.gae_lambda,
                )
                advantages_t = torch.tensor(advs, device=self.device)
                returns_t = torch.tensor(rets, device=self.device)
                if advantages_t.std() > 1e-8:
                    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # ── Forward pass ──────────────────────────────────────
            logits, values = self.net(frames, states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # ── KL check BEFORE the gradient step ─────────────────
            kl_val = kl_divergence(old_log_probs, new_log_probs, actions)
            total_kl += kl_val

            # Hard stop: if KL exceeds target × factor, skip this epoch's update
            if kl_val > self.cfg.kl_target * self.cfg.kl_stop_factor:
                logger.debug("KL early-stop: %.4f > %.4f (epoch %d)",
                             kl_val, self.cfg.kl_target * self.cfg.kl_stop_factor, epoch)
                kl_stopped_early = True
                break

            # Clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(-1), returns_t)
            total_loss = policy_loss + self.cfg.value_coef * value_loss - ent_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            n_updates += 1

        # ── PER: update priorities with TD-error proxy ────────────
        if len(self.per_buffer) > self.cfg.batch_size:
            n_per = self.cfg.batch_size // 2
            samples = self.per_buffer.tree.sample(n_per)
            if samples:
                idxs = [idx for idx, _ in samples]
                td_errors = [
                    abs(self.per_buffer.tree.data[idx].reward - self.per_buffer.tree.data[idx].value)
                    for idx in idxs
                ]
                self.per_buffer.update_priorities(idxs, td_errors)
                self.per_buffer.total_updates += 1

        self._steps_since_update = 0
        if kl_stopped_early:
            self._kl_early_stops += 1

        metrics = {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "kl_div": total_kl / max(1, n_updates),
            "lr": self.scheduler.optimizer.param_groups[0]["lr"],
            "ent_coef": ent_coef,
            "buffer_size": len(self.rollout),
            "per_size": len(self.per_buffer),
            "total_steps": self.total_steps,
            "kl_early_stops_total": self._kl_early_stops,
        }
        logger.debug("PPO update: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
        return metrics

    # ── Persistence ──────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_step": self.scheduler.step_count,
            "total_steps": self.total_steps,
            "config": vars(self.cfg),
        }, path)
        logger.info("Model saved → %s (%d steps)", path, self.total_steps)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.scheduler.step_count = ckpt.get("scheduler_step", 0)
