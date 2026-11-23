"""
Intrinsic Motivation – Curiosity Module for Green Vision.
Implements ICM (Intrinsic Curriculum of Exploration, Pathak et al. 2017):
a small forward-dynamics model predicts the next state embedding; the
prediction error becomes an *intrinsic* reward bonus on top of the
game's extrinsic reward. This drives the agent toward novel / surprising
states without hand-crafted exploration heuristics.
Architecture:
    State encoder:  Linear(128 → 128) → ReLU → Linear(128 → 64)
    Forward model:  concat(enc(s_t), action_onehot) → MLP(64+31 → 128 → 64)
    Intrinsic reward = || enc(s_{t+1}) - pred(s_t, a_t) ||_2
The module is lightweight (~50k params) and trains alongside the main
PPO policy using its own Adam optimizer.
Usage (in ppo_agent or agent loop):
    curiosity = CuriosityModule(state_dim=128, num_actions=31)
    intrinsic_r = curiosity.compute_intrinsic_reward(state_vec, action_idx, next_state_vec)
    total_reward = extrinsic_r + curiosity.coeff * intrinsic_r
"""
from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger("brain.curiosity")
class _StateEncoder(nn.Module):
    """Encodes a 128-dim game state into a compact 64-dim embedding."""
    def __init__(self, in_dim: int = 128, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class _ForwardModel(nn.Module):
    """Predicts next-state embedding given current embedding + action one-hot."""
    def __init__(self, enc_dim: int = 64, num_actions: int = 31, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim + num_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, enc_dim),
        )
    def forward(self, emb: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([emb, action_onehot], dim=-1)
        return self.net(inp)
class CuriosityModule:
    """
    ICM-style curiosity: learns a forward model of state transitions and
    uses prediction error as intrinsic reward.
    Parameters
    ----------
    state_dim : dimensionality of the game state vector (default 128).
    num_actions : size of discrete action space (default 31).
    lr : learning rate for the internal Adam optimizer.
    coeff : scaling factor applied to the raw prediction-error reward.
    clip_max : upper clamp on intrinsic reward to prevent runaway bonuses.
    device : 'cuda' or 'cpu'.
    """
    def __init__(
        self,
        state_dim: int = 128,
        num_actions: int = 31,
        lr: float = 1e-4,
        coeff: float = 0.5,
        clip_max: float = 1.0,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.coeff = coeff
        self.clip_max = clip_max
        self.encoder = _StateEncoder(state_dim).to(self.device)
        self.forward_model = _ForwardModel(64, num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.forward_model.parameters()),
            lr=lr,
        )
        # Running stats for logging
        self._total_intrinsic_reward = 0.0
        self._num_steps = 0
        logger.info("CuriosityModule initialised (device=%s, coeff=%.2f)", self.device, coeff)
    # ── Core API ───────────────────────────────────────────────────────
    @torch.no_grad()
    def compute_intrinsic_reward(
        self,
        state_vec: np.ndarray,
        action_idx: int,
        next_state_vec: np.ndarray,
    ) -> float:
        """
        Compute the intrinsic reward bonus for a transition.
        Returns a scalar in [0, clip_max] scaled by `coeff`.
        """
        s_t = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
        s_tp1 = torch.from_numpy(np.asarray(next_state_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
        onehot = torch.zeros(1, self.forward_model.net[0].in_features - 64, device=self.device)
        onehot[0, min(action_idx, onehot.shape[1] - 1)] = 1.0
        emb_t = self.encoder(s_t)              # (1, 64)
        with torch.no_grad():
            target = self.encoder(s_tp1).detach()   # true next-state embedding
        pred_emb = self.forward_model(emb_t, onehot)  # (1, 64)
        err = F.mse_loss(pred_emb, target, reduction="sum").item() ** 0.5
        intrinsic = min(err * self.coeff, self.clip_max)
        self._total_intrinsic_reward += intrinsic
        self._num_steps += 1
        return float(intrinsic)
    def train_step(
        self,
        states: np.ndarray,
        actions: list[int],
        next_states: np.ndarray,
    ) -> dict[str, float]:
        """
        Train the encoder + forward model on a batch of transitions.
        Called periodically (e.g., every PPO update or every N steps).
        Returns loss metrics.
        """
        B = len(states)
        if B == 0:
            return {"curiosity_loss": 0.0}
        s_t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        s_tp1 = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(self.device)
        n_act = self.forward_model.net[0].in_features - 64
        onehots = torch.zeros(B, n_act, device=self.device)
        for i, a in enumerate(actions):
            onehots[i, min(a, n_act - 1)] = 1.0
        emb_t = self.encoder(s_t)
        pred = self.forward_model(emb_t, onehots)
        # Target: encode next state (stop-grad to avoid collapsing encoder)
        with torch.no_grad():
            target = self.encoder(s_tp1).detach()
        loss = F.mse_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.forward_model.parameters()), 1.0
        )
        self.optimizer.step()
        return {"curiosity_loss": float(loss.item())}
    @torch.no_grad()
    def novelty_score(self, state_vec: np.ndarray) -> float:
        """
        How 'surprising' is this state? High score → agent has rarely seen it.
        Useful for the planner to decide when to explore vs exploit.
        Returns a value in [0, 1] (higher = more novel).
        """
        s = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
        emb = self.encoder(s)
        # Norm of embedding as proxy for novelty (calibrated over training)
        norm = emb.norm(dim=-1).item()
        # Soft-normalise: typical norms are ~5-15; map to [0,1]
        return min(1.0, max(0.0, (norm - 3.0) / 12.0))
    # ── Persistence ────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save({
            "encoder": self.encoder.state_dict(),
            "forward_model": self.forward_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "coeff": self.coeff,
            "clip_max": self.clip_max,
        }, path)
        logger.info("CuriosityModule saved → %s", path)
    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.forward_model.load_state_dict(ckpt["forward_model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.coeff = ckpt.get("coeff", self.coeff)
        self.clip_max = ckpt.get("clip_max", self.clip_max)
        logger.info("CuriosityModule loaded ← %s", path)
    # ── Dashboard ──────────────────────────────────────────────────────
    def summary(self) -> dict[str, float]:
        avg = self._total_intrinsic_reward / max(1, self._num_steps)
        return {
            "steps_tracked": self._num_steps,
            "avg_intrinsic_reward": round(avg, 5),
            "coeff": self.coeff,
        }