"""
World Model – Learned Forward Dynamics for Green Vision.
Unlike curiosity.py (which computes 1-step prediction *error* as a reward
bonus), this module learns to **simulate** future states so the planner can
lookahead before committing to an action sequence.
Architecture:
    State encoder:   Linear(128→128) → GELU → Linear(128→96)
    Transition net:  concat(enc(s_t), onehot(a_t)) → MLP(96+31→192→192→96)
    Decoder:         Linear(96→128)  (reconstruction target, stop-grad)
Multi-step rollout:
    The planner calls `rollout(state_vec, actions, horizon)` which iterates
    the transition net `horizon` times, producing predicted state embeddings.
    A lightweight value head on top of each predicted embedding gives an
    estimated return, enabling greedy / beam search over action sequences.
Usage:
    wm = WorldModel(state_dim=128, num_actions=31)
    preds = wm.rollout(current_state, [A_NAV_ECONOMY, A_APPLY_POLICY], horizon=4)
    best_seq = wm.plan(current_state, candidate_seqs, horizon=4)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.getLogger("brain.world_model")
# ── Sub-networks ───────────────────────────────────────────────────────────
class _Encoder(nn.Module):
    """Maps raw 128-dim game state → compact 96-dim latent."""
    def __init__(self, in_dim: int = 128, hidden: int = 128, out_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class _TransitionNet(nn.Module):
    """Predicts next latent given current latent + action one-hot."""
    def __init__(self, lat_dim: int = 96, num_actions: int = 31, hidden: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lat_dim + num_actions, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, lat_dim),
        )
    def forward(self, z: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, a_onehot], dim=-1)
        return self.net(inp)
class _Decoder(nn.Module):
    """Latent → reconstructed state (used only as training target)."""
    def __init__(self, lat_dim: int = 96, out_dim: int = 128):
        super().__init__()
        self.net = nn.Linear(lat_dim, out_dim)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
class _ValueHead(nn.Module):
    """Estimates expected cumulative reward from a latent state."""
    def __init__(self, lat_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
# ── Main module ────────────────────────────────────────────────────────────
class WorldModel:
    """
    Learned dynamics model for multi-step lookahead planning.
    Parameters
    ----------
    state_dim : raw game-state vector dimensionality (default 128).
    num_actions : discrete action space size (default 31).
    lr : Adam learning rate.
    gamma : discount factor used by the value head.
    device : 'cuda' or 'cpu'.
    """
    def __init__(
        self,
        state_dim: int = 128,
        num_actions: int = 31,
        lr: float = 5e-4,
        gamma: float = 0.99,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.encoder = _Encoder(state_dim).to(self.device)
        self.transition = _TransitionNet(96, num_actions).to(self.device)
        self.decoder = _Decoder(96, state_dim).to(self.device)
        self.value_head = _ValueHead(96).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.transition.parameters()) +
            list(self.decoder.parameters()) +
            list(self.value_head.parameters()),
            lr=lr, weight_decay=1e-5,
        )
        # Training stats
        self._total_loss = 0.0
        self._num_updates = 0
        # Running reconstruction loss (EMA) – used as uncertainty proxy
        self._recon_ema: float = 1.0   # start pessimistic (untrained)
        self._recon_alpha: float = 0.05
        logger.info("WorldModel initialised (device=%s, actions=%d)", self.device, num_actions)
    # ── Inference / Planning API ───────────────────────────────────────
    @torch.no_grad()
    def rollout(
        self,
        state_vec: np.ndarray,
        actions: list[int],
        horizon: int | None = None,
    ) -> list[np.ndarray]:
        """
        Simulate `len(actions)` future states starting from `state_vec`.
        Returns a list of predicted raw-state vectors (one per step).
        """
        h = horizon or len(actions)
        s = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
        z = self.encoder(s)  # (1, 96)
        preds: list[np.ndarray] = []
        for i in range(min(h, len(actions))):
            onehot = torch.zeros(1, self.num_actions, device=self.device)
            onehot[0, min(actions[i], self.num_actions - 1)] = 1.0
            z = self.transition(z, onehot)
            recon = self.decoder(z)
            preds.append(recon.squeeze(0).cpu().numpy())
        return preds
    @torch.no_grad()
    def plan(
        self,
        state_vec: np.ndarray,
        candidate_seqs: list[list[int]],
        horizon: int = 4,
        beam_width: int = 5,
    ) -> tuple[list[int], float]:
        """
        Score each candidate action sequence via the value head and return
        the best one.
        Parameters
        ----------
        candidate_seqs : list of action-index sequences to evaluate.
        horizon : max steps to simulate per sequence.
        beam_width : keep top-N during evaluation (prunes expensive rollouts).
        Returns
        -------
        (best_action_seq, estimated_value)
        """
        if not candidate_seqs:
            return [], 0.0
        # Sort by a quick heuristic first (prefer shorter / known-good), then full eval
        scored: list[tuple[float, list[int]]] = []
        for seq in candidate_seqs[:beam_width * 3]:  # cap candidates
            h = min(horizon, len(seq))
            s = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
            z = self.encoder(s)
            total_val = 0.0
            discount = 1.0
            for i in range(h):
                onehot = torch.zeros(1, self.num_actions, device=self.device)
                onehot[0, min(seq[i], self.num_actions - 1)] = 1.0
                z = self.transition(z, onehot)
                v = float(self.value_head(z).item())
                total_val += discount * v
                discount *= self.gamma
            scored.append((total_val, seq))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_val, best_seq = scored[0]
        return best_seq, best_val
    @torch.no_grad()
    def predict_next_state(self, state_vec: np.ndarray, action_idx: int) -> np.ndarray:
        """Convenience: single-step prediction."""
        preds = self.rollout(state_vec, [action_idx], horizon=1)
        return preds[0] if preds else state_vec.copy()
    @torch.no_grad()
    def state_uncertainty(self, state_vec: np.ndarray) -> float:
        """
        Per-state uncertainty in [0, 1]. High when the world model has not
        learned this region well (poor auto-reconstruction of the input state)
        or is globally under-trained. The agent uses this to gate WM overrides
        and fall back to curiosity-driven exploration.
        """
        s = torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
        z = self.encoder(s)
        recon = self.decoder(z)
        local_err = F.mse_loss(recon, s, reduction="mean").item()
        # Normalise: trained err ~0.05-0.3; untrained ~1+.
        local_norm = min(1.0, max(0.0, (local_err - 0.05) / 1.0))
        global_norm = min(1.0, max(0.0, (self._recon_ema - 0.05) / 1.0))
        return round(max(local_norm, global_norm), 4)
    @torch.no_grad()
    def confidence(self, state_vec: np.ndarray) -> float:
        """Model-based confidence in [0, 1] for a state region (= 1 - uncertainty)."""
        return round(1.0 - self.state_uncertainty(state_vec), 4)
    # ── Training ───────────────────────────────────────────────────────
    def train_step(
        self,
        states: np.ndarray,
        actions: list[int],
        next_states: np.ndarray,
        rewards: list[float] | None = None,
    ) -> dict[str, float]:
        """
        One gradient step on a batch of (s, a, s') transitions.
        Losses:
          - reconstruction: || decoder(enc(s')) - s' ||²  (stop-grad target)
          - transition:     || trans(enc(s), a) - enc(s').detach() ||²
          - value regression: MSE(value_head(z'), reward + γ·V(s''))
                              (simplified: just regress to cumulative reward)
        """
        B = len(states)
        if B == 0:
            return {"wm_loss": 0.0}
        s_t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(self.device)
        s_tp1 = torch.from_numpy(np.asarray(next_states, dtype=np.float32)).to(self.device)
        n_act = self.num_actions
        onehots = torch.zeros(B, n_act, device=self.device)
        for i, a in enumerate(actions):
            onehots[i, min(a, n_act - 1)] = 1.0
        # Encode
        z_t = self.encoder(s_t)
        with torch.no_grad():
            z_tp1_target = self.encoder(s_tp1).detach()
        # Transition loss
        z_pred = self.transition(z_t, onehots)
        trans_loss = F.mse_loss(z_pred, z_tp1_target)
        # Reconstruction loss (decoder on predicted latent → should match next state)
        recon = self.decoder(z_pred)
        recon_loss = F.mse_loss(recon, s_tp1)
        # Value regression: target = immediate reward (simple baseline)
        v_pred = self.value_head(z_pred).squeeze(-1)  # (B,)
        if rewards is not None:
            r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            value_loss = F.mse_loss(v_pred, r_t)
        else:
            value_loss = torch.tensor(0.0, device=self.device)
        total_loss = trans_loss + recon_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        # Update running reconstruction EMA (uncertainty proxy).
        self._recon_ema = ((1 - self._recon_alpha) * self._recon_ema
                           + self._recon_alpha * float(recon_loss.item()))
        self._total_loss += float(total_loss.item())
        self._num_updates += 1
        return {
            "wm_loss": round(float(total_loss.item()), 6),
            "trans_loss": round(float(trans_loss.item()), 6),
            "recon_loss": round(float(recon_loss.item()), 6),
            "value_loss": round(float(value_loss.item()), 6),
        }
    def parameters(self):
        """All trainable params (for clip_grad_norm_)."""
        yield from self.encoder.parameters()
        yield from self.transition.parameters()
        yield from self.decoder.parameters()
        yield from self.value_head.parameters()
    # ── Persistence ────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save({
            "encoder": self.encoder.state_dict(),
            "transition": self.transition.state_dict(),
            "decoder": self.decoder.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gamma": self.gamma,
        }, path)
        logger.info("WorldModel saved → %s", path)
    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.transition.load_state_dict(ckpt["transition"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.value_head.load_state_dict(ckpt["value_head"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.gamma = ckpt.get("gamma", self.gamma)
        logger.info("WorldModel loaded ← %s", path)
    # ── Dashboard ──────────────────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        avg_loss = self._total_loss / max(1, self._num_updates)
        return {
            "updates": self._num_updates,
            "avg_loss": round(avg_loss, 6),
            "params": sum(p.numel() for p in self.parameters()),
        }