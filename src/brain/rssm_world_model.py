"""
RSSM World Model – Recurrent State-Space Model for Green Vision.
Inspired by DreamerV3 (Hafner et al., 2023). Replaces the simple feedforward
transition net with a proper RSSM that separates:
  • Deterministic state h_t : GRU hidden vector (long-horizon memory)
  • Stochastic latent   z_t : diagonal Gaussian (short-term uncertainty)
This gives the planner two benefits over the old WM:
  1. **Calibrated uncertainty** – σ² of the stochastic posterior tells us how
     surprised the model is; high σ → low confidence → fall back to curiosity.
  2. **Better multi-step rollouts** – the GRU carries context across steps so
     8-16 step predictions don't diverge as quickly.

Implements DreamerV3-style techniques:
  • symlog/symexp normalisation for unbounded continuous values
  • Two-hot action encoding (smoother gradients than one-hot)
  • Free-bits KL balancing (prevents posterior collapse in early training)

Architecture (compact, fits on RTX 4090 at 300 W):
    Encoder:      obs(128) + h_{t-1}(deter) → μ, σ  (posterior)
    Prior:        h_{t-1} + a_{t-1}          → μ', σ'  (prediction)
    GRU core:     [z_{t-1}, a_{t-1}]         → h_t
    Decoder:      [h_t, z_t]                 → recon_obs(128), reward(1), cont(1)
Training loss:
    L = λ_recon · MSE(recon, symlog(obs))
      + λ_KL   · free_bits_KL(q(z|s,a,h) ‖ p(z|h,a))
      + λ_rwd  · Huber(pred_r, r)
      + λ_cont · BCE(pred_c, c)
Usage:
    wm = RSSMWorldModel(state_dim=128, num_actions=31, device="cuda")
    wm.train_step(obs, action_idx, next_obs, reward, done)
    preds = wm.rollout(current_state, actions, horizon=8)
    conf  = wm.confidence(current_state)   # 1 - normalised σ²
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("brain.rssm_wm")


# ── DreamerV3 utility functions ─────────────────────────────────────────────
def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform: sign(x) * log(|x| + 1).
    Compresses unbounded values into a bounded range while preserving sign."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def two_hot(action_idx: int, num_actions: int, device: str | torch.device) -> torch.Tensor:
    """Two-hot action encoding (DreamerV3 standard).
    Instead of one 1.0, sets two adjacent positions to 0.5 each.
    Provides smoother gradient flow than one-hot for discrete actions.
    Returns shape (num_actions,)."""
    vec = torch.zeros(num_actions, device=device)
    i = min(action_idx, num_actions - 1)
    vec[i] = 0.5
    vec[(i + 1) % num_actions] = 0.5
    return vec


# ── Sub-networks ────────────────────────────────────────────────────────────
class _GRUCore(nn.Module):
    """Deterministic recurrent core: [z_{t-1}, a_{t-1}] → h_t.
    Uses LayerNorm on input for training stability (DreamerV3 reference)."""

    def __init__(self, deter: int, stoch: int, num_actions: int, hidden: int = 512):
        super().__init__()
        self.norm_in = nn.LayerNorm(stoch + num_actions, eps=1e-03)
        self.gru = nn.GRUCell(stoch + num_actions, deter)

    def forward(self, z_prev: torch.Tensor, a_enc: torch.Tensor,
                h_prev: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z_prev, a_enc], dim=-1)  # (B, stoch+A)
        inp = self.norm_in(inp)
        return self.gru(inp, h_prev)                 # (B, deter)


class _PosteriorHead(nn.Module):
    """q(z | obs, h): produces μ and log σ² for the stochastic latent."""

    def __init__(self, in_dim: int, stoch: int, min_log_std: float = -7.0,
                 max_log_std: float = 2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, stoch * 4),
            nn.SiLU(),
            nn.Linear(stoch * 4, stoch * 2),
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)                          # (B, 2*stoch)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return mu, log_std


class _PriorHead(nn.Module):
    """p(z | h, a): prediction of next stochastic latent."""

    def __init__(self, deter: int, num_actions: int, stoch: int,
                 min_log_std: float = -7.0, max_log_std: float = 2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter + num_actions, stoch * 4),
            nn.SiLU(),
            nn.Linear(stoch * 4, stoch * 2),
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, h: torch.Tensor, a_enc: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([h, a_enc], dim=-1)
        out = self.net(inp)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return mu, log_std


class _DecoderHead(nn.Module):
    """[h_t, z_t] → (recon_obs, reward, continue). Decoder outputs in symlog space."""

    def __init__(self, deter: int, stoch: int, obs_dim: int):
        super().__init__()
        in_dim = deter + stoch
        self.obs_net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2), nn.SiLU(),
            nn.Linear(in_dim * 2, obs_dim),
        )
        self.rwd_head = nn.Linear(in_dim, 1)
        self.cont_head = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, h: torch.Tensor, z: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = torch.cat([h, z], dim=-1)
        recon = self.obs_net(inp)       # (B, obs_dim) – in symlog space
        rwd = self.rwd_head(inp)        # (B, 1)
        cont = torch.sigmoid(self.cont_head(inp))  # (B, 1)
        return recon, rwd, cont


# ── Main module ─────────────────────────────────────────────────────────────
class RSSMWorldModel:
    """
    Recurrent State-Space Model world model.

    Parameters
    ----------
    state_dim   : observation vector dimensionality (default 128).
    num_actions : discrete action space size (default 31).
    deter       : GRU hidden / deterministic state size (default 512).
    stoch       : stochastic latent size (default 64).
    lr          : AdamW learning rate.
    gamma       : discount factor for value estimation in rollouts.
    device      : 'cuda' or 'cpu'.
    lambda_recon: weight on reconstruction loss.
    lambda_kl   : weight on KL divergence between posterior and prior.
    lambda_rwd  : weight on reward prediction loss.
    lambda_cont : weight on episode-continuation BCE loss.
    free_bits   : per-element KL floor in nats (DreamerV3 default 1.0).
                  Prevents posterior collapse by ignoring small KL values.
    """

    def __init__(
        self,
        state_dim: int = 128,
        num_actions: int = 31,
        deter: int = 512,
        stoch: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cuda",
        lambda_recon: float = 1.0,
        lambda_kl: float = 1.0,
        lambda_rwd: float = 5.0,
        lambda_cont: float = 1.0,
        free_bits: float = 1.0,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.deter = deter
        self.stoch = stoch
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.lambda_rwd = lambda_rwd
        self.lambda_cont = lambda_cont
        self.free_bits = free_bits

        # Sub-networks
        self.gru_core = _GRUCore(deter, stoch, num_actions).to(self.device)
        # Posterior sees obs + h → (μ, logσ²)
        self.posterior = _PosteriorHead(state_dim + deter, stoch).to(self.device)
        # Prior sees h + a → (μ', logσ')
        self.prior = _PriorHead(deter, num_actions, stoch).to(self.device)
        # Decoder
        self.decoder = _DecoderHead(deter, stoch, state_dim).to(self.device)
        # Value head for planning (on top of [h, z])
        self.value_head = nn.Sequential(
            nn.Linear(deter + stoch, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1),
        ).to(self.device)

        all_params = (
            list(self.gru_core.parameters())
            + list(self.posterior.parameters())
            + list(self.prior.parameters())
            + list(self.decoder.parameters())
            + list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
        # Training stats
        self._num_updates: int = 0
        self._total_loss: float = 0.0
        # Running σ² EMA – used as a global uncertainty proxy
        self._sigma_ema: float = 1.0
        self._sigma_alpha: float = 0.02
        logger.info(
            "RSSMWorldModel init (device=%s, deter=%d, stoch=%d, actions=%d, free_bits=%.1f)",
            self.device, deter, stoch, num_actions, free_bits,
        )

    # ── Inference / Planning API ──────────────────────────────────────────
    @torch.no_grad()
    def rollout(
        self,
        state_vec: np.ndarray,
        actions: list[int],
        horizon: int | None = None,
    ) -> list[np.ndarray]:
        """
        Simulate future states using the prior (no observations available).
        Returns a list of predicted raw-state vectors (symexp applied to exit symlog space).
        """
        h = self._init_deter()  # zero-init GRU hidden
        z = torch.zeros(1, self.stoch, device=self.device)
        obs_t = symlog(torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device))
        # First step: use posterior to get initial z from observation
        mu, log_std = self.posterior(torch.cat([obs_t, h], dim=-1))
        std = torch.exp(log_std)
        z = self._sample_gaussian(mu, std)
        preds: list[np.ndarray] = []
        for i in range(min(horizon or len(actions), len(actions))):
            a_enc = two_hot(actions[i], self.num_actions, self.device).unsqueeze(0)
            # Prior prediction
            mu_p, log_std_p = self.prior(h, a_enc)
            std_p = torch.exp(log_std_p)
            z_next = self._sample_gaussian(mu_p, std_p)  # sample for rollout diversity
            # Update deterministic state
            h = self.gru_core(z, a_enc, h)
            z = z_next
            recon, _, _ = self.decoder(h, z)
            # Transform back to original observation space
            preds.append(symexp(recon.squeeze(0)).cpu().numpy())
        return preds

    @torch.no_grad()
    def plan(
        self,
        state_vec: np.ndarray,
        candidate_seqs: list[list[int]],
        horizon: int = 8,
        beam_width: int = 5,
    ) -> tuple[list[int], float]:
        """
        Score candidate action sequences via the value head and return the best.
        Uses mean (not sampled) stochastic latent for stable planning.
        """
        if not candidate_seqs:
            return [], 0.0
        obs_t = symlog(torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device))
        scored: list[tuple[float, list[int]]] = []
        for seq in candidate_seqs[:beam_width * 3]:
            h = self._init_deter()
            mu, log_std = self.posterior(torch.cat([obs_t, h], dim=-1))
            z = mu  # use mean for deterministic planning
            total_val = 0.0
            discount = 1.0
            for i in range(min(horizon, len(seq))):
                a_enc = two_hot(seq[i], self.num_actions, self.device).unsqueeze(0)
                mu_p, _ = self.prior(h, a_enc)
                z = mu_p
                h = self.gru_core(z, a_enc, h)
                v = float(self.value_head(torch.cat([h, z], dim=-1)).item())
                total_val += discount * v
                discount *= self.gamma
            scored.append((total_val, seq))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], scored[0][0]

    @torch.no_grad()
    def confidence(self, state_vec: np.ndarray) -> float:
        """
        Model-based confidence in [0, 1].
        High σ² (posterior spread) → low confidence.
        Combines per-state and global EMA uncertainty.
        """
        obs_t = symlog(torch.from_numpy(np.asarray(state_vec, dtype=np.float32)).unsqueeze(0).to(self.device))
        h = self._init_deter()
        _, log_std = self.posterior(torch.cat([obs_t, h], dim=-1))
        local_var = float(torch.exp(2.0 * log_std).mean().item())
        # Normalise: trained var ~0.01-0.5; untrained ~1+
        local_norm = min(1.0, max(0.0, (local_var - 0.01) / 1.0))
        global_norm = min(1.0, max(0.0, (self._sigma_ema - 0.01) / 1.0))
        return round(1.0 - max(local_norm, global_norm), 4)

    @torch.no_grad()
    def state_uncertainty(self, state_vec: np.ndarray) -> float:
        """Per-state uncertainty in [0, 1] (= 1 - confidence)."""
        return round(1.0 - self.confidence(state_vec), 4)

    # ── Training ──────────────────────────────────────────────────────────
    def train_step(
        self,
        obs: np.ndarray,
        action_idx: int,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
    ) -> dict[str, float]:
        """
        Single-step training update.
        Observations are symlog-transformed before entering the network.
        Actions use two-hot encoding for smoother gradients.
        KL uses free-bits to prevent posterior collapse.
        """
        B = 1
        o_t = symlog(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device))
        o_tp1 = symlog(torch.from_numpy(np.asarray(next_obs, dtype=np.float32)).unsqueeze(0).to(self.device))
        r_t = torch.tensor([[reward]], device=self.device)
        c_t = torch.tensor([[0.0 if done else 1.0]], device=self.device)
        a_enc = two_hot(action_idx, self.num_actions, self.device).unsqueeze(0)

        # Deterministic state from previous step (zero-init for single step;
        # in practice you'd carry h across an episode – see train_sequence)
        h_prev = self._init_deter()
        # Prior: p(z_t | h_{t-1}, a_{t-1})
        mu_p, log_std_p = self.prior(h_prev, a_enc)
        # GRU update → h_t
        z_prev = torch.zeros(B, self.stoch, device=self.device)
        h_t = self.gru_core(z_prev, a_enc, h_prev)
        # Posterior: q(z_t | o_t, h_t)
        mu_q, log_std_q = self.posterior(torch.cat([o_t, h_t], dim=-1))
        # Decoder predictions at t+1 using prior sample
        mu_p_s = mu_p + torch.exp(log_std_p) * torch.randn_like(mu_p)
        recon_tp1, rwd_pred, cont_pred = self.decoder(h_t, mu_p_s)
        # ── Losses ────────────────────────────────────────────────────────
        loss_recon = F.mse_loss(recon_tp1, o_tp1)  # both in symlog space
        loss_rwd = F.smooth_l1_loss(rwd_pred.squeeze(-1), r_t.squeeze(-1))
        loss_cont = F.binary_cross_entropy(cont_pred.squeeze(-1), c_t.squeeze(-1))
        # KL(q ‖ p) with free-bits floor per element
        kl = self._kl_free_bits(mu_q, log_std_q, mu_p, log_std_p).mean()
        total = (
            self.lambda_recon * loss_recon
            + self.lambda_kl * kl
            + self.lambda_rwd * loss_rwd
            + self.lambda_cont * loss_cont
        )
        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self._all_params(), max_norm=5.0)
        self.optimizer.step()
        # Update σ EMA
        mean_var = float(torch.exp(2.0 * log_std_q).mean().item())
        self._sigma_ema = (1 - self._sigma_alpha) * self._sigma_ema + self._sigma_alpha * mean_var
        self._num_updates += 1
        self._total_loss += float(total.item())
        return {
            "loss_total": round(float(total.item()), 6),
            "loss_recon": round(float(loss_recon.item()), 6),
            "loss_kl": round(float(kl.item()), 6),
            "loss_rwd": round(float(loss_rwd.item()), 6),
            "loss_cont": round(float(loss_cont.item()), 6),
            "sigma_ema": round(self._sigma_ema, 6),
            "updates": self._num_updates,
        }

    def train_sequence(
        self,
        obs_seq: list[np.ndarray],
        action_seq: list[int],
        reward_seq: list[float],
        done_flags: list[bool],
    ) -> dict[str, float]:
        """
        Train on a full episode / trajectory while carrying the GRU hidden state.
        Much more effective than independent single-step updates because the
        deterministic state accumulates context across the sequence.
        Uses symlog observations, two-hot actions, and free-bits KL.
        """
        T = len(obs_seq)
        if T < 2:
            return {"loss_total": 0.0, "skipped": True}
        all_losses: list[dict[str, float]] = []
        h = self._init_deter()
        z_prev = torch.zeros(1, self.stoch, device=self.device)
        for t in range(T - 1):
            o_tp1 = symlog(torch.from_numpy(np.asarray(obs_seq[t + 1], dtype=np.float32)).unsqueeze(0).to(self.device))
            r_t = torch.tensor([[reward_seq[t]]], device=self.device)
            c_t = torch.tensor([[0.0 if done_flags[t] else 1.0]], device=self.device)
            a_enc = two_hot(action_seq[t], self.num_actions, self.device).unsqueeze(0)

            # Prior at step t (predicts z_{t+1} from h_t, a_t)
            mu_p, log_std_p = self.prior(h, a_enc)
            # GRU → h_{t+1}
            h_next = self.gru_core(z_prev, a_enc, h)
            # Posterior at t+1: q(z_{t+1} | o_{t+1}, h_{t+1})
            mu_q, log_std_q = self.posterior(torch.cat([o_tp1, h_next], dim=-1))
            std_q = torch.exp(log_std_q)
            eps = torch.randn_like(std_q)
            z_sampled = mu_q + std_q * eps
            # Decoder at t+1
            recon, rwd_pred, cont_pred = self.decoder(h_next, z_sampled)
            loss_recon = F.mse_loss(recon, o_tp1)  # symlog space
            loss_rwd = F.smooth_l1_loss(rwd_pred.squeeze(-1), r_t.squeeze(-1))
            loss_cont = F.binary_cross_entropy(cont_pred.squeeze(-1), c_t.squeeze(-1))
            kl = self._kl_free_bits(mu_q, log_std_q, mu_p, log_std_p).mean()
            total = (
                self.lambda_recon * loss_recon
                + self.lambda_kl * kl
                + self.lambda_rwd * loss_rwd
                + self.lambda_cont * loss_cont
            )
            if t == 0:
                self.optimizer.zero_grad()
            total.backward()  # accumulate gradients across the sequence
            all_losses.append({
                "loss_total": float(total.item()),
                "loss_recon": float(loss_recon.item()),
                "loss_kl": float(kl.item()),
            })
            h = h_next
            z_prev = z_sampled.detach()
        torch.nn.utils.clip_grad_norm_(self._all_params(), max_norm=5.0)
        self.optimizer.step()
        mean_var = float(torch.exp(2.0 * log_std_q).mean().item())
        self._sigma_ema = (1 - self._sigma_alpha) * self._sigma_ema + self._sigma_alpha * mean_var
        self._num_updates += 1
        avg = {k: sum(d[k] for d in all_losses) / len(all_losses) for k in all_losses[0]}
        avg["sigma_ema"] = round(self._sigma_ema, 6)
        avg["updates"] = self._num_updates
        return avg

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        state = {
            "gru_core": self.gru_core.state_dict(),
            "posterior": self.posterior.state_dict(),
            "prior": self.prior.state_dict(),
            "decoder": self.decoder.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "meta": {
                "state_dim": self.state_dim,
                "num_actions": self.num_actions,
                "deter": self.deter,
                "stoch": self.stoch,
                "gamma": self.gamma,
                "free_bits": self.free_bits,
                "sigma_ema": self._sigma_ema,
                "num_updates": self._num_updates,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        logger.info("RSSM saved → %s (%d updates)", path, self._num_updates)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.gru_core.load_state_dict(state["gru_core"])
        self.posterior.load_state_dict(state["posterior"])
        self.prior.load_state_dict(state["prior"])
        self.decoder.load_state_dict(state["decoder"])
        self.value_head.load_state_dict(state["value_head"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        meta = state.get("meta", {})
        self._sigma_ema = meta.get("sigma_ema", 1.0)
        self._num_updates = meta.get("num_updates", 0)
        self.free_bits = meta.get("free_bits", 1.0)
        logger.info("RSSM loaded ← %s (%d updates)", path, self._num_updates)

    # ── Internal helpers ──────────────────────────────────────────────────
    def _init_deter(self) -> torch.Tensor:
        return torch.zeros(1, self.deter, device=self.device)

    @staticmethod
    def _sample_gaussian(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def _kl_gaussian(
        mu_q: torch.Tensor, log_std_q: torch.Tensor,
        mu_p: torch.Tensor, log_std_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(q ‖ p) for diagonal Gaussians (raw, no free-bits).
        q: (B, D), p: (B, D). Returns per-element KL summed over D → (B,).
        Kept for backward compatibility / debugging.
        """
        var_q = torch.exp(2.0 * log_std_q)
        var_p = torch.exp(2.0 * log_std_p)
        kl = 0.5 * (
            var_q / var_p
            + (mu_q - mu_p) ** 2 / var_p
            - 1.0
            + torch.log(var_p / var_q)
        )
        return kl.sum(dim=-1)  # (B,)

    def _kl_free_bits(
        self,
        mu_q: torch.Tensor, log_std_q: torch.Tensor,
        mu_p: torch.Tensor, log_std_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(q ‖ p) with free-bits floor (DreamerV3 standard).
        Per-element KL is clamped to at least `free_bits` nats before summing.
        This prevents the posterior from collapsing to the prior in early training
        when the model hasn't learned meaningful structure yet.
        Default free_bits = 1.0 nat per element.
        Returns shape (B,).
        """
        var_q = torch.exp(2.0 * log_std_q)
        var_p = torch.exp(2.0 * log_std_p)
        kl_per_elem = 0.5 * (
            var_q / var_p
            + (mu_q - mu_p) ** 2 / var_p
            - 1.0
            + torch.log(var_p / var_q)
        )
        # Free bits: ignore KL below threshold per element
        kl_clipped = torch.clamp(kl_per_elem, min=self.free_bits)
        return kl_clipped.sum(dim=-1)  # (B,)

    def _all_params(self):
        return (
            list(self.gru_core.parameters())
            + list(self.posterior.parameters())
            + list(self.prior.parameters())
            + list(self.decoder.parameters())
            + list(self.value_head.parameters())
        )
