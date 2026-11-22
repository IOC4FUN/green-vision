"""
Gymnasium environment wrapper for Green Vision / Geoplanetical.

Exposes the 128-dim state vector as observation space and 31 discrete actions.
Uses the project's existing reward_shaper for consistent reward computation.

Two operating modes:
  - "live":       delegates to a running GreenVisionAgent instance
  - "simulated":  random-walk state model for offline testing / CI
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# Project imports (relative from src/)
from ..utils.state_schema import STATE_DIM, build_vector, MEMORY_KEYS
from ..utils.reward_shaper import RewardWeights, compute_reward
from ..brain.network import NUM_ACTIONS, ACTION_NAMES

logger = logging.getLogger("gym_env.geoplanetical")


class GeoplaneticalEnv(gym.Env):
    """
    Gymnasium-compatible environment wrapping the Green Vision agent's
    interaction loop with Geoplanetical.

    Observation: Box(low=0, high=1, shape=(128,), dtype=float32)
    Action:      Discrete(31)

    Parameters
    ----------
    mode : str
        "live" – delegate actions to a real agent + game window.
        "simulated" – internal random-walk model (no game required).
    agent : object | None
        A running ``GreenVisionAgent`` instance (required when mode="live").
    reward_weights : RewardWeights | None
        Override default reward coefficients; loaded from config if omitted.
    max_steps : int
        Episode length cap (truncation threshold). Default 10 000.
    render_mode : str | None
        "human", "rgb_array", or None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        mode: str = "simulated",
        agent: Optional[Any] = None,
        reward_weights: Optional[RewardWeights] = None,
        max_steps: int = 10_000,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if mode not in ("live", "simulated"):
            raise ValueError(f"mode must be 'live' or 'simulated', got {mode!r}")
        if mode == "live" and agent is None:
            raise ValueError("agent= is required when mode='live'")

        self.mode = mode
        self.agent = agent
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = Discrete(NUM_ACTIONS)

        # Reward shaping – reuse project module for consistency
        self.reward_weights = reward_weights or RewardWeights()

        # Internal bookkeeping
        self._prev_state_dict: dict[str, float] = {}
        self._curr_state_dict: dict[str, float] = {}
        self._step_count: int = 0
        self._terminated: bool = False
        self._truncated: bool = False

        logger.info(
            "GeoplaneticalEnv initialised (mode=%s, obs_dim=%d, actions=%d)",
            mode, STATE_DIM, NUM_ACTIONS,
        )

    # ── Gymnasium API ───────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        if self.mode == "live" and self.agent is not None:
            obs_vec = np.asarray(self.agent.state_vector, dtype=np.float32).copy()
        else:
            obs_vec = self._initial_simulated_state()

        self._prev_state_dict = self._vec_to_dict(obs_vec)
        self._curr_state_dict = dict(self._prev_state_dict)
        return obs_vec, {}

    def step(self, action: int):
        """Execute one environment step. Returns (obs, reward, terminated, truncated, info)."""
        if self._terminated or self._truncated:
            raise RuntimeError("Env is done; call reset() first.")

        self._step_count += 1

        # Obtain next observation
        if self.mode == "live" and self.agent is not None:
            self.agent.perform_action(int(action))
            obs_vec = np.asarray(self.agent.state_vector, dtype=np.float32).copy()
        else:
            obs_vec = self._simulate_step(int(action))

        self._curr_state_dict = self._vec_to_dict(obs_vec)

        # Reward via project's shaper
        was_noop = (int(action) == 0)  # ACTION_NAMES[0] == "noop"
        reward = compute_reward(
            self.reward_weights,
            self._prev_state_dict,
            self._curr_state_dict,
            action_was_noop=was_noop,
        )

        # Advance bookkeeping
        self._prev_state_dict = dict(self._curr_state_dict)
        self._terminated = False
        self._truncated = self._step_count >= self.max_steps

        info: dict[str, Any] = {
            "action_name": ACTION_NAMES[int(action)],
            "step": self._step_count,
        }
        return obs_vec, float(reward), self._terminated, self._truncated, info

    def render(self):
        """Render the current state."""
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            print(f"[Step {self._step_count}] State summary:")
            for k in MEMORY_KEYS:
                v = self._curr_state_dict.get(k, 0.0)
                print(f"  {k:>20s}: {v:.1f}")
            return None
        if self.render_mode == "rgb_array":
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            for i, key in enumerate(MEMORY_KEYS):
                val = int(np.clip(self._curr_state_dict.get(key, 0.0), 0, 255))
                row_start = i * 7
                img[row_start : row_start + 6, :, :] = val
            return img
        return None

    def close(self) -> None:
        """Release resources held by the environment."""
        if self.mode == "live" and self.agent is not None:
            try:
                self.agent.shutdown()
            except Exception:  # noqa: BLE001
                logger.warning("agent.shutdown() raised; ignoring.", exc_info=True)

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _vec_to_dict(vec: np.ndarray) -> dict[str, float]:
        """Extract named memory_vars from the state vector (reverse /100 norm)."""
        d: dict[str, float] = {}
        for i, key in enumerate(MEMORY_KEYS):
            d[key] = float(vec[32 + i]) * 100.0
        return d

    def _initial_simulated_state(self) -> np.ndarray:
        """Build a neutral starting state for simulated mode."""
        obs = build_vector()
        defaults = {
            "budget_balance": 50.0,
            "approval_rating": 50.0,
            "gdp_growth": 1.0,
            "unemployment": 5.0,
            "inflation": 2.0,
            "military_strength": 50.0,
            "army_loyalty": 70.0,
            "war_support": 0.0,
            "stability_index": 60.0,
        }
        for i, key in enumerate(MEMORY_KEYS):
            obs[32 + i] = defaults[key] / 100.0
        return obs

    def _simulate_step(self, action: int) -> np.ndarray:
        """Random-walk simulation with small action-specific effects."""
        rng = self.np_random  # Gymnasium-managed RNG (reproducible via seed)
        obs = build_vector()

        prev_vals = np.array(
            [self._prev_state_dict.get(k, 50.0) / 100.0 for k in MEMORY_KEYS],
            dtype=np.float32,
        )
        noise = rng.normal(0, 0.01, size=len(MEMORY_KEYS)).astype(np.float32)
        new_vals = np.clip(prev_vals + noise, 0.0, 1.0)

        # Simplified action effects on specific memory keys
        idx_budget = MEMORY_KEYS.index("budget_balance")       # 0
        idx_approval = MEMORY_KEYS.index("approval_rating")     # 1
        idx_stability = MEMORY_KEYS.index("stability_index")    # 8

        if action == ACTION_NAMES.index("increase_tax"):
            new_vals[idx_budget] -= 0.02
        elif action == ACTION_NAMES.index("decrease_tax"):
            new_vals[idx_budget] += 0.01
        elif action == ACTION_NAMES.index("apply_policy"):
            new_vals[idx_approval] += 0.01
            new_vals[idx_stability] += 0.01
        elif action == ACTION_NAMES.index("declare_war"):
            new_vals[MEMORY_KEYS.index("war_support")] += 0.05
            new_vals[idx_stability] -= 0.03

        obs[32 : 32 + len(MEMORY_KEYS)] = new_vals

        # Time encoding segment [64:80]
        t = self._step_count * 0.1
        obs[64] = float(np.sin(t)) * 0.5 + 0.5
        obs[65] = float(np.cos(t)) * 0.5 + 0.5

        return obs
