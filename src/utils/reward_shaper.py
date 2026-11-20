"""
Reward shaper — externalises reward weights into settings.yaml.

Instead of hard-coding coefficients inside _compute_reward(), the agent
reads a `reward_weights` block from config and applies it here.

Example YAML:
    reward_weights:
      approval_delta: 1.0
      budget_health: 0.8
      stability: 0.6
      gdp_growth: 0.4
      unemployment_penalty: -0.5
      inflation_penalty: -0.3
      war_risk: -2.0
      noop_penalty: -0.1

> Keep in sync with code — update this file whenever the module API changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("utils.reward")

# Sensible defaults if no config is provided.
_DEFAULTS: dict[str, float] = {
    "approval_delta": 1.0,
    "budget_health": 0.8,
    "stability": 0.6,
    "gdp_growth": 0.4,
    "unemployment_penalty": -0.5,
    "inflation_penalty": -0.3,
    "war_risk": -2.0,
    "noop_penalty": -0.1,
}


@dataclass
class RewardWeights:
    approval_delta: float = _DEFAULTS["approval_delta"]
    budget_health: float = _DEFAULTS["budget_health"]
    stability: float = _DEFAULTS["stability"]
    gdp_growth: float = _DEFAULTS["gdp_growth"]
    unemployment_penalty: float = _DEFAULTS["unemployment_penalty"]
    inflation_penalty: float = _DEFAULTS["inflation_penalty"]
    war_risk: float = _DEFAULTS["war_risk"]
    noop_penalty: float = _DEFAULTS["noop_penalty"]
    extra: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RewardWeights":
        raw = cfg.get("reward_weights", {}) or {}
        w = cls()
        for k, v in raw.items():
            if hasattr(w, k):
                setattr(w, k, float(v))
            else:
                w.extra[k] = float(v)
        logger.debug("Reward weights loaded: %s", vars(w))
        return w


def compute_reward(
    weights: RewardWeights,
    prev_state: dict[str, float],
    curr_state: dict[str, float],
    action_was_noop: bool = False,
) -> float:
    """
    Compute a scalar reward from state deltas.

    All values are expected on a 0-100 scale (matching memory_vars).
    Returns a single float; the PPO agent normalises internally.
    """
    r = 0.0

    # Approval change (positive delta is good)
    prev_app = prev_state.get("approval_rating", 50.0)
    curr_app = curr_state.get("approval_rating", 50.0)
    r += weights.approval_delta * (curr_app - prev_app) / 10.0

    # Budget health: closer to 50 (balanced) is better
    budget = curr_state.get("budget_balance", 50.0)
    r += weights.budget_health * (1.0 - abs(budget - 50.0) / 50.0)

    # Stability
    stab = curr_state.get("stability_index", 50.0)
    r += weights.stability * (stab / 100.0 - 0.5)

    # GDP growth (positive is good)
    gdp = curr_state.get("gdp_growth", 0.0)
    r += weights.gdp_growth * max(0.0, min(gdp, 20.0)) / 20.0

    # Unemployment penalty (higher is worse)
    unemp = curr_state.get("unemployment", 5.0)
    r += weights.unemployment_penalty * (unemp / 20.0)

    # Inflation penalty
    infl = curr_state.get("inflation", 2.0)
    r += weights.inflation_penalty * (infl / 10.0)

    # War risk: if war_support spiked, heavy penalty
    war_sup = curr_state.get("war_support", 0.0)
    prev_war = prev_state.get("war_support", 0.0)
    if war_sup > prev_war + 10:
        r += weights.war_risk

    # Noop penalty to discourage stalling
    if action_was_noop:
        r += weights.noop_penalty

    return float(r)
