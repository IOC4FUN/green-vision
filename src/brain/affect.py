"""
Affective State Module – Emotional Modulation for Green Vision.
A lightweight "emotion engine" that tracks the agent's internal arousal /
confidence level based on recent performance and uses it to modulate
behavioural parameters (exploration rate, risk tolerance, goal priority).
This is NOT a full LLM-based emotion model; it's a compact state machine
with continuous variables inspired by the Homeostatic Artificial Emotions
framework (de Haan & van den Herik, 2003).
State variables (all in [0, 1]):
    confidence   – how sure the agent is about its current strategy.
    frustration  – builds up on repeated failures / stalls.
    excitement   – spikes on novel successes or big positive rewards.
Derived outputs:
    exploration_bonus : added to entropy coefficient when frustrated.
    risk_multiplier   : scales down aggressive actions when low-confidence.
    urgency           : boosts crisis-response goal priority when excited/frustrated.
Usage:
    affect = AffectModule()
    affect.update(reward=+0.8, action_was_risky=True)
    ent_bonus = affect.exploration_bonus()   # add to PPO entropy coef
    risk_scale = affect.risk_multiplier()    # multiply into risky-action logits
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
logger = logging.getLogger("brain.affect")
@dataclass
class AffectSnapshot:
    """Point-in-time emotional state for journaling."""
    ts: float
    confidence: float
    frustration: float
    excitement: float
    exploration_bonus: float
    risk_multiplier: float
    urgency: float
class AffectModule:
    """
    Tracks and modulates the agent's behavioural parameters via a simple
    continuous emotion model.
    Parameters
    ----------
    decay_rate : per-step exponential decay toward neutral (0.5).
    success_boost / failure_penalty : magnitude of state shifts.
    window : number of recent rewards to consider for trend detection.
    """
    def __init__(
        self,
        decay_rate: float = 0.97,
        success_boost: float = 0.12,
        failure_penalty: float = 0.18,
        window: int = 20,
    ):
        self.decay_rate = decay_rate
        self.success_boost = success_boost
        self.failure_penalty = failure_penalty
        self.window = window
        # Core affective states – start at neutral
        self.confidence: float = 0.5
        self.frustration: float = 0.3
        self.excitement: float = 0.2
        # History for trend detection
        self._recent_rewards: list[float] = []
        self._consecutive_failures: int = 0
        self._total_updates: int = 0
        logger.info("AffectModule initialised")
    # ── Update ───────────────────────────────────────────────────────
    def update(
        self,
        reward: float,
        action_was_risky: bool = False,
        done: bool = False,
    ) -> None:
        """
        Call once per decision step with the received reward.
        Adjusts internal states based on outcome and history.
        """
        self._recent_rewards.append(reward)
        if len(self._recent_rewards) > self.window:
            self._recent_rewards = self._recent_rewards[-self.window:]
        # Decay toward neutral
        self.confidence += (0.5 - self.confidence) * (1 - self.decay_rate)
        self.frustration += (0.3 - self.frustration) * (1 - self.decay_rate)
        self.excitement += (0.2 - self.excitement) * (1 - self.decay_rate)
        # Reward-driven shifts
        if reward > 0.1:
            self.confidence = min(1.0, self.confidence + self.success_boost)
            self.excitement = min(1.0, self.excitement + self.success_boost * 0.8)
            self.frustration = max(0.0, self.frustration - self.success_boost * 0.5)
            self._consecutive_failures = 0
        elif reward < -0.1:
            self.frustration = min(1.0, self.frustration + self.failure_penalty)
            self.confidence = max(0.0, self.confidence - self.failure_penalty * 0.6)
            self.excitement = max(0.0, self.excitement - self.failure_penalty * 0.4)
            self._consecutive_failures += 1
        else:
            # Neutral: slight confidence drift based on trend
            if len(self._recent_rewards) >= 3:
                recent_avg = sum(self._recent_rewards[-3:]) / 3
                if recent_avg > 0:
                    self.confidence = min(1.0, self.confidence + 0.02)
                elif recent_avg < 0:
                    self.frustration = min(1.0, self.frustration + 0.03)
        # Risky action penalty when already low-confidence
        if action_was_risky and self.confidence < 0.3:
            self.frustration = min(1.0, self.frustration + 0.05)
        # Episode end: partial reset toward neutral (faster decay)
        if done:
            self.confidence = 0.5 * 0.7 + self.confidence * 0.3
            self.frustration = 0.3 * 0.7 + self.frustration * 0.3
            self.excitement = 0.2 * 0.7 + self.excitement * 0.3
            self._consecutive_failures = 0
        self._total_updates += 1
    # ── Derived behavioural modifiers ────────────────────────────────
    def exploration_bonus(self) -> float:
        """
        Extra entropy coefficient to add on top of the PPO schedule.
        Higher frustration → more exploration (agent is stuck).
        Range: [0, 0.03]
        """
        return round(self.frustration * 0.03, 5)
    def risk_multiplier(self) -> float:
        """
        Multiplier applied to logits of risky actions (declare_war, call_election,
        break_alliance, etc.). Low confidence → suppress risk.
        Range: [0.4, 1.0]
        """
        base = 0.4 + 0.6 * self.confidence
        # Excitement can override some caution
        if self.excitement > 0.7:
            base = min(1.0, base + 0.1)
        return round(base, 3)
    def urgency(self) -> float:
        """
        Boost factor for crisis-response goal priority.
        High frustration or excitement → act faster on crises.
        Range: [1.0, 2.0] (multiply into priority weight)
        """
        boost = max(self.frustration, self.excitement)
        return round(1.0 + boost, 3)
    def should_take_break(self) -> bool:
        """
        If frustration is very high and confidence very low, the agent should
        pause / reset strategy rather than keep grinding.
        """
        return self.frustration > 0.85 and self.confidence < 0.2
    # ── Snapshot & Dashboard ─────────────────────────────────────────
    def snapshot(self) -> AffectSnapshot:
        return AffectSnapshot(
            ts=time.time(),
            confidence=round(self.confidence, 4),
            frustration=round(self.frustration, 4),
            excitement=round(self.excitement, 4),
            exploration_bonus=self.exploration_bonus(),
            risk_multiplier=self.risk_multiplier(),
            urgency=self.urgency(),
        )
    def summary(self) -> dict[str, float]:
        return {
            "confidence": round(self.confidence, 4),
            "frustration": round(self.frustration, 4),
            "excitement": round(self.excitement, 4),
            "consecutive_failures": self._consecutive_failures,
            "total_updates": self._total_updates,
            "exploration_bonus": self.exploration_bonus(),
            "risk_multiplier": self.risk_multiplier(),
            "urgency": self.urgency(),
        }
"""
