"""
Adaptive Curriculum Manager for Green Vision.

Uses ReflectionModule hints (per-scenario success rates, failure patterns)
to dynamically adjust which scenarios the agent practises next and how hard
they are. The idea: spend more time on weak areas before moving on.

Responsibilities:
  • Maintain a weighted scenario queue (weak scenarios get higher weight).
  • Adjust difficulty knobs (starting budget deficit, crisis frequency,
    opponent aggression) based on recent performance.
  • Provide `next_scenario()` so the game controller / training harness can
    pick what to play next.
  • Log curriculum decisions for post-hoc analysis.
"""
from __future__ import annotations
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("brain.curriculum")

# ── Scenario definition ────────────────────────────────────────────────────
@dataclass
class ScenarioConfig:
    """One playable scenario with tunable difficulty."""
    name: str                    # e.g. "Germany/government", "Canada/government"
    country: str = "Germany"
    role: str = "government"
    starting_budget_deficit: float = -5.0   # % of GDP negative
    crisis_frequency_mult: float = 1.0      # multiplier on crisis spawn rate
    opponent_aggression: float = 0.3       # 0..1 how aggressive AI neighbours are
    max_years: int = 4                     # episode length in game-years
    notes: str = ""

# Default scenario pool (mirrors COUNTRY_TIERS in game_knowledge.py)
DEFAULT_SCENARIOS: list[ScenarioConfig] = [
    ScenarioConfig(name="Canada/government", country="Canada",
                   starting_budget_deficit=-2.0, crisis_frequency_mult=0.7,
                   opponent_aggression=0.2, notes="Beginner – fiscal focus"),
    ScenarioConfig(name="Australia/government", country="Australia",
                   starting_budget_deficit=-3.0, crisis_frequency_mult=0.8,
                   opponent_aggression=0.25, notes="Trade & export policy"),
    ScenarioConfig(name="Germany/government", country="Germany",
                   starting_budget_deficit=-5.0, crisis_frequency_mult=1.0,
                   opponent_aggression=0.35, notes="Coalition politics + energy"),
    ScenarioConfig(name="France/government", country="France",
                   starting_budget_deficit=-6.0, crisis_frequency_mult=1.1,
                   opponent_aggression=0.4, notes="Social unrest risk"),
    ScenarioConfig(name="Brazil/government", country="Brazil",
                   starting_budget_deficit=-8.0, crisis_frequency_mult=1.3,
                   opponent_aggression=0.5, notes="Inflation + inequality"),
    ScenarioConfig(name="India/government", country="India",
                   starting_budget_deficit=-7.0, crisis_frequency_mult=1.4,
                   opponent_aggression=0.45, notes="Scale + diversity challenges"),
]

# ── Curriculum state ───────────────────────────────────────────────────────
@dataclass
class CurriculumState:
    """Tracks the curriculum's internal bookkeeping."""
    scenario_weights: dict[str, float] = field(default_factory=dict)
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)  # last N
    difficulty_level: int = 1          # 1=easiest … 5=hardest
    total_episodes_run: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

# ── Main module ────────────────────────────────────────────────────────────
class CurriculumManager:
    """
    Adaptive difficulty & scenario selection.

    Usage (in agent.py or training harness):
        curriculum = CurriculumManager(reflector)
        scenario = curriculum.next_scenario()
        # ... play episode with scenario params ...
        curriculum.record_result(scenario.name, outcome_str, reward)
    """

    def __init__(self, reflector=None, scenarios: list[ScenarioConfig] | None = None,
                 min_episodes_before_promote: int = 5,
                 success_threshold: float = 0.6):
        self.reflector = reflector  # ReflectionModule (optional)
        self.scenarios: list[ScenarioConfig] = scenarios or list(DEFAULT_SCENARIOS)
        self.min_episodes_before_promote = min_episodes_before_promote
        self.success_threshold = success_threshold
        self.state = CurriculumState()
        # Initialise weights uniformly
        for s in self.scenarios:
            self.state.scenario_weights[s.name] = 1.0

    # ── Core API ───────────────────────────────────────────────────────
    def next_scenario(self) -> ScenarioConfig:
        """
        Pick the next scenario to practise using weighted random sampling.
        Weak scenarios (low success rate) get boosted weight.
        """
        names = [s.name for s in self.scenarios]
        weights = [max(0.1, self.state.scenario_weights.get(n, 1.0)) for n in names]
        chosen_name = random.choices(names, weights=weights, k=1)[0]
        scenario = next(s for s in self.scenarios if s.name == chosen_name)

        # Apply current difficulty scaling
        diff_mult = 1.0 + (self.state.difficulty_level - 1) * 0.25
        scaled = ScenarioConfig(
            name=scenario.name,
            country=scenario.country,
            role=scenario.role,
            starting_budget_deficit=round(scenario.starting_budget_deficit * diff_mult, 1),
            crisis_frequency_mult=round(scenario.crisis_frequency_mult * diff_mult, 2),
            opponent_aggression=min(1.0, round(scenario.opponent_aggression * diff_mult, 2)),
            max_years=scenario.max_years,
            notes=f"{scenario.notes} | diff={self.state.difficulty_level}",
        )
        logger.info("Curriculum → %s (diff=%d, deficit=%.1f%%, crisis×%.2f)",
                    scaled.name, self.state.difficulty_level,
                    scaled.starting_budget_deficit, scaled.crisis_frequency_mult)
        return scaled

    def record_result(self, scenario_name: str, outcome: str, reward: float) -> None:
        """
        Called after each episode completes.
        Updates weights and potentially adjusts difficulty level.
        """
        self.state.total_episodes_run += 1
        entry = {
            "ts": time.time(),
            "scenario": scenario_name,
            "outcome": outcome,       # success | stall | failure
            "reward": round(reward, 3),
            "difficulty": self.state.difficulty_level,
        }
        self.state.recent_outcomes.append(entry)
        if len(self.state.recent_outcomes) > 50:
            self.state.recent_outcomes = self.state.recent_outcomes[-50:]

        # Update weight for this scenario
        w = self.state.scenario_weights.get(scenario_name, 1.0)
        if outcome == "failure":
            w *= 1.4   # boost weak scenarios
        elif outcome == "stall":
            w *= 1.1
        else:  # success
            w *= 0.85  # reduce over-practised strong scenarios
        self.state.scenario_weights[scenario_name] = max(0.1, min(5.0, w))

        # Check if we should promote / demote difficulty level
        self._maybe_adjust_difficulty()

    def get_status(self) -> dict[str, Any]:
        """Dashboard-friendly summary."""
        return {
            "total_episodes": self.state.total_episodes_run,
            "difficulty_level": self.state.difficulty_level,
            "scenario_weights": {k: round(v, 3) for k, v in self.state.scenario_weights.items()},
            "recent_5": self.state.recent_outcomes[-5:],
            "reflector_hints": self.reflector.get_curriculum_hints() if self.reflector else [],
        }

    # ── Internal ───────────────────────────────────────────────────────
    def _maybe_adjust_difficulty(self) -> None:
        """Promote or demote difficulty based on recent performance window."""
        window = self.state.recent_outcomes[-self.min_episodes_before_promote:]
        if len(window) < self.min_episodes_before_promote:
            return
        success_rate = sum(1 for e in window if e["outcome"] == "success") / len(window)

        if success_rate >= self.success_threshold and self.state.difficulty_level < 5:
            self.state.difficulty_level += 1
            logger.info("Curriculum PROMOTED to difficulty %d (rate=%.2f)",
                        self.state.difficulty_level, success_rate)
        elif success_rate < 0.3 and self.state.difficulty_level > 1:
            self.state.difficulty_level -= 1
            logger.warning("Curriculum DEMOTED to difficulty %d (rate=%.2f)",
                           self.state.difficulty_level, success_rate)
