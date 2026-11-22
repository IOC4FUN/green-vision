"""
Reflection Module for Green Vision.
Post-episode self-analysis: after each episode ends, the agent "thinks back"
on what happened, extracts lessons, and feeds curriculum signals.

Responsibilities:
  • Analyse the just-completed episode step-by-step.
  • Identify action→reward correlations (which moves helped / hurt).
  • Detect repeated failure patterns (same sub-action failing across episodes).
  • Write a structured Reflection record attached to the episode.
  • Track per-scenario success rates → flag weak areas for curriculum.
  • Provide `get_curriculum_hints()` so the planner can prioritise practice.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .episodic_memory import EpisodicMemory, Episode, Outcome

logger = logging.getLogger("brain.reflection")

# ── Data containers ────────────────────────────────────────────────────────
@dataclass
class ActionStat:
    """Aggregated stats for one action within an episode."""
    action_idx: int
    action_name: str
    count: int = 0
    total_reward_after: float = 0.0   # sum of rewards on steps where this action was taken
    avg_reward_after: float = 0.0
    best_step: int = -1               # step index with highest reward after this action
    worst_step: int = -1

@dataclass
class FailurePattern:
    """A repeated failure detected across episodes."""
    description: str
    occurrences: int = 1
    last_seen_ts: float = 0.0
    affected_scenarios: list[str] = field(default_factory=list)

@dataclass
class Reflection:
    """Structured output of a post-episode analysis pass."""
    episode_id: int
    ts: float
    outcome: str                       # success | stall | failure
    total_reward: float
    num_steps: int
    top_positive_actions: list[ActionStat] = field(default_factory=list)
    top_negative_actions: list[ActionStat] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)       # human-readable takeaways
    failure_patterns: list[FailurePattern] = field(default_factory=list)
    scenario_success_rate: Optional[float] = None          # 0..1 for this scenario
    raw_summary: str = ""

# ── Main module ────────────────────────────────────────────────────────────
class ReflectionModule:
    """
    Called after each `end_episode()` to perform self-analysis.

    Usage (in agent.py):
        reflector = ReflectionModule(episodic_mem)
        ep = episodic_mem.end_episode(...)
        if ep is not None:
            reflection = reflector.reflect(ep)
            logger.info("Reflection: %s", reflection.raw_summary[:200])
    """

    def __init__(self, memory: EpisodicMemory):
        self.memory = memory
        self.reflections: list[Reflection] = []
        # Per-scenario tracking: scenario → [successes, total]
        self._scenario_stats: dict[str, list[int]] = {}
        # Cross-episode failure pattern tracker
        self._failure_patterns: dict[str, FailurePattern] = {}

    # ── Core analysis ──────────────────────────────────────────────
    def reflect(self, episode: Episode) -> Reflection:
        """Analyse a completed episode and return a structured Reflection."""
        ts = time.time()
        refl = Reflection(
            episode_id=episode.id,
            ts=ts,
            outcome=episode.outcome.value,
            total_reward=round(episode.total_reward, 4),
            num_steps=len(episode.steps),
        )

        if not episode.steps:
            refl.raw_summary = "Empty episode – no steps recorded."
            self.reflections.append(refl)
            return refl

        # 1. Per-action reward aggregation
        action_stats = self._aggregate_actions(episode)
        sorted_by_reward = sorted(action_stats, key=lambda s: s.avg_reward_after, reverse=True)
        refl.top_positive_actions = sorted_by_reward[:3]
        refl.top_negative_actions = sorted_by_reward[-3:] if len(sorted_by_reward) > 3 else []

        # 2. Extract lessons (heuristic rules)
        refl.lessons = self._extract_lessons(episode, action_stats)

        # 3. Detect failure patterns across episodes
        refl.failure_patterns = self._detect_failure_patterns(episode)

        # 4. Scenario success rate
        scenario = episode.scenario or "unknown"
        if scenario not in self._scenario_stats:
            self._scenario_stats[scenario] = [0, 0]
        self._scenario_stats[scenario][1] += 1
        if episode.outcome == Outcome.SUCCESS:
            self._scenario_stats[scenario][0] += 1
        refl.scenario_success_rate = (
            self._scenario_stats[scenario][0] / max(1, self._scenario_stats[scenario][1])
        )

        # 5. Build human-readable summary
        parts = [
            f"Ep{episode.id} [{refl.outcome}] r={refl.total_reward:+.2f} steps={refl.num_steps}",
        ]
        if refl.top_positive_actions:
            best = refl.top_positive_actions[0]
            parts.append(f"best: {best.action_name}(avg_r={best.avg_reward_after:+.3f})")
        if refl.top_negative_actions and refl.top_negative_actions[0].avg_reward_after < -0.05:
            worst = refl.top_negative_actions[-1]
            parts.append(f"worst: {worst.action_name}(avg_r={worst.avg_reward_after:+.3f})")
        if refl.lessons:
            parts.append(f"lessons: {'; '.join(refl.lessons[:2])}")
        refl.raw_summary = " | ".join(parts)

        self.reflections.append(refl)
        if len(self.reflections) > 200:
            self.reflections = self.reflections[-100:]

        logger.debug("Reflection: %s", refl.raw_summary)
        return refl

    # ── Curriculum hints ───────────────────────────────────────────
    def get_curriculum_hints(self, threshold: float = 0.4) -> list[str]:
        """
        Return scenario names whose success rate is below `threshold`.
        The planner / dashboard can use these to prioritise practice.
        """
        weak: list[str] = []
        for scenario, (succ, total) in self._scenario_stats.items():
            if total < 3:
                continue  # not enough data yet
            rate = succ / total
            if rate < threshold:
                weak.append(scenario)
        return sorted(weak, key=lambda s: self._scenario_stats[s][0] / max(1, self._scenario_stats[s][1]))

    # ── Dashboard / inspection ─────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        total = len(self.reflections)
        successes = sum(1 for r in self.reflections if r.outcome == "success")
        failures = sum(1 for r in self.reflections if r.outcome == "failure")
        return {
            "total_reflections": total,
            "successes": successes,
            "failures": failures,
            "stalls": total - successes - failures,
            "scenario_stats": {
                k: {"successes": v[0], "total": v[1], "rate": round(v[0]/max(1,v[1]), 3)}
                for k, v in self._scenario_stats.items()
            },
            "active_failure_patterns": [
                {"desc": p.description, "count": p.occurrences}
                for p in list(self._failure_patterns.values())[-5:]
            ],
            "recent_5": [
                {"ep_id": r.episode_id, "outcome": r.outcome,
                 "reward": r.total_reward, "summary": r.raw_summary[:120]}
                for r in self.reflections[-5:]
            ],
        }

    # ── Internal helpers ───────────────────────────────────────────
    @staticmethod
    def _aggregate_actions(episode: Episode) -> list[ActionStat]:
        """Compute per-action reward stats within one episode."""
        from collections import defaultdict
        buckets: dict[int, list[float]] = defaultdict(list)
        names: dict[int, str] = {}

        for i, step in enumerate(episode.steps):
            buckets[step.action_idx].append(step.reward)
            names[step.action_idx] = step.action_name

        stats: list[ActionStat] = []
        for idx, rewards in buckets.items():
            s = ActionStat(
                action_idx=idx,
                action_name=names[idx],
                count=len(rewards),
                total_reward_after=sum(rewards),
                avg_reward_after=sum(rewards) / len(rewards),
            )
            if rewards:
                s.best_step = int(np.argmax(rewards))
                s.worst_step = int(np.argmin(rewards))
            stats.append(s)
        return stats

    @staticmethod
    def _extract_lessons(episode: Episode, action_stats: list[ActionStat]) -> list[str]:
        """Heuristic lesson extraction from a single episode."""
        lessons: list[str] = []

        # Rule 1: if the episode failed and there was a long noop streak early on
        noop_count = sum(1 for s in episode.steps if s.action_name == "noop")
        if episode.outcome == Outcome.FAILURE and noop_count > len(episode.steps) * 0.4:
            lessons.append("Too much stalling (noop > 40% of steps) – act more decisively.")

        # Rule 2: identify the worst action that appeared multiple times
        for stat in sorted(action_stats, key=lambda x: x.avg_reward_after):
            if stat.count >= 3 and stat.avg_reward_after < -0.1:
                lessons.append(
                    f"'{stat.action_name}' consistently hurt (avg_r={stat.avg_reward_after:+.3f}, n={stat.count})."
                )
                break

        # Rule 3: if success came after a specific action sequence
        if episode.outcome == Outcome.SUCCESS and len(episode.steps) >= 4:
            last_good = max(episode.steps[-4:], key=lambda s: s.reward)
            if last_good.reward > 0.1:
                lessons.append(
                    f"Late-game '{last_good.action_name}' contributed strongly (r={last_good.reward:+.2f})."
                )

        # Rule 4: crisis severity correlation
        if episode.peak_crisis_severity >= 3:
            lessons.append(f"Peak crisis severity {episode.peak_crisis_severity} was reached – consider earlier intervention.")

        return lessons[:5]  # cap at 5 lessons per episode

    def _detect_failure_patterns(self, episode: Episode) -> list[FailurePattern]:
        """
        Check whether the current episode's failure mode matches any previously seen pattern.
        Pattern key = scenario + dominant negative action.
        """
        patterns: list[FailurePattern] = []
        if episode.outcome != Outcome.FAILURE:
            return patterns

        # Find the most damaging action in this episode
        neg_actions = [s for s in self._aggregate_actions(episode) if s.avg_reward_after < -0.05]
        if not neg_actions:
            return patterns

        worst = min(neg_actions, key=lambda x: x.avg_reward_after)
        key = f"{episode.scenario}:{worst.action_name}"

        if key in self._failure_patterns:
            fp = self._failure_patterns[key]
            fp.occurrences += 1
            fp.last_seen_ts = time.time()
            if episode.scenario and episode.scenario not in fp.affected_scenarios:
                fp.affected_scenarios.append(episode.scenario)
        else:
            fp = FailurePattern(
                description=f"Repeated failure in '{episode.scenario}' via '{worst.action_name}'",
                occurrences=1,
                last_seen_ts=time.time(),
                affected_scenarios=[episode.scenario] if episode.scenario else [],
            )
            self._failure_patterns[key] = fp

        patterns.append(fp)
        return patterns
