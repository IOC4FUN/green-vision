"""
Episodic Memory for Green Vision.

Stores completed episodes (sequences of decisions + outcomes) so the agent
can:
  • Retrieve similar past situations to inform planning (few-shot analogy).
  • Provide context to the reward shaper (was this situation good before?).
  • Feed curriculum signals to PPO (which scenarios need more practice?).
  • Log structured experience for post-hoc analysis / dashboard display.

Design:
  - Each Episode = list of (state_vec, action_idx, reward) tuples + metadata.
  - Similarity retrieval uses cosine distance on the mean state vector.
  - Episodes are tagged by outcome class: success | stall | failure.
  - Bounded capacity (LRU eviction) to keep RAM bounded.
"""
from __future__ import annotations
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("brain.episodic")

# ── Outcome classification ─────────────────────────────────────────────────
class Outcome(Enum):
    SUCCESS = "success"      # net positive reward over episode
    STALL = "stall"          # roughly neutral
    FAILURE = "failure"      # net negative / crisis triggered

@dataclass
class StepRecord:
    """One decision step within an episode."""
    ts: float
    state: np.ndarray       # (STATE_DIM,) float32
    action_idx: int
    action_name: str
    reward: float
    thought: str = ""

@dataclass
class Episode:
    id: int
    start_ts: float
    end_ts: float
    steps: list[StepRecord] = field(default_factory=list)
    outcome: Outcome = Outcome.STALL
    total_reward: float = 0.0
    tags: list[str] = field(default_factory=list)   # e.g. ["economy", "crisis"]
    scenario: str = ""
    peak_crisis_severity: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def mean_state(self) -> np.ndarray:
        if not self.steps:
            return np.zeros(128, dtype=np.float32)
        states = np.stack([s.state for s in self.steps])
        return states.mean(axis=0).astype(np.float32)

# ── Similarity helper ──────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (higher = more similar)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ── Main memory store ──────────────────────────────────────────────────────
class EpisodicMemory:
    """
    Bounded episodic memory with similarity-based retrieval.

    Usage:
        mem = EpisodicMemory(capacity=500)
        # During play:
        mem.begin_episode(scenario="Germany/government")
        mem.record_step(state_vec, action_idx, action_name, reward, thought)
        mem.end_episode(outcome=Outcome.SUCCESS, tags=["economy"])

        # Before deciding:
        similar = mem.retrieve_similar(current_state_vec, k=3)
        for ep in similar:
            logger.info("Similar past: %s (reward=%.2f)", ep.tags, ep.total_reward)
    """
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self._episodes: deque[Episode] = deque(maxlen=capacity)
        self._current: Optional[Episode] = None
        self._counter = 0

    # ── Episode lifecycle ──────────────────────────────────────────
    def begin_episode(self, scenario: str = "") -> None:
        if self._current is not None:
            # Auto-close previous episode if still open
            self.end_episode()
        self._counter += 1
        self._current = Episode(
            id=self._counter,
            start_ts=time.time(),
            end_ts=time.time(),
            scenario=scenario,
        )

    def record_step(
        self,
        state: np.ndarray,
        action_idx: int,
        action_name: str,
        reward: float,
        thought: str = "",
    ) -> None:
        if self._current is None:
            return
        rec = StepRecord(
            ts=time.time(),
            state=state.copy(),
            action_idx=action_idx,
            action_name=action_name,
            reward=reward,
            thought=thought[:200],  # cap length
        )
        self._current.steps.append(rec)
        self._current.total_reward += reward

    def end_episode(
        self,
        outcome: Outcome | None = None,
        tags: list[str] | None = None,
        peak_crisis_severity: int = 0,
    ) -> Optional[Episode]:
        if self._current is None:
            return None
        ep = self._current
        ep.end_ts = time.time()
        if outcome is not None:
            ep.outcome = outcome
        else:
            # Auto-classify from total reward
            if ep.total_reward > 1.0:
                ep.outcome = Outcome.SUCCESS
            elif ep.total_reward < -1.0:
                ep.outcome = Outcome.FAILURE
            else:
                ep.outcome = Outcome.STALL
        if tags:
            ep.tags.extend(tags)
        ep.peak_crisis_severity = peak_crisis_severity
        self._episodes.append(ep)
        logger.debug(
            "Episode %d closed: outcome=%s reward=%.2f steps=%d",
            ep.id, ep.outcome.value, ep.total_reward, len(ep.steps),
        )
        self._current = None
        return ep

    # ── Retrieval ──────────────────────────────────────────────────
    def retrieve_similar(self, state_vec: np.ndarray, k: int = 3) -> list[tuple[Episode, float]]:
        """
        Return up to k episodes most similar to the given state vector,
        sorted by cosine similarity (descending).
        Each result is (episode, similarity_score).
        """
        if not self._episodes:
            return []
        scored: list[tuple[Episode, float]] = []
        for ep in self._episodes:
            sim = cosine_similarity(state_vec, ep.mean_state)
            scored.append((ep, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def retrieve_by_outcome(self, outcome: Outcome, k: int = 5) -> list[Episode]:
        """Return up to k recent episodes with a specific outcome."""
        matches = [ep for ep in self._episodes if ep.outcome == outcome]
        return matches[-k:]  # most recent first

    def retrieve_failures(self, k: int = 5) -> list[Episode]:
        """Shortcut: get recent failure episodes for replanning context."""
        return self.retrieve_by_outcome(Outcome.FAILURE, k=k)

    # ── Statistics / dashboard ─────────────────────────────────────
    @property
    def size(self) -> int:
        return len(self._episodes)

    def summary(self) -> dict[str, Any]:
        total = len(self._episodes)
        successes = sum(1 for e in self._episodes if e.outcome == Outcome.SUCCESS)
        failures = sum(1 for e in self._episodes if e.outcome == Outcome.FAILURE)
        stalls = total - successes - failures
        avg_reward = (
            sum(e.total_reward for e in self._episodes) / max(1, total)
        )
        return {
            "total_episodes": total,
            "successes": successes,
            "failures": failures,
            "stalls": stalls,
            "avg_total_reward": round(avg_reward, 3),
            "recent_5": [
                {"id": e.id, "outcome": e.outcome.value,
                 "reward": round(e.total_reward, 2),
                 "steps": len(e.steps), "tags": e.tags}
                for e in list(self._episodes)[-5:]
            ],
        }

    # ── Persistence (simple JSONL append) ──────────────────────────
    def save_to_file(self, path: str) -> None:
        """Append all episodes to a JSONL file (one line per episode)."""
        import json
        from pathlib import Path as P
        p = P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for ep in self._episodes:
                row = {
                    "id": ep.id,
                    "start_ts": ep.start_ts,
                    "end_ts": ep.end_ts,
                    "outcome": ep.outcome.value,
                    "total_reward": round(ep.total_reward, 4),
                    "num_steps": len(ep.steps),
                    "tags": ep.tags,
                    "scenario": ep.scenario,
                    "peak_crisis": ep.peak_crisis_severity,
                }
                f.write(json.dumps(row) + "\n")
