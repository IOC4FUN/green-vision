"""
Autonomous Planner for Green Vision (v2).
Sits above the raw PPO policy and provides:
  • Goal management with multi-step decomposition.
  • **Dynamic goal generation**: auto-creates goals from crisis detection
    and game-state analysis when no active goal matches the situation.
  • **Failure detection & replanning**: tracks consecutive failed sub-actions;
    abandons stuck goals and generates alternative approaches.
  • Contextual reasoning: combines NN output + rule-based heuristics +
    knowledge-base lookups + episodic memory analogies.
  • Action sequencing with timeout-based advancement.
  • Safety checks via GameKnowledge.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger("brain.planner")

# ── Enums & data classes ───────────────────────────────────────────────────
class GoalPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class SubAction:
    action_idx: int
    description: str
    done: bool = False
    max_wait_s: float = 5.0
    started_at: float = 0.0
    attempts: int = 0       # how many times this step has been attempted
    last_failure_ts: float = 0.0

@dataclass
class Goal:
    id: str
    description: str
    priority: GoalPriority
    created_at: float
    steps_total: int = 1
    steps_done: int = 0
    status: str = "active"   # active | blocked | completed | abandoned
    context: dict[str, Any] = field(default_factory=dict)
    sub_actions: list[SubAction] = field(default_factory=list)
    current_step: int = 0
    failure_count: int = 0   # consecutive failures across all steps
    max_failures_before_abandon: int = 6

    @property
    def progress(self) -> float:
        return self.steps_done / max(1, self.steps_total)

    @property
    def is_multi_step(self) -> bool:
        return len(self.sub_actions) > 1

    def next_sub_action(self) -> Optional[SubAction]:
        if self.current_step < len(self.sub_actions):
            return self.sub_actions[self.current_step]
        return None

    def advance(self) -> None:
        if self.current_step < len(self.sub_actions):
            self.sub_actions[self.current_step].done = True
            self.current_step += 1
            self.steps_done = self.current_step
            self.failure_count = 0  # reset on successful advance
            if self.current_step >= len(self.sub_actions):
                self.status = "completed"

    def record_failure(self) -> None:
        """Called when a sub-action's expected effect was NOT observed."""
        sa = self.next_sub_action()
        if sa is not None:
            sa.attempts += 1
            sa.last_failure_ts = time.time()
        self.failure_count += 1
        if self.failure_count >= self.max_failures_before_abandon:
            self.status = "abandoned"
            logger.warning("Goal %s ABANDONED after %d consecutive failures: %s",
                           self.id, self.failure_count, self.description)

@dataclass
class Thought:
    timestamp: float
    observation_summary: str
    reasoning: str
    decision: str
    confidence: float
    goal_id: Optional[str] = None
    memory_analogies: list[str] = field(default_factory=list)

# ── Goal templates ─────────────────────────────────────────────────────────
GOAL_TEMPLATES: dict[str, list[tuple[int, str]]] = {
    "stabilize_economy": [
        (5,  "Navigate to budget screen"),
        (12, "Increase tax slightly"),
        (10, "Apply policy"),
        (9,  "Check economy overview"),
    ],
    "boost_approval": [
        (6,  "Navigate to legislation"),
        (14, "Increase spending (social)"),
        (10, "Apply policy"),
        (7,  "Check diplomacy for goodwill"),
    ],
    "prepare_military": [
        (8,  "Navigate to military"),
        (14, "Increase defence spending"),
        (10, "Apply policy"),
        (20, "Form alliance"),
    ],
    "pass_legislation": [
        (6,  "Navigate to legislation"),
        (16, "File bill"),
        (10, "Apply / confirm"),
    ],
    "seek_peace_treaty": [
        (7,  "Navigate to diplomacy"),
        (19, "Seek peace"),
        (10, "Confirm treaty"),
    ],
    # ── New templates for dynamic generation ───────────────────────
    "reduce_inflation": [
        (5,  "Navigate to budget screen"),
        (13, "Decrease tax (stimulus off)"),
        (15, "Decrease spending"),
        (10, "Apply policy"),
    ],
    "cut_deficit": [
        (5,  "Navigate to budget screen"),
        (12, "Increase tax"),
        (15, "Decrease non-essential spending"),
        (10, "Apply policy"),
        (9,  "Verify economy overview"),
    ],
    "boost_military_loyalty": [
        (8,  "Navigate to military"),
        (14, "Increase defence spending"),
        (10, "Apply policy"),
        (23, "Appoint loyal minister"),
    ],
    "diplomatic_reset": [
        (7,  "Navigate to diplomacy"),
        (20, "Form alliance with neighbour"),
        (10, "Confirm"),
        (19, "Seek peace if at war"),
    ],
}

# ── Crisis → goal template mapping (for dynamic generation) ────────────────
CRISIS_TO_GOAL: dict[str, tuple[str, GoalPriority]] = {
    "economy_collapse":   ("stabilize_economy", GoalPriority.CRITICAL),
    "high_inflation":     ("reduce_inflation", GoalPriority.HIGH),
    "budget_deficit":     ("cut_deficit", GoalPriority.HIGH),
    "low_approval":       ("boost_approval", GoalPriority.HIGH),
    "military_threat":    ("prepare_military", GoalPriority.CRITICAL),
    "low_army_loyalty":   ("boost_military_loyalty", GoalPriority.HIGH),
    "war_active":         ("seek_peace_treaty", GoalPriority.MEDIUM),
    "diplomatic_isolation": ("diplomatic_reset", GoalPriority.MEDIUM),
}

# ── Planner ────────────────────────────────────────────────────────────────
class AutonomousPlanner:
    """
    High-level decision layer with dynamic goal management.
    Each cycle:
      1. Observe current game state.
      2. Check for failures → abandon stuck goals.
      3. Dynamic goal generation from crisis detection.
      4. If an active multi-step goal has a pending sub-action, follow it.
      5. Query knowledge base + episodic memory for context.
      6. Get neural network's action suggestion.
      7. Apply safety checks and heuristics to override if needed.
      8. Produce final action + thought trace.
    """
    def __init__(self):
        self.goals: list[Goal] = []
        self.thoughts: list[Thought] = []
        self._goal_counter = 0
        self.current_scenario: str = ""
        self.total_decisions = 0
        self.action_history: list[tuple[float, int, str]] = []
        self._last_goal_advance_ts: float = 0.0
        # Failure tracking per goal step
        self._step_effect_check_delay_s: float = 3.0
        self._pending_effect_checks: dict[str, tuple[int, float]] = {}  # goal_id → (expected_action, ts)
        # Episodic memory + affect references (set externally by agent)
        self.episodic_mem = None  # type: Optional[Any]
        self.affect = None        # type: Optional[Any]
    # ── Goal management ──────────────────────────────────────────
    def set_scenario(self, scenario_name: str) -> None:
        self.current_scenario = scenario_name
        logger.info("Scenario set: %s", scenario_name)

    def add_goal(
        self, description: str, priority: GoalPriority,
        steps: int = 1, context: dict | None = None,
        template: str | None = None,
    ) -> Goal:
        self._goal_counter += 1
        g = Goal(
            id=f"g{self._goal_counter}",
            description=description,
            priority=priority,
            created_at=time.time(),
            steps_total=steps,
            context=context or {},
        )
        if template and template in GOAL_TEMPLATES:
            for idx, desc in GOAL_TEMPLATES[template]:
                g.sub_actions.append(SubAction(action_idx=idx, description=desc))
            g.steps_total = len(g.sub_actions)
        self.goals.append(g)
        return g

    def complete_goal(self, goal_id: str) -> None:
        for g in self.goals:
            if g.id == goal_id:
                g.status = "completed"
                g.steps_done = g.steps_total
                break

    def active_goals(self) -> list[Goal]:
        return [g for g in self.goals if g.status == "active"]

    def _get_current_sub_action(self) -> Optional[tuple[Goal, SubAction]]:
        for g in sorted(self.active_goals(), key=lambda x: x.priority.value):
            sa = g.next_sub_action()
            if sa is not None and not sa.done:
                return g, sa
        return None

    # ── Dynamic goal generation ──────────────────────────────────
    def generate_goals_from_state(self, game_state: dict[str, Any]) -> list[Goal]:
        """
        Scan game state for crisis conditions and auto-generate goals.
        Only creates a goal if no equivalent active goal already exists.
        Returns list of newly created goals.
        """
        from .game_knowledge import GameKnowledge
        alerts = GameKnowledge.detect_crisis(game_state)
        new_goals: list[Goal] = []

        existing_descs = {g.description.lower() for g in self.active_goals()}

        for alert in alerts:
            if alert.severity < 2:
                continue
            # Map system + condition to a template
            template_key = self._map_alert_to_template(alert.system, game_state)
            if template_key is None:
                continue
            tpl_name, priority = CRISIS_TO_GOAL.get(template_key, (None, GoalPriority.MEDIUM))
            if tpl_name is None or tpl_name not in GOAL_TEMPLATES:
                continue
            desc = f"Auto: respond to {alert.system} crisis ({template_key})"
            if desc.lower() in existing_descs:
                continue  # already have an active goal for this
            g = self.add_goal(desc, priority, template=tpl_name,
                              context={"crisis_system": alert.system, "severity": alert.severity})
            new_goals.append(g)
            existing_descs.add(desc.lower())
            logger.info("DYNAMIC GOAL created: %s [priority=%s]", desc, priority.name)

        return new_goals

    @staticmethod
    def _map_alert_to_template(system: str, state: dict) -> Optional[str]:
        """Map a crisis alert's system + current values to a template key."""
        if system == "economy":
            inflation = state.get("inflation", 0)
            budget = state.get("budget_balance", 0)
            if inflation > 8.0:
                return "high_inflation"
            elif budget < -10.0:
                return "budget_deficit"
            else:
                return "economy_collapse"
        elif system == "politics":
            approval = state.get("approval_rating", 50)
            if approval < 35:
                return "low_approval"
            return None
        elif system == "military":
            loyalty = state.get("army_loyalty", 70)
            strength = state.get("military_strength", 50)
            if loyalty < 40:
                return "low_army_loyalty"
            elif strength < 30:
                return "military_threat"
            return None
        elif system == "diplomacy":
            war_support = state.get("war_support", 50)
            if war_support < 30:
                return "war_active"
            return "diplomatic_isolation"
        return None

    # ── Failure detection & replanning ───────────────────────────
    def check_goal_progress(self, game_state: dict[str, Any]) -> list[Goal]:
        """
        Check whether the current sub-action's expected effect has landed.
        If not (after timeout), record a failure. If too many failures,
        abandon the goal and trigger replanning.
        Returns list of goals that were abandoned this cycle.
        """
        now = time.time()
        abandoned: list[Goal] = []
        for g in self.active_goals():
            # Affect-coupled abandonment threshold: frustrated → drop stuck
            # goals sooner (min 3); calm → allow up to 8 attempts.
            if self.affect is not None:
                try:
                    frust = float(getattr(self.affect, "frustration", 0.3))
                    dyn = int(round(8 - 5 * min(1.0, max(0.0, frust))))
                    g.max_failures_before_abandon = max(3, min(8, dyn))
                except Exception:  # noqa: BLE001
                    pass
            sa = g.next_sub_action()
            if sa is None or sa.done:
                continue
            if sa.started_at <= 0:
                continue
            elapsed = now - sa.started_at
            if elapsed < sa.max_wait_s:
                continue

            # Effect check: did the relevant state variable improve?
            improved = self._check_effect(g, sa, game_state)
            if improved:
                g.advance()
                logger.info("Goal %s advanced to step %d/%d", g.id, g.current_step, len(g.sub_actions))
                if g.status == "completed":
                    logger.info("Goal %s COMPLETED: %s", g.id, g.description)
            else:
                g.record_failure()
                if g.status == "abandoned":
                    abandoned.append(g)
                    self._replan_after_abandonment(g, game_state)

        return abandoned

    @staticmethod
    def _check_effect(goal: Goal, sa: SubAction, state: dict) -> bool:
        """
        Heuristic: did the action's target metric move in the right direction?
        Returns True if effect observed (or if we can't determine → assume OK).
        """
        # Map action type to expected state change
        checks = {
            12: lambda s: s.get("budget_balance", -999) > -50,   # increase_tax → budget improves
            13: lambda s: s.get("inflation", 999) < 15,           # decrease_tax → inflation down
            14: lambda s: s.get("approval_rating", 0) > 25,      # increase_spending → approval up
            15: lambda s: s.get("budget_balance", -999) > -80,   # decrease_spending → deficit shrinks
            16: lambda s: True,                                   # file_bill → hard to verify quickly
            19: lambda s: s.get("war_support", 0) > 20,          # seek_peace → war support stabilises
            20: lambda s: True,                                   # form_alliance → diplomatic check
        }
        checker = checks.get(sa.action_idx)
        if checker is None:
            return True  # unknown action → don't penalise
        try:
            return bool(checker(state))
        except Exception:
            return True

    def _replan_after_abandonment(self, failed_goal: Goal, game_state: dict[str, Any]) -> None:
        """
        When a goal is abandoned, generate an alternative approach.
        Strategy: pick a different template that addresses the same system.
        """
        logger.info("Replanning after abandonment of '%s'", failed_goal.description)
        # Find an alternative template for the same crisis system
        alt_templates = {
            "economy": ["stabilize_economy", "cut_deficit", "reduce_inflation"],
            "politics": ["boost_approval", "pass_legislation"],
            "military": ["prepare_military", "boost_military_loyalty"],
            "diplomacy": ["seek_peace_treaty", "diplomatic_reset"],
        }
        system = failed_goal.context.get("crisis_system", "")
        candidates = alt_templates.get(system, [])
        used_tpl = failed_goal.context.get("template", "")
        for tpl in candidates:
            if tpl != used_tpl and tpl in GOAL_TEMPLATES:
                new_desc = f"Alt plan for '{failed_goal.description}'"
                self.add_goal(new_desc, failed_goal.priority, template=tpl,
                              context={"replan_of": failed_goal.id, "crisis_system": system})
                logger.info("Replacement goal created: %s (template=%s)", new_desc, tpl)
                break

    # ── Decision cycle ───────────────────────────────────────────
    def decide(
        self,
        nn_action: int,
        nn_confidence: float,
        game_state: dict[str, Any],
        knowledge_hints: list[str] | None = None,
        urgency_boost: float = 1.0,
    ) -> tuple[int, Thought]:
        from .network import ACTION_NAMES
        ts = time.time()
        obs_summary = self._summarize_state(game_state)

        final_action = nn_action
        override_reason = ""
        active_goal_id: Optional[str] = None
        memory_analogies: list[str] = []
        avoid_actions: set[int] = set()

        # ── 1. Check goal progress / failures ─────────────────────
        self.check_goal_progress(game_state)

        # ── 2. Dynamic goal generation ────────────────────────────
        if not self.active_goals():
            self.generate_goals_from_state(game_state)

        # ── 3. Safety check via GameKnowledge ─────────────────────
        from .game_knowledge import GameKnowledge
        recent = [a for _, a, _ in self.action_history[-6:]]
        evaln = GameKnowledge.evaluate_action(nn_action, game_state, recent)
        if not evaln.allowed:
            final_action = evaln.override_action if evaln.override_action is not None else 0
            override_reason = f"Blocked by game knowledge: {evaln.reason}"

        # ── 4. Multi-step goal sequencing ─────────────────────────
        pair = self._get_current_sub_action()
        if pair is not None and not override_reason:
            g, sa = pair
            active_goal_id = g.id
            if sa.started_at == 0.0 or (ts - sa.started_at) >= sa.max_wait_s:
                final_action = sa.action_idx
                override_reason = (
                    f"Goal '{g.description}' step {g.current_step + 1}/{len(g.sub_actions)}: "
                    f"{sa.description}" + (f" (attempt {sa.attempts + 1})" if sa.attempts > 0 else "")
                )
                if sa.started_at == 0.0:
                    sa.started_at = ts

        # ── 5. Episodic memory analogies + failure avoidance ───────
        if self.episodic_mem is not None:
            try:
                import numpy as _np
                state_vec = _np.array(list(game_state.values())[:128], dtype=_np.float32)
                similar = self.episodic_mem.retrieve_similar(state_vec, k=2)
                fail_counts: dict[int, int] = {}
                for ep, sim in similar:
                    memory_analogies.append(
                        f"ep{ep.id}({ep.outcome.value},r={ep.total_reward:.1f}): {', '.join(ep.tags[:3])}"
                    )
                    if ep.outcome.value == "failure" and ep.steps:
                        for sr in ep.steps[-6:]:   # weight recent steps more
                            fail_counts[sr.action_idx] = fail_counts.get(sr.action_idx, 0) + 1
                if fail_counts:
                    top_fail = max(fail_counts.items(), key=lambda kv: kv[1])[0]
                    if fail_counts[top_fail] >= 2:
                        avoid_actions.add(top_fail)
            except Exception:
                pass
        # ── 5b. Avoid actions that repeatedly failed in similar states ──
        if final_action in avoid_actions and not override_reason:
            avoided_name = ACTION_NAMES[final_action]
            alt = self._pick_safe_alternative(final_action, game_state)
            if alt is not None and alt != final_action:
                final_action = alt
                override_reason = (f"Avoiding '{avoided_name}' "
                                   f"(failed in similar past states)")
        # ── 6. Knowledge-based boost (low NN confidence) ──────────
        if knowledge_hints and nn_confidence < 0.4 and not override_reason:
            kb_action = self._interpret_knowledge(knowledge_hints, game_state)
            if kb_action is not None:
                final_action = kb_action
                override_reason = "Knowledge base override (low NN confidence)"

        # Build thought trace
        reasoning_parts = [f"NN: '{ACTION_NAMES[nn_action]}' (conf={nn_confidence:.2f})"]
        if override_reason:
            reasoning_parts.append(f"OVERRIDE: {override_reason}")
        if memory_analogies:
            reasoning_parts.append(f"MEM: {memory_analogies[0]}")
        reasoning_parts.append(f"→ '{ACTION_NAMES[final_action]}'")

        thought = Thought(
            timestamp=ts,
            observation_summary=obs_summary,
            reasoning=" | ".join(reasoning_parts),
            decision=ACTION_NAMES[final_action],
            confidence=nn_confidence,
            goal_id=active_goal_id,
            memory_analogies=memory_analogies,
        )
        self.thoughts.append(thought)
        if len(self.thoughts) > 500:
            self.thoughts = self.thoughts[-300:]
        self.action_history.append((ts, final_action, ACTION_NAMES[final_action]))
        self.total_decisions += 1
        return final_action, thought

    # ── Helpers ──────────────────────────────────────────────────
    def _pick_safe_alternative(self, avoided: int, state: dict[str, Any]) -> Optional[int]:
        """Choose a different action to take instead of one that keeps failing."""
        pair = self._get_current_sub_action()
        if pair is not None:
            _, sa = pair
            if sa.action_idx != avoided:
                return sa.action_idx
        # Fallback: reassess via the economy overview rather than repeat a failure.
        return 9
    @staticmethod
    def _summarize_state(state: dict[str, Any]) -> str:
        parts = []
        for k in ["budget_balance", "approval_rating", "gdp_growth",
                  "unemployment", "inflation", "military_strength"]:
            if k in state and isinstance(state[k], (int, float)):
                parts.append(f"{k}={state[k]:.1f}")
        return ", ".join(parts) if parts else "(no numeric state)"

    @staticmethod
    def _interpret_knowledge(hints: list[str], state: dict) -> Optional[int]:
        combined = " ".join(hints).lower()
        mapping = {
            "tax": 12, "spending": 14, "bill": 16, "war": 18,
            "peace": 19, "alliance": 20, "election": 22,
            "budget": 5, "diplomacy": 7, "military": 8,
        }
        for kw, idx in mapping.items():
            if kw in combined:
                return idx
        return None

    # ── Manual action recording (from dashboard) ────────────────
    def record_manual_action(self, action_name: str) -> None:
        """Record a user-injected action in the history without incrementing decisions."""
        ts = time.time()
        self.action_history.append((ts, -1, f"MANUAL:{action_name}"))
        logger.info("Manual action recorded: %s", action_name)

    # ── Dashboard data ───────────────────────────────────────────
    def get_dashboard_data(self) -> dict:
        abandoned_count = sum(1 for g in self.goals if g.status == "abandoned")
        return {
            "total_decisions": self.total_decisions,
            "active_goals": [
                {"id": g.id, "desc": g.description, "priority": g.priority.name,
                 "progress": round(g.progress, 2),
                 "current_step": g.current_step + 1 if g.sub_actions else 0,
                 "steps_total": len(g.sub_actions) or g.steps_total,
                 "failures": g.failure_count}
                for g in self.active_goals()[:10]
            ],
            "recent_thoughts": [
                {"ts": t.timestamp, "obs": t.observation_summary,
                 "reasoning": t.reasoning, "decision": t.decision,
                 "confidence": t.confidence, "goal_id": t.goal_id,
                 "mem_analogies": t.memory_analogies}
                for t in self.thoughts[-20:]
            ],
            "action_history_last_50": [
                {"ts": ts, "action": name} for ts, _, name in self.action_history[-50:]
            ],
            "scenario": self.current_scenario,
            "goals_abandoned_total": abandoned_count,
        }
