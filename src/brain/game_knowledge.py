"""
Structured a priori knowledge of Geoplanetical's systems.

This module encodes what the agent *knows* about how the game works before
it plays – system thresholds, decision rules, safety checks, console command
references, and strategic patterns derived from extensive research into the
game's mechanics, community guides, and patch notes (Nov 2026).

It complements the KnowledgeDB (runtime episodic/semantic memory) by providing
the "textbook" layer: stable, well-understood facts about the game world.

Research sources:
  • geoplanetical.com/guides/beginner/          – first-hour protocol
  • geoplanetical.com/cheats/console-commands/  – full 58-command reference
  • geoplaneticalgame.wiki/systems/             – budget/GDP, elections/strikes,
                                                   laws/parliament, war/diplomacy
  • geoplanetical.wiki tips & tricks            – strategy patterns, country tiers
  • Steam store page + patch notes              – current version state
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("brain.game_knowledge")

# ── Action index constants (must match network.ACTION_NAMES order) ────────
A_NOOP = 0
A_PAUSE = 1
A_RESUME = 2
A_OPEN_MENU = 3
A_CLOSE_MENU = 4
A_NAV_BUDGET = 5
A_NAV_LEGISLATION = 6
A_NAV_DIPLOMACY = 7
A_NAV_MILITARY = 8
A_NAV_ECONOMY = 9
A_APPLY_POLICY = 10
A_DISCARD_POLICY = 11
A_INCREASE_TAX = 12
A_DECREASE_TAX = 13
A_INCREASE_SPENDING = 14
A_DECREASE_SPENDING = 15
A_FILE_BILL = 16
A_VETO_BILL = 17
A_DECLARE_WAR = 18
A_SEEK_PEACE = 19
A_FORM_ALLIANCE = 20
A_BREAK_ALLIANCE = 21
A_CALL_ELECTION = 22
A_APPOINT_MINISTER = 23
A_FIRE_MINISTER = 24
A_USE_CONSOLE = 25
A_SAVE_GAME = 26
A_LOAD_GAME = 27
A_SPEED_1 = 28
A_SPEED_2 = 29
A_SPEED_3 = 30

# ── State variable thresholds ─────────────────────────────────────────────
# Each entry: (warning_low, warning_high, critical_low, critical_high)
# Values in the same units as the game state dict (mostly 0-100 scale).
STATE_THRESHOLDS: dict[str, tuple[float, float, float, float]] = {
    "approval_rating":   (35.0, 75.0, 25.0, 85.0),
    "budget_balance":    (-10.0, 50.0, -30.0, 100.0),
    "gdp_growth":        (0.5, 5.0, -2.0, 8.0),
    "unemployment":      (3.0, 10.0, 15.0, 25.0),
    "inflation":         (1.0, 4.0, 8.0, 15.0),
    "military_strength": (30.0, 80.0, 15.0, 95.0),
    "army_loyalty":      (50.0, 90.0, 35.0, 95.0),
    "war_support":       (40.0, 80.0, 25.0, 90.0),
    "stability_index":   (40.0, 80.0, 25.0, 90.0),
}

# ── Console command reference (official wiki + Oct/Nov 2026 patches) ─────
CONSOLE_COMMANDS: dict[str, list[dict[str, str]]] = {
    "economy": [
        {"cmd": "money [percent]", "effect": "Add 1-10% of GDP to treasury"},
        {"cmd": "fill_stockpiles", "effect": "Fill strategic and military stockpiles"},
        {"cmd": "own_currency [type|name]", "effect": "Issue own currency at same value"},
        {"cmd": "adopt_euro", "effect": "Request euro adoption"},
        {"cmd": "inflation_shock <percent>", "effect": "Move price level -50% to +1000%"},
    ],
    "society": [
        {"cmd": "love_people", "effect": "Reset public opinion to clean slate"},
        {"cmd": "set_stability <0-100>", "effect": "Set stability to given value"},
        {"cmd": "lock_love", "effect": "Freeze approval and stability at max (Oct 2026)"},
    ],
    "diplomacy": [
        {"cmd": "love_world", "effect": "All countries' opinion → 100, lifts sanctions"},
        {"cmd": "yesman on|off", "effect": "AI accepts all diplomacy you start"},
        {"cmd": "remove_sanctions", "effect": "Lift sanctions targeting you"},
    ],
    "research": [
        {"cmd": "fast_research on|off", "effect": "National research at 10× speed"},
        {"cmd": "research_now", "effect": "Finish highest-priority active research"},
        {"cmd": "research_all [domain]", "effect": "Discover missing tech or one domain"},
    ],
    "politics": [
        {"cmd": "adopt_now", "effect": "Pass bills/reforms already in debate"},
        {"cmd": "adopt_always on|off", "effect": "Future bills auto-pass"},
    ],
    "war": [
        {"cmd": "peace_out", "effect": "White peace on your wars, troops go home"},
        {"cmd": "world_peace", "effect": "End every active war"},
        {"cmd": "win_war [warId]", "effect": "Win one of your wars with peace terms"},
        {"cmd": "nuclear_power", "effect": "Complete nuclear program, 2500 warheads"},
        {"cmd": "ready_army", "effect": "Restore readiness, organization, supplies"},
        {"cmd": "add_equipment <type> <qty>", "effect": "Deliver equipment (capped by real fleet)"},
    ],
    "construction": [
        {"cmd": "fast_build on|off", "effect": "Terrestrial construction at 10×"},
        {"cmd": "fast_space on|off", "effect": "Space projects at 10×"},
        {"cmd": "space_now", "effect": "Complete oldest active space project"},
    ],
    "territory": [
        {"cmd": "provat [lon lat]", "effect": "Read province under cursor"},
        {"cmd": "debug_map on|off", "effect": "Hover shows country tag + province ID"},
        {"cmd": "annex_prov <provId>", "effect": "Take one province by ID"},
        {"cmd": "annex_country <country>", "effect": "Take all provinces of a country"},
        {"cmd": "give_prov <provId> <country>", "effect": "Give province to any country"},
        {"cmd": "regions [country]", "effect": "List regional questions"},
        {"cmd": "secede <region>", "effect": "Region declares independence"},
    ],
    "health": [
        {"cmd": "outbreak <disease> [country] [people]", "effect": "Infected people arrive"},
        {"cmd": "epidemic_truth [country]", "effect": "Show clinical numbers vs declared"},
        {"cmd": "pandemic [family]", "effect": "Start new pathogen (corona, flu, henipa, filovirus)"},
        {"cmd": "epidemic_clear [country]", "effect": "All acute cases recover now"},
    ],
}

# ── Country difficulty tiers ──────────────────────────────────────────────
COUNTRY_TIERS: dict[str, list[dict[str, str]]] = {
    "beginner": [
        {"name": "Canada", "focus": "Fiscal policy without territorial pressure"},
        {"name": "Germany", "focus": "Energy and coalition politics"},
        {"name": "Australia", "focus": "Trade and export policy"},
    ],
    "intermediate": [
        {"name": "France", "focus": "Domestic politics + global influence"},
        {"name": "Brazil", "focus": "Growth while controlling deficits"},
        {"name": "India", "focus": "Growth, demographics, regional security"},
    ],
    "advanced": [
        {"name": "Ukraine", "focus": "Survive external pressure, rebuild resilience"},
        {"name": "Iran", "focus": "Navigate sanctions and regional diplomacy"},
        {"name": "North Korea", "focus": "Sustainability under isolation"},
    ],
}


@dataclass(frozen=True)
class CrisisAlert:
    """A detected crisis condition requiring immediate attention."""
    severity: int          # 1 = warning, 2 = danger, 3 = critical
    system: str            # economy | politics | military | diplomacy | compound
    message: str
    recommended_actions: tuple[int, ...]
    console_command: Optional[str] = None


@dataclass(frozen=True)
class ActionEvaluation:
    """Result of evaluating whether an action is safe/appropriate."""
    allowed: bool
    reason: str = ""
    override_action: Optional[int] = None


class GameKnowledge:
    """
    A priori knowledge of Geoplanetical's systems.

    Provides the planner with:
      • Crisis detection from numeric game state
      • Action safety evaluation (multi-condition pre-checks)
      • Strategic priority ordering
      • Console command suggestions
      • Sequencing rules to prevent contradictory moves
    """

    # ── Crisis detection ────────────────────────────────────────────
    @staticmethod
    def detect_crisis(state: dict[str, float]) -> list[CrisisAlert]:
        """Scan game state for conditions requiring immediate attention."""
        alerts: list[CrisisAlert] = []

        # Skip if all values are at default 50.0 – no real data available yet
        vals = [v for v in state.values() if isinstance(v, (int, float))]
        if vals and all(abs(v - 50.0) < 0.01 for v in vals):
            return alerts

        for var_name, (warn_lo, warn_hi, crit_lo, crit_hi) in STATE_THRESHOLDS.items():
            val = state.get(var_name)
            if val is None or not isinstance(val, (int, float)):
                continue

            if val <= crit_lo:
                severity, direction = 3, "critically low"
            elif val >= crit_hi:
                severity, direction = 3, "critically high"
            elif val <= warn_lo:
                severity, direction = 2, "dangerously low"
            elif val >= warn_hi:
                severity, direction = 2, "dangerously high"
            else:
                continue

            system, message, actions, cmd = GameKnowledge._crisis_response(
                var_name, val, direction, severity,
            )
            if system:
                alerts.append(CrisisAlert(
                    severity=severity,
                    system=system,
                    message=message,
                    recommended_actions=tuple(actions),
                    console_command=cmd,
                ))

        # Compound crisis: economic + political simultaneously
        econ_bad = state.get("budget_balance", 0) < -15
        pol_bad = state.get("approval_rating", 50) < 30
        if econ_bad and pol_bad:
            alerts.insert(0, CrisisAlert(
                severity=3,
                system="compound",
                message=("Economic AND political crisis – "
                         "prioritize stabilization over reform"),
                recommended_actions=(A_NAV_BUDGET, A_DECREASE_SPENDING, A_APPLY_POLICY),
            ))

        return sorted(alerts, key=lambda a: -a.severity)

    @staticmethod
    def _crisis_response(
        var: str, val: float, direction: str, severity: int,
    ) -> tuple[str, str, list[int], Optional[str]]:
        """Map a variable breach to (system, message, actions, console_cmd)."""
        responses: dict[str, tuple[str, str, list[int], Optional[str]]] = {
            "approval_rating": (
                "politics",
                f"Approval {direction} at {val:.0f}% – government stability at risk",
                [A_NAV_LEGISLATION, A_FILE_BILL] if severity < 3 else [A_NAV_ECONOMY, A_INCREASE_SPENDING],
                None,
            ),
            "budget_balance": (
                "economy",
                f"Budget {direction} at {val:.1f} – fiscal pressure building",
                [A_NAV_BUDGET, A_DECREASE_SPENDING] if val < 0 else [A_NAV_BUDGET, A_INCREASE_TAX],
                "money 5" if severity >= 3 and val < -20 else None,
            ),
            "gdp_growth": (
                "economy",
                f"GDP growth {direction} at {val:.1f}%",
                [A_NAV_ECONOMY, A_APPLY_POLICY],
                None,
            ),
            "unemployment": (
                "economy",
                f"Unemployment {direction} at {val:.1f}%",
                [A_NAV_ECONOMY, A_INCREASE_SPENDING],
                None,
            ),
            "inflation": (
                "economy",
                f"Inflation {direction} at {val:.1f}% – monetary policy needed",
                [A_NAV_ECONOMY, A_DISCARD_POLICY],
                None,
            ),
            "army_loyalty": (
                "military",
                f"Army loyalty {direction} at {val:.0f}% – mutiny risk",
                [A_NAV_MILITARY, A_NOOP],
                "ready_army" if severity >= 3 else None,
            ),
            "war_support": (
                "military",
                f"War support {direction} at {val:.0f}% – front may collapse",
                [A_SEEK_PEACE] if val < 30 else [A_NAV_DIPLOMACY],
                "peace_out" if severity >= 3 and val < 25 else None,
            ),
            "stability_index": (
                "politics",
                f"Stability {direction} at {val:.0f}%",
                [A_NAV_LEGISLATION, A_APPLY_POLICY],
                "set_stability 60" if severity >= 3 else None,
            ),
        }
        return responses.get(var, ("unknown", "", [], None))

    # ── Action safety evaluation ────────────────────────────────────
    @staticmethod
    def evaluate_action(
        action_idx: int,
        state: dict[str, float],
        recent_actions: list[int],
    ) -> ActionEvaluation:
        """
        Check whether an action is safe/appropriate given current game state
        and the sequence of recent actions.
        """
        from .network import ACTION_NAMES

        name = (ACTION_NAMES[action_idx]
                if action_idx < len(ACTION_NAMES) else f"unknown_{action_idx}")

        # ── War declaration: multi-condition pre-check ─────────────
        if action_idx == A_DECLARE_WAR:
            checks: list[str] = []
            loyalty = state.get("army_loyalty", 50)
            if loyalty < 60:
                checks.append(f"Army loyalty {loyalty:.0f}% < 60% required")
            support = state.get("war_support", 50)
            if support < 40:
                checks.append(f"War support {support:.0f}% < 40% required")
            budget = state.get("budget_balance", 0)
            if budget < -10:
                checks.append(f"Budget deficit {budget:.1f} – cannot sustain war costs")
            stability = state.get("stability_index", 50)
            if stability < 35:
                checks.append(f"Stability {stability:.0f}% too low for escalation")
            if checks:
                return ActionEvaluation(
                    allowed=False,
                    reason=" | ".join(checks),
                    override_action=A_NOOP,
                )

        # ── Election call: approval threshold ──────────────────────
        if action_idx == A_CALL_ELECTION:
            approval = state.get("approval_rating", 50)
            if approval < 35:
                return ActionEvaluation(
                    allowed=False,
                    reason=f"Approval {approval:.0f}% < 35% – calling election risks losing power",
                    override_action=A_NAV_LEGISLATION,
                )

        # ── Fiscal sequencing: don't stack changes ────────────────
        fiscal_actions = {A_INCREASE_TAX, A_DECREASE_TAX,
                          A_INCREASE_SPENDING, A_DECREASE_SPENDING}
        if action_idx in fiscal_actions:
            recent_fiscal = [a for a in recent_actions[-4:] if a in fiscal_actions]
            if len(recent_fiscal) >= 2:
                return ActionEvaluation(
                    allowed=False,
                    reason=("Already made 2+ fiscal changes recently – "
                            "let effects settle before adjusting again"),
                    override_action=A_NAV_BUDGET,
                )
            last_fiscal = recent_fiscal[-1] if recent_fiscal else None
            if action_idx == A_INCREASE_TAX and last_fiscal == A_DECREASE_SPENDING:
                return ActionEvaluation(
                    allowed=False,
                    reason="Combining tax hike with spending cuts creates maximum voter backlash",
                    override_action=A_NOOP,
                )
            if action_idx == A_DECREASE_SPENDING and last_fiscal == A_INCREASE_TAX:
                return ActionEvaluation(
                    allowed=False,
                    reason="Spending cuts right after tax hike – sequence them separately",
                    override_action=A_NOOP,
                )

        # ── Minister firing: not during crisis ────────────────────
        if action_idx == A_FIRE_MINISTER:
            stability = state.get("stability_index", 50)
            approval = state.get("approval_rating", 50)
            if stability < 40 or approval < 35:
                return ActionEvaluation(
                    allowed=False,
                    reason=(f"Firing minister during instability "
                            f"(stab={stability:.0f}%, appr={approval:.0f}%) risks cabinet collapse"),
                    override_action=A_APPOINT_MINISTER,
                )

        # ── Alliance breaking: check military position ───────────
        if action_idx == A_BREAK_ALLIANCE:
            military = state.get("military_strength", 50)
            if military < 40:
                return ActionEvaluation(
                    allowed=False,
                    reason=f"Military strength {military:.0f}% too low to break alliance safely",
                    override_action=A_NAV_DIPLOMACY,
                )

        # ── Speed: avoid fastest during critical moments ─────────
        if action_idx == A_SPEED_3:
            alerts = GameKnowledge.detect_crisis(state)
            if any(a.severity >= 2 for a in alerts):
                return ActionEvaluation(
                    allowed=False,
                    reason="Critical situation active – slow down to x1 or x2",
                    override_action=A_SPEED_1,
                )

        return ActionEvaluation(allowed=True)

    # ── Priority ordering ───────────────────────────────────────────
    @staticmethod
    def get_priority_actions(state: dict[str, float]) -> list[int]:
        """
        Given current game state, return ordered list of what should be done first.
        Based on the research principle: stabilize before reforming.
        """
        priorities: list[tuple[int, int]] = []

        loyalty = state.get("army_loyalty", 50)
        budget = state.get("budget_balance", 0)
        approval = state.get("approval_rating", 50)
        stability = state.get("stability_index", 50)

        # Survival-level (lower score = more urgent)
        if loyalty < 40:
            priorities.append((0, A_NAV_MILITARY))
        if budget < -25:
            priorities.append((1, A_NAV_BUDGET))
        if approval < 30:
            priorities.append((2, A_NAV_ECONOMY))
        if stability < 30:
            priorities.append((3, A_NAV_LEGISLATION))

        # Strategic-level
        if -25 <= budget < 0:
            priorities.append((10, A_NAV_BUDGET))
        if 30 <= approval < 40:
            priorities.append((15, A_NAV_ECONOMY))

        # Operational-level (no crisis – focus on development)
        if not priorities:
            priorities.append((30, A_NAV_ECONOMY))
            priorities.append((35, A_NAV_DIPLOMACY))
            priorities.append((40, A_NAV_LEGISLATION))

        priorities.sort(key=lambda x: x[0])
        return [a for _, a in priorities]

    # ── Console command suggestions ────────────────────────────────
    @staticmethod
    def console_command_for(situation: str) -> Optional[str]:
        """Map a situation description to the most useful console command."""
        s = situation.lower()
        mappings = [
            ("money", "money 5"),
            ("funds", "money 5"),
            ("treasury", "money 5"),
            ("stockpile", "fill_stockpiles"),
            ("ammo", "ready_army"),
            ("army ready", "ready_army"),
            ("research", "fast_research on"),
            ("tech", "research_now"),
            ("peace", "peace_out"),
            ("war end", "world_peace"),
            ("sanction", "remove_sanctions"),
            ("diplomacy accept", "yesman on"),
            ("stability fix", "set_stability 60"),
            ("approval fix", "love_people"),
            ("intel", "full_intel on"),
            ("bill pass", "adopt_now"),
            ("construction", "fast_build on"),
            ("space", "fast_space on"),
        ]
        for keyword, cmd in mappings:
            if keyword in s:
                return cmd
        return None

    # ── Strategic advice (human-readable) ──────────────────────────
    @staticmethod
    def strategic_advice(state: dict[str, float]) -> str:
        """Generate a brief strategic assessment of the current situation."""
        parts: list[str] = []

        approval = state.get("approval_rating", 50)
        budget = state.get("budget_balance", 0)
        growth = state.get("gdp_growth", 2.0)
        inflation = state.get("inflation", 2.0)
        loyalty = state.get("army_loyalty", 70)

        if budget < -15:
            parts.append(f"Fiscal pressure: balance {budget:.1f}. Prioritize revenue or cut non-essential spending.")
        elif budget > 30:
            parts.append(f"Surplus {budget:.1f} – room for investment or tax relief.")
        else:
            parts.append(f"Economy stable (balance {budget:.1f}, growth {growth:.1f}%).")

        if inflation > 5:
            parts.append(f"Inflation at {inflation:.1f}% – avoid money creation, consider tightening.")

        if approval < 35:
            parts.append(f"Approval {approval:.0f}% – pass popular measures, avoid controversial reforms.")
        elif approval > 60:
            parts.append(f"Strong mandate ({approval:.0f}%) – window for structural reform.")

        if loyalty < 50:
            parts.append(f"Army loyalty {loyalty:.0f}% – improve conditions before any military action.")

        return " ".join(parts) if parts else "No critical issues. Focus on long-term development."


# ── Keybinds (documented controls) ───────────────────────────────
KEYBINDS: dict[str, str] = {
    "Space":   "Pause / resume the simulation",
    "1":       "Simulation speed x1 (slowest)",
    "2":       "Simulation speed x2",
    "3":       "Simulation speed x3 (fastest)",
    "Escape":  "Open / close the pause menu (stops time)",
    "F5":      "Create a manual named save",
    "`":       "Open the cheat console (disabled by default in Green Vision)",
}

# ── Main menus / screens and how to reach them ───────────────────
MENUS: dict[str, str] = {
    "budget":       "State cash / budget screen – edit revenue & spending proposals; Apply commits, Discard abandons",
    "legislation":  "Laws screen – check support, submit bill, follow parliamentary procedure",
    "diplomacy":    "Foreign relations – alliances, treaties, sanctions, peace negotiations",
    "military":     "Armed forces – readiness, loyalty, equipment, war decisions",
    "economy":      "Economic overview – GDP growth, inflation, unemployment, sectors",
    "cablegram":    "Minister advice – select a minister, Ask for advice, Open file jumps to their portfolio",
    "settings":     "Autosave cadence (1/3/6/12 months), retained copies, game rules",
    "save_load":    "Named saves + autosaves; Load returns to a saved campaign",
}

# ── Playable-world facts (documented) ─────────────────────────────
PLAYABLE_NOTES: list[str] = [
    "Every new campaign starts on January 1, 2026 with real-world starting data.",
    "The Russia-Ukraine war is already active at the start of every world.",
    "More than 200 nations and territories are playable, including self-administered ones (Greenland, Puerto Rico, Hong Kong).",
    "Some microstates are NOT playable (Monaco, Vatican).",
    "You can switch country or political role mid-campaign from the pause menu without restarting the world.",
    "Losing an election does not end the campaign; you continue as your party with reduced powers.",
    "Game rules adjustable at start/mid-game: AI war declarations, nuclear first use, disasters, pandemics. Leaders' temperament is fixed at start.",
    "Budget forecasts vs enacted policies differ – Apply commits a package, Discard abandons it; borrowing creates debt+interest separately.",
]


def seed_base_knowledge(kb) -> int:
    """
    Load stable, documented game knowledge into the KnowledgeDB so the agent
    has a working base *before* any live Serper research succeeds.
    Idempotent-ish: safe to call on every startup (adds snippets; vector store
    dedupes by content in practice for our purposes).
    Returns number of snippets added.
    """
    n = 0
    # Keybinds
    for key, desc in KEYBINDS.items():
        kb.add_knowledge(content=f"Keybind '{key}': {desc}", source="base_keybinds")
        n += 1
    # Menus
    for name, desc in MENUS.items():
        kb.add_knowledge(content=f"Menu '{name}': {desc}", source="base_menus")
        n += 1
    # Playable-world facts
    for note in PLAYABLE_NOTES:
        kb.add_knowledge(content=note, source="base_world")
        n += 1
    # Country archetypes + tiers
    for tier, entries in COUNTRY_TIERS.items():
        for c in entries:
            kb.add_knowledge(
                content=f"Country tier [{tier}] {c['name']}: {c['focus']}",
                source="base_countries",
            )
            n += 1
    # Crisis response rules (so the KB can override a low-confidence NN)
    crisis_rules = [
        "If budget_balance is negative and large, navigate to budget and cut non-essential spending before raising taxes."
        "If approval_rating drops below ~35%, pass popular measures and avoid controversial reforms; do NOT call an early election."
        "Never declare war unless army_loyalty >= 60% AND war_support >= 40% AND budget is not deeply in deficit."
        "Do not stack fiscal changes: after two tax/spending moves, let effects settle before adjusting again."
        "Firing a minister during instability (stability < 40 or approval < 35) risks cabinet collapse – appoint instead."
        "Breaking an alliance while military_strength < 40% is unsafe – strengthen diplomacy first."
        "Inflation above ~5%: avoid money creation, consider tightening monetary policy."
        "Army loyalty below ~50%: improve conditions before any military action."
    ]
    for rule in crisis_rules:
        kb.add_knowledge(content=rule, source="base_strategy")
        n += 1
    logger.info("Seeded %d base knowledge snippets", n)
    return n
