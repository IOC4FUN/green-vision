"""
Green Vision – Web Dashboard Server (Flask).
Serves the static front-end and exposes a JSON API for live agent state.
Usage:
    python main.py --web-dashboard          # integrated with agent loop
    python src/dashboard/server.py --port 8090   # standalone (reads log file)
API endpoints:
    GET  /            → index.html
    GET  /style.css   → static assets
    GET  /app.js      → static assets
    GET  /api/state   → full agent state snapshot
    GET  /api/health  → subsystem health metrics
    GET  /api/thoughts→ recent thought stream
    GET  /api/goals   → active + completed goals
    GET  /api/actions → action history
    GET  /api/actions-available → list of all actions the user can trigger
    POST /api/action  → inject a manual action {"action": "increase_tax"}
    POST /api/control → pause/resume autonomous loop {"cmd": "pause"|"resume"|"stop"}
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional
ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))
logger = logging.getLogger("dashboard.server")

# ── Action groups for the UI ────────────────────────────────────────────────
ACTION_GROUPS: dict[str, list[dict[str, str]]] = {
    "Navigation": [
        {"name": "navigate_budget",      "label": "Budget (F1×3)"},
        {"name": "navigate_legislation", "label": "Legislation (F2×3)"},
        {"name": "navigate_diplomacy",   "label": "Diplomacy (F3×3)"},
        {"name": "navigate_military",    "label": "Military (F4×3)"},
        {"name": "navigate_economy",     "label": "Economy (F5×3)"},
    ],
    "Economy": [
        {"name": "increase_tax",       "label": "+ Tax Rate"},
        {"name": "decrease_tax",       "label": "− Tax Rate"},
        {"name": "increase_spending",  "label": "+ Spending"},
        {"name": "decrease_spending",  "label": "− Spending"},
        {"name": "apply_policy",       "label": "Apply Policy ⏎"},
        {"name": "discard_policy",     "label": "Discard Esc"},
    ],
    "Legislation & Government": [
        {"name": "file_bill",      "label": "File Bill"},
        {"name": "veto_bill",      "label": "Veto Bill"},
        {"name": "call_election",  "label": "Call Election"},
        {"name": "appoint_minister","label": "Appoint Minister"},
        {"name": "fire_minister",  "label": "Fire Minister"},
    ],
    "Diplomacy & Military": [
        {"name": "declare_war",    "label": "⚔ Declare War"},
        {"name": "seek_peace",     "label": "🕊 Seek Peace"},
        {"name": "form_alliance",  "label": "Form Alliance"},
        {"name": "break_alliance", "label": "Break Alliance"},
    ],
    "System": [
        {"name": "noop",           "label": "No-Op (wait)"},
        {"name": "pause_game",     "label": "Pause (Space)"},
        {"name": "resume_game",    "label": "Resume (Space)"},
        {"name": "open_menu",      "label": "Open Menu (Esc)"},
        {"name": "close_menu",     "label": "Close Menu (Esc)"},
        {"name": "use_console",    "label": "Console (`)"},
        {"name": "save_game",      "label": "Save Game"},
        {"name": "load_game",      "label": "Load Game"},
        {"name": "switch_speed_1", "label": "Speed 1×"},
        {"name": "switch_speed_2", "label": "Speed 2×"},
        {"name": "switch_speed_3", "label": "Speed 3×"},
    ],
}

# ── Flask app factory ───────────────────────────────────────────────────────
def create_app(agent: Optional[Any] = None) -> Any:
    from flask import Flask, jsonify, request, send_from_directory
    app = Flask(
        __name__,
        static_folder=str(Path(__file__).parent / "static"),
        static_url_path="",
    )

    # Allow the front-end to poll from any origin (local dev convenience)
    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    # ── Static files ──────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(app.static_folder, filename)

    # ── API endpoints ─────────────────────────────────────────────────────
    @app.route("/api/state")
    def api_state():
        if agent is None:
            return jsonify({"error": "No agent attached", "status": "offline"})
        try:
            dash = agent.planner.get_dashboard_data() or {}
            state_vec = getattr(agent, "_state_vec", None)
            import numpy as np
            state_summary = {
                "frame_count": agent._frame_count,
                "decision_interval_s": agent._decision_interval,
                "episode_active": agent._episode_active,
                "no_op_streak": agent._no_op_streak,
                "console_enabled": agent.console_enabled,
                "autonomous_paused": getattr(agent, "_manual_pause", False),
            }
            if state_vec is not None and isinstance(state_vec, np.ndarray):
                state_summary["state_dim"] = int(state_vec.shape[0])
                state_summary["state_norm"] = round(float(np.linalg.norm(state_vec)), 4)
            payload = {**dash, **state_summary, "ts": time.time()}
            return jsonify(payload)
        except Exception as e:  # noqa: BLE001
            logger.exception("/api/state failed")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/health")
    def api_health():
        if agent is None:
            return jsonify({"status": "offline"})
        try:
            health_data: dict[str, Any] = {}
            hw = getattr(agent, "health", None)
            if hw is not None:
                health_data["ocr_empty_threshold"] = hw.ocr_empty_threshold
                health_data["mem_fail_timeout_s"] = hw.mem_fail_timeout_s
                health_data["capture_min_fps"] = hw.capture_min_fps
            # GPU / VRAM quick check (non-blocking)
            import subprocess
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,clocks.current.graphics,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                ).stdout.strip()
                parts = out.split(",")
                health_data["gpu"] = {
                    "vram_used_mib": int(parts[0]),
                    "vram_total_mib": int(parts[1]),
                    "clock_mhz": int(parts[2]),
                    "power_w": float(parts[3]),
                }
            except Exception:  # noqa: BLE001
                health_data["gpu"] = {"error": "unavailable"}
            health_data["ts"] = time.time()
            return jsonify(health_data)
        except Exception as e:  # noqa: BLE001
            logger.exception("/api/health failed")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/thoughts")
    def api_thoughts():
        if agent is None:
            return jsonify([])
        try:
            dash = agent.planner.get_dashboard_data() or {}
            thoughts = dash.get("recent_thoughts", [])
            return jsonify(thoughts[-20:])
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 500

    @app.route("/api/goals")
    def api_goals():
        if agent is None:
            return jsonify([])
        try:
            dash = agent.planner.get_dashboard_data() or {}
            goals = dash.get("active_goals", [])
            completed = dash.get("completed_goals", [])
            return jsonify({"active": goals, "completed": completed})
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 500

    @app.route("/api/actions")
    def api_actions():
        if agent is None:
            return jsonify([])
        try:
            dash = agent.planner.get_dashboard_data() or {}
            actions = dash.get("action_history_last_50", [])
            return jsonify(actions[-30:])
        except Exception as e:  # noqa: BLE001
            return jsonify({"error": str(e)}), 500

    # ── Manual control endpoints ──────────────────────────────────────────
    @app.route("/api/actions-available")
    def api_actions_available():
        """Return the grouped action list for the UI."""
        from src.brain.network import ACTION_NAMES
        groups = []
        for group_name, items in ACTION_GROUPS.items():
            valid = [it for it in items if it["name"] in ACTION_NAMES]
            groups.append({"group": group_name, "actions": valid})
        return jsonify(groups)

    @app.route("/api/action", methods=["POST"])
    def api_action():
        """Inject a manual action. Body: {\"action\": \"increase_tax\"}"""
        if agent is None:
            return jsonify({"error": "No agent attached"}), 400
        data = request.get_json(silent=True) or {}
        action_name = data.get("action", "").strip()
        if not action_name:
            return jsonify({"error": "Missing 'action' field"}), 400
        try:
            from src.brain.network import ACTION_NAMES
            if action_name not in ACTION_NAMES:
                return jsonify({"error": f"Unknown action '{action_name}'"}), 400
            idx = ACTION_NAMES.index(action_name)
            # Execute on the main thread via a flag so we don't race the loop
            agent._manual_action_queue = getattr(agent, "_manual_action_queue", [])
            agent._manual_action_queue.append((idx, time.time()))
            logger.info("Manual action queued: %s (idx=%d)", action_name, idx)
            return jsonify({"ok": True, "action": action_name, "index": idx})
        except Exception as e:  # noqa: BLE001
            logger.exception("/api/action failed")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/control", methods=["POST"])
    def api_control():
        """Pause / resume / stop the autonomous loop. Body: {\"cmd\": \"pause\"}"""
        if agent is None:
            return jsonify({"error": "No agent attached"}), 400
        data = request.get_json(silent=True) or {}
        cmd = data.get("cmd", "").strip().lower()
        if cmd == "pause":
            agent._manual_pause = True
            logger.info("Autonomous loop PAUSED by user")
            return jsonify({"ok": True, "status": "paused"})
        elif cmd == "resume":
            agent._manual_pause = False
            logger.info("Autonomous loop RESUMED by user")
            return jsonify({"ok": True, "status": "resumed"})
        elif cmd == "stop":
            agent._stop_event = True
            logger.info("Agent STOP requested by user")
            return jsonify({"ok": True, "status": "stopping"})
        else:
            return jsonify({"error": f"Unknown cmd '{cmd}' (use pause|resume|stop)"}), 400

    # ── Learning statistics endpoint ─────────────────────────────────────
    @app.route("/api/stats")
    def api_stats():
        if agent is None:
            return jsonify({"error": "No agent attached", "status": "offline"})
        try:
            import numpy as np
            data: dict[str, Any] = {}

            # Reward history from checkpoint manager (rolling window of 200)
            ckpt = getattr(agent, "checkpoints", None)
            rewards_hist: list[float] = []
            best_reward = -999.0
            checkpoints_list: list[str] = []
            if ckpt is not None:
                rewards_hist = [round(r, 4) for r in list(ckpt._rewards)]
                best_reward = round(ckpt.best_reward, 4) if ckpt.best_reward > -1e6 else 0.0
                checkpoints_list = ckpt.list_checkpoints()
            data["reward_history"] = rewards_hist
            data["best_reward"] = best_reward
            data["checkpoints"] = checkpoints_list
            data["total_steps_tracked"] = len(rewards_hist)

            # Episodic memory summary
            ep_mem = getattr(agent, "episodic_mem", None)
            episodes_summary = {"total": 0, "success": 0, "stall": 0, "failure": 0,
                                "avg_reward_success": 0, "avg_reward_stall": 0, "avg_reward_failure": 0}
            episode_rewards: list[float] = []
            if ep_mem is not None and hasattr(ep_mem, '_episodes'):
                eps = list(ep_mem._episodes)
                episodes_summary["total"] = len(eps)
                sums = {"success": [], "stall": [], "failure": []}
                for ep in eps:
                    outcome_name = ep.outcome.value if hasattr(ep.outcome, 'value') else str(ep.outcome)
                    if outcome_name in sums:
                        sums[outcome_name].append(ep.total_reward)
                    episode_rewards.append(round(ep.total_reward, 3))
                for k in ["success", "stall", "failure"]:
                    episodes_summary[k] = len(sums[k])
                    episodes_summary[f"avg_reward_{k}"] = round(sum(sums[k])/len(sums[k]), 4) if sums[k] else 0
            data["episodes"] = episodes_summary
            data["episode_rewards_recent"] = episode_rewards[-50:]

            # Curiosity module stats
            curiosity_mod = getattr(agent, "curiosity", None)
            curiosity_data = {"total_intrinsic": 0.0, "num_steps": 0, "avg_intrinsic": 0.0}
            if curiosity_mod is not None:
                n_steps = getattr(curiosity_mod, '_num_steps', 0)
                total_intr = getattr(curiosity_mod, '_total_intrinsic_reward', 0.0)
                curiosity_data = {
                    "total_intrinsic": round(total_intr, 4),
                    "num_steps": n_steps,
                    "avg_intrinsic": round(total_intr / max(1, n_steps), 6),
                }
            data["curiosity"] = curiosity_data

            # PPO agent metrics (last update losses if cached)
            ppo = getattr(agent, "agent", None)
            ppo_metrics: dict[str, Any] = {}
            if ppo is not None:
                rollout_size = len(ppo.rollout) if hasattr(ppo, 'rollout') else 0
                replay_size = len(ppo.replay) if hasattr(ppo, 'replay') else 0
                ppo_metrics = {
                    "rollout_buffer": rollout_size,
                    "replay_buffer": replay_size,
                    "device": str(ppo.device),
                    "update_every_steps": ppo.cfg.update_every_steps,
                    "batch_size": ppo.cfg.batch_size,
                }
                # Last update metrics if the agent stored them
                last_upd = getattr(ppo, '_last_update_metrics', None)
                if last_upd and isinstance(last_upd, dict):
                    for k in ["policy_loss", "value_loss", "entropy", "kl_divergence", "approx_kl"]:
                        if k in last_upd:
                            ppo_metrics[k] = round(float(last_upd[k]), 6)
            data["ppo"] = ppo_metrics

            # Planner / decision stats
            planner = getattr(agent, "planner", None)
            planner_stats: dict[str, Any] = {}
            if planner is not None:
                total_decisions = getattr(planner, 'total_decisions', 0)
                goals_list = getattr(planner, 'goals', []) or []
                completed = sum(1 for g in goals_list if g.status == "completed")
                abandoned = sum(1 for g in goals_list if g.status == "abandoned")
                active = sum(1 for g in goals_list if g.status == "active")
                thoughts_count = len(getattr(planner, 'thoughts', []) or [])
                action_hist_len = len(getattr(planner, 'action_history', []) or [])
                # Action distribution (top actions by frequency)
                from collections import Counter
                action_counts = Counter(name for _, _, name in (getattr(planner, 'action_history', []) or []))
                top_actions = action_counts.most_common(10)
                planner_stats = {
                    "total_decisions": total_decisions,
                    "goals_completed": completed,
                    "goals_abandoned": abandoned,
                    "goals_active": active,
                    "thoughts_recorded": thoughts_count,
                    "actions_total": action_hist_len,
                    "top_actions": [{"name": n, "count": c} for n, c in top_actions],
                }
            data["planner"] = planner_stats

            # State vector info
            state_vec = getattr(agent, "_state_vec", None)
            state_info: dict[str, Any] = {"dim": 0, "norm": 0.0}
            if state_vec is not None and isinstance(state_vec, np.ndarray):
                state_info = {"dim": int(state_vec.shape[0]), "norm": round(float(np.linalg.norm(state_vec)), 4)}
            data["state"] = state_info

            # Health watchdog counters
            hw = getattr(agent, "health", None)
            health_counters: dict[str, Any] = {}
            if hw is not None:
                health_counters = {
                    "ocr_empty_threshold": getattr(hw, 'ocr_empty_threshold', 30),
                    "mem_fail_timeout_s": getattr(hw, 'mem_fail_timeout_s', 10.0),
                    "capture_min_fps": getattr(hw, 'capture_min_fps', 2.0),
                }
            data["health_config"] = health_counters

            data["ts"] = time.time()
            return jsonify(data)
        except Exception as e:  # noqa: BLE001
            logger.exception("/api/stats failed")
            return jsonify({"error": str(e)}), 500

    return app

# ── Standalone mode (no live agent; reads log tail) ────────────────────────
def _standalone(port: int = 8090, host: str = "127.0.0.1") -> None:
    """Run the dashboard without a live agent – serves static files + stub API."""
    app = create_app(agent=None)
    logger.info("Standalone dashboard on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False, use_reloader=False)

# ── CLI entry point ─────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Green Vision Web Dashboard")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    _standalone(port=args.port, host=args.host)
    return 0

if __name__ == "__main__":
    sys.exit(main())
