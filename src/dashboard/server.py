"""
Green Vision – Web Dashboard Server (Flask).
Serves the static front-end and exposes a JSON API for live agent state.

Usage:
    python main.py --web-dashboard          # integrated with agent loop
    python src/dashboard/server.py --port 8090   # standalone (reads log file)

API endpoints:
    GET /            → index.html
    GET /style.css   → static assets
    GET /app.js      → static assets
    GET /api/state   → full agent state snapshot
    GET /api/health  → subsystem health metrics
    GET /api/thoughts→ recent thought stream
    GET /api/goals   → active + completed goals
    GET /api/actions → action history
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("dashboard.server")

# ── Flask app factory ───────────────────────────────────────────────────────

def create_app(agent: Optional[Any] = None) -> "Flask":
    from flask import Flask, jsonify, send_from_directory

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
            health_data = {}
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
