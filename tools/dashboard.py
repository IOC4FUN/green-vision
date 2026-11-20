#!/usr/bin/env python3
"""
Green Vision – Live Dashboard (terminal-based).
Shows real-time agent state, recent thoughts, and game metrics.
Run alongside the agent or in dashboard-only mode:
    python tools/dashboard.py              # tail the log file for structured output
    python tools/dashboard.py --standalone # observe-only mode with live rendering
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))


def print_dashboard(data: dict) -> None:
    """Render a compact terminal dashboard."""
    lines = []
    lines.append("\033[2J\033[H")  # clear screen
    lines.append("┌─────────────────────────────────────────────────────┐")
    lines.append(f"│  GREEN VISION – Agent Dashboard                      │")
    scenario = data.get('scenario', 'N/A')
    lines.append(f"│  Scenario: {scenario:<41} │")
    lines.append(f"│  Decisions made: {data.get('total_decisions', 0):<37} │")
    lines.append("├─────────────────────────────────────────────────────┤")
    lines.append("│ Active Goals:")
    for g in data.get("active_goals", [])[:5]:
        prog = g.get("progress", 0)
        bar = "█" * int(prog * 10) + "░" * (10 - int(prog * 10))
        desc = g.get("desc", g.get("description", "?"))[:30]
        pri = g.get("priority", "?")
        lines.append(f"│   [{pri}] {bar} {desc}")
    lines.append("├─────────────────────────────────────────────────────┤")
    lines.append("│ Recent Thoughts:")
    for t in data.get("recent_thoughts", [])[-5:]:
        ts = time.strftime("%H:%M:%S", time.localtime(t["ts"]))
        conf = t.get("confidence", 0)
        reasoning = t.get("reasoning", "")
        decision = t.get("decision", "?")
        lines.append(f"│   {ts} → {decision} (conf={conf:.2f})")
        if len(reasoning) > 60:
            lines.append(f"│         {reasoning[:60]}...")
    lines.append("├─────────────────────────────────────────────────────┤")
    lines.append("│ Last Actions:")
    for a in data.get("action_history_last_50", [])[-8:]:
        ts = time.strftime("%H:%M:%S", time.localtime(a["ts"]))
        action = a.get("action", "?")
        lines.append(f"│   {ts}  {action}")
    lines.append("└─────────────────────────────────────────────────────┘")
    print("\n".join(lines))


def _run_standalone(agent) -> None:
    """Observe-only loop with dashboard rendering."""
    agent.capture.start()
    try:
        while True:
            fobj = agent.capture.latest_frame()
            if fobj is not None and fobj.raw is not None:
                ocr_text = ""
                if agent.config.get("use_ocr", True) and agent.ocr.available:
                    ocr_res = agent.ocr.process(fobj.raw)
                    ocr_text = ocr_res.raw_text or ""
                mem_state: dict = {}
                if agent.mem_reader.pid:
                    try:
                        mem_state = agent.mem_reader.read_state().values or {}
                    except Exception:  # noqa: BLE001
                        pass
                resized = fobj.resized if fobj.resized is not None else fobj.raw
                agent._update_state_vector(resized, ocr_text, mem_state)
            dash = agent.planner.get_dashboard_data()
            print_dashboard(dash)
            time.sleep(2)
    except KeyboardInterrupt:
        print()
    finally:
        agent.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Green Vision Dashboard")
    parser.add_argument("--standalone", action="store_true",
                        help="Observe-only mode with live rendering")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    from src.agent import GreenVisionAgent
    agent = GreenVisionAgent(config_path=str(ROOT / "config" / "settings.yaml"))

    if args.standalone:
        _run_standalone(agent)
    else:
        # Tail the log file for structured output
        log_file = ROOT / "logs" / "green_vision.log"
        print(f"Tailing {log_file} – Ctrl+C to stop\n")
        try:
            with open(log_file) as f:
                f.seek(0, 2)  # seek to end
                while True:
                    line = f.readline()
                    if line:
                        print(line.rstrip())
                    else:
                        time.sleep(1)
        except (KeyboardInterrupt, FileNotFoundError):
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
