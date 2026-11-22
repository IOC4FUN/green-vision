#!/usr/bin/env python3
"""
Green Vision – Entry Point.
Usage:
    python main.py                  # full autonomous mode (launches game)
    python main.py --no-launch      # attach to already-running game
    python main.py --dashboard      # show live dashboard instead of playing
    python main.py --train-steps N  # run N training steps then exit
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(name)-20s] %(levelname)-5s %(message)s"
    datefmt = "%H:%M:%S"
    # Ensure log directory exists before creating the FileHandler
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "green_vision.log", encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


def _run_dashboard(agent) -> int:
    """Observe-only loop that prints a lightweight status line."""
    import time
    logger = logging.getLogger("main")
    logger.info("Dashboard mode – monitoring only, no input.")
    try:
        agent.capture.start()
    except Exception as e:  # noqa: BLE001
        logger.warning("Screen capture failed to start: %s", e)
    try:
        while True:
            fobj = agent.capture.latest_frame()
            if fobj is not None and fobj.raw is not None:
                ocr_text = ""
                if agent.config.get("use_ocr", True) and agent.ocr.available:
                    ocr_text = (agent.ocr.process(fobj.raw).raw_text or "")
                mem_state = {}
                if agent.mem_reader.pid:
                    try:
                        mem_state = agent.mem_reader.read_state().values or {}
                    except Exception:  # noqa: BLE001
                        pass
                agent._update_state_vector(
                    fobj.resized if fobj.resized is not None else fobj.raw,
                    ocr_text, mem_state,
                )
                dash = agent.planner.get_dashboard_data() or {}
                print(
                    f"\r[{dash.get('total_decisions', agent.planner.total_decisions)} decisions] "
                    f"scenario={dash.get('scenario', agent.planner.current_scenario)}",
                    end="", flush=True,
                )
            time.sleep(1)
    except KeyboardInterrupt:
        print()
    finally:
        agent.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Green Vision – Autonomous Geoplanetical AI")
    parser.add_argument("--config", default=str(ROOT / "config" / "settings.yaml"), help="Config file path")
    parser.add_argument("--no-launch", action="store_true", help="Don't auto-launch the game")
    parser.add_argument("--dashboard", action="store_true", help="Show dashboard only (no input injection)")
    parser.add_argument("--web-dashboard", action="store_true", help="Start live web dashboard (Flask) on --port")
    parser.add_argument("--port", type=int, default=8090, help="Port for web dashboard (default 8090)")
    parser.add_argument("--train-steps", type=int, default=0, help="Exit after N decision steps")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    # Ensure log dir exists
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "data" / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "knowledge").mkdir(parents=True, exist_ok=True)

    from src.agent import GreenVisionAgent
    agent = GreenVisionAgent(config_path=args.config)
    if args.no_launch:
        agent.config["auto_launch_game"] = False

    if args.dashboard:
        return _run_dashboard(agent)

    # ── Web dashboard (Flask) ─────────────────────────────────────────────
    if args.web_dashboard:
        import threading
        from src.dashboard import create_app
        flask_app = create_app(agent=agent)
        def _serve():
            flask_app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
        t = threading.Thread(target=_serve, daemon=True, name="web-dashboard")
        t.start()
        logger.info("Web dashboard live at http://127.0.0.1:%d", args.port)

    max_steps = args.train_steps if args.train_steps > 0 else None
    if max_steps is not None:
        logger.info("Training mode: %d steps then exit.", max_steps)
    agent.start(max_steps=max_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
