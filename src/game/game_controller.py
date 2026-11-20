"""
Game controller for Green Vision.

Handles:
  • Auto-launching Geoplanetical via Steam (steam://rungameid/940230).
  • Waiting for the Proton/Wine process to appear and stabilize.
  • Tracking game lifecycle (launching → running → paused → exited).
  • Scenario rotation (different countries / roles per session).
  • Focus management (ensure game window is focused before input injection).
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

logger = logging.getLogger("game.controller")


class GameStatus(Enum):
    NOT_LAUNCHED = auto()
    LAUNCHING = auto()
    RUNNING = auto()
    PAUSED = auto()
    EXITED = auto()
    ERROR = auto()


@dataclass
class ScenarioConfig:
    country: str
    role: str  # "government" | "opposition"
    difficulty: str = "normal"


class GameController:
    """
    Manages the Geoplanetical game lifecycle.

    Usage:
        gc = GameController(steam_app_id=940230)
        gc.launch()                    # starts via Steam
        gc.wait_for_ready(timeout=120) # blocks until process detected
        ...
        gc.is_running                  # poll status
    """

    def __init__(self, steam_app_id: int = 940230, launch_timeout_s: int = 120):
        self._app_id = steam_app_id
        self._timeout = launch_timeout_s
        self.status = GameStatus.NOT_LAUNCHED
        self._pid: Optional[int] = None
        self.scenario: ScenarioConfig = ScenarioConfig("Germany", "government")
        self.session_start: float = 0.0
        self.total_play_time_s: float = 0.0

    # ── Launch ───────────────────────────────────────────────────

    def launch(self) -> bool:
        """Launch the game via Steam protocol handler."""
        logger.info("Launching Geoplanetical (Steam app %d)...", self._app_id)
        self.status = GameStatus.LAUNCHING
        try:
            subprocess.Popen(
                ["steam", f"steam://rungameid/{self._app_id}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except FileNotFoundError:
            logger.error("Steam binary not found in PATH")
            self.status = GameStatus.ERROR
            return False

    def wait_for_ready(self, timeout: int | None = None) -> bool:
        """Block until the game process is detected and stable."""
        timeout = timeout or self._timeout
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._find_process():
                # Wait a few extra seconds for Proton to fully init
                time.sleep(5)
                if self._find_process():
                    self.status = GameStatus.RUNNING
                    self.session_start = time.time()
                    logger.info("Game ready! PID=%d (waited %.1fs)", self._pid, time.time()-t0)
                    return True
            time.sleep(2)
        logger.error("Timed out waiting for game (%ds)", timeout)
        self.status = GameStatus.ERROR
        return False

    @property
    def is_running(self) -> bool:
        if self.status != GameStatus.RUNNING:
            return False
        return self._find_process()

    @property
    def pid(self) -> Optional[int]:
        return self._pid

    # ── Focus management ─────────────────────────────────────────

    def focus_game_window(self) -> bool:
        """Bring the game window to front (for input injection)."""
        try:
            # Try wmctrl first, fall back to xdotool
            result = subprocess.run(
                ["wmctrl", "-a", "Geoplanetical"],
                capture_output=True, timeout=3
            )
            if result.returncode == 0:
                return True
            # Fallback: xdotool search + windowactivate
            subprocess.run(
                ["xdotool", "search", "--name", "Geoplanetical", "windowactivate"],
                capture_output=True, timeout=3
            )
            return True
        except Exception as e:
            logger.warning("Could not focus game window: %s", e)
            return False

    def restore_focus(self) -> None:
        """Restore focus to whatever was previously focused."""
        try:
            subprocess.run(["wmctrl", "-R", ":ACTIVE:"], capture_output=True, timeout=2)
        except Exception:
            pass

    # ── Scenario management ──────────────────────────────────────

    def set_scenario(self, country: str, role: str = "government") -> None:
        self.scenario = ScenarioConfig(country=country, role=role)
        logger.info("Scenario: %s (%s)", country, role)

    def rotate_scenario(self, scenarios: list[dict]) -> None:
        """Cycle through a list of scenario dicts {country, role}."""
        idx = int(time.time()) % len(scenarios)
        s = scenarios[idx]
        self.set_scenario(s.get("country", "Germany"), s.get("role", "government"))

    # ── Internal ─────────────────────────────────────────────────

    def _find_process(self) -> bool:
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", "Geoplanetical.exe"], text=True, timeout=3
            ).strip()
            pids = [int(p) for p in out.split() if p.isdigit()]
            if pids:
                self._pid = max(pids)  # largest PID = most recent launch
                return True
        except Exception:
            pass
        self._pid = None
        return False
