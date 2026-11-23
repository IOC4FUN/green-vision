"""
Game controller for Green Vision – Windows edition.
Handles:
  • Auto-launching Geoplanetical via Steam (steam://rungameid/940230).
  • Waiting for the game process to appear and stabilize.
  • Tracking game lifecycle (launching → running → paused → exited).
  • Scenario rotation (different countries / roles per session).
  • Focus management (ensure game window is focused before input injection).
"""
from __future__ import annotations
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger("game.controller")

# Default Steam path on Windows
_STEAM_CANDIDATES = [
    r"C:\Program Files (x86)\Steam\steam.exe",
    r"C:\Program Files\Steam\steam.exe",
]


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


def _find_steam() -> Optional[str]:
    """Locate steam.exe on this machine."""
    for p in _STEAM_CANDIDATES:
        if os.path.isfile(p):
            return p
    # Try PATH
    try:
        out = subprocess.check_output(["where", "steam.exe"], text=True, timeout=5).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            return lines[0]
    except Exception:
        pass
    return None


def _find_game_process() -> Optional[int]:
    """Find Geoplanetical.exe PID via tasklist (Windows-native)."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq Geoplanetical.exe", "/FO", "CSV", "/NH"],
            text=True, timeout=5,
        ).strip()
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split(',')]
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    except Exception:
        pass
    return None


class GameController:
    """
    Manages the Geoplanetical game lifecycle on Windows.
    Usage:
        gc = GameController(steam_app_id=940230)
        gc.launch()                    # starts via Steam protocol handler
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
        """Launch the game via Steam protocol handler (Windows)."""
        logger.info("Launching Geoplanetical (Steam app %d)...", self._app_id)
        self.status = GameStatus.LAUNCHING
        try:
            # Method 1: steam:// URL protocol (works if Steam is running or installed)
            os.startfile(f"steam://rungameid/{self._app_id}")
            return True
        except Exception as e:
            logger.warning("os.startfile failed (%s), trying direct Steam path...", e)
            steam_exe = _find_steam()
            if steam_exe:
                try:
                    subprocess.Popen(
                        [steam_exe, "-applaunch", str(self._app_id)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
                except Exception as e2:
                    logger.error("Failed to launch via Steam binary: %s", e2)
            else:
                logger.error("Steam not found. Install Steam and own Geoplanetical.")
            self.status = GameStatus.ERROR
            return False

    def wait_for_ready(self, timeout: int | None = None) -> bool:
        """Block until the game process is detected and stable."""
        timeout = timeout or self._timeout
        t0 = time.time()
        while time.time() - t0 < timeout:
            pid = _find_game_process()
            if pid:
                # Wait a few extra seconds for the game to fully initialise
                time.sleep(5)
                pid2 = _find_game_process()
                if pid2:
                    self._pid = pid2
                    self.status = GameStatus.RUNNING
                    self.session_start = time.time()
                    logger.info("Game ready! PID=%d (waited %.1fs)", self._pid, time.time() - t0)
                    return True
            time.sleep(2)
        logger.error("Timed out waiting for game (%ds)", timeout)
        self.status = GameStatus.ERROR
        return False

    @property
    def is_running(self) -> bool:
        if self.status != GameStatus.RUNNING:
            return False
        return _find_game_process() is not None

    @property
    def pid(self) -> Optional[int]:
        return self._pid

    # ── Focus management ─────────────────────────────────────────
    def focus_game_window(self) -> bool:
        """Bring the game window to front (for input injection)."""
        try:
            from ..input.virtual_input import VirtualInputDevice
            return VirtualInputDevice.focus_game("Geoplanetical")
        except Exception as e:
            logger.warning("Could not focus game window: %s", e)
            return False

    def restore_focus(self) -> None:
        """Restore focus to whatever was previously focused (best-effort)."""
        pass  # On Windows we don't track previous window; user can Alt-Tab back.

    # ── Scenario management ──────────────────────────────────────
    def set_scenario(self, country: str, role: str = "government") -> None:
        self.scenario = ScenarioConfig(country=country, role=role)
        logger.info("Scenario: %s (%s)", country, role)

    def rotate_scenario(self, scenarios: list[dict]) -> None:
        """Cycle through a list of scenario dicts {country, role}."""
        idx = int(time.time()) % len(scenarios)
        s = scenarios[idx]
        self.set_scenario(s.get("country", "Germany"), s.get("role", "government"))
