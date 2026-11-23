"""
Subsystem health watchdog — monitors OCR, memory-reader, capture FPS.

If a subsystem degrades beyond thresholds the agent can pause decisions
or log warnings instead of acting on stale data.

> Keep in sync with code — update this file whenever the module API changes.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("utils.health")


@dataclass
class HealthStatus:
    ocr_ok: bool = True
    mem_reader_ok: bool = True
    capture_fps_ok: bool = True
    last_ocr_empty_streak: int = 0
    last_mem_fail_ts: float = 0.0
    last_capture_ts: float = 0.0
    overall_ok: bool = True

    @property
    def degraded_subsystems(self) -> list[str]:
        out = []
        if not self.ocr_ok:
            out.append("ocr")
        if not self.mem_reader_ok:
            out.append("memory_reader")
        if not self.capture_fps_ok:
            out.append("capture")
        return out


class HealthWatchdog:
    """
    Call .tick() once per main-loop iteration.
    Pass optional signals (ocr_text, mem_state, frame_ts).
    """

    def __init__(
        self,
        ocr_empty_threshold: int = 30,       # consecutive empty OCR frames before flagging
        mem_fail_timeout_s: float = 10.0,     # seconds since last successful read
        capture_min_fps: float = 2.0,         # below this → degraded
        check_interval_s: float = 5.0,        # how often to log a summary
    ):
        self.ocr_empty_threshold = ocr_empty_threshold
        self.mem_fail_timeout_s = mem_fail_timeout_s
        self.capture_min_fps = capture_min_fps
        self.check_interval_s = check_interval_s

        self._ocr_empty_streak = 0
        self._last_mem_ok_ts = time.time()
        self._last_frame_ts = time.time()
        self._frame_count = 0
        self._fps_window_start = time.time()
        self._fps_window_frames = 0
        self._current_fps = 0.0
        self._last_log_ts = 0.0

    def tick(
        self,
        ocr_text: str = "",
        mem_state: Optional[dict] = None,
        frame_received: bool = True,
    ) -> HealthStatus:
        now = time.time()

        # ── OCR ─────────────────────────────────────────────────
        if ocr_text and ocr_text.strip():
            self._ocr_empty_streak = 0
        else:
            self._ocr_empty_streak += 1

        # ── Memory reader ───────────────────────────────────────
        if mem_state is not None and len(mem_state) > 0:
            self._last_mem_ok_ts = now

        # ── Capture FPS ─────────────────────────────────────────
        if frame_received:
            self._frame_count += 1
            self._fps_window_frames += 1
            elapsed = now - self._fps_window_start
            if elapsed >= 2.0:
                self._current_fps = self._fps_window_frames / elapsed
                self._fps_window_frames = 0
                self._fps_window_start = now

        # ── Evaluate ────────────────────────────────────────────
        status = HealthStatus(
            ocr_ok=self._ocr_empty_streak < self.ocr_empty_threshold,
            mem_reader_ok=(now - self._last_mem_ok_ts) < self.mem_fail_timeout_s,
            capture_fps_ok=self._current_fps >= self.capture_min_fps or self._current_fps == 0.0,  # 0 = not enough data yet
            last_ocr_empty_streak=self._ocr_empty_streak,
            last_mem_fail_ts=now - self._last_mem_ok_ts,
            last_capture_ts=now - self._last_frame_ts,
        )
        status.overall_ok = status.ocr_ok and status.mem_reader_ok and status.capture_fps_ok

        # Periodic log
        if now - self._last_log_ts >= self.check_interval_s:
            self._last_log_ts = now
            if not status.overall_ok:
                logger.warning(
                    "Health degraded: %s (ocr_streak=%d, mem_age=%.1fs, fps=%.1f)",
                    ",".join(status.degraded_subsystems),
                    status.last_ocr_empty_streak,
                    status.last_mem_fail_ts,
                    self._current_fps,
                )
            else:
                logger.debug("All subsystems healthy (fps=%.1f)", self._current_fps)

        return status
