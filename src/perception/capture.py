"""
Screen capture for Green Vision.
Uses `mss` (Windows GDI/DXGI backend).
On Windows this captures the primary monitor via BitBlt – no admin needed.
The game should run in borderless-windowed mode so it's always visible
to the screen grabber even if another window has focus.
Key design:
  • Captures at configurable FPS in a background thread.
  • Stores only the latest frame (ring buffer of N for temporal context).
  • Resizes to CNN input size (224×224) for the brain.
  • Exposes both raw PIL image and numpy array.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("perception.capture")

try:
    import mss
    import mss.tools
except ImportError:
    mss = None
    logger.warning("mss not installed – screen capture unavailable until: pip install mss")

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class Frame:
    """A single captured frame with metadata."""
    timestamp: float
    raw: np.ndarray            # H×W×3 uint8 (BGR or RGB depending on source)
    resized: Optional[np.ndarray] = None   # 224×224×3 for CNN
    ocr_text: str = ""
    window_title: str = ""


class ScreenCapture:
    """
    Background-thread screen capturer.

    Usage:
        cap = ScreenCapture(fps=5, target_size=(224, 224))
        cap.start()
        frame = cap.latest_frame()   # non-blocking, returns most recent
        ...
        cap.stop()
    """

    def __init__(
        self,
        fps: int = 5,
        monitor_index: int = 1,       # 1-based; 0 = all monitors combined
        target_size: tuple[int, int] = (224, 224),
        buffer_len: int = 16,         # keep last N frames for temporal context
    ):
        if mss is None:
            raise RuntimeError("mss library not available")

        self._fps = fps
        self._interval = 1.0 / fps
        self._monitor_index = monitor_index
        self._target_size = target_size
        self._buffer: deque[Frame] = deque(maxlen=buffer_len)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sct: Optional[mss.mss] = None
        self.frame_count = 0

    # ── lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ScreenCapture")
        self._thread.start()
        logger.info("Screen capture started @ %d fps", self._fps)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._sct:
            try:
                self._sct.close()
            except Exception:
                pass
        logger.info("Screen capture stopped (%d frames captured total)", self.frame_count)

    # ── public API ───────────────────────────────────────────────

    def latest_frame(self) -> Optional[Frame]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def recent_frames(self, n: int = 4) -> list[Frame]:
        """Return up to N most recent frames (oldest first)."""
        with self._lock:
            items = list(self._buffer)[-n:]
            return items

    @property
    def is_capturing(self) -> bool:
        return self._running

    # ── internal loop ────────────────────────────────────────────

    def _loop(self) -> None:
        self._sct = mss.mss()
        monitor = self._sct.monitors[self._monitor_index]
        interval = self._interval

        while self._running:
            t0 = time.perf_counter()
            try:
                img = self._sct.grab(monitor)
                arr = np.array(img, dtype=np.uint8)[:, :, :3]  # BGRA → BGR
                ts = time.time()

                # Resize for CNN input
                resized = self._resize(arr)

                frame = Frame(timestamp=ts, raw=arr, resized=resized)

                with self._lock:
                    self._buffer.append(frame)
                self.frame_count += 1

            except Exception as e:
                logger.error("Capture error: %s", e)

            # Sleep to maintain FPS
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    @staticmethod
    def _resize(arr: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        """Resize using PIL (fast, no OpenCV dependency at import time)."""
        if Image is None:
            return arr[:target[1], :target[0]]
        pil_img = Image.fromarray(arr[:, :, ::-1])  # BGR→RGB
        pil_img = pil_img.resize(target, Image.LANCZOS)
        return np.array(pil_img)  # H×W×3 RGB
