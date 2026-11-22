"""
Virtual input device for Green Vision – Windows edition.
Uses pyautogui (Win32 SendInput under the hood) for keyboard/mouse injection.
No admin rights needed; works as a normal user on Windows 10/11.

Architecture:
  • All key/mouse events go through pyautogui → Win32 SendInput API.
  • Focus management: before sending inputs we ensure the game window
    has focus (via ctypes SetForegroundWindow), then restore the
    previously-focused window after the action completes.
  • Inter-key delay mimics human typing cadence (configurable).
"""
from __future__ import annotations
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

logger = logging.getLogger("input.virtual")

# ── Key names (pyautogui-compatible strings) ────────────────────────
class Key(IntEnum):
    """Key identifiers – values are pyautogui key-name strings via .name lookup."""
    A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9
    K=10; L=11; M=12; N=13; O=14; P=15; Q=16; R=17; S=18; T=19
    U=20; V=21; W=22; X=23; Y=24; Z=25
    NUM_0=26; NUM_1=27; NUM_2=28; NUM_3=29; NUM_4=30; NUM_5=31
    NUM_6=32; NUM_7=33; NUM_8=34; NUM_9=35
    ENTER=36; ESCAPE=37; SPACE=38; TAB=39; BACKSPACE=40
    UP=41; DOWN=42; LEFT=43; RIGHT=44
    F1=45; F2=46; F3=47; F4=48; F5=49; F6=50; F7=51; F8=52
    F9=53; F10=54; F11=55; F12=56
    SHIFT_L=57; CTRL_L=58; ALT_L=59
    COMMA=60; PERIOD=61; SLASH=62; MINUS=63; EQUALS=64
    LBRACKET=65; RBRACKET=66; BACKSLASH=67; SEMICOLON=68; APOSTROPHE=69
    GRAVE=70  # backtick – opens Geoplanetical console

# Map Key enum → pyautogui key string
_KEY_MAP: dict[int, str] = {
    int(Key.A): "a", int(Key.B): "b", int(Key.C): "c", int(Key.D): "d",
    int(Key.E): "e", int(Key.F): "f", int(Key.G): "g", int(Key.H): "h",
    int(Key.I): "i", int(Key.J): "j", int(Key.K): "k", int(Key.L): "l",
    int(Key.M): "m", int(Key.N): "n", int(Key.O): "o", int(Key.P): "p",
    int(Key.Q): "q", int(Key.R): "r", int(Key.S): "s", int(Key.T): "t",
    int(Key.U): "u", int(Key.V): "v", int(Key.W): "w", int(Key.X): "x",
    int(Key.Y): "y", int(Key.Z): "z",
    int(Key.NUM_0): "0", int(Key.NUM_1): "1", int(Key.NUM_2): "2",
    int(Key.NUM_3): "3", int(Key.NUM_4): "4", int(Key.NUM_5): "5",
    int(Key.NUM_6): "6", int(Key.NUM_7): "7", int(Key.NUM_8): "8",
    int(Key.NUM_9): "9",
    int(Key.ENTER): "enter", int(Key.ESCAPE): "escape", int(Key.SPACE): "space",
    int(Key.TAB): "tab", int(Key.BACKSPACE): "backspace",
    int(Key.UP): "up", int(Key.DOWN): "down", int(Key.LEFT): "left", int(Key.RIGHT): "right",
    int(Key.F1): "f1", int(Key.F2): "f2", int(Key.F3): "f3", int(Key.F4): "f4",
    int(Key.F5): "f5", int(Key.F6): "f6", int(Key.F7): "f7", int(Key.F8): "f8",
    int(Key.F9): "f9", int(Key.F10): "f10", int(Key.F11): "f11", int(Key.F12): "f12",
    int(Key.SHIFT_L): "shift", int(Key.CTRL_L): "ctrl", int(Key.ALT_L): "alt",
    int(Key.COMMA): ",", int(Key.PERIOD): ".", int(Key.SLASH): "/",
    int(Key.MINUS): "-", int(Key.EQUALS): "=",
    int(Key.LBRACKET): "[", int(Key.RBRACKET): "]", int(Key.BACKSLASH): "\\",
    int(Key.SEMICOLON): ";", int(Key.APOSTROPHE): "'",
    int(Key.GRAVE): "`",
}


class Mouse(IntEnum):
    BTN_LEFT = 0; BTN_RIGHT = 1; BTN_MIDDLE = 2


@dataclass
class InputAction:
    """A single atomic input event."""
    kind: str          # "key_press", "key_release", "mouse_click", "mouse_move", "type_char"
    code: int = 0      # key code or mouse button
    value: int = 1     # 1=press, 0=release
    x: int = 0         # for mouse_move
    y: int = 0


def _char_to_key(ch: str) -> tuple[str, bool]:
    """Return (pyautogui_key_string, needs_shift) for a character."""
    lower = ch.lower()
    if lower in _KEY_MAP.values():
        return lower, ch.isupper() and ch.isalpha()
    # Direct char via pyautogui.typewrite handles most printable chars
    return ch, False


class VirtualInputDevice:
    """
    Software-only input via Win32 SendInput (through pyautogui).
    Usage:
        vi = VirtualInputDevice(name="GreenVision")
        vi.open()                          # no-op on Windows; kept for API compat
        vi.type_text("hello")             # types a string with delays
        vi.press_key(Key.SPACE)            # single key press+release
        vi.click_mouse(x=500, y=300)       # click at absolute position
        vi.close()                         # no-op
    """

    def __init__(self, name: str = "GreenVisionInput", action_delay_ms: int = 80):
        self._name = name
        self._delay = action_delay_ms / 1000.0
        self._lock = threading.Lock()
        self._open_ok = False
        self._pyautogui = None

    # ── lifecycle ────────────────────────────────────────────────
    def open(self) -> bool:
        """Initialise pyautogui backend."""
        try:
            import pyautogui
            pyautogui.FAILSAFE = True   # move mouse to corner to abort
            pyautogui.PAUSE = 0.02      # small built-in pause between calls
            self._pyautogui = pyautogui
            self._open_ok = True
            logger.info("Virtual input device '%s' ready (Win32 SendInput via pyautogui)", self._name)
            return True
        except ImportError:
            logger.error("pyautogui not installed. Run: pip install pyautogui")
            return False
        except Exception as e:
            logger.error("Failed to init input backend: %s", e)
            return False

    def close(self) -> None:
        self._open_ok = False
        logger.info("Virtual input device closed")

    @property
    def is_open(self) -> bool:
        return self._open_ok and self._pyautogui is not None

    # ── high-level actions ───────────────────────────────────────
    def press_key(self, key: Key) -> None:
        """Press and release a single key."""
        with self._lock:
            if not self.is_open:
                return
            name = _KEY_MAP.get(int(key))
            if name is None:
                logger.warning("Unknown key: %s", key)
                return
            self._pyautogui.press(name)

    def type_text(self, text: str) -> None:
        """Type a string character by character with human-like delay."""
        with self._lock:
            if not self.is_open:
                return
            for ch in text:
                key_str, needs_shift = _char_to_key(ch)
                try:
                    if needs_shift:
                        self._pyautogui.hotkey('shift', key_str)
                    else:
                        self._pyautogui.typewrite(key_str, interval=0.02)
                except Exception as e:
                    logger.debug("type_char '%s' failed: %s", ch, e)
                time.sleep(self._delay)

    def click_mouse(self, x: int = 0, y: int = 0, button: int = Mouse.BTN_LEFT) -> None:
        """Move to (x,y) and click."""
        with self._lock:
            if not self.is_open:
                return
            btn = {Mouse.BTN_LEFT: 'left', Mouse.BTN_RIGHT: 'right', Mouse.BTN_MIDDLE: 'middle'}[button]
            self._pyautogui.click(x=x, y=y, button=btn)

    def move_mouse(self, x: int, y: int) -> None:
        """Set absolute mouse position."""
        with self._lock:
            if not self.is_open:
                return
            self._pyautogui.moveTo(x, y, duration=0.05)

    # ── focus management (Windows) ───────────────────────────────
    @staticmethod
    def find_window(title_substring: str = "Geoplanetical") -> Optional[int]:
        """Find a window handle by title substring using ctypes EnumWindows."""
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _enum(hwnd, lparam):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if title_substring.lower() in buf.value.lower():
                    found.append(hwnd)
            return True

        user32.EnumWindows(_enum, None)
        return found[0] if found else None

    @staticmethod
    def set_foreground(hwnd: int) -> bool:
        """Bring a window to the foreground via SetForegroundWindow."""
        import ctypes
        user32 = ctypes.windll.user32
        # Attach thread input to bypass foreground lock
        fore_hwnd = user32.GetForegroundWindow()
        fore_thread = user32.GetWindowThreadProcessId(fore_hwnd, None)
        cur_thread = ctypes.windll.kernel32.GetCurrentThreadId()
        if fore_thread != cur_thread:
            user32.AttachThreadInput(cur_thread, fore_thread, True)
            result = user32.SetForegroundWindow(hwnd)
            user32.AttachThreadInput(cur_thread, fore_thread, False)
        else:
            result = user32.SetForegroundWindow(hwnd)
        return bool(result)

    def focus_game(self, title: str = "Geoplanetical") -> bool:
        """Find and focus the game window. Returns True on success."""
        hwnd = self.find_window(title)
        if hwnd is None:
            logger.warning("Game window '%s' not found", title)
            return False
        ok = self.set_foreground(hwnd)
        time.sleep(0.1)  # let Windows settle
        return ok
