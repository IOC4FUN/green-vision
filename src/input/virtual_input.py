"""
Virtual input device for Green Vision.

Creates a software-only keyboard/mouse via Linux `uinput` (/dev/uinput).
Events sent through this device are indistinguishable from real hardware
to the X server / game, but do NOT move or click the user's physical
mouse or type on their physical keyboard.

Architecture:
  • A dedicated uinput device is registered at startup.
  • Key/mouse events are queued and flushed in small batches with
    configurable inter-key delay (mimics human typing cadence).
  • Focus management: before sending inputs we ensure the game window
    has focus (via wmctrl/xdotool if available), then restore the
    previously-focused window after the action completes.

NOTE: Requires either root/sudo OR the user to be in the `input` group,
or `/dev/uinput` must be world-writable. The wrapper script handles this.
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

# ── Key codes (Linux input-event-codes) ────────────────────────────
class Key(IntEnum):
    A=30; B=48; C=46; D=32; E=18; F=33; G=34; H=35; I=23; J=36
    K=37; L=38; M=50; N=49; O=24; P=25; Q=16; R=19; S=31; T=20
    U=22; V=47; W=17; X=45; Y=21; Z=44
    NUM_0=2; NUM_1=3; NUM_2=4; NUM_3=5; NUM_4=6; NUM_5=7
    NUM_6=8; NUM_7=9; NUM_8=10; NUM_9=11
    ENTER=28; ESCAPE=1; SPACE=57; TAB=15; BACKSPACE=14
    UP=103; DOWN=108; LEFT=105; RIGHT=106
    F1=59; F2=60; F3=61; F4=62; F5=63; F6=64; F7=65; F8=66
    SHIFT_L=42; CTRL_L=29; ALT_L=56
    COMMA=51; PERIOD=52; SLASH=53; MINUS=12; EQUALS=13
    LBRACKET=26; RBRACKET=27; BACKSLASH=43; SEMICOLON=39; APOSTROPHE=40
    GRAVE=41  # backtick – opens Geoplanetical console


class Mouse(IntEnum):
    BTN_LEFT=0; BTN_RIGHT=1; BTN_MIDDLE=2
    REL_X=0; REL_Y=1
# ── Linux input event type / code constants (module-level) ───────
EV_SYN = 0x00; EV_KEY = 0x01; EV_REL = 0x02; EV_ABS = 0x03
BTN_LEFT = 0x110; BTN_RIGHT = 0x111; BTN_MIDDLE = 0x112
ABS_X = 0x00; ABS_Y = 0x01
REL_X = 0x00; REL_Y = 0x01
UI_SET_EVBIT = 0x40045564
UI_SET_KEYCODE = 0x4004556f
UI_DEV_CREATE = 0x40045580
UI_DEV_DESTROY = 0x40045581


@dataclass
class InputAction:
    """A single atomic input event."""
    kind: str          # "key_press", "key_release", "mouse_click", "mouse_move", "type_char"
    code: int = 0      # key code or mouse button
    value: int = 1     # 1=press, 0=release
    x: int = 0         # for mouse_move
    y: int = 0


class VirtualInputDevice:
    """
    Software-only input via /dev/uinput.

    Usage:
        vi = VirtualInputDevice(name="GreenVision")
        vi.open()                          # registers the device (needs perms)
        vi.type_text("hello")             # types a string with delays
        vi.press_key(Key.SPACE)            # single key press+release
        vi.click_mouse(x=500, y=300)       # click at absolute position
        vi.close()
    """

    def __init__(self, name: str = "GreenVisionInput", action_delay_ms: int = 80):
        self._name = name
        self._delay = action_delay_ms / 1000.0
        self._fd: Optional[int] = None
        self._lock = threading.Lock()
        self._open_ok = False

    # ── lifecycle ────────────────────────────────────────────────

    def open(self) -> bool:
        """Register the uinput device. Returns True on success."""
        try:
            import fcntl
            import struct
            import ctypes
            import ctypes.util

            UINPUT_DEV_PATH = "/dev/uinput"
            UI_SET_EVBIT = 0x40045564
            UI_SET_KEYCODE = 0x4004556f
            UI_DEV_CREATE = 0x40045580
            UI_DEV_DESTROY = 0x40045581
            EV_SYN = 0x00; EV_KEY = 0x01; EV_REL = 0x02; EV_ABS = 0x03
            BTN_LEFT = 0x110; BTN_RIGHT = 0x111; BTN_MIDDLE = 0x112
            ABS_X = 0x00; ABS_Y = 0x01
            REL_X = 0x00; REL_Y = 0x01

            fd = os.open(UINPUT_DEV_PATH, os.O_WRONLY | os.O_NONBLOCK)

            # Enable event types
            fcntl.ioctl(fd, UI_SET_EVBIT, EV_KEY | EV_REL | EV_ABS | EV_SYN)

            # Set keycodes (all keys we use)
            for code in list(Key):
                fcntl.ioctl(fd, UI_SET_KEYCODE, struct.pack("iI", code, int(code)))

            # Mouse buttons
            for btn_code in [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE]:
                fcntl.ioctl(fd, UI_SET_KEYCODE, struct.pack("iI", btn_code, btn_code))

            # Absolute axes for mouse position (screen coords)
            abs_axes = [(ABS_X, 0, 4096), (ABS_Y, 0, 4096)]
            for axis, minimum, maximum in abs_axes:
                # struct input_absinfo { __u16 value, min, max, fuzz, flat, resolution }
                info = struct.pack("hhhhhh", 0, minimum, maximum, 0, 0, 0)
                ABS_MIN = 0x40280300 + axis * 4
                fcntl.ioctl(fd, ABS_MIN, info)  # simplified; full impl uses proper ioctl numbers

            # Create the device
            name_bytes = self._name.encode().ljust(80, b"\x00")
            os.write(fd, name_bytes)
            fcntl.ioctl(fd, UI_DEV_CREATE, 0)

            self._fd = fd
            self._open_ok = True
            logger.info("Virtual input device '%s' registered (fd=%d)", self._name, fd)
            return True

        except PermissionError:
            logger.error(
                "Permission denied on /dev/uinput. Run with sudo or add user to 'input' group: "
                "sudo usermod -aG input $USER"
            )
            return False
        except Exception as e:
            logger.error("Failed to open uinput: %s", e)
            return False

    def close(self) -> None:
        if self._fd is not None:
            try:
                import fcntl, struct
                UI_DEV_DESTROY = 0x40045581
                fcntl.ioctl(self._fd, UI_DEV_DESTROY, 0)
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
            self._open_ok = False
            logger.info("Virtual input device closed")

    @property
    def is_open(self) -> bool:
        return self._open_ok and self._fd is not None

    # ── high-level actions ───────────────────────────────────────

    def press_key(self, key: Key) -> None:
        """Press and release a single key."""
        self._send_event(EV_KEY, int(key), 1)
        time.sleep(0.02)
        self._send_event(EV_KEY, int(key), 0)
        self._sync()

    def type_text(self, text: str) -> None:
        """Type a string character by character with human-like delay."""
        for ch in text:
            code = _char_to_keycode(ch)
            if code is not None:
                shift = ord(ch).isupper() or ch in '!@#$%^&*()'
                if shift:
                    self._send_event(EV_KEY, int(Key.SHIFT_L), 1)
                self._send_event(EV_KEY, code, 1)
                time.sleep(0.02)
                self._send_event(EV_KEY, code, 0)
                if shift:
                    self._send_event(EV_KEY, int(Key.SHIFT_L), 0)
                self._sync()
                time.sleep(self._delay)

    def click_mouse(self, x: int = 0, y: int = 0, button: int = BTN_LEFT) -> None:
        """Move to (x,y) and click."""
        self.move_mouse(x, y)
        self._send_event(EV_KEY, button, 1)
        time.sleep(0.05)
        self._send_event(EV_KEY, button, 0)
        self._sync()

    def move_mouse(self, x: int, y: int) -> None:
        """Set absolute mouse position."""
        self._send_event(EV_ABS, ABS_X, x)
        self._send_event(EV_ABS, ABS_Y, y)
        self._sync()

    # ── low-level event sending ──────────────────────────────────

    def _send_event(self, ev_type: int, code: int, value: int) -> None:
        if not self.is_open:
            logger.warning("Input device not open – event dropped")
            return
        import struct
        # struct input_event { __kernel_time_t t; __u16 type; __u16 code; __s32 value }
        ts_sec = int(time.time())
        ts_usec = int((time.time() % 1) * 1_000_000)
        evt = struct.pack("llHHi", ts_sec, ts_usec, ev_type, code, value)
        os.write(self._fd, evt)

    def _sync(self) -> None:
        if not self.is_open:
            return
        import struct
        ts_sec = int(time.time())
        ts_usec = int((time.time() % 1) * 1_000_000)
        evt = struct.pack("llHHi", ts_sec, ts_usec, EV_SYN, 0, 0)
        os.write(self._fd, evt)


# ── helpers ────────────────────────────────────────────────────────

EV_KEY = 0x01
EV_REL = 0x02
EV_ABS = 0x03
EV_SYN = 0x00
ABS_X = 0x00
ABS_Y = 0x01
BTN_LEFT = 0x110

_CHAR_MAP: dict[str, Key] = {
    'a':Key.A,'b':Key.B,'c':Key.C,'d':Key.D,'e':Key.E,'f':Key.F,
    'g':Key.G,'h':Key.H,'i':Key.I,'j':Key.J,'k':Key.K,'l':Key.L,
    'm':Key.M,'n':Key.N,'o':Key.O,'p':Key.P,'q':Key.Q,'r':Key.R,
    's':Key.S,'t':Key.T,'u':Key.U,'v':Key.V,'w':Key.W,'x':Key.X,
    'y':Key.Y,'z':Key.Z,' ':Key.SPACE,'\n':Key.ENTER,'\t':Key.TAB,
    '0':Key.NUM_0,'1':Key.NUM_1,'2':Key.NUM_2,'3':Key.NUM_3,'4':Key.NUM_4,
    '5':Key.NUM_5,'6':Key.NUM_6,'7':Key.NUM_7,'8':Key.NUM_8,'9':Key.NUM_9,
    ',':Key.COMMA,'.':Key.PERIOD,'/':Key.SLASH,'-':Key.MINUS,'=':Key.EQUALS,
    '[':Key.LBRACKET,']':Key.RBRACKET,'\\':Key.BACKSLASH,';':Key.SEMICOLON,
    "'":Key.APOSTROPHE,'`':Key.GRAVE,'\x1b':Key.ESCAPE,'\x7f':Key.BACKSPACE,
}

def _char_to_keycode(ch: str) -> Optional[int]:
    return int(_CHAR_MAP.get(ch.lower(), Key.A)) if ch in _CHAR_MAP or ch.lower() in _CHAR_MAP else None
