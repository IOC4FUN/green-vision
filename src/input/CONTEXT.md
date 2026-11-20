# input/ — hands

Drives keyboard/mouse without touching the user's real devices.

## Files
- `virtual_input.py` — `VirtualInputDevice`: software-only keyboard+mouse via Linux `/dev/uinput`. Events look like real hardware to X/the game but never move the physical mouse or type on the physical keyboard.
  - `Key(IntEnum)` = raw Linux input-event-codes (A=30 … GRAVE=41 for the Geoplanetical backtick console). `Mouse(IntEnum)` buttons.
  - API: `open()` (registers device), `press_key(Key)`, `type_text(str)` (human cadence w/ shift handling), `click_mouse(x,y,btn)`, `move_mouse(x,y)`, `close()`.
  - **Perms:** needs root OR user in `input` group OR world-writable /dev/uinput. Wrapper script handles this.

## Dev notes
- All low-level events go through `_send_event` + `_sync`; keep that pattern if you add event types.
- The uinput ioctl constants are duplicated at module level and inside `open()` — if you change one, check both.
- Focus management (bring game window front before acting, restore after) is expected by callers; see `game/game_controller.focus_game_window`.
