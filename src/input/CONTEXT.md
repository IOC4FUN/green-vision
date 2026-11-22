# input/ — hands
Drives keyboard/mouse via Win32 SendInput without touching the user's real devices.
## Files
- `virtual_input.py` — `VirtualInputDevice`: software-only keyboard+mouse via **pyautogui** (Win32 SendInput under the hood). Events look like real hardware to the game but never move the physical mouse or type on the physical keyboard beyond what we send.
  - `Key(IntEnum)` = internal indices mapped to pyautogui key-name strings (A=0 … GRAVE=66 for the Geoplanetical backtick console). `Mouse(IntEnum)` buttons.
  - API: `open()` (initialises pyautogui), `press_key(Key)`, `type_text(str)` (human cadence w/ shift handling), `click_mouse(x,y,btn)`, `move_mouse(x,y)`, `close()`.
  - **Perms:** none special — works as a normal Windows user. No admin/root needed.
  - **Focus:** `find_window(title_substring)` uses ctypes EnumWindows; `focus_game(title)` brings the window front before injection.
## Dev notes
- All low-level events go through `_KEY_MAP` lookup + pyautogui calls; keep that pattern if you add event types.
- Focus management (bring game window front before acting, restore after) is expected by callers; see `game/game_controller.focus_game_window`.
- pyautogui.FAILSAFE is enabled: moving the mouse to the top-left corner aborts input.
