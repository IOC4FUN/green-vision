# game/ — lifecycle
Owns launching and tracking the Geoplanetical process on Windows.
## Files
- `game_controller.py` — `GameController`: launches via Steam protocol (`steam://rungameid/940230`) or direct steam.exe path, waits for `Geoplanetical.exe` to appear+stabilize (via `tasklist /FI IMAGENAME eq Geoplanetical.exe`), tracks status (`GameStatus` enum: NOT_LAUNCHED→LAUNCHING→RUNNING→PAUSED→EXITED/ERROR), manages focus (ctypes SetForegroundWindow via VirtualInputDevice), rotates scenarios.
  - `ScenarioConfig(country, role, difficulty)`; default Germany/government.
  - API: `launch()`, `wait_for_ready(timeout)`, `.is_running`, `.pid`, `focus_game_window()`, `restore_focus()`, `set_scenario()`, `rotate_scenario(list[dict])`.
  - Finds PID with `tasklist` (Windows-native); largest = most recent launch.
  - Steam candidates: `C:\Program Files (x86)\Steam\steam.exe`, `C:\Program Files\Steam\steam.exe`, or PATH lookup.
## Dev notes
- Steam app id 940230 is hardcoded in `__init__`; keep it consistent with config/settings.yaml `steam_app_id`.
- Focus must be ensured before any input injection or keys go nowhere.
- Game install path from config: `F:\SteamLibrary\steamapps\common\Geoplanetical`.
