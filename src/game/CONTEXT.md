# game/ — lifecycle

Owns launching and tracking the Geoplanetical process.

## Files
- `game_controller.py` — `GameController`: launches via Steam protocol (`steam://rungameid/940230`), waits for the Proton/Wine child to appear+stabilize, tracks status (`GameStatus` enum: NOT_LAUNCHED→LAUNCHING→RUNNING→PAUSED→EXITED/ERROR), manages focus (wmctrl w/ xdotool fallback), rotates scenarios.
  - `ScenarioConfig(country, role, difficulty)`; default Germany/government.
  - API: `launch()`, `wait_for_ready(timeout)`, `.is_running`, `.pid`, `focus_game_window()`, `restore_focus()`, `set_scenario()`, `rotate_scenario(list[dict])`.
  - Finds PID with `pgrep -f Geoplanetical.exe`, largest = most recent launch.

## Dev notes
- Steam app id 940230 is hardcoded in `__init__`; keep it consistent with config/settings.yaml `steam_app_id`.
- Focus must be ensured before any input injection or keys go nowhere.
