# dashboard/ — control panel

Web UI for watching and steering the live agent.

## Files
- `__init__.py` — package init (see tools/dashboard.py for the runnable server).
- `static/` — front-end assets served by the dashboard.

The dashboard exposes runtime toggles, most importantly **console_enabled** (lets the AI open the backtick cheat console; OFF by default). It reads/writes the same config the agent uses.

## Dev notes
- Any new user-facing toggle should be mirrored into `config/settings.yaml` so it survives restarts.
- Keep the dashboard read-mostly on hot paths; don't block the decision loop serving requests.
