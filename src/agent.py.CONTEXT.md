# agent.py — top-level loop (the body)
Wires perception → memory → brain → input into a running game session.
## Flow (per frame / per decision)
1. `ScreenCapture.latest_frame()` → resized 224² frame + raw.
2. `OCREngine.process(raw)` → UI text fields.
3. `GameProcessReader.read_state().values` → numeric state (best-effort).
4. `_update_state_vector(resized, ocr_text, mem_state)` → fixed 128-dim vector.
5. At `decision_interval_s` (0.5s = 2 Hz): `_think_and_decide(...)` → action_idx;
   idle watchdog may override with `_corrective_action()`; then `_execute_action`,
   `_compute_reward`, `_learn`.
6. **Episodic memory**: if `_episode_active`, calls `episodic_mem.record_step(state, action_idx, name, reward, thought)` after each learn step.
7. Journal every decision to JSONL.
## Episodic memory lifecycle
- `begin_episode(scenario)` called in `start()` right after initial goals are set.
- `record_step(...)` called every decision cycle while `_episode_active=True`.
- On `stop()`: `end_episode()` → `reflector.reflect(ep)` → `save_to_file(data/memory/episodes.jsonl)`.
- Planner reads from `self.episodic_mem` for analogy retrieval per-decision.
## Key knobs (from config/settings.yaml)
- `decision_interval_s` (0.5), `console_enabled` (false default), `idle_watchdog_s` (30).
- Action handlers live in the `actions_map` inside `_execute_action`; `use_console`
  is gated by `self.console_enabled` via `_maybe_use_console`.
- Idle watchdog: if no meaningful action for `idle_watchdog_s` AND no-op streak ≥ 4,
  force a corrective move from `GameKnowledge.get_priority_actions`.
## Dev notes
- Adding an action: append to `brain/network.ACTION_NAMES` + add a handler in `actions_map`.
- State-vector layout is implicit here — document any new field's slot when you extend it.
- Episodic memory path configurable via `episodic_memory_path` in settings.yaml (default `data/memory/episodes.jsonl`).
