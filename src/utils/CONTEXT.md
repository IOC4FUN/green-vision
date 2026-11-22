# utils/ — shared helpers
Cross-cutting, dependency-light modules used by the agent loop.

## Files
- `state_schema.py` — Single source of truth for the 128-dim state vector layout.
  - `SEGMENTS`: list of `StateSegment(name, start, end, description)` covering all 6 blocks.
  - `MEMORY_KEYS`: the 9 named game-process variables read into [32:64].
  - `build_vector()` → zeroed `(128,) float32`; `validate(vec)`; `describe()` for logs.

- `reward_shaper.py` — Externalises reward coefficients to `settings.yaml` (`reward_weights:` block).
  - `RewardWeights.from_config(cfg)` loads weights (falls back to defaults).
  - `compute_reward(weights, prev_state, curr_state, action_was_noop)` → scalar float.
  - Weights cover: approval_delta, budget_health, stability, gdp_growth, unemployment_penalty, inflation_penalty, war_risk, noop_penalty.

- `journal.py` — Append-only JSONL decision log (`data/journal/decisions.jsonl`).
  - `DecisionJournal(path).log(action, confidence, state_snapshot, ocr_text, reward, thought)`.
  - One line per decision; useful for post-hoc debugging.
  - `.close()` on shutdown.

- `checkpoints.py` — Periodic model snapshots + best-reward tracking.
  - `CheckpointManager(dir_path, save_every_steps, window)`.
  - `.record_reward(r)` feeds a rolling deque; `.maybe_save(step, model, optimizer)` saves every N steps and promotes to `best.pt` when rolling avg improves.

- `health.py` — Subsystem health watchdog (OCR empty streak, memory-reader timeout, capture FPS).
  - `HealthWatchdog(ocr_empty_threshold, mem_fail_timeout_s, capture_min_fps, check_interval_s)`.
  - Call `.tick(ocr_text=..., mem_state=...)` once per main-loop iteration → `HealthStatus` dataclass.
  - Logs warnings when any subsystem degrades.

- `video_pretrain.py` — `VideoPretrainer`: builds supervised-pretraining batches from ingested video clips.
  - Two-stage strategy: Stage A (CNN reconstruction on frame pairs), Stage B (action-prediction SFT on boundary transitions).
  - Loads manifests from clip dirs; filters frames with `boundary=True`; heuristic action labeller (v1: weighted random over common actions).
  - `get_batch(index?)` → `PretrainBatch(frames(B,T,3,H,W), state_vec(B,D), actions(B,), rewards, dones, log_probs, values)`.
  - `num_batches()` → total non-overlapping batches per epoch.
  - `_build_frame_stack(end_idx)` → T-frame stack ending at transition.
  - `_load_frame(idx)` → resize to target_size CHW float32 [0,1].
  - **Known issue:** `_load_frame` does O(n) linear scan of `_samples`; should be dict-cached for >50K samples.
  - **Known issue:** `_heuristic_label` is essentially random; improve with OCR diff + cursor-region change detection.

## Scripts (project root /scripts/)
- `gather_videos.py` — Autonomous YouTube→frames gatherer. Unlimited by default (`--max-videos 0`). Searches multiple queries, dedupes, downloads at 480p, extracts 1fps JPEGs, detects boundaries (pixel-diff ≥ 0.05), writes manifest.json per clip, deletes source video.
- `pretrain_from_video.py` — Two-stage training: Stage A (CNN recon on frame pairs, AdamW+CosineAnneal), Stage B (action SFT on boundary transitions, loads Stage-A CNN weights). Outputs `data/models/cnn_recon.pt` and `data/models/policy_sft.pt`.
- `train_autonomous.py` — Self-managing loop: resource check → gather → pretrain → PPO warm-up → checkpoint → log to .convolog.md → repeat. Run unattended via a background terminal or Windows Task Scheduler.

## Dev notes
- All modules are importable from the package root: `from .utils.state_schema import build_vector`, etc.
- Keep these dependency-light (numpy, stdlib only); no torch imports here — EXCEPT video_pretrain.py which needs torch+PIL by design.
- Update this file whenever a module's public API changes.
