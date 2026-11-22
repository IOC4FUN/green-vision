# Changelog

All notable changes to Green Vision are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Entries are appended automatically by the post-commit hook under `[Unreleased]`.

## [Unreleased]
- 2026-11-22 — chore: auto-update changelog (`6c94e29`)
- 2026-11-22 — chore: auto-update changelog after initial commit (`d9b0fae`)
- 2026-11-22 — feat: Green Vision v0.1 — Windows-only AI game agent (PPO + RSSM + dashboard) (`02a7f2c`)

- 2026-11-21 — chore: update changelog with CI commit entry (`821aa36ba7171c576a66332cbdb3b388e5848613`)

- 2026-11-21 — chore(ci): add gymnasium dep + dedupe changelog entry (`3d8b849708041643269172ad3c4ec43822229e11`)
- 2026-11-21 — feat(gym): Gymnasium-compatible environment wrapper (`src/gym_env/`) with live & simulated modes; README + CHANGELOG updated
- 2026-11-21 — fix: PPOAgent accepts optional network param + pre-commit secret-scan hook (`d9c5fd5d75e930c5cd6a41ebc063a79140c08ba8`)
- 2026-11-21 — fix: PPOAgent accepts optional `network` param; train_autonomous passes `config=cfg, network=net`
- 2026-11-21 — feat(security): pre-commit hook scans staged files for secrets before push
- 2026-11-21 — fix: total_mem → total_memory for torch >= 2.x CUDA device props (`e9462cd0da029774117c902c64890d97afff8f9a`)

- 2026-11-21 — feat(dashboard): Flask web dashboard with live agent state API (`e9b15ba`)
- 2026-11-20 — RSSM cleanup + LLM swap manager + training scripts + config updates (`1a96c93`)
- 2026-11-20 — perf: reduce default batch to 8 + grad-accum + bf16 flags for concurrent GPU use (`0cea448`)
- 2026-11-20 — fix: refuse CPU fallback — CUDA-only guard in pretrain (`b729c5a`)
- 2026-11-20 — feat: gpu_handoff.sh auto-pause llama-server during training (`a598126`)
- 2026-11-20 — fix: affect.py line-170 crash + yt-dlp firefox cookie flags (`f0091ce`)
