# Changelog

All notable changes to Green Vision are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Entries are appended automatically by the post-commit hook under `[Unreleased]`.

## [Unreleased]
- 2026-11-23 — docs(session_boot): point curiosity at src/brain/curiosity.py; remove dead scripts/.venv-cuda (`704bcc539`)
- 2026-11-23 — chore: auto-update CHANGELOG (`a4bf0f2ef`)
- 2026-11-23 — chore(config): expose tuning knobs (curiosity_coeff, ocr_every_n_frames, click_x/y) in settings.yaml (`c6b60d309`)
- 2026-11-23 — chore: full lint cleanup + __init__ exports + f-string/import hygiene (`c5e2fda6a`)
- 2026-11-23 — chore: full lint cleanup + integration test fixes (`1de282ee3`)
- 2026-11-23 — fix(world_model): correct plan() tuple unpacking order; fix(knowledge_db): safe numeric trigger matching with ValueError fallback (`128ed2a98`)
- 2026-11-23 — docs: add WORK_LOG.md — tracks all added/removed/generated files for cleanup reference (`65205174b`)
- 2026-11-23 — fix(ocr): add _find_tesseract() fallback + explicit pytesseract_cmd (`9bf8e1ede`)
- 2026-11-22 — feat(pretrain): auto-stage B launcher (`9be0ce835`)
- 2026-11-22 — chore(pretrain): add manifest generator + gitignore for training artifacts (`44d6f5165`)
- 2026-11-22 — chore: add session_boot.py — single source of truth for agent re-orientation (`26121896b`)
- 2026-11-22 — feat(brain): add curiosity module, curriculum manager & pretraining launchers (`c72931bff`)
- 2026-11-22 — docs(brain): fully document Curiosity Module & Curriculum Manager in README (`022d256`)
- 2026-11-22 — docs(brain): expand brain modules table + add curiosity & curriculum details (`c68ce74`)
- 2026-11-22 — feat(tracker): add AI improvement progress tracker with pipeline milestones, risks & metrics (`68f8608`)
- 2026-11-22 — docs: expand troubleshooting into categorized tables + add Known Limitations & Caveats section (`7ffc9dc`)
- 2026-11-22 — feat(data): expand video targets to 20 — added menu/start coverage tutorials + long livestreams for maximum frame diversity (`1e01e9d`)
- 2026-11-22 — docs: rewrite README with full project structure, system requirements, security workflow, troubleshooting & companion docs (`b967d28`)
- 2026-11-22 — fix: break post-commit changelog infinite loop — skip when only CHANGELOG.md changed (`7b0061c`)
- 2026-11-22 — chore: auto-update changelog (`d958629`)
- 2026-11-22 — chore: auto-update changelog (`4ed0185`)
- 2026-11-22 — chore: auto-update changelog (`4faf3bc`)
- 2026-11-22 — chore: auto-update changelog (`0495154`)
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
