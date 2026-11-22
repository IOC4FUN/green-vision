# 🎯 Green Vision — AI Improvement Progress Tracker

> Auto-maintained by the agent. Updated after each milestone is completed.
> Last updated: 2026-11-22

## Current Phase: **Pretraining (Stage A)**

Overall readiness for autonomous play: **~35 %**

---

## Pipeline Milestones

| # | Milestone | Status | Priority | Notes |
|---|-----------|--------|----------|-------|
| 1 | Project scaffold + architecture | ✅ Done | — | All modules written; smoke test passes imports |
| 2 | Video collection (≥ 100K frames) | ✅ Done | — | 111,275 frames from 16 sources; menu/tutorial coverage included |
| 3 | Pretraining Stage A — CNN reconstruction | ⬜ Pending | 🔴 Critical | `pretrain_from_video.py --stage a`; needs GPU dry-run first |
| 4 | Pretraining Stage B — Action-prediction SFT | ⬜ Pending | 🔴 Critical | Depends on Stage A checkpoint; boundary labels from pixel-diff |
| 5 | PPO warm-start (offline replay) | ⬜ Pending | 🔴 Critical | `train_autonomous.py` cycle; loads pretrained encoder into policy |
| 6 | RSSM world-model training | ⬜ Pending | 🟡 High | Separate from policy; needed for imagination rollouts in planner |
| 7 | Live agent loop (`main.py`) against game | ⬜ Pending | 🔴 Critical | Requires game running; 5 fps capture → PPO update every 8 steps |
| 8 | Dashboard end-to-end verification | ⬜ Pending | 🟡 High | Flask app code exists; not tested with live data flowing |
| 9 | Process-memory offset mapping | ⬜ Pending | 🟡 High | Post-Steam-update offsets may be stale; need debugger session |
| 10 | Knowledge DB population + embedding index | ⬜ Pending | 🟢 Medium | `data/knowledge/db.sqlite` empty; game_mechanics.md needs chunking |
| 11 | Curiosity module wired into PPO loss | ⬜ Pending | 🟢 Medium | Code in `src/brain/curiosity.py`; not yet added to reward computation |
| 12 | Adaptive curriculum (auto-difficulty) | ⬜ Pending | 🟢 Low | Disabled by default (`use_curriculum: false`); nice-to-have |
| 13 | Long-horizon stability test (≥ 4 hr continuous play) | ⬜ Pending | 🟡 High | Catches memory leaks, VRAM drift, idle-watchdog edge cases |
| 14 | Multi-scenario generalisation (US, Brazil, Germany) | ⬜ Pending | 🟢 Low | Current config hardcodes Germany; curriculum would handle this |

---

## Immediate Next Steps (when user returns)

- [ ] **Smoke test**: load 200 random frames → encoder forward pass → confirm no shape/OOM errors
- [ ] **Stage A pretraining**: run `pretrain_from_video.py --stage a` on full 111K-frame set (~2–4 hr on RTX)
- [ ] **Verify checkpoint**: inspect reconstruction quality on held-out menu + gameplay frames
- [ ] **Stage B pretraining**: action-prediction SFT using boundary labels
- [ ] **PPO warm-start**: 5-min quick train to confirm loss decreases
- [ ] **First live session**: launch game, attach agent, watch dashboard for 10 min

---

## Known Gaps / Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Stage A OOM at batch=8 × 224×224 | Medium | Lower batch to 4 or use gradient accumulation (already in code) |
| Boundary labels noisy from fast-cut videos | Medium | Pixel-diff threshold filters most; prefer long clips (done in target list) |
| Game patch breaks memory offsets | High after any Steam update | Agent degrades to visual-only; re-map when convenient |
| PPO reward hacking (war_risk = −2.0 dominates) | Medium | Monitor per-segment rewards in dashboard; rebalance weights |
| VRAM contention with other GPU workloads | Low (dedicated machine) | `gpu_handoff` pattern exists; close browser during training |

---

## Metrics to Track During Training

| Metric | Target | Where to check |
|--------|--------|----------------|
| Stage A reconstruction PSNR | > 25 dB on held-out set | Console output + `data/models/stage_a_*.pt` |
| Stage B action accuracy | > 70 % top-1 on boundary frames | Console output |
| PPO loss (first 1K steps) | Decreasing, no NaN | Dashboard `/api/state` or `logs/` |
| Live agent decision latency | < 200 ms per step | Dashboard timing panel |
| VRAM usage (steady state) | < 80 % of total | `nvidia-smi` or dashboard |
| Frames processed / min (live) | ≥ 300 (5 fps × 60 s) | Dashboard throughput counter |

---

## Changelog Cross-Reference

See [`CHANGELOG.md`](CHANGELOG.md) for commit-level history.
This tracker is the **strategic** view; CHANGELOG is the **tactical** log.
