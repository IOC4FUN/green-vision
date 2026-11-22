# 🟢 Green Vision

> **Autonomous Geoplanetical AI Agent** — fully local, self-improving deep RL.
>
> Watches, thinks, and plays [Geoplanetical](https://store.steampowered.com/app/940230/) on Windows.
> No cloud. No API calls at inference. Every frame makes the next decision sharper.

| | |
|---|---|
| **Platform** | Windows 10 / 11 (x64) |
| **Python** | ≥ 3.10 (venv at `.venv`) |
| **GPU** | NVIDIA CUDA (required for training; CPU inference only) |
| **Game** | [Geoplanetical](https://store.steampowered.com/app/940230/) — Steam app `940230` |
| **License** | Private |

---

## ✨ What It Does

| Phase | Detail |
|-------|--------|
| 👁️ **Perceives** | 5 fps screen capture (`mss`) + Tesseract OCR + Win32 process-memory reads → 128-dim state vector |
| 🧠 **Thinks** | CNN+Transformer policy/value net (ResNet-lite backbone, 4-layer transformer, 8 heads over 8-frame sequence), goal-decomposition planner, RSSM world model for imagination rollouts, episodic memory, reflection & curiosity modules |
| ⚡ **Acts** | Injects keyboard/mouse via pyautogui (Win32 SendInput) into the game window — 31 discrete actions |
| 📈 **Learns** | PPO (GAE λ=0.95, PER α=0.6, KL early-stop, entropy annealing) every 8 env steps; two-stage video pretraining; autonomous training loop with GPU/disk resource guards |

### How It Learns (in one paragraph)

The agent starts by watching YouTube gameplay videos. `gather_videos.py` downloads clips and extracts JPEG frames; `pretrain_from_video.py` runs a two-stage curriculum — Stage A teaches the CNN to reconstruct visual features from consecutive frame pairs, Stage B fine-tunes action prediction on scene-boundary transitions. Once the network has useful priors, `train_autonomous.py` enters an endless cycle: gather more data → re-pretrain → PPO warm-up → checkpoint. In live play (`main.py`), the full perception→thinking→action loop runs at ~2 Hz, updating the policy every 8 environment steps using shaped rewards derived from in-game state improvements (approval, budget, stability, GDP).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GREEN VISION AGENT                                  │
├──────────────┬──────────────────┬───────────────────────────────────────────┤
│  PERCEPTION  │     MEMORY       │              BRAIN                        │
│              │                  │                                           │
│ mss capture  │ ProcessReader    │ GreenVisionNetwork                        │
│  (5 fps,     │  (Win32         │  ├─ CNNEncoder (ResNet-lite 64→128→256)  │
│   224² RGB)  │   ReadProcess-  │  ├─ Fusion (state ⊕ visual)               │
│ Tesseract OCR│   Memory →      │  └─ Transformer (4L, 8h, T=8)            │
│  (UI text)   │   state_vec)    │       → policy_logits[31] + value[1]     │
│ MemReader    │ KnowledgeDB      │                                           │
│  (structs)   │  (SQLite tips)  │ PPOAgent                                  │
│              │ EpisodicMemory   │  GAE · PER · KL-stop · entropy anneal    │
│ VideoIngest  │ Reflection       │                                           │
│  (yt-dlp +   │ Curiosity        │ Planner                                   │
│   ffmpeg)    │ CurriculumMgr    │  goal templates · crisis detection        │
│              │ GameKnowledge    │                                           │
│              │  (rule priors,  │ AffectModule                              │
│              │   58 cmds)      │  valence/arousal modulates exploration    │
├──────────────┴──────────────────┴───────────────────────────────────────────┤
│  RSSM World Model (deter=512, stoch=64) — imagination rollouts             │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT: pyautogui (Win32 SendInput) → keys + mouse to game window          │
│  GAME CONTROLLER: Steam launch, tasklist PID poll, focus mgmt              │
│  RESEARCH: Serper.dev web search → strategy KB (offline at inference)     │
│  DASHBOARD: Flask :8090 live state API + terminal dashboard                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎓 Training Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| 📹 Data collection | `scripts/gather_videos.py` | YouTube→frame extraction via yt-dlp + ffmpeg. Pixel-diff boundary detection per clip. 20 GB disk floor. |
| 📦 Bulk collect | `scripts/collect_all_videos.py` | Multi-query bulk download with dedup and manifest tracking. |
| 🔬 Pretrain A | `scripts/pretrain_from_video.py --stage a` | CNN reconstruction on consecutive frame pairs (stride 3). Teaches visual features. CUDA-only, bf16 default. |
| 🎯 Pretrain B | `scripts/pretrain_from_video.py --stage b` | Action-prediction SFT on boundary transitions with weighted action priors. |
| 🔄 Autonomous loop | `scripts/train_autonomous.py` | Endless cycle: resource check → gather → pretrain A+B → PPO warm-up → checkpoint → convolog update. Guards: ≥20 GB disk, ≥8 GB free VRAM. |
| ⚡ Quick smoke | `scripts/quick_train_5min.py` | 5-min Stage-A smoke run — verifies the training path end-to-end. |
| 🕹️ Live RL | `main.py --train-steps N` | Full agent in the game; PPO updates every 8 steps from real rewards. |

---

## 🎮 Gymnasium API

Green Vision exposes Geoplanetical as a standard [Gymnasium](https://gymnasium.farama.org/) environment.

| Property | Value |
|----------|-------|
| Observation space | `Box(0, 1, shape=(128,), dtype=float32)` — normalised state vector |
| Action space | `Discrete(31)` — all game actions |
| Modes | `"live"` (full agent + game) · `"simulated"` (random-walk, no game needed) |
| Reward | Reuses `src/utils/reward_shaper.py` weights for consistency |

```python
from src.gym_env import GeoplaneticalEnv

# Simulated mode – no game window required; great for unit tests & CI
env = GeoplaneticalEnv(mode="simulated", max_steps=500)
obs, info = env.reset(seed=42)
for _ in range(env.max_steps):
    action = env.action_space.sample()   # or your policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
print(f"Episode reward: {info['episode_reward']:.2f}")
```

> **Note:** `mode="live"` needs the game running and all perception/input dependencies installed.

---

## 📊 State Vector (128 dims)

Single source of truth: `src/utils/state_schema.py`.

| Segment | Dims | Content |
|---------|------|----------|
| 🖼️ Screen features | 0–15 | 4×4 downsampled grayscale cell means (/255) |
| 🔢 OCR numbers | 16–31 | First 16 numeric values from OCR text, clamped [0,1] via /100 |
| 💾 Memory vars | 32–63 | Game-process reads: budget_balance, approval_rating, gdp_growth, unemployment, inflation, military_strength, army_loyalty, war_support — each /100 |
| ⏱️ Time encoding | 64–79 | sin/cos wall-clock + reserved in-game date slots |
| 🎬 Action history | 80–95 | One-hot history of last 16 actions (flattened) |
| 📦 Reserved | 96–127 | Zero-padded; future use (research embeddings, crisis flags) |

### Action Space (31 actions)

```
noop · pause · resume · open_menu · close_menu
navigate_{budget,legislation,diplomacy,military,economy}
apply_policy · discard_policy · increase_tax · decrease_tax
increase_spending · decrease_spending · file_bill · veto_bill
declare_war · seek_peace · form_alliance · break_alliance
call_election · appoint_minister · fire_minister
use_console · save_game · load_game · switch_speed_{1,2,3}
```

---

## 🚀 Quick Start

```bat
:: ── One-time setup (creates venv, installs deps, checks system tools) ──
setup_windows.bat

:: ── Full autonomous mode – launches the game and plays ──
run_agent.bat

:: ── Web dashboard on http://127.0.0.1:8090 ──
run_dashboard.bat

:: ── Monitor only (no input injection) – safe to try first ──
run_agent.bat --dashboard

:: ── Attach to an already-running game window ──
run_agent.bat --no-launch

:: ── Live terminal dashboard in a second terminal ──
.venv\Scripts\python.exe tools\dashboard.py

:: ── Kick off the autonomous training loop ──
.venv\Scripts\python.exe scripts\train_autonomous.py --cycles 0 --max-hours 24

:: ── Gather more video data (unlimited until disk floor) ──
.venv\Scripts\python.exe scripts\gather_videos.py

:: ── Run pretraining manually ──
.venv\Scripts\python.exe scripts\pretrain_from_video.py --stage both --epochs-a 2 --epochs-b 4

:: ── Quick smoke test (verifies imports, forward pass, PPO step) ──
.venv\Scripts\python.exe scripts\smoke_test.py
```

### System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10 / 11 x64 |
| GPU | NVIDIA with ≥ 8 GB VRAM (CUDA 11.8+); CPU-only for inference |
| RAM | ≥ 16 GB recommended |
| Disk | ≥ 50 GB free (video frames + checkpoints) |
| Python | 3.10 – 3.12 (managed in `.venv`) |
| Tesseract OCR | `C:\Program Files\Tesseract-OCR` (system-wide install) |
| ffmpeg | In PATH (used by video ingestion pipeline) |
| Steam | Geoplanetical installed (`F:\SteamLibrary\steamapps\common\Geoplanetical` default) |

---

## ⚙️ Configuration

All tunables live in `config/settings.yaml`. Key sections:

| Section | Notable keys |
|---------|-------------|
| **Game** | `steam_app_id`, `auto_launch_game`, `game_path`, `scenario.country/role` |
| **Perception** | `capture_fps` (5), `capture_target_size` (224), `use_ocr`, `ocr_lang` |
| **Brain / Training** | `device` (cuda/cpu), `learning_rate`, `update_every_steps` (8), `model_path` |
| **PPO extras** | `min_lr`, `warmup_steps`, `total_training_steps`, `entropy_coef_start/end`, `rollout_capacity` |
| **Reward shaping** | `reward_weights.*` — per-segment coefficients (approval, budget, stability, war_risk, etc.) |
| **Input** | `input_backend` (pyautogui), `focus_before_input`, `console_enabled`, `idle_watchdog_s` |
| **Video / Pretrain** | `video_data_dir`, `rssm_deter/stoch`, `wm_lr`, `keep_source_video`, `min_free_gb`, `jpeg_quality` |
| **Memory** | `episodic_memory_path`, `knowledge_db`, `knowledge_vectors` |

API keys go in `.env` (gitignored). Currently: Serper for the research phase.

---

## 📁 Project Structure

```
green-vision/
├── main.py                     # CLI entry point (--dashboard / --no-launch / --web-dashboard / --train-steps)
├── setup_windows.bat           # One-time venv + deps installer
├── run_agent.bat               # Launch full autonomous agent
├── run_dashboard.bat           # Launch Flask web dashboard (:8090)
├── requirements.txt            # Python dependencies
├── config/
│   └── settings.yaml           # All tunable parameters
├── src/
│   ├── agent.py                # GreenVisionAgent – top-level orchestrator (~2 Hz loop)
│   ├── brain/                  # Decision-making core
│   │   ├── network.py          # CNN(Hadamax)+Transformer, 31 actions, STATE_DIM=128
│   │   ├── ppo_agent.py        # PPO w/ GAE(λ=0.95), PER, KL-stop, entropy annealing
│   │   ├── planner.py          # AutonomousPlanner – goal priority queue, crisis detection
│   │   ├── rssm_world_model.py # DreamerV3-style RSSM (deter=512, stoch=64)
│   │   ├── world_model.py      # Simple feedforward WM fallback
│   │   ├── episodic_memory.py  # LRU 500-episode store, cosine retrieval
│   │   ├── reflection.py       # Post-episode self-critique → policy adjustments
│   │   ├── curiosity.py        # Intrinsic reward (prediction error) for exploration
│   │   ├── curriculum.py       # Progressive difficulty / task unlocking
│   │   ├── affect.py           # Emotional valence/arousal modulates risk appetite
│   │   └── game_knowledge.py   # Rule-based priors, 58 console commands
│   ├── perception/
│   │   ├── capture.py          # mss background thread @ 5 fps, ring buffer 16, 224² resize
│   │   ├── ocr_engine.py       # Tesseract OCR wrapper, UI pattern regexes
│   │   └── video_ingest.py     # yt-dlp + ffmpeg frame extraction pipeline
│   ├── memory/
│   │   ├── process_reader.py   # Win32 ReadProcessMemory → state_vec (offset + string-scan)
│   │   └── knowledge_db.py     # SQLite-backed persistent strategy KB
│   ├── input/
│   │   └── virtual_input.py    # pyautogui / Win32 SendInput key+mouse, focus mgmt
│   ├── game/
│   │   └── game_controller.py  # Steam launch, tasklist PID poll, scenario rotation
│   ├── research/
│   │   └── serper_client.py    # Google Search API for offline strategy research
│   ├── gym_env/
│   │   └── geoplanetical_env.py# Gymnasium-compatible env wrapper (live & simulated)
│   ├── dashboard/
│   │   ├── server.py           # Flask app – live state API, decision log, metrics
│   │   └── static/             # index.html, app.js, style.css
│   └── utils/
│       ├── state_schema.py     # STATE_DIM=128 layout definition
│       ├── reward_shaper.py    # RewardWeights + compute_reward()
│       ├── video_pretrain.py   # Contrastive pretraining from gameplay clips
│       ├── journal.py          # DecisionJournal – logs every decision with context
│       ├── checkpoints.py      # CheckpointManager – save/load model+optimizer state
│       └── health.py           # HealthWatchdog – idle detection, crash recovery, GPU monitor
├── scripts/
│   ├── gather_videos.py        # YouTube search → download → ffmpeg frames → manifest
│   ├── collect_all_videos.py   # Multi-query bulk collection with dedup
│   ├── pretrain_from_video.py  # Stage A (CNN recon) + Stage B (action SFT), CUDA-only
│   ├── train_autonomous.py     # Endless autonomous training loop with resource guards
│   ├── quick_train_5min.py     # 5-min PPO smoke run
│   ├── smoke_test.py           # Import all modules, forward pass, PPO step
│   ├── pre_commit_hook.py      # Git hook: blocks secrets & sensitive paths
│   ├── post_commit_changelog.py# Git hook: auto-appends CHANGELOG entry
│   └── update_changelog.py     # Manual changelog regeneration from git log
├── tools/
│   ├── context.py              # Encrypted local context store (key at ~/.config/green-vision/)
│   └── dashboard.py            # Standalone terminal dashboard
├── data/                       # All runtime data (gitignored except .gitkeep)
│   ├── models/                 # .pt checkpoints
│   ├── knowledge/              # SQLite DBs + embedding vectors
│   ├── videos/                 # Collected gameplay frames
│   ├── memory/                 # Episodic memory JSONL
│   └── journal/                # Decision logs
├── logs/                       # green_vision.log (gitignored)
├── .github/workflows/ci.yml    # CI: syntax check + smoke test on push/PR
├── SESSION_BOOT.md             # Agent session bootstrap doc
├── PROJECT_MAP.md              # Auto-generated file index for fast navigation
└── CHANGELOG.md                # Keep-a-Changelog format, auto-updated by post-commit hook
```

---

## 🔒 Security & Git Workflow

Every commit is guarded by two hooks installed in `.git/hooks/`:

| Hook | Script | What it does |
|------|--------|--------------|
| **pre-commit** | `scripts/pre_commit_hook.py` | Blocks staged files matching sensitive paths (`data/models/*.pt`, `logs/*`, `.env`, etc.) and scans content for secret patterns (API keys, PEM blocks, AWS/GitHub tokens). Aborts the commit if anything is found. |
| **post-commit** | `scripts/post_commit_changelog.py` | Appends a dated one-liner to `CHANGELOG.md` under `[Unreleased]`. Idempotent — skips if only `CHANGELOG.md` changed (no infinite loop). Stages the update for the next commit. |

The agent auto-pushes after every code change. The remote is `https://github.com/IOC4FUN/green-vision.git` (private).

### CI (GitHub Actions)

`.github/workflows/ci.yml` runs on every push/PR to `master` or `main`:
1. Syntax-checks all Python files (`py_compile`)
2. Runs `scripts/smoke_test.py` (import + forward pass + PPO step)

---

## 📦 Dependencies

Installed in `.venv` via `requirements.txt`:

| Category | Packages |
|----------|----------|
| Core ML | `torch≥2.2`, `torchvision≥0.17`, `numpy≥1.24` |
| Perception | `opencv-python-headless≥4.8`, `pytesseract≥0.3.10`, `Pillow≥10`, `mss≥9` |
| Input (Windows) | `pyautogui≥0.9.54` |
| Process inspection | `psutil≥5.9` |
| Web / Research | `requests≥2.31` |
| Config / Utils | `PyYAML≥6`, `python-dotenv≥1`, `rich≥13` |
| RL environment | `gymnasium≥0.29` |
| Dashboard | `flask≥3` |
| Video download | `yt-dlp≥2024.8.6` |
| Encrypted context | `cryptography≥41` |
| System tools | Tesseract OCR (`C:\Program Files\Tesseract-OCR`), ffmpeg (in PATH) |

---

## 🧪 Testing & Verification

```bat
:: Full smoke test — imports every module, runs a forward pass, does one PPO step
.venv\Scripts\python.exe scripts\smoke_test.py

:: Compile-check all Python files (no execution)
.venv\Scripts\python.exe -m compileall -q src scripts tools main.py

:: Quick 5-min training run (verifies the full training path on GPU)
.venv\Scripts\python.exe scripts\quick_train_5min.py
```

CI mirrors the first two steps automatically on every push.

---

## 📖 Companion Docs

| File | Purpose |
|------|---------|
| [`SESSION_BOOT.md`](SESSION_BOOT.md) | Read this first in any new agent session — verified project state, key files, environment details |
| [`PROJECT_MAP.md`](PROJECT_MAP.md) | Auto-generated file index with per-module descriptions and key constants |
| [`CHANGELOG.md`](CHANGELOG.md) | Keep-a-Changelog format; auto-updated by post-commit hook |
| `src/*/CONTEXT.md` | Per-module design notes (brain, perception, input, memory, game, research, dashboard, utils) |

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available()` returns False | Install CUDA 11.8+ driver; verify `nvidia-smi` works; ensure venv torch is the CUDA build (`pip install torch --index-url https://download.pytorch.org/whl/cu118`) |
| Tesseract not found | Install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki); default path `C:\Program Files\Tesseract-OCR` |
| Game window not detected | Ensure Geoplanetical is running and visible; check `game_path` in `settings.yaml`; try `run_agent.bat --no-launch` to attach manually |
| Pre-commit blocks a commit | Review the blocked paths/secrets listed in output; remove them from staging or fix the content |
| CHANGELOG keeps growing with "auto-update" entries | The post-commit hook skips when only CHANGELOG changed — if you see duplicates, run `git log --oneline -5` and squash the noise commits |
| Out of VRAM during training | Reduce `capture_target_size`, lower batch size in PPO config, or close other GPU workloads |
| Disk space warning | Delete old frames in `data/videos/`; set `keep_source_video: false` (default) to auto-delete .mp4 after extraction |

---

## 📄 License

Private. All rights reserved.
