# 🟢 Green Vision

> **Autonomous Geoplanetical AI Agent** — fully local, self-improving deep RL.
>
> Watches, thinks, and plays [Geoplanetical](https://store.steampowered.com/app/940230/) on Linux.
> No cloud. No API calls at inference. Every frame makes the next decision sharper.

---

## ✨ What It Does

| Phase | Detail |
|-------|--------|
| 👁️ **Perceives** | 5 fps screen capture + Tesseract OCR + process-memory reading → 128-dim state vector |
| 🧠 **Thinks** | CNN+Transformer policy/value net (ResNet-lite backbone, 4-layer transformer, 8 heads over 8-frame sequence), goal-decomposition planner, RSSM world model for imagination rollouts, episodic memory, reflection module |
| ⚡ **Acts** | Injects keyboard/mouse via `/dev/uinput` or `xdotool` into the game window (31 discrete actions) |
| 📈 **Learns** | PPO (GAE λ=0.95, PER α=0.6, KL early-stop, entropy annealing) every 8 env steps; two-stage video pretraining; autonomous training loop with GPU/disk management |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GREEN VISION AGENT                                  │
├──────────────┬──────────────────┬───────────────────────────────────────────┤
│  PERCEPTION  │     MEMORY       │              BRAIN                        │
│              │                  │                                           │
│ mss capture  │ ProcessReader    │ GreenVisionNetwork                        │
│  (5 fps,     │  (game RAM →    │  ├─ CNNEncoder (ResNet-lite 64→128→256)  │
│   224² RGB)  │   state_vec)    │  ├─ Fusion (state ⊕ visual)               │
│ Tesseract OCR│ KnowledgeDB      │  └─ Transformer (4L, 8h, T=8)            │
│  (UI text)   │  (SQLite tips)  │       → policy_logits[31] + value[1]     │
│ MemReader    │ EpisodicMemory   │                                           │
│  (structs)   │  (LRU 500 eps)  │ PPOAgent                                  │
│              │ Reflection       │  GAE · PER · KL-stop · entropy anneal    │
│ VideoIngest  │ GameKnowledge    │                                           │
│  (yt-dlp +   │  (rule priors,  │ Planner                                   │
│   ffmpeg)    │   58 cmds)      │  goal templates · crisis detection        │
├──────────────┴──────────────────┴───────────────────────────────────────────┤
│  RSSM World Model (deter=512, stoch=64) — imagination rollouts             │
│  Affect Module — emotional valence/arousal modulates exploration           │
├─────────────────────────────────────────────────────────────────────────────┤
│  INPUT: /dev/uinput or xdotool → keys + mouse to game window               │
│  GAME CONTROLLER: Steam launch, focus mgmt, lifecycle                      │
│  RESEARCH: Serper.dev web search → strategy KB (offline at inference)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎓 Training Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| 📹 Data collection | `scripts/gather_videos.py` | Unlimited YouTube→frame extraction via yt-dlp + ffmpeg. Pixel-diff boundary detection per clip. 20 GB disk floor. |
| 🔬 Pretrain A | `scripts/pretrain_from_video.py --stage a` | CNN reconstruction on consecutive frame pairs (stride 3). Teaches visual features. |
| 🎯 Pretrain B | `scripts/pretrain_from_video.py --stage b` | Action-prediction SFT on boundary transitions with weighted action priors. |
| 🔄 Autonomous loop | `scripts/train_autonomous.py` | Endless cycle: gather → pretrain A+B → PPO warm-up → checkpoint → convolog update. Auto-stops/restarts llama-server for VRAM. Resource guards: ≥20 GB disk, ≥8 GB free VRAM, power-draw warning. |
| 🕹️ Live RL | `main.py --train-steps N` | Full agent in the game; PPO updates every 8 steps from real rewards. |

### How It Learns

1. **PPO reinforcement learning** – policy updates every 8 env steps; reward derived from game-state improvements (approval, budget, stability).
2. **Knowledge base** – web-searched strategy tips stored in SQLite, consulted when neural confidence is low.
3. **Model persistence** – weights save on shutdown; reload next session. The agent gets better the more you let it play.
---
## 🎮 Gymnasium API

Green Vision exposes Geoplanetical as a standard [Gymnasium](https://gymnasium.farama.org/) environment, making it compatible with any RL framework or research tooling.

| Property | Value |
|----------|-------|
| Observation space | `Box(0, 1, shape=(128,), dtype=float32)` — normalised state vector |
| Action space | `Discrete(31)` — all game actions |
| Modes | `"live"` (delegates to full agent + game) · `"simulated"` (random-walk, no game needed) |
| Reward | Reuses `src/utils/reward_shaper.py` weights for consistency |

### Quick example

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

### Live mode

```python
env = GeoplaneticalEnv(mode="live")      # requires a running Geoplanetical window
obs, info = env.reset()
# … same step loop; each action is injected via /dev/uinput or xdotool
```

> **Note:** `mode="live"` needs the game running and all perception/input dependencies installed. Use `mode="simulated"` for headless testing.
---
## 📊 State Vector (128 dims)

| Segment | Dims | Content |
|---------|------|----------|
| 💰 Economy | 0–23 | Budget balance, tax rate, spending, GDP growth, debt, reserves… |
| 🤝 Diplomacy | 24–47 | Relations matrix (top-6 nations), alliances, war status… |
| ⚔️ Military | 48–63 | Army/navy/air strength, morale, conscription… |
| 🏛️ Politics | 64–87 | Approval, stability, party support, bill queue… |
| 🖥️ UI / Screen | 88–111 | Active panel, cursor region, menu depth, speed setting… |
| 🔢 Meta | 112–127 | Turn count, crisis flags, last-action one-hot residual… |

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

```bash
# One-time setup (creates venv, installs deps, checks system tools)
./scripts/setup.sh

# Full autonomous mode – launches the game and plays
./scripts/run.sh

# Monitor only (no input injection) – safe to try first
./scripts/run.sh --dashboard

# Attach to an already-running game window
./scripts/run.sh --no-launch

# Live terminal dashboard in a second terminal
python tools/dashboard.py

# Kick off the autonomous training loop (background)
nohup .venv/bin/python scripts/train_autonomous.py > logs/train_$(date +%Y%m%d).log 2>&1 &

# Gather more video data (unlimited until disk floor)
.venv/bin/python scripts/gather_videos.py

# Run pretraining manually
.venv/bin/python scripts/pretrain_from_video.py --stage both --epochs-a 2 --epochs-b 4
```

---

## ⚙️ Configuration

Edit `config/settings.yaml`:

| Key | Description |
|-----|-------------|
| `scenario` | Which country/role to play |
| `decision_interval_s` | How often the agent acts (default 0.5 s = 2 Hz) |
| `device` | `cuda` or `cpu` |
| `capture_fps` | Screen grab rate |
| `reward_weights` | Per-segment reward shaping coefficients |

API keys go in `.env` (gitignored; currently Serper for research phase).

---

## 📁 Project Structure

```
green-vision/
├── main.py                     # entry point (--dashboard / --no-launch / --train-steps)
├── config/settings.yaml        # all tunables
├── src/
│   ├── agent.py                # GreenVisionAgent orchestrator
│   ├── brain/
│   │   ├── network.py          # CNN+Transformer policy/value net
│   │   ├── ppo_agent.py        # PPO w/ GAE, PER, KL-stop, entropy anneal
│   │   ├── planner.py          # goal decomposition + crisis auto-gen
│   │   ├── rssm_world_model.py # DreamerV3-style RSSM (deter=512, stoch=64)
│   │   ├── world_model.py      # simple feedforward WM fallback
│   │   ├── episodic_memory.py  # LRU 500-episode store, cosine retrieval
│   │   ├── reflection.py       # post-episode analysis → curriculum hints
│   │   ├── affect.py           # emotional valence/arousal modulation
│   │   └── game_knowledge.py   # rule-based priors, 58 console cmds
│   ├── perception/
│   │   ├── capture.py          # mss screen grab, ring buffer 16, resize 224²
│   │   ├── ocr_engine.py       # Tesseract OCR wrapper
│   │   └── video_ingest.py     # yt-dlp + ffmpeg frame extraction pipeline
│   ├── memory/
│   │   ├── process_reader.py   # read game process memory → state_vec
│   │   └── knowledge_db.py     # SQLite strategy KB
│   ├── input/
│   │   └── virtual_input.py    # /dev/uinput or xdotool key+mouse
│   ├── research/
│   │   └── serper_client.py    # Google search via Serper.dev
│   ├── game/
│   │   └── game_controller.py  # Steam launch, focus, lifecycle
│   ├── dashboard/              # live terminal monitor
│   ├── gym_env/                # Gymnasium-compatible RL env wrapper
│   │   └── geoplanetical_env.py  # GeoplaneticalEnv (live + simulated)
│   └── utils/
│       ├── video_pretrain.py   # VideoPretrainer: SFT batches from boundaries
│       ├── state_schema.py     # 128-dim state vector layout
│       ├── reward_shaper.py    # configurable reward weights
│       ├── checkpoints.py      # save/load helpers
│       └── health.py           # GPU/disk monitoring
├── scripts/
│   ├── setup.sh                # one-time install
│   ├── run.sh                  # launch the agent
│   ├── gather_videos.py        # autonomous YouTube→frames (unlimited)
│   ├── pretrain_from_video.py  # Stage A (CNN recon) + Stage B (action SFT)
│   ├── train_autonomous.py     # self-managing train loop w/ LLM mgmt
│   ├── quick_train_5min.py     # short smoke-training for CI/dev
│   └── smoke_test.py           # import & instantiation checks
├── tools/
│   └── dashboard.py            # live terminal monitor
├── data/                       # models, KB, replays, videos (gitignored)
├── logs/                       # runtime logs (gitignored)
└── .convolog.md                # local session memory (gitignored)
```

---

## 📋 Requirements

- **Linux** with X11 (Wayland needs `xdotool` + XWayland)
- **NVIDIA GPU** recommended (RTX 4090 ideal; falls back to CPU)
- **Tesseract OCR**, **xdotool**, **wmctrl** installed
- **Steam** running (for auto-launch)
- Python 3.10+, PyTorch with CUDA

---

## 🔒 Data & Privacy

- All learning is **local**; no frames or state leave your machine.
- The only external call during the *research* phase is Serper.dev web search for strategy tips (stored in SQLite, never sent at inference time).
- Video data collected via yt-dlp from public YouTube gameplay; stored locally under `data/videos/` (gitignored, ~1 GB+ and growing).

---

## 📝 Notes

- The agent runs *alongside* your desktop – it captures the screen even when the game window isn't focused.
- Input injection briefly focuses the game window, acts, then restores focus.
- All learning is local; no data leaves your machine except Serper search queries.
