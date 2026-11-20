# 🟢 Green Vision – Autonomous Geoplanetical AI Agent

> A fully local, self-improving deep-reinforcement-learning agent that watches,
thinks, and plays **Geoplanetical** (Steam 940230) on Linux — no cloud, no API
calls at inference time. Every frame it sees makes the next decision sharper.

## What It Does

1. **Perceives** the live game via 5 fps screen capture + OCR + process-memory
   reading → a 128-dim state vector.
2. **Thinks** through a CNN+Transformer policy/value network (ResNet-lite backbone,
   4-layer transformer with 8 heads over an 8-frame sequence), a goal-decomposition
   planner, an RSSM world model for imagination rollouts, episodic memory for
   analogy retrieval, and a reflection module that writes curriculum hints after
   each episode.
3. **Acts** by injecting keyboard/mouse events through `/dev/uinput` or `xdotool`
   into the game window (31 discrete actions: navigate panels, apply policies,
declare war, manage economy, use console commands, switch speed, etc.).
4. **Learns** continuously:
   - **PPO** (GAE λ=0.95, PER α=0.6, KL early-stop, entropy annealing 0.05→0.005)
     updates every 8 env steps during live play.
   - **Two-stage video pretraining** on thousands of real gameplay frames:
     Stage A teaches the CNN to reconstruct UI; Stage B weakly supervises the
     action head on scene-boundary transitions.
   - **Autonomous training loop** (`train_autonomous.py`) cycles gather →
     pretrain → PPO warm-up → checkpoint indefinitely, auto-managing GPU VRAM
     (stops/restarts your local LLM server) and disk usage.

## Architecture at a Glance

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

## Training Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| Data collection | `scripts/gather_videos.py` | Unlimited YouTube→frame extraction via yt-dlp + ffmpeg. Pixel-diff boundary detection per clip. 20 GB disk floor. |
| Pretrain A | `scripts/pretrain_from_video.py --stage a` | CNN reconstruction on consecutive frame pairs (stride 3). Teaches visual features. |
| Pretrain B | `scripts/pretrain_from_video.py --stage b` | Action-prediction SFT on boundary transitions with weighted action priors. |
| Autonomous loop | `scripts/train_autonomous.py` | Endless cycle: gather → pretrain A+B → PPO warm-up → checkpoint → convolog update. Auto-stops/restarts llama-server for VRAM. Resource guards: ≥20 GB disk, ≥8 GB free VRAM, power-draw warning. |
| Live RL | `main.py --train-steps N` | Full agent in the game; PPO updates every 8 steps from real rewards. |

## State Vector (128 dims)

| Segment | Dims | Content |
|---------|------|----------|
| Economy | 0–23 | Budget balance, tax rate, spending, GDP growth, debt, reserves… |
| Diplomacy | 24–47 | Relations matrix (top-6 nations), alliances, war status… |
| Military | 48–63 | Army/navy/air strength, morale, conscription… |
| Politics | 64–87 | Approval, stability, party support, bill queue… |
| UI / Screen | 88–111 | Active panel, cursor region, menu depth, speed setting… |
| Meta | 112–127 | Turn count, crisis flags, last-action one-hot residual… |

## Action Space (31 actions)

`noop · pause · resume · open_menu · close_menu · navigate_{budget,legislation,diplomacy,military,economy} · apply_policy · discard_policy · increase_tax · decrease_tax · increase_spending · decrease_spending · file_bill · veto_bill · declare_war · seek_peace · form_alliance · break_alliance · call_election · appoint_minister · fire_minister · use_console · save_game · load_game · switch_speed_{1,2,3}`

## Quick Start
```bash
# One-time setup (creates venv, installs deps, checks system tools)
./scripts/setup.sh

# Full autonomous mode – launches the game and plays
./scripts/run.sh

# Monitor only (no input injection) – safe to try first
./scripts/run.sh --dashboard

# Attach to an already-running game window
./scripts/run.sh --no-launch

# Kick off the autonomous training loop (background)
nohup .venv/bin/python scripts/train_autonomous.py > logs/train_$(date +%Y%m%d).log 2>&1 &

# Gather more video data (unlimited until disk floor)
.venv/bin/python scripts/gather_videos.py

# Run pretraining manually
.venv/bin/python scripts/pretrain_from_video.py --stage both --epochs-a 2 --epochs-b 4
```

## Configuration
Edit `config/settings.yaml`:
- **scenario** – which country/role to play
- **decision_interval_s** – how often the agent acts (default 0.5 s = 2 Hz)
- **device** – `cuda` or `cpu`
- **capture_fps** – screen grab rate
- **reward_weights** – per-segment reward shaping coefficients

API keys go in `.env` (gitignored; currently Serper for research phase).

## Project Structure
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
│   └── train_autonomous.py     # self-managing train loop w/ LLM mgmt
├── tools/
│   └── dashboard.py            # live terminal monitor
├── data/                       # models, KB, replays, videos (gitignored)
├── logs/                       # runtime logs (gitignored)
└── .convolog.md                # local session memory (gitignored)
```

## Requirements
- **Linux** with X11 (Wayland needs `xdotool` + XWayland)
- **NVIDIA GPU** recommended (RTX 4090 ideal; falls back to CPU)
- **Tesseract OCR**, **xdotool**, **wmctrl** installed
- **Steam** running (for auto-launch)
- Python 3.10+, PyTorch with CUDA

## Data & Privacy
- All learning is **local**; no frames or state leave your machine.
- The only external call during the *research* phase is Serper.dev web search
  for strategy tips (stored in SQLite, never sent at inference time).
- Video data collected via yt-dlp from public YouTube gameplay; stored locally
  under `data/videos/` (gitignored, ~1 GB+ and growing).

## Roadmap / Next Steps
- [ ] First full pretraining run on 4090 (Stage A+B over 105K+ frames)
- [ ] Wire PPO warm-up to real Geoplanetical process (replace synthetic transitions)
- [ ] Improve Stage B action labels: OCR-diff + cursor-region IDM instead of weighted random
- [ ] RSSM imagination rollouts integrated into PPO update loop
- [ ] Multi-agent self-play for diplomacy/military scenarios
- [ ] CI: GitHub Actions smoke-test (syntax check + unit tests)

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌───────────────┐
│  PERCEPTION  │───▶│   MEMORY     │───▶│     BRAIN     │
│ Screen+OCR   │    │ ProcReader   │    │ PPO Network   │
│ mss capture  │    │ KnowledgeDB  │    │ Planner       │
└──────────────┘    └──────────────┘    └───────┬───────┘
                                                │
                    ┌──────────────┐            ▼
                    │   INPUT      │◀─── Action (30 discrete)
                    │ xdotool/uinp │
                    └──────────────┘

  + Research: Serper.dev web search for strategy knowledge
  + Game Controller: auto-launch via Steam, focus management
```

## Quick Start

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
```

## Configuration

Edit `config/settings.yaml`:
- **scenario** – which country/role to play
- **decision_interval_s** – how often the agent acts (default 0.5s = 2 Hz)
- **device** – `cuda` or `cpu`
- **capture_fps** – screen grab rate

API keys go in `.env` (already has your Serper key).

## How It Learns

1. **PPO reinforcement learning** – the network updates its policy every 8 steps.
   Reward is derived from game-state improvements (approval, budget, stability).
2. **Knowledge base** – web-searched strategy tips are stored in SQLite and
   consulted when neural confidence is low.
3. **Model persistence** – weights save on shutdown; reload next session.
   The agent literally gets better the more you let it play.

## Project Structure

```
green-vision/
├── main.py                 # entry point
├── config/settings.yaml    # all tunables
├── .env                    # API keys (Serper)
├── src/
│   ├── agent.py            # orchestrator (the "brain loop")
│   ├── brain/
│   │   ├── network.py      # CNN + Transformer policy/value net
│   │   ├── ppo_agent.py    # PPO training/inference
│   │   └── planner.py      # goal-based reasoning layer
│   ├── perception/
│   │   ├── capture.py      # mss screen grabber
│   │   └── ocr_engine.py   # Tesseract OCR wrapper
│   ├── memory/
│   │   ├── process_reader.py  # read game process memory
│   │   └── knowledge_db.py    # SQLite strategy KB
│   ├── input/
│   │   └── virtual_input.py   # xdotool/uinput key+mouse
│   ├── research/
│   │   └── serper_client.py   # Google search via Serper.dev
│   └── game/
│       └── game_controller.py # Steam launch, focus, lifecycle
├── scripts/
│   ├── setup.sh            # one-time install
│   └── run.sh              # launch the agent
├── tools/
│   └── dashboard.py        # live terminal monitor
├── data/                   # models, KB, replays (gitignored)
└── logs/                   # runtime logs
```

## Requirements

- **Linux** with X11 (Wayland needs `xdotool` + XWayland)
- **NVIDIA GPU** recommended (RTX 4090 ideal) – falls back to CPU
- **Tesseract OCR**, **xdotool**, **wmctrl** installed
- **Steam** running (for auto-launch)
- Python 3.10+

## Notes

- The agent runs *alongside* your desktop – it captures the screen even when
  the game window isn't focused.
- Input injection briefly focuses the game window, acts, then restores focus.
- All learning is local; no data leaves your machine except Serper search queries.
