# 🟢 Green Vision – Autonomous Geoplanetical AI

A local, self-learning neural agent that plays **Geoplanetical** on your PC.
It watches the screen, reads game memory, makes its own decisions, and gets
smarter every session.

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
