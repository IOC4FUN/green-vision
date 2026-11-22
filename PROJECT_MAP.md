# Green Vision – Project Map (Quick Reference)

> Auto-generated index for fast agent navigation.
> Last updated: 2026-11-22

## Entry Points
| File | Purpose |
|------|---------|
| `main.py` | CLI entry: `--no-launch`, `--dashboard`, `--web-dashboard --port N`, `--train-steps N` |
| `run_agent.bat` | Windows launcher for full autonomous mode |
| `run_dashboard.bat` | Windows launcher for web dashboard (Flask :8090) |
| `setup_windows.bat` | One-time venv + deps setup |

## Source Tree (`src/`)
```
src/
├── __init__.py
├── agent.py              # GreenVisionAgent – top-level orchestrator (observe→think→act→learn loop @ ~2 Hz)
│                         #   Wires: capture, ocr, input, mem_reader, brain, planner, game_ctrl
│
├── brain/                # Decision-making core
│   ├── network.py        # GreenVisionNetwork v2.5 – CNN(Hadamax)+Transformer, 31 actions, STATE_DIM=128
│   │                     #   ACTION_NAMES[31], NUM_ACTIONS, SEQ_LEN=8, HIDDEN_DIM=256
│   ├── ppo_agent.py      # PPO with GAE(λ=0.95), entropy annealing, KL early-stop, prioritised replay
│   ├── planner.py        # AutonomousPlanner – goal priority queue, scenario-aware decision logic
│   ├── game_knowledge.py # GameKnowledge + seed_base_knowledge() – static rules for Geoplanetical
│   ├── episodic_memory.py# EpisodicMemory – stores Outcome episodes for reflection
│   ├── reflection.py     # ReflectionModule – post-episode self-critique → policy adjustments
│   ├── curriculum.py     # CurriculumManager – progressive difficulty / task unlocking
│   ├── curiosity.py      # CuriosityModule – intrinsic reward (prediction error) for exploration
│   ├── world_model.py    # WorldModel – learned transition model (simple)
│   ├── rssm_world_model.py # RSSMWorldModel – recurrent state-space model (advanced)
│   └── affect.py         # AffectModule – emotional valence/arousal modulating risk appetite
│
├── perception/           # Sensory input
│   ├── capture.py        # ScreenCapture – mss background thread @ 5 fps, ring buffer 16 frames, 224×224 resize
│   ├── ocr_engine.py     # OCREngine – pytesseract wrapper, UI_PATTERNS regex dict, OCRResult dataclass
│   └── video_ingest.py   # VideoIngestor – ingests gameplay videos for offline pretraining
│
├── memory/               # State extraction
│   ├── process_reader.py # GameProcessReader – ReadProcessMemory (Win32), offset-based + string-scan fallback
│   └── knowledge_db.py   # KnowledgeDB – SQLite-backed persistent game knowledge store
│
├── input/                # Motor output
│   └── virtual_input.py  # VirtualInputDevice – pyautogui/SendInput, Key enum(67 keys), focus mgmt via ctypes
│
├── game/                 # Game lifecycle
│   └── game_controller.py# GameController – Steam launch, tasklist PID poll, scenario rotation, window focus
│
├── research/             # External info gathering
│   ├── serper_client.py  # SerperClient – Google Search API for in-game research queries
│   └── CONTEXT.md        # Notes on research integration design
│
├── gym_env/              # Gymnasium-compatible env wrapper
│   └── geoplanetical_env.py # GeoplaneticalEnv(gym.Env) – wraps agent as RL environment
│
├── dashboard/            # Web UI (Flask)
│   └── ...               # Live state display, decision log, model metrics
│
└── utils/
    ├── state_schema.py   # STATE_DIM=128 layout: screen[0:16], ocr[16:32], memory[32:64], time[64:80], action_hist[80:96], reserved[96:128]
    │                     #   MEMORY_KEYS[8]: budget_balance, approval_rating, gdp_growth, unemployment,
    │                     #                   inflation, military_strength, army_loyalty, war_support
    ├── reward_shaper.py  # RewardWeights + compute_reward() – shaped reward signal
    ├── video_pretrain.py # VideoPretrainer – contrastive pretraining from gameplay clips
    ├── journal.py        # DecisionJournal – logs every decision with context for post-hoc analysis
    ├── checkpoints.py    # CheckpointManager – save/load model+optimizer state
    └── health.py         # HealthWatchdog – idle detection, crash recovery, GPU monitoring
```

## Config
| File | Key settings |
|------|-------------|
| `config/settings.yaml` | capture_fps=5, input_backend=sendinput, ocr_lang=eng, steam_app_id=940230 |
| `config/memory_offsets.yaml` | (optional) known game struct offsets for direct reads |

## Scripts (`scripts/`)
| Script | Purpose |
|--------|---------|
| `smoke_test.py` | Quick import + init sanity check (run after any change) |
| `quick_train_5min.py` | 5-min PPO training smoke run |
| `pretrain_from_video.py` | Contrastive pretraining on collected videos |
| `gather_videos.py` | Download/collect gameplay video data |
| `train_autonomous.py` | Full autonomous training loop |
| `update_changelog.py` | Auto-generate CHANGELOG from git log |
| `pre_commit_hook.py` | Git hook: lint + quick test before commit |

## Tools (`tools/`)
| File | Purpose |
|-------|---------|
| `context.py` | Encrypted local context store (key at ~/.config/green-vision/) |
| `dashboard.py` | Standalone dashboard utilities |

## Key Constants (single source of truth)
- **STATE_DIM** = 128 → `src/utils/state_schema.py`
- **NUM_ACTIONS** = 31 → `src/brain/network.py` ACTION_NAMES list
- **SEQ_LEN** = 8 → temporal window for transformer
- **HIDDEN_DIM** = 256, **NHEAD** = 8, **LAYERS** = 4
- **Capture**: 5 fps, 224×224, buffer 16 frames
- **Decision rate**: ~2 Hz (0.5 s cycle)
- **Game**: Geoplanetical (Steam app 940230), runs natively on Windows

## Data Directories
```
data/
├── models/       # .pt checkpoints (gitignored)
├── knowledge/    # SQLite DBs (gitignored)
├── replays/      # Episode replay buffers (gitignored)
└── logs/         # Training/inference logs
logs/             # green_vision.log (main log file)
```

## Dependencies (installed in `.venv`)
- torch, torchvision (CUDA) – neural network + training
- numpy, opencv-python-headless, Pillow – image processing
- mss – screen capture
- pytesseract (+ Tesseract OCR binary at C:\Program Files\Tesseract-OCR)
- pyautogui – Win32 SendInput keyboard/mouse
- flask – web dashboard
- gymnasium – RL environment interface
- PyYAML, python-dotenv, rich, psutil, requests
