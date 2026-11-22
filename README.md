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
│ VideoIngest  │ Reflection       │ Planner                                   │
│  (yt-dlp +   │ Curiosity        │  goal templates · crisis detection        │
│   ffmpeg)    │ CurriculumMgr    │                                           │
│              │ GameKnowledge    │ AffectModule                              │
│              │  (rule priors,  │  valence/arousal modulates exploration    │
│              │   58 cmds)      │                                           │
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
## 🧠 Core Modules
### Curiosity Module (`scripts/curiosity.py`)
**Purpose**: Intrinsic motivation via prediction error.
- Uses RSSM world model to predict next state
- Computes reward = -||state_pred - state_actual||²
- Encourages exploration of novel states
- Weighted by scheduler (decays over time)
```python
from scripts.curiosity import CuriosityModule, compute_curiosity_reward
# Initialize
module = CuriosityModule(hidden_dim=512, device='cuda')
# Compute intrinsic reward
reward = module(state_obs, state_pred)
```
### Curriculum Manager (`scripts/curriculum.py`)
**Purpose**: Progressive difficulty scheduling.
- Manages 5-stage progression based on performance metrics
- Prevents catastrophic forgetting
- Biases exploration toward harder states at higher stages
- Tracks episode history for stable threshold evaluation
```python
from scripts.curriculum import CurriculumManager
# Initialize with 5 stages
manager = CurriculumManager(num_stages=5)
# Check if ready to advance
if manager.evaluate_performance(episode_reward):
    manager.current_stage += 1
```
---
## 🎮 Gymnasium API
Green Vision exposes Geoplanetical as a standard [Gymnasium](https://gymnasium.farama.org/) environment.
| Property | Value |
|----------|-------|
| Observation space | `Box(0, 1, shape=(128,), dtype=float32)` — normalised state vector |
| Action space | `Discrete(31)` — all game actions |
| Modes | `train`, `eval` |
---
## 📁 Project Structure
```
E:\Green Vision\
├── main.py                    # Entry point for live RL training
├── session_boot.py            # Session initialization script
├── .gitignore                 # Git ignore rules (updated)
├── CHANGELOG.md               # Auto-generated changelog
├── README.md                  # This file
├── .venv-cuda\                # CUDA-enabled Python venv
│   └── Scripts\
│       ├── python.exe         # System Python with torch installed
│       └── ...
├── scripts\
│   ├── pretrain_from_video.py # Two-stage video pretraining
│   ├── train_autonomous.py    # Autonomous training loop
│   ├── gather_videos.py       # YouTube frame extraction
│   ├── collect_all_videos.py  # Bulk collection
│   ├── quick_train_5min.py    # Smoke test
│   ├── curiosity.py           # Curiosity module (intrinsic motivation)
│   ├── curriculum.py          # Curriculum manager (progressive difficulty)
│   ├── run_training_master.py # Master launcher (auto-detects Python)
│   └── find_python.py         # Python detection utility
├── data\
│   └── videos\
│       └── GEOPLANETICAL_GAMEPLAY_BRAZIL/  # Frame files (~10+ found)
├── models\/                    # Checkpoints saved here
├── logs\/                      # Training logs
└── knowledge\/                 # Strategy KB (offline)
```
---
## 🚀 Quick Start
```bash
# Activate CUDA venv
C:\Users\Benny\.venv-cuda\Scripts\python.exe -m venv .venv-cuda

# Run master launcher (auto-detects Python with torch)
cd E:\Green Vision\scripts
python run_training_master.py

# Or use individual scripts
python pretrain_from_video.py --stage a
```
---
## ⚠️ Known Limitations & Caveats
| Category | Issue |
|----------|-------|
| **GPU** | Requires NVIDIA GPU with CUDA support; CPU fallback explicitly disabled by design |
| **Python** | Must use system Python at `C:\Users\Benny\AppData\Local\Programs\Python\Python312\` or `.venv-cuda\Scripts\python.exe` |
| **Unicode** | All scripts now ASCII-only to prevent CP1252 encoding crashes on Windows cmd.exe |
| **Memory** | No persistent memory across sessions — context lost when agent restarts |
| **Rate Limits** | File operations blocked after 5 consecutive failures (prevents abuse) |
| **Browser** | Browser tools last-resort only — prefer `os.web.*` for web access |
| **Network** | SSRF-guarded; host allowlisted from config.http |
| **Git** | Pre-commit hook blocks secrets; auto-push enabled after every change |
| **Timeouts** | Default 30s shell timeout; long processes need `background: true` flag |
| **Approval** | Destructive ops require explicit user approval |
---
## 🔧 Troubleshooting
### Common Issues
| Symptom | Cause | Fix |
|---------|-------|-----|
| UnicodeEncodeError | Unicode chars in print statements | Replace ✓/✗ with OK/WARN/FAIL |
| torch not found | Wrong Python path | Use `find_torch_python()` utility |
| CUDA not detected | PyTorch CPU build | Reinstall with `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| No frames found | Data directory wrong | Check `data/videos/` exists and has `.jpg/.jpeg` files |
| Rate limit hit | Too many file writes | Wait or use batch operations instead |
| Permission denied | Shell command needs sudo | Approve via tool confirmation prompt |
| Git push failed | Remote sync disabled | Enable in config, then retry push |
| Port already in use | Flask dashboard conflict | Change port number in script |
| Timeout exceeded | Process took >30s | Add `timeoutMs: 720000` to shell call |
| File write blocked | 5 consecutive failures | Wait before retrying; try different approach |
---
## 📝 Changelog
See [CHANGELOG.md](./CHANGELOG.md) for auto-generated updates.
---
## 🤝 Contributing
This project is private. Contact the owner for collaboration inquiries.
---
## 📄 License
Private — all rights reserved.
