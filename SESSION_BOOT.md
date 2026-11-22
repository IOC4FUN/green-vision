# SESSION BOOT — Read This First

> Any new agent session: read this file in ONE call, then go straight to work.
> No re-discovery needed. Everything below is verified as of 2026-11-22.

## Project Identity
- **Name:** Green Vision
- **Location:** `E:\Green Vision` (dedicated drive, ~1.8 TB free)
- **Game target:** Geoplanetical (Steam app 940230) — political/strategy sim by Polytale Games
- **Goal:** Self-learning AI agent that plays Geoplanetical from screen pixels + OCR state, using PPO RL

## Architecture Pipeline
```
YouTube videos → gather_videos.py → JPEG frames + manifests
    → pretrain_from_video.py --stage a  (CNN reconstruction, visual features)
    → pretrain_from_video.py --stage b  (Action-prediction SFT on boundaries)
    → train_autonomous.py               (Full loop: gather→pretrain→PPO warmup→checkpoint)
    → main.py                           (Live in-game agent with PPO updates every 8 steps)
```

## Key Files
| Path | Purpose |
|------|--------|
| `main.py` | Entry point: live agent or web dashboard (`--web-dashboard --port 8090`) |
| `scripts/gather_videos.py` | YouTube search → download → ffmpeg frame extraction → manifest.json per clip |
| `scripts/pretrain_from_video.py` | Stage A (CNN recon) + Stage B (action SFT). CUDA-only. bf16 default. |
| `scripts/train_autonomous.py` | Endless cycle: resource check → gather → pretrain A+B → PPO warmup → save |
| `scripts/quick_train_5min.py` | Quick Stage-A smoke training (1 epoch) |
| `scripts/smoke_test.py` | Import all modules, forward pass, PPO step — verify nothing broke |
| `src/brain/network.py` | GreenVisionNetwork: CNN backbone + Transformer encoder + policy/value heads |
| `src/brain/ppo_agent.py` | PPOAgent: GAE, clipped objective, replay buffer, rollout buffer |
| `src/brain/rssm_world_model.py` | RSSM world model for planning |
| `src/brain/planner.py` | AutonomousPlanner: goal generation from state |
| `src/brain/curiosity.py` | CuriosityModule: intrinsic reward (prediction error) |
| `src/perception/capture.py` | Screen capture via mss |
| `src/perception/ocr_engine.py` | Tesseract OCR for UI text extraction |
| `src/game/game_controller.py` | Steam launch + window management |
| `src/input/virtual_input.py` | pyautogui keyboard/mouse injection |
| `src/memory/process_reader.py` | Read game process memory (psutil-based) |
| `config/settings.yaml` | All tunable params (fps, batch size, learning rate, etc.) |
| `data/knowledge/game_mechanics.md` | Geoplanetical mechanics reference (budget, legislation, economy, military) |
| `data/videos/` | Collected gameplay frames (gitignored) |
| `data/models/` | Saved checkpoints (gitignored) |

## Environment
- **Python venv:** `E:\Green Vision\.venv`
- **Key packages:** torch 2.15 (CUDA), torchvision, numpy, opencv-python-headless, pytesseract, pyautogui, mss, flask, gymnasium, PyYAML, rich, psutil, requests, yt-dlp
- **Tesseract OCR:** `C:\Program Files\Tesseract-OCR` (system-wide)
- **ffmpeg:** installed via winget (Gyan.FFmpeg 9.1), in PATH
- **GPU:** CUDA-capable (RTX class). Training is CUDA-only by design.

## Quick Commands (run from `E:\Green Vision`)
```bat
:: Smoke test (verify everything imports & runs)
.venv\Scripts\python scripts\smoke_test.py

:: Collect YouTube video frames (5 videos as test)
.venv\Scripts\python scripts\gather_videos.py --max-videos 5

:: Pretrain Stage A only (CNN recon)
.venv\Scripts\python scripts\pretrain_from_video.py --stage a --epochs-a 3

:: Full pretrain A+B
.venv\Scripts\python scripts\pretrain_from_video.py --stage both

:: Autonomous training loop (runs forever until killed)
.venv\Scripts\python scripts\train_autonomous.py --cycles 0 --max-hours 24

:: Live agent (game must be running or will launch Steam)
run_agent.bat

:: Web dashboard on :8090
run_dashboard.bat
```

## Network Architecture (current)
- CNN backbone: Conv layers → feature maps
- Transformer encoder over SEQ_LEN frame sequence
- Policy head: logits over NUM_ACTIONS discrete actions
- Value head: scalar state value
- ~6.9M parameters, 31 named actions
- Input: (B, SEQ_LEN, 3, 224, 224) frames + (B, STATE_DIM) vector state

## PPO Config Defaults (from src/brain/ppo_agent.py PPOConfig)
- learning_rate: 3e-4 → min_lr: 3e-5 (cosine decay after 200-step warmup)
- update_every_steps: 8
- batch_size: 64
- rollout_capacity: 512
- replay_capacity: 100,000 (PER sum-tree)
- num_epochs_per_update: 4
- GAE lambda: 0.95, gamma: 0.99
- Clip range: 0.2
- Entropy: 0.05 → 0.005 (annealed over 100K steps)
- KL early-stop: target 0.02 × factor 3.0
- PER alpha: 0.6, beta: 0.4→1.0

## Research Insights (from GitHub, saved in memory note #148)
- **ImpalaCNN** (IMPALA ResNet + Global Avg Pool) > plain CNN for visual backbones
- **DrAC** random crop augmentation improves generalization
- **Curriculum reward shaping**: start simple, add complexity over training phases
- **Sticky actions** (25% repeat last action): reduces UI flicker in strategy games
- **Linear LR schedule**: decay from initial over total timesteps
- **Weight decay 1e-4**: prevents plasticity loss / value function collapse
- **Entropy bonus early**: encourages exploration before exploitation
- **Value function collapse risk** at ~1M steps — monitor and adjust weight decay

## Autonomous Rules (pinned in profile)
1. NO human in loop. Never ask/hedge/say 'should I'. Decide everything.
2. Vague topic → pick most useful interpretation. Broad scope → narrow to practical.
3. Missing context → use best available info, proceed.
4. Overlaps → read existing work first, add NEW value only.
5. Install/download ANY software that helps the project.
6. Search Google + GitHub for similar projects. Learn, copy what helps.
7. Optimize everything: speed, storage, settings.
8. Store information persistently for faster future sessions.

## What's Done vs Pending
### Done ✅
- Project moved to E:\Green Vision, venv rebuilt, smoke test passing
- LLM swap fully removed (no ollama/llama.cpp references remain)
- gather_videos.py Windows-ready (_find_ytdlp(), no Linux-only flags)
- Game mechanics reference saved
- All .bat launchers point to E:\ paths
- train_autonomous.py uses Scripts\python.exe on Windows
- smoke_test.py uses relative path (portable)
- ffmpeg + Tesseract installed system-wide
- yt-dlp installed in venv

### Pending / Next Steps ⏳
- [ ] Run video gathering (background job was started but unverified — check `data/videos/` for manifests)
- [ ] First Stage A pretrain run once frames exist
- [ ] Consider upgrading CNN backbone to ImpalaCNN architecture
- [ ] Add DrAC augmentation to frame preprocessing
- [ ] Implement curriculum reward shaping in PPO config
- [ ] Add sticky action logic to virtual_input.py
- [ ] Linear LR schedule in training scripts
- [ ] Monitor value function stability during long runs
- [ ] Wire live game loop (main.py) to real Geoplanetical process reads
