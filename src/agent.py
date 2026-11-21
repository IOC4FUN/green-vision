"""
Green Vision – Autonomous Game Agent Orchestrator.
Ties together all subsystems:
  • Perception (screen capture + OCR)
  • Memory (process reader + knowledge DB)
  • Brain (PPO network + planner)
  • Input (virtual keyboard/mouse via /dev/uinput)
  • Research (Serper web search)
  • Game controller (launch, focus, lifecycle)
The agent runs a continuous loop:
    observe → think → act → learn
at ~2 Hz decision rate (configurable).
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from .brain.network import NUM_ACTIONS, STATE_DIM, ACTION_NAMES
from .brain.ppo_agent import PPOAgent, PPOConfig
from .brain.planner import AutonomousPlanner, GoalPriority
from .brain.game_knowledge import GameKnowledge, seed_base_knowledge, A_NAV_ECONOMY
from .brain.episodic_memory import EpisodicMemory, Outcome
from .brain.reflection import ReflectionModule
from .brain.curriculum import CurriculumManager
from .brain.curiosity import CuriosityModule
from .brain.world_model import WorldModel
from .brain.rssm_world_model import RSSMWorldModel
from .brain.affect import AffectModule
from .game.game_controller import GameController
from .input.virtual_input import VirtualInputDevice, Key
from .memory.knowledge_db import KnowledgeDB
from .memory.process_reader import GameProcessReader
from .perception.capture import ScreenCapture
from .perception.ocr_engine import OCREngine
from .perception.video_ingest import VideoIngestor
from .research.serper_client import SerperClient
from .utils.state_schema import MEMORY_KEYS as _MEM_KEYS, build_vector as _build_vec
from .utils.reward_shaper import RewardWeights, compute_reward as _shaped_reward
from .utils.video_pretrain import VideoPretrainer
from .utils.journal import DecisionJournal
from .utils.checkpoints import CheckpointManager
from .utils.health import HealthWatchdog

logger = logging.getLogger("green_vision.agent")

# Default click target for mouse-based actions when no region is configured.
_DEFAULT_CLICK_X = 960
_DEFAULT_CLICK_Y = 540


class GreenVisionAgent:
    """
    Top-level orchestrator for the Green Vision AI.
    Lifecycle:
        agent = GreenVisionAgent(config_path="config/settings.yaml")
        agent.start()          # launches game + enters main loop
        agent.stop()           # graceful shutdown
    """

    def __init__(self, config_path: str | Path = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self._running = False
        self._stop_event = False

        # ── Subsystems ───────────────────────────────────────────
        self.game_ctrl = GameController(
            steam_app_id=self.config.get("steam_app_id", 940230),
            launch_timeout_s=self.config.get("launch_timeout_s", 120),
        )
        self.capture = ScreenCapture(fps=self.config.get("capture_fps", 5))
        self.ocr = OCREngine(lang=self.config.get("ocr_lang", "eng"))
        self.input = VirtualInputDevice()
        self.kb = KnowledgeDB(
            db_path=self.config.get("knowledge_db", "data/knowledge/db.sqlite"),
            vector_dir=self.config.get("knowledge_vectors", "data/knowledge/vectors"),
        )
        self.serper = SerperClient(api_key=self.config.get("serper_api_key"))

        # Brain
        ppo_cfg = PPOConfig(
            device=self.config.get("device", "cuda"),
            learning_rate=self.config.get("learning_rate", 3e-4),
            update_every_steps=self.config.get("update_every_steps", 8),
        )
        model_path = self.config.get("model_path", "data/models/green_vision.pt")
        self.agent = PPOAgent(config=ppo_cfg, model_path=model_path)
        self.planner = AutonomousPlanner()
        # Episodic memory + reflection
        self.episodic_mem = EpisodicMemory(capacity=500)
        self.reflector = ReflectionModule(self.episodic_mem)
        self.curriculum = CurriculumManager(reflector=self.reflector)
        self.planner.episodic_mem = self.episodic_mem
        # Intrinsic motivation + world model + affective state
        self.curiosity = CuriosityModule(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            coeff=float(self.config.get("curiosity_coeff", 0.5)),
            device=self.config.get("device", "cuda"),
        )
        # Primary world model: RSSM (DreamerV3-style) with calibrated uncertainty.
        # Falls back to the simpler feedforward WM if RSSM init fails.
        try:
            self.world_model = RSSMWorldModel(
                state_dim=STATE_DIM,
                num_actions=NUM_ACTIONS,
                deter=int(self.config.get("rssm_deter", 512)),
                stoch=int(self.config.get("rssm_stoch", 64)),
                lr=float(self.config.get("wm_lr", 3e-4)),
                device=self.config.get("device", "cuda"),
            )
            logger.info("Using RSSMWorldModel as primary world model")
        except Exception as e:  # noqa: BLE001
            logger.warning("RSSM init failed (%s); falling back to feedforward WM", e)
            self.world_model = WorldModel(
                state_dim=STATE_DIM,
                num_actions=NUM_ACTIONS,
                device=self.config.get("device", "cuda"),
            )
        # Video ingestion pipeline (YouTube → frames → pretraining data)
        self.video_ingestor = VideoIngestor(
            output_dir=self.config.get("video_data_dir", "data/videos"),
        )
        self.affect = AffectModule()
        self.planner.affect = self.affect
        # Transition tracking for curiosity / world-model training
        self._prev_state_vec: np.ndarray = np.zeros(STATE_DIM, dtype=np.float32)
        self._wm_buffer: list[tuple[np.ndarray, int, np.ndarray, float]] = []
        self._wm_train_every: int = 4   # train WM + curiosity every N decisions
        self._wm_step_counter: int = 0
        self._episode_active = False

        # Memory reader (reads game process memory if accessible)
        self.mem_reader = GameProcessReader(process_name="Geoplanetical.exe")

        # State tracking
        self._frame_count = 0
        self._decision_interval = float(self.config.get("decision_interval_s", 0.5))  # 2 Hz
        self._last_decision_ts = 0.0
        self._state_vec: np.ndarray = np.zeros(STATE_DIM, dtype=np.float32)
        self._game_state_dict: dict[str, Any] = {}
        # Cached observation from the last decision cycle (for _learn).
        self._last_obs_frame: Optional[np.ndarray] = None   # (3,H,W) float [0,1]
        self._last_logits: Optional[Any] = None              # torch tensor
        self._last_value: Optional[Any] = None               # torch tensor
        # Idle-watchdog state (forces a corrective move if the agent stalls)
        self._last_meaningful_ts: float = time.time()
        self._no_op_streak: int = 0
        self.console_enabled: bool = bool(self.config.get("console_enabled", False))

        # ── Utils subsystems ───────────────────────────────────────
        self.reward_weights = RewardWeights.from_config(self.config)
        self.journal = DecisionJournal(
            path=self.config.get("journal_path", "data/journal/decisions.jsonl")
        )
        self.checkpoints = CheckpointManager(
            dir_path=self.config.get("checkpoint_dir", "data/models/checkpoints"),
            save_every_steps=int(self.config.get("checkpoint_every_steps", 500)),
        )
        self.health = HealthWatchdog(
            ocr_empty_threshold=int(self.config.get("ocr_empty_threshold", 30)),
            mem_fail_timeout_s=float(self.config.get("mem_fail_timeout_s", 10.0)),
            capture_min_fps=float(self.config.get("capture_min_fps", 2.0)),
        )
        self._prev_state_dict: dict[str, float] = {}
    @staticmethod
    def _load_config(path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            logger.warning("Config %s not found – using defaults", path)
            return {}
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        # Inject .env values
        env_file = path.parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg.setdefault(k.strip(), v.strip())
        return cfg

    # ── Main loop ────────────────────────────────────────────────
    def start(self, max_steps: int | None = None) -> None:
        """Launch the game and enter the main agent loop.

        Args:
            max_steps: If set, stop after this many decision steps (training mode).
        """
        logger.info("═" * 60)
        logger.info("  GREEN VISION – Autonomous Game Agent")
        logger.info("═" * 60)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Launch game
        if self.config.get("auto_launch_game", True):
            if not self.game_ctrl.launch():
                logger.error("Failed to launch game – exiting")
                return
            if not self.game_ctrl.wait_for_ready():
                logger.error("Game did not become ready – exiting")
                return

        # Set scenario – optionally driven by the adaptive curriculum.
        scenario = self.config.get("scenario", {}) or {}
        if self.config.get("use_curriculum", False):
            scen_cfg = self.curriculum.next_scenario()
            scenario = {"country": scen_cfg.country, "role": scen_cfg.role}
            logger.info("Curriculum selected scenario: %s", scen_cfg.name)
        self._current_scenario_name = (
            f"{scenario.get('country', 'Germany')}/{scenario.get('role', 'government')}"
        )
        self.planner.set_scenario(self._current_scenario_name)

        # Initial goals
        self.planner.add_goal("Survive first year in office", GoalPriority.CRITICAL, steps=12)
        self.planner.add_goal("Maintain approval above 40%", GoalPriority.HIGH, steps=999)
        self.planner.add_goal("Pass at least one major bill per quarter", GoalPriority.MEDIUM, steps=4)
        # Begin episodic episode tracking
        self.episodic_mem.begin_episode(
            scenario=f"{scenario.get('country', 'Germany')}/{scenario.get('role', 'government')}"
        )
        self._episode_active = True
        # Seed documented base knowledge so the AI has a working model before any live research.
        try:
            seeded = seed_base_knowledge(self.kb)
            logger.info("Base knowledge ready (%d snippets). Console enabled: %s", seeded, self.console_enabled)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to seed base knowledge: %s", e)
        # Research initial strategy (if Serper available)
        if self.serper.available:
            self._initial_research()

        # Start perception + input subsystems
        try:
            self.capture.start()
        except Exception as e:  # noqa: BLE001
            logger.warning("Screen capture failed to start: %s", e)
        if not self.input.open():
            logger.warning("Virtual input device unavailable – actions will be logged only")
        self.mem_reader.find_process()  # best-effort attach

        self._running = True
        logger.info("Agent loop started. Decision interval: %.2fs", self._decision_interval)
        try:
            while self._running and not self._stop_event:
                t_cycle_start = time.perf_counter()

                # Check game is still alive
                if not self.game_ctrl.is_running:
                    logger.warning("Game process lost – waiting for restart...")
                    time.sleep(5)
                    continue

                # ── OBSERVE ──────────────────────────────────────
                fobj = self.capture.latest_frame()
                if fobj is None or fobj.raw is None:
                    time.sleep(0.1)
                    continue

                raw = fobj.raw          # H×W×3 uint8 (BGR from mss)
                resized = fobj.resized  # 224×224×3 RGB for the CNN
                if resized is None:
                    import cv2
                    resized = cv2.resize(raw, (224, 224))[:, :, ::-1]  # BGR→RGB + resize

                ocr_text = ""
                if self.config.get("use_ocr", True) and self.ocr.available:
                    ocr_res = self.ocr.process(raw)
                    ocr_text = ocr_res.raw_text or ""

                # Read memory (best-effort)
                mem_state: dict[str, float] = {}
                if self.mem_reader.pid:
                    try:
                        mem_state = self.mem_reader.read_state().values or {}
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Memory read failed: %s", e)

                # Health watchdog tick
                health_status = self.health.tick(ocr_text=ocr_text, mem_state=mem_state or None)

                # Build state vector
                self._update_state_vector(resized, ocr_text, mem_state)

                # ── THINK / ACT / LEARN at decision rate ─────────
                now = time.time()
                if now - self._last_decision_ts >= self._decision_interval:
                    self._last_decision_ts = now
                    action_idx = self._think_and_decide(resized, ocr_text)
                    # Idle watchdog: if we've been stalling, force a corrective move.
                    if self._idle_too_long(now):
                        forced = self._corrective_action()
                        logger.warning(
                            "Idle watchdog fired (no-op streak=%d) – forcing %s",
                            self._no_op_streak, ACTION_NAMES[forced],
                        )
                        action_idx = forced
                    self._track_activity(action_idx, now)
                    self._execute_action(action_idx)
                    reward = self._compute_reward()
                    self._learn(action_idx, reward)
                    # Episodic memory recording
                    if self._episode_active:
                        try:
                            thought_str = ""
                            if self.planner.thoughts:
                                thought_str = self.planner.thoughts[-1].reasoning[:200]
                            self.episodic_mem.record_step(
                                state=self._state_vec.copy(),
                                action_idx=action_idx,
                                action_name=ACTION_NAMES[action_idx],
                                reward=reward,
                                thought=thought_str,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    # Journal every decision
                    try:
                        thought_text = self.planner.thoughts[-1].reasoning if self.planner.thoughts else ""
                        self.journal.log(
                            action=ACTION_NAMES[action_idx],
                            confidence=float(probs.max().item()) if hasattr(self, '_last_conf') else 0.5,
                            state_snapshot=self._game_state_dict,
                            ocr_text=ocr_text[:200],
                            reward=reward,
                            thought=thought_text,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    # Checkpoint manager
                    self.checkpoints.record_reward(reward)
                    self.checkpoints.maybe_save(
                        step=self._frame_count,
                        model=self.agent.net,
                        optimizer=self.agent.optimizer,
                    )
                    self._frame_count += 1
                    if max_steps is not None and self._frame_count >= max_steps:
                        logger.info("Reached max_steps=%d – stopping.", max_steps)
                        break
                    if self._frame_count % 50 == 0:
                        logger.info(
                            "[step %d] decisions=%d | buffer=%d | last_action=%s",
                            self._frame_count,
                            self.planner.total_decisions,
                            len(self.agent.buffer),
                            ACTION_NAMES[action_idx],
                        )

                # Sleep to maintain target rate
                elapsed = time.perf_counter() - t_cycle_start
                sleep_time = max(0.0, self._decision_interval * 0.3 - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down Green Vision...")
        self._running = False
        # End episodic episode + reflect
        if self._episode_active:
            ep = None
            try:
                ep = self.episodic_mem.end_episode()
                if ep is not None:
                    refl = self.reflector.reflect(ep)
                    logger.info("Final reflection: %s", refl.raw_summary[:200])
                self.episodic_mem.save_to_file(
                    self.config.get("episodic_memory_path", "data/memory/episodes.jsonl")
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Episodic memory shutdown failed: %s", e)
            # Record episode outcome into the adaptive curriculum.
            try:
                if ep is not None:
                    self.curriculum.record_result(
                        scenario_name=self._current_scenario_name,
                        outcome=ep.outcome.value,
                        reward=ep.total_reward,
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug("Curriculum record_result failed: %s", e)
            self._episode_active = False
        try:
            self.agent.save(self.config.get("model_path", "data/models/green_vision.pt"))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to save model: %s", e)
        try:
            self.capture.stop()
        except Exception as e:  # noqa: BLE001
            logger.debug("capture.stop failed: %s", e)
        try:
            self.input.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("input.close failed: %s", e)
        try:
            self.journal.close()
        except Exception:  # noqa: BLE001
            pass
        logger.info("Model saved. Journal closed. Goodbye.")

    def _signal_handler(self, signum, frame):
        logger.info("Received signal %d – stopping...", signum)
        self._stop_event = True

    # ── Observation → State vector ───────────────────────────────
    def _update_state_vector(
        self, frame: np.ndarray, ocr_text: str, mem_state: dict
    ) -> None:
        """
        Build the fixed-size state vector from all available sources.
        Layout defined in src/utils/state_schema.py – single source of truth.
        """
        import re
        vec = _build_vec()  # zeroed (STATE_DIM,) float32

        # [0:16] Screen features: 4×4 downsampled grayscale cell means.
        gray = frame.mean(axis=2) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        cells = []
        for i in range(4):
            for j in range(4):
                block = gray[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
                cells.append(block.mean() if block.size else 0.0)
        vec[0:16] = np.array(cells, dtype=np.float32) / 255.0

        # [16:32] OCR numbers (first 16 parsed values, clamped).
        nums = [float(x.replace(",", "")) for x in re.findall(r'\d+[.,]\d+', ocr_text)]
        for i, n in enumerate(nums[:16]):
            vec[16 + i] = min(n / 100.0, 1.0)

        # [32:64] Memory-read game variables (named keys from state_schema).
        for i, key in enumerate(_MEM_KEYS):
            if key in mem_state:
                try:
                    vec[32 + i] = float(mem_state[key]) / 100.0
                except (TypeError, ValueError):
                    pass

        # [64:80] Time encoding.
        t = time.time()
        vec[64] = np.sin(t * 0.001)
        vec[65] = np.cos(t * 0.001)

        self._state_vec = vec
        self._game_state_dict = {k: float(mem_state.get(k, 50)) for k in _MEM_KEYS}

    # ── Decision cycle ───────────────────────────────────────────
    def _think_and_decide(self, resized: np.ndarray, ocr_text: str) -> int:
        """Run the full think pipeline and return an action index."""
        import torch.nn.functional as F
        # Normalize frame to (3, H, W) float [0,1].
        f = resized.astype(np.float32) / 255.0
        if f.ndim == 3 and f.shape[-1] == 3:
            f = f.transpose(2, 0, 1)
        self._last_obs_frame = f
        # Neural network forward pass.
        logits, value = self.agent.observe(f, self._state_vec)
        self._last_logits = logits
        self._last_value = value
        # ── Affect modulation: suppress risky actions when low-confidence ──
        risk_scale = self.affect.risk_multiplier()
        if risk_scale < 1.0:
            risky_idx = {ACTION_NAMES.index(n) for n in
                         ("declare_war", "call_election", "break_alliance",
                          "fire_minister", "veto_bill")}
            logits = logits.clone()
            for ri in risky_idx:
                if ri < logits.shape[-1]:
                    logits[0, ri] *= risk_scale
        nn_action, _log_prob = self.agent.select_action(logits)
        # Confidence: softmax max probability.
        probs = F.softmax(logits.squeeze(0), dim=-1)
        confidence = float(probs.max().item())
        # ── World-model lookahead: score top-K NN candidates over 3 steps ──
        wm_bonus_action: int | None = None
        wm_conf = 0.0
        try:
            wm_conf = float(self.world_model.confidence(self._state_vec))
            top_k = torch.topk(logits.squeeze(0), k=5).indices.tolist()
            candidate_seqs = [[a] * 3 for a in top_k]
            best_seq, best_val = self.world_model.plan(
                self._state_vec, candidate_seqs, horizon=3, beam_width=5
            )
            # Trust the model-based plan only when the world model is confident
            # about this state region; otherwise leave it to policy/curiosity.
            if best_seq and best_val > 0.1 and wm_conf >= 0.5:
                wm_bonus_action = best_seq[0]
        except Exception as e:  # noqa: BLE001
            logger.debug("World-model plan failed: %s", e)
        # Knowledge base hints.
        kb_hints: list[str] = []
        try:
            query = f"{self.planner.current_scenario} {str(self._game_state_dict)[:160]}"
            for r in self.kb.search_knowledge(query, k=3):
                content = r.get("content")
                if content:
                    kb_hints.append(str(content))
        except Exception as e:  # noqa: BLE001
            logger.debug("KB search failed: %s", e)
        # Planner decides final action (with affect urgency boost).
        urgency = self.affect.urgency()
        final_action, thought = self.planner.decide(
            nn_action=nn_action,
            nn_confidence=confidence,
            game_state=self._game_state_dict,
            knowledge_hints=kb_hints or None,
            urgency_boost=urgency,
        )
        # World-model override: if WM strongly prefers a different action and
        # NN confidence is low (< 0.4), trust the model-based plan.
        if (wm_bonus_action is not None
                and wm_bonus_action != final_action
                and confidence < 0.4):
            logger.info("WM override: NN=%s → WM=%s (conf=%.2f)",
                        ACTION_NAMES[final_action], ACTION_NAMES[wm_bonus_action], confidence)
            final_action = wm_bonus_action
        # Curiosity fallback: NN unsure AND world model uncertain → explore.
        if confidence < 0.3 and wm_conf < 0.5:
            try:
                final_action = self._curious_explore(int(final_action))
                logger.info("Curiosity fallback exploring (NN conf=%.2f, WM conf=%.2f)",
                            confidence, wm_conf)
            except Exception as e:  # noqa: BLE001
                logger.debug("Curiosity fallback failed: %s", e)
        # Update prev state vector for next curiosity computation
        self._prev_state_vec = self._state_vec.copy()
        logger.debug("Thought: %s | affect: conf=%.2f frust=%.2f excite=%.2f",
                     (thought.reasoning[:120] if thought else ""),
                     self.affect.confidence, self.affect.frustration, self.affect.excitement)
        return int(final_action)

    def _execute_action(self, action_idx: int) -> None:
        """Map action index to actual input events."""
        name = ACTION_NAMES[action_idx]
        cx = int(self.config.get("click_x", _DEFAULT_CLICK_X))
        cy = int(self.config.get("click_y", _DEFAULT_CLICK_Y))

        actions_map = {
            "noop": lambda: None,
            "pause_game": lambda: self.input.press_key(Key.SPACE),
            "resume_game": lambda: self.input.press_key(Key.SPACE),
            "open_menu": lambda: self.input.press_key(Key.ESCAPE),
            "close_menu": lambda: self.input.press_key(Key.ESCAPE),
            "navigate_budget": lambda: self._press_seq([Key.F1] * 3),
            "navigate_legislation": lambda: self._press_seq([Key.F2] * 3),
            "navigate_diplomacy": lambda: self._press_seq([Key.F3] * 3),
            "navigate_military": lambda: self._press_seq([Key.F4] * 3),
            "navigate_economy": lambda: self._press_seq([Key.F5] * 3),
            "apply_policy": lambda: self.input.press_key(Key.ENTER),
            "discard_policy": lambda: self.input.press_key(Key.ESCAPE),
            "increase_tax": lambda: self.input.click_mouse(cx, cy),
            "decrease_tax": lambda: self.input.click_mouse(cx, cy),
            "increase_spending": lambda: self.input.click_mouse(cx, cy),
            "decrease_spending": lambda: self.input.click_mouse(cx, cy),
            "file_bill": lambda: self.input.press_key(Key.ENTER),
            "veto_bill": lambda: self.input.press_key(Key.ESCAPE),
            "declare_war": lambda: self.input.press_key(Key.ENTER),
            "seek_peace": lambda: self.input.press_key(Key.ENTER),
            "form_alliance": lambda: self.input.press_key(Key.ENTER),
            "break_alliance": lambda: self.input.press_key(Key.ESCAPE),
            "call_election": lambda: self.input.press_key(Key.ENTER),
            "appoint_minister": lambda: self.input.press_key(Key.ENTER),
            "fire_minister": lambda: self.input.press_key(Key.ESCAPE),
            "use_console": self._maybe_use_console,
            "save_game": lambda: self._press_seq([Key.F5]),
            "load_game": lambda: self._press_seq([Key.F9] if hasattr(Key, 'F9') else [Key.F1]),
            "switch_speed_1": lambda: self.input.press_key(Key.NUM_1),
            "switch_speed_2": lambda: self.input.press_key(Key.NUM_2),
            "switch_speed_3": lambda: self.input.press_key(Key.NUM_3),
        }
        fn = actions_map.get(name)
        if fn is None:
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            logger.warning("Action '%s' failed: %s", name, e)

    def _press_seq(self, keys) -> None:
        for k in keys:
            self.input.press_key(k)
            time.sleep(0.1)

    # ── Console gating + idle watchdog ───────────────────────────
    def _maybe_use_console(self) -> None:
        """Open the cheat console only if it has been enabled (dashboard toggle)."""
        if not self.console_enabled:
            logger.debug("Console action requested but console_enabled=false – skipping")
            return
        self.input.press_key(Key.GRAVE)

    def _track_activity(self, action_idx: int, now: float) -> None:
        name = ACTION_NAMES[action_idx]
        if name == "noop":
            self._no_op_streak += 1
        else:
            self._no_op_streak = 0
            self._last_meaningful_ts = now

    def _idle_too_long(self, now: float) -> bool:
        limit = float(self.config.get("idle_watchdog_s", 30))
        return (now - self._last_meaningful_ts) > limit and self._no_op_streak >= 4

    def _curious_explore(self, default_action: int) -> int:
        """Pick a novel, low-risk action when both NN and WM are uncertain."""
        import random
        risky = {"declare_war", "call_election", "break_alliance",
                 "fire_minister", "veto_bill"}
        candidates = [i for i in range(NUM_ACTIONS)
                      if ACTION_NAMES[i] not in risky and i != 0]
        best_a, best_n = int(default_action), -1.0
        for a in random.sample(candidates, k=min(8, len(candidates))):
            try:
                nxt = self.world_model.predict_next_state(self._state_vec, a)
                nov = self.curiosity.novelty_score(nxt)
            except Exception:  # noqa: BLE001
                continue
            if nov > best_n:
                best_n, best_a = float(nov), a
        return int(best_a)
    def _corrective_action(self) -> int:
        """Pick a sensible move to un-stall the agent."""
        try:
            prios = GameKnowledge.get_priority_actions(self._game_state_dict)
            if prios:
                return int(prios[0])
        except Exception:  # noqa: BLE001
            pass
        return A_NAV_ECONOMY

    def _compute_reward(self) -> float:
        """
        Derive a scalar reward using the externalised RewardWeights (settings.yaml).
        Adds intrinsic curiosity bonus on top of extrinsic game reward.
        """
        curr = self._game_state_dict
        prev = self._prev_state_dict or {k: 50.0 for k in _MEM_KEYS}
        action_was_noop = (self.planner.action_history and
                           self.planner.action_history[-1][2] == "noop")
        r = _shaped_reward(self.reward_weights, prev, curr, action_was_noop)
        # Intrinsic curiosity reward (ICM prediction-error bonus)
        try:
            last_action = self.planner.action_history[-1][1] if self.planner.action_history else 0
            intrinsic = self.curiosity.compute_intrinsic_reward(
                self._prev_state_vec, last_action, self._state_vec
            )
            r += intrinsic
        except Exception as e:  # noqa: BLE001
            logger.debug("Curiosity reward failed: %s", e)
        # Store current as previous for next cycle
        self._prev_state_dict = dict(curr)
        return float(np.clip(r, -1.0, 1.0))

    # ── Learning ─────────────────────────────────────────────────
    def _learn(self, action_idx: int, reward: float) -> None:
        """Store transition, train curiosity/world-model, trigger PPO update."""
        import torch
        from torch.distributions import Categorical
        if self._last_obs_frame is None or self._last_logits is None:
            return
        try:
            dist = Categorical(logits=self._last_logits.squeeze(0))
            log_prob = float(dist.log_prob(torch.tensor(action_idx)).item())
            val = float(self._last_value.item())
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not compute log-prob/value: %s", e)
            return
        done = not self.game_ctrl.is_running
        self.agent.store_transition(
            self._last_obs_frame, self._state_vec,
            action_idx, reward, done, log_prob, val,
        )
        # ── Affect update (emotional state modulation) ──────────────
        risky_actions = {"declare_war", "call_election", "break_alliance",
                         "fire_minister", "veto_bill"}
        was_risky = ACTION_NAMES[action_idx] in risky_actions
        self.affect.update(reward=reward, action_was_risky=was_risky, done=done)
        # ── World-model / curiosity training buffer ─────────────────
        self._wm_buffer.append((
            self._prev_state_vec.copy(), action_idx,
            self._state_vec.copy(), reward,
        ))
        if len(self._wm_buffer) > 256:
            self._wm_buffer.pop(0)
        self._wm_step_counter += 1
        if self._wm_step_counter >= self._wm_train_every and len(self._wm_buffer) >= 8:
            try:
                batch = self._wm_buffer[-32:]
                states_b = np.array([t[0] for t in batch], dtype=np.float32)
                actions_b = [t[1] for t in batch]
                nexts_b = np.array([t[2] for t in batch], dtype=np.float32)
                rewards_b = [t[3] for t in batch]
                wm_metrics = self.world_model.train_step(states_b, actions_b, nexts_b, rewards_b)
                cur_metrics = self.curiosity.train_step(states_b, actions_b, nexts_b)
                logger.debug("WM/curiosity train: %s | %s", wm_metrics, cur_metrics)
            except Exception as e:  # noqa: BLE001
                logger.debug("WM/curiosity training failed: %s", e)
            self._wm_step_counter = 0
        # ── PPO update ──────────────────────────────────────────────
        if self.agent.should_update():
            metrics = self.agent.update()
            logger.debug("PPO update: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

    # ── Research ─────────────────────────────────────────────────
    def _initial_research(self) -> None:
        """Do a one-time web search to seed the knowledge base."""
        try:
            results = self.serper.research_game_strategy("beginner tips first year")
            n_strat = 0
            for r in list(results)[:5]:
                content = f"{r.title}. {r.snippet}"
                self.kb.add_knowledge(content=content, source=r.link or "research")
                n_strat += 1
            patches = self.serper.check_for_patches()
            n_patch = 0
            for p in list(patches)[:3]:
                self.kb.add_knowledge(content=f"{p.title}. {p.snippet}", source=p.link or "patches")
                n_patch += 1
            logger.info("Initial research complete: %d strategy + %d patch entries",
                        n_strat, n_patch)
        except Exception as e:  # noqa: BLE001
            logger.warning("Research failed (non-fatal): %s", e)
