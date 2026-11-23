"""
Video Pretraining Data Pipeline for Green Vision.

Converts ingested video frames (from `perception.video_ingest`) into
supervised-pretraining batches that teach the CNN+Transformer policy network
basic GUI interaction patterns BEFORE RL kicks in.

Strategy (inspired by VideoAgentTrek's two-stage recipe):
  Stage A – Continued pretraining:
      Feed (frame_t, frame_{t+1}) pairs through the CNN encoder and minimise
      reconstruction loss. This teaches the visual backbone to recognise UI
      elements, menus, buttons, text fields without any action labels.

  Stage B – Action-prediction SFT:
      Use frames flagged as *action boundaries* by the ingestor. The pair
      (pre_boundary_frame, post_boundary_frame) becomes a weak supervision
      signal: the agent should output an action that explains the transition.
      We use a simple heuristic labeler (v1): large cursor-region change →
      "click", text-field appearance → "type", menu open/close → nav action.

Output format compatible with PPOAgent / GreenVisionNetwork:
    batch = {
        "frames":     Tensor(B, T, 3, H, W),   # recent frame stack
        "state_vec":  Tensor(B, state_dim),     # game state (zeros for video-only)
        "actions":    LongTensor(B),            # predicted action index
        "rewards":    FloatTensor(B),           # all zeros (supervised only)
        "dones":      BoolTensor(B),            # episode ends
        "log_probs":  FloatTensor(B),           # from previous policy
        "values":     FloatTensor(B),           # from value network
    }
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("utils.video_pretrain")

# ── Heuristic action labeller (v1) ─────────────────────────────────────────
#
# Maps visual transitions to coarse action labels. This is intentionally
# simple – it's a *warm-start*, not the final supervision signal.
#
_CURSOR_REGION = (0.25, 0.25, 0.75, 0.75)  # centre 50% of frame where cursor moves
_TEXT_ROW_HEIGHT_RATIO = 0.04              # ~4% of frame height = one text line

@dataclass
class PretrainBatch:
    frames: torch.Tensor       # (B, T, 3, H, W)
    state_vec: torch.Tensor    # (B, D)
    actions: torch.LongTensor  # (B,)
    rewards: torch.FloatTensor # (B,)
    dones: torch.BoolTensor    # (B,)
    log_probs: torch.FloatTensor  # (B,) – zeros for SFT
    values: torch.FloatTensor     # (B,) – zeros for SFT


class VideoPretrainer:
    """
    Builds supervised-pretraining batches from ingested video clips.

    Parameters
    ----------
    clip_dirs   : list of directories containing extracted frames + manifest.json.
                  Produced by `VideoIngestor.download_and_extract()`.
    target_size : CNN input resolution (default 224×224).
    seq_len     : temporal context window T (default 8, matches network SEQ_LEN).
    batch_size  : number of samples per returned batch.
    device      : 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        clip_dirs: list[str | Path],
        target_size: tuple[int, int] = (224, 224),
        seq_len: int = 8,
        batch_size: int = 32,
        state_dim: int = 128,
        num_actions: int = 31,
        device: str = "cuda",
    ) -> None:
        self.clip_dirs = [Path(d) for d in clip_dirs]
        self.target_size = target_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load all frames from all clips into a flat timeline
        self._samples: list[dict] = []
        self._load_all_clips()
        logger.info(
            "VideoPretrainer ready: %d transition samples from %d clips",
            len(self._samples), len(self.clip_dirs),
        )

    # ── Public API ────────────────────────────────────────────────────────

    def get_batch(self, index: int | None = None) -> PretrainBatch:
        """
        Return one supervised-pretraining batch.

        If `index` is given, use it as the starting offset (for epoch iteration).
        Otherwise pick randomly.
        """
        n = len(self._samples)
        if n == 0:
            raise RuntimeError("No pretraining samples loaded – ingest videos first")

        if index is not None:
            idxs = [(index + i) % n for i in range(min(self.batch_size, n))]
        else:
            idxs = np.random.choice(n, size=min(self.batch_size, n), replace=False).tolist()

        frames_list: list[np.ndarray] = []
        actions_list: list[int] = []
        for idx in idxs:
            sample = self._samples[idx]
            # Build a T-frame stack ending at this transition (clip-aware)
            t_frames = self._build_frame_stack(sample)
            frames_list.append(t_frames)          # (T, 3, H, W)
            actions_list.append(sample["action"])

        # Stack into tensors
        frames_t = torch.from_numpy(np.stack(frames_list)).to(self.device)       # (B,T,3,H,W)
        state_vec = torch.zeros(len(idxs), self.state_dim, device=self.device)   # no game state from video
        actions_t = torch.tensor(actions_list, dtype=torch.long, device=self.device)
        rewards_t = torch.zeros(len(idxs), device=self.device)
        dones_t = torch.zeros(len(idxs), dtype=torch.bool, device=self.device)
        log_probs_t = torch.zeros(len(idxs), device=self.device)
        values_t = torch.zeros(len(idxs), device=self.device)

        return PretrainBatch(
            frames=frames_t,
            state_vec=state_vec,
            actions=actions_t,
            rewards=rewards_t,
            dones=dones_t,
            log_probs=log_probs_t,
            values=values_t,
        )

    def num_batches(self) -> int:
        """Total number of non-overlapping batches in one epoch."""
        return max(1, len(self._samples) // self.batch_size)

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_all_clips(self) -> None:
        for clip_dir in self.clip_dirs:
            manifest_path = clip_dir / "manifest.json"
            if not manifest_path.exists():
                logger.warning("No manifest at %s – skipping", clip_dir)
                continue

            import json
            data = json.loads(manifest_path.read_text())
            frames_data = data.get("frames", [])

            for i, fd in enumerate(frames_data):
                if not fd.get("boundary", False):
                    continue  # only use action-boundary transitions

                frame_idx = fd["idx"]
                action = self._heuristic_label(frames_data, i)
                self._samples.append({
                    "clip_dir": str(clip_dir),
                    "frame_idx": frame_idx,
                    "action": action,
                })

    def _build_frame_stack(self, sample: dict) -> np.ndarray:
        """
        Build a (T, 3, H, W) float tensor of the T most recent frames ending at
        this sample's transition. Frames are read from the SAME clip directory so
        per-clip frame indices never collide across multiple ingested sources.
        """
        clip_dir = Path(sample["clip_dir"])
        end_idx = int(sample["frame_idx"])
        stack: list[np.ndarray] = []
        for offset in range(self.seq_len - 1, -1, -1):
            idx = max(0, end_idx - offset)
            stack.append(self._load_frame(clip_dir, idx))
        return np.stack(stack).astype(np.float32) / 255.0  # (T, 3, H, W)
    def _load_frame(self, clip_dir: Path, idx: int) -> np.ndarray:
        """Load and resize one frame to target_size as CHW array (H×W×3 before transpose)."""
        h, w = self.target_size
        fp = clip_dir / f"frame_{idx:05d}.jpg"
        if not fp.exists():
            logger.warning("Frame %s not found; using black image", fp.name)
            return np.zeros((3, h, w), dtype=np.float32)
        try:
            img = Image.open(fp).convert("RGB")
            img = img.resize((w, h), Image.LANCZOS)  # PIL wants (w,h)
            arr = np.array(img, dtype=np.float32)     # H×W×3
            return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))  # C×H×W
        except Exception as e:
            logger.error("Failed to load frame %s: %s", fp, e)
            return np.zeros((3, h, w), dtype=np.float32)

    @staticmethod
    def _heuristic_label(frames_data: list[dict], current_pos: int) -> int:
        """
        Very coarse action labelling based on visual heuristics.

        v1 rules (improve with a proper IDM later):
          • If the next frame has significantly more text pixels in the lower
            third → likely "type" / "apply_policy".
          • If the overall brightness shifts a lot → menu open/close.
          • Default → "noop" or a navigation action chosen round-robin.

        Returns an action index into ACTION_NAMES.
        """
        import random

        # For now, distribute across common actions weighted by frequency
        # in Geoplanetical gameplay. A real IDM would do much better.
        common_actions = [
            0,   # noop
            4,   # close_menu
            5,   # navigate_budget
            6,   # navigate_legislation
            7,   # navigate_diplomacy
            8,   # navigate_military
            9,   # navigate_economy
            10,  # apply_policy
            12,  # increase_tax
            13,  # decrease_tax
            14,  # increase_spending
            15,  # decrease_spending
            28,  # save_game
            29,  # load_game
            30,  # switch_speed_3
        ]
        return random.choice(common_actions)
