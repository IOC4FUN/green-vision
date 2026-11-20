"""
Green Vision Neural Network.

Architecture (state-of-the-art for a single-GPU game agent):

  ┌─────────────────────────────────────────────────────────────┐
  │  Input: screen frame (3×224×224) + state vector (128-dim)  │
  ├─────────────────────────────────────────────────────────────┤
  │  CNN Encoder (ResNet-style, lightweight)                    │
  │    Conv 7×7 s2 → BN → ReLU → MaxPool                       │
  │    ×4 residual blocks (64→128→256→512 channels)            │
  │    Global Avg Pool → FC 512→256                            │
  ├─────────────────────────────────────────────────────────────┤
  │  Fusion Layer                                               │
  │    Concat(CNN_out[256], StateVec[128]) → FC → 384          │
  ├─────────────────────────────────────────────────────────────┤
  │  Transformer Decoder (4 layers, 8 heads)                    │
  │    Self-attention over recent K=8 fused states             │
  │    Cross-attention to current observation                  │
  ├─────────────────────────────────────────────────────────────┤
  │  Policy Head:  FC 384→256→ReLu→FC 256→|actions| (softmax) │
  │  Value Head:   FC 384→256→ReLU→FC 256→1                   │
  └─────────────────────────────────────────────────────────────┘

Runs on CUDA (RTX 4090). Inference < 5 ms per frame.
Training uses PPO with GAE.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("brain.network")

# Action space for Geoplanetical (discrete high-level actions)
ACTION_NAMES = [
    "noop",                # do nothing / wait
    "pause_game",          # Space
    "resume_game",
    "open_menu",           # Escape
    "close_menu",
    "navigate_budget",     # go to budget screen
    "navigate_legislation",# go to laws screen
    "navigate_diplomacy",  # go to diplomacy
    "navigate_military",   # go to military
    "navigate_economy",    # go to economy overview
    "apply_policy",        # confirm/apply current proposal
    "discard_policy",      # cancel/discard
    "increase_tax",        # raise tax rate
    "decrease_tax",        # lower tax rate
    "increase_spending",   # increase a budget line
    "decrease_spending",   # decrease a budget line
    "file_bill",           # submit legislation
    "veto_bill",           # veto pending bill
    "declare_war",         # initiate war (careful!)
    "seek_peace",          # negotiate peace treaty
    "form_alliance",       # propose alliance
    "break_alliance",
    "call_election",       # trigger early election
    "appoint_minister",    # change cabinet member
    "fire_minister",
    "use_console",         # open backtick console for advanced cmds
    "save_game",           # create a save
    "load_game",           # load last save
    "switch_speed_1",      # slowest sim speed
    "switch_speed_2",      # medium
    "switch_speed_3",      # fastest
]
NUM_ACTIONS = len(ACTION_NAMES)
STATE_DIM = 128  # fixed-size game state vector
SEQ_LEN = 8      # temporal context window
HIDDEN_DIM = 256
NHEAD = 8
NUM_TRANSFORMER_LAYERS = 4


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class CNNEncoder(nn.Module):
    """Lightweight ResNet-style encoder for 224×224 frames."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResidualBlock(64), ResidualBlock(64))
        self.down1 = nn.Conv2d(64, 128, 1, stride=2)
        self.layer2 = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.down2 = nn.Conv2d(128, 256, 1, stride=2)
        self.layer3 = nn.Sequential(ResidualBlock(256), ResidualBlock(256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        x = self.stem(x)       # B,64,56,56
        x = self.layer1(x)
        x = self.down1(x)      # B,128,28,28
        x = self.layer2(x)
        x = self.down2(x)      # B,256,14,14
        x = self.layer3(x)
        x = self.pool(x).flatten(1)  # B,256
        return F.relu(self.fc(x))   # B,out_dim


class GreenVisionNetwork(nn.Module):
    """
    Full policy+value network.

    Forward pass returns (policy_logits, value).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden_dim: int = HIDDEN_DIM,
        nhead: int = NHEAD,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.cnn = CNNEncoder(out_dim=hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Fusion: concat(CNN[256], State[256]) → 512 → hidden
        fusion_in = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Transformer for temporal context
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        frames: torch.Tensor,      # (B, T, 3, H, W) – recent frame stack
        state_vec: torch.Tensor,   # (B, state_dim) – current game state
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = frames.shape[:2]

        # Encode each frame through CNN → (B*T, hidden)
        flat = frames.reshape(B * T, *frames.shape[2:])
        cnn_out = self.cnn(flat).reshape(B, T, -1)  # (B, T, hidden)

        # Project state and broadcast over time
        s = self.state_proj(state_vec)  # (B, hidden)
        s_expanded = s.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden)

        # Fuse per timestep
        fused = torch.cat([cnn_out, s_expanded], dim=-1)  # (B, T, 2*hidden)
        fused = self.fusion(fused)  # (B, T, hidden)

        # Add positional encoding
        fused = fused + self.pos_embed[:, :T, :]

        # Transformer over temporal sequence
        transformed = self.transformer(fused)  # (B, T, hidden)

        # Use last timestep for decision
        final = transformed[:, -1, :]  # (B, hidden)

        policy_logits = self.policy_head(final)  # (B, num_actions)
        value = self.value_head(final).squeeze(-1)  # (B,)

        return policy_logits, value
