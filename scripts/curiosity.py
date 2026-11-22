#!/usr/bin/env python3
"""
Curiosity Module — Intrinsic Motivation via Prediction Error.
Implements an intrinsic reward signal based on RSSM prediction error.
Integrates with PPO agent to encourage exploration of novel states.
"""
import torch
import torch.nn as nn
from typing import Optional
import logging

log = logging.getLogger(__name__)

class CuriosityModule(nn.Module):
    """
    Computes intrinsic reward using world model prediction error.
    Uses RSSM (Recurrent State-Space Model) to predict next state.
    Reward = -||state_pred - state_actual||^2
    """
    def __init__(self, hidden_dim: int = 512, device: str = 'cuda'):
        super().__init__()
        self.device = device
        # Simple MLP for prediction head if not provided by external model
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        
        log.info("CuriosityModule initialized")
        
    def forward(self, state_obs: torch.Tensor, state_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_obs: Actual observed state vector [B, STATE_DIM]
            state_pred: Predicted state from RSSM [B, STATE_DIM]
        Returns:
            Intrinsic reward scalar per batch element
        """
        diff = state_obs - state_pred
        # Squared Euclidean distance (negative for maximization in RL)
        intrinsic_reward = -(diff ** 2).mean(dim=1)
        return intrinsic_reward.to(self.device)

class CuriosityScheduler(nn.Module):
    """
    Scales curiosity weight over time to prevent early dominance.
    Decay factor prevents agent from ignoring extrinsic rewards too quickly.
    """
    def __init__(self, initial_weight: float = 0.5, decay_rate: float = 0.999):
        super().__init__()
        self.initial_weight = initial_weight
        self.decay_rate = decay_rate
        self.step_count = 0
        
    def get_weight(self, step: int) -> float:
        """Returns current curiosity weight."""
        weight = self.initial_weight * (self.decay_rate ** step)
        return min(weight, 1.0)

def compute_curiosity_reward(
    rssm_model,
    obs_batch: torch.Tensor,
    action_batch: torch.Tensor,
    scheduler: Optional[CuriosityScheduler] = None
) -> torch.Tensor:
    """
    Compute total intrinsic reward given RSSM model and scheduler.
    Args:
        rssm_model: Trained RSSM world model instance
        obs_batch: [B, T, STATE_DIM]
        action_batch: [B, T, NUM_ACTIONS]
        scheduler: CuriosityScheduler instance
    Returns:
        Intrinsic reward tensor [B]
    """
    # Run forward pass through RSSM to get predictions
    with torch.no_grad():
        state_pred, _ = rssm_model(obs_batch, action_batch)
    
    # Compute prediction error
    diff = obs_batch - state_pred
    intrinsic = -(diff ** 2).mean(dim=2)  # Mean over time steps
    
    if scheduler is not None:
        w = scheduler.get_weight(scheduler.step_count)
        intrinsic = intrinsic * w
        scheduler.step_count += 1
        
    return intrinsic
