#!/usr/bin/env python3
"""
Curriculum Manager — Progressive Difficulty Scheduler.
Manages training stages based on agent performance metrics.
Prevents catastrophic forgetting by gradually increasing complexity.
"""
import logging
from typing import Dict, List, Optional
import numpy as np

log = logging.getLogger(__name__)

class CurriculumManager:
    """
    Manages progressive difficulty levels for Green Vision training.
    Stages are defined by game complexity (e.g., simple states -> crises).
    """
    def __init__(self, num_stages: int = 5):
        self.num_stages = num_stages
        self.current_stage = 0
        self.performance_history: List[float] = []
        self.stage_thresholds: List[float] = [0.6, 0.7, 0.8, 0.9]
        log.info(f"Curriculum initialized with {num_stages} stages")
        
    def get_difficulty_multiplier(self) -> float:
        """
        Returns current difficulty multiplier (1.0 = baseline).
        Higher stage = higher multiplier for exploration noise or action space.
        """
        # Simple linear scaling
        return 1.0 + (self.current_stage / self.num_stages) * 2.0
        
    def evaluate_performance(self, episode_reward: float) -> bool:
        """
        Evaluates if agent has met threshold to advance stage.
        Args:
            episode_reward: Normalized reward from last episode
        Returns:
            True if ready to advance stage
        """
        self.performance_history.append(episode_reward)
        
        # Keep last 10 episodes for stability
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
            
        avg_reward = np.mean(self.performance_history[-5:])
        
        # Check if we've crossed the threshold for next stage
        if self.current_stage < self.num_stages - 1:
            threshold = self.stage_thresholds[self.current_stage]
            if avg_reward >= threshold:
                log.info(f"Performance {avg_reward:.3f} >= Threshold {threshold:.3f}. Advancing to Stage {self.current_stage + 2}")
                self.current_stage += 1
                return True
        
        return False
        
    def get_state_complexity_weight(self, state_vec: np.ndarray) -> float:
        """
        Computes weight based on current state complexity.
        Used to bias exploration toward harder states at higher stages.
        """
        # Simple heuristic: variance in memory variables indicates complexity
        mem_vars = state_vec[32:64]  # Memory segment indices
        complexity = np.std(mem_vars)
        
        # Scale by current stage
        base_weight = 1.0 + (self.current_stage * 0.1)
        return complexity * base_weight
