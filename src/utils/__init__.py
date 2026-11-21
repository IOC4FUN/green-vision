"""
Green Vision shared utilities.

Modules:
    state_schema   – named 128-dim vector layout registry
    reward_shaper  – YAML-driven reward weights + compute_reward()
    journal        – append-only JSONL decision log
    checkpoints    – periodic model snapshot manager
    health         – subsystem watchdog (OCR / memory / capture FPS)
"""
from .state_schema import STATE_DIM, SEGMENTS, MEMORY_KEYS, build_vector, validate, describe
from .reward_shaper import RewardWeights, compute_reward
from .journal import DecisionJournal
from .checkpoints import CheckpointManager
from .health import HealthWatchdog, HealthStatus
from .llm_manager import LLMManager, LLMConfig
__all__ = [
    "STATE_DIM", "SEGMENTS", "MEMORY_KEYS",
    "build_vector", "validate", "describe",
    "RewardWeights", "compute_reward",
    "DecisionJournal",
    "CheckpointManager",
    "HealthWatchdog", "HealthStatus",
    "LLMManager", "LLMConfig",
]
