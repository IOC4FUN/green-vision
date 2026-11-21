"""
gym_env – Gymnasium-compatible environment wrapper for Green Vision.

Exposes Geoplanetical as a standard RL environment:
  Observation: Box(0, 1, (128,), float32)
  Action:      Discrete(31)

Usage:
    from src.gym_env import GeoplaneticalEnv
    env = GeoplaneticalEnv(mode="simulated")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from .geoplanetical_env import GeoplaneticalEnv

__all__ = ["GeoplaneticalEnv"]
