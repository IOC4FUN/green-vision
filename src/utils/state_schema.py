"""
State vector schema — single source of truth for the 128-dim layout.

Every writer (_update_state_vector) and reader (network, dashboard, journal)
must reference these constants instead of hard-coding offsets.

> Keep in sync with code — update this file whenever the module API changes.
"""
from __future__ import annotations

import numpy as np

# ── Total dimensionality ──────────────────────────────────────────────
STATE_DIM = 128


class StateSegment:
    """Describes one contiguous block of the state vector."""

    def __init__(self, name: str, start: int, end: int, description: str):
        self.name = name
        self.start = start
        self.end = end          # exclusive
        self.size = end - start
        self.description = description

    def __repr__(self) -> str:
        return f"{self.name}[{self.start}:{self.end}] ({self.size}d) – {self.description}"


# ── Segment definitions ───────────────────────────────────────────────
SEGMENTS: list[StateSegment] = [
    StateSegment("screen_features", 0, 16,
                 "4×4 downsampled grayscale cell means, normalised /255"),
    StateSegment("ocr_numbers", 16, 32,
                 "First 16 numeric values parsed from OCR text, clamped to [0,1] via /100"),
    StateSegment("memory_vars", 32, 64,
                 "Game-process memory reads (budget, approval, GDP, etc.), each /100"),
    StateSegment("time_encoding", 64, 80,
                 "sin/cos of wall-clock time + reserved slots for in-game date"),
    StateSegment("action_history", 80, 96,
                 "One-hot history of last 16 actions (flattened)"),
    StateSegment("reserved", 96, 128,
                 "Zero-padded; future use (e.g. research embeddings, crisis flags)"),
]

# ── Named keys within the memory_vars segment ────────────────────────
MEMORY_KEYS: list[str] = [
    "budget_balance",
    "approval_rating",
    "gdp_growth",
    "unemployment",
    "inflation",
    "military_strength",
    "army_loyalty",
    "war_support",
    "stability_index",
]


def build_vector() -> np.ndarray:
    """Return a zeroed state vector with correct dtype/shape."""
    return np.zeros(STATE_DIM, dtype=np.float32)


def validate(vec: np.ndarray) -> None:
    """Raise if shape/dtype are wrong (cheap sanity check)."""
    assert vec.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"


def describe() -> str:
    """Human-readable layout dump for logs / dashboard."""
    lines = [f"State vector – {STATE_DIM} dims"]
    for seg in SEGMENTS:
        lines.append(f"  {seg}")
    return "\n".join(lines)
