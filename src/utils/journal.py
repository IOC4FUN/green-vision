"""
Decision journal — append-only JSONL log of every agent decision.

Each line is a self-contained record:
    {ts, step, action, confidence, state_snapshot, ocr_excerpt, reward, thought}

Useful for post-hoc debugging: "why did it declare war at step 482?"

> Keep in sync with code — update this file whenever the module API changes.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("utils.journal")


class DecisionJournal:
    """Thread-unsafe (single-threaded agent loop) JSONL writer."""

    def __init__(self, path: str | Path = "data/journal/decisions.jsonl", max_ocr_chars: int = 200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_ocr_chars = max_ocr_chars
        self._fh = open(self.path, "a", encoding="utf-8")
        self._step = 0
        logger.info("Decision journal → %s", self.path)

    def log(
        self,
        action: str,
        confidence: float,
        state_snapshot: dict[str, float],
        ocr_text: str = "",
        reward: float = 0.0,
        thought: str = "",
    ) -> None:
        self._step += 1
        record: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "step": self._step,
            "action": action,
            "confidence": round(confidence, 4),
            "state": {k: round(v, 2) for k, v in state_snapshot.items()},
            "reward": round(reward, 4),
        }
        if ocr_text:
            record["ocr"] = ocr_text[:self.max_ocr_chars]
        if thought:
            record["thought"] = thought[:300]
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    @property
    def step(self) -> int:
        return self._step

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
