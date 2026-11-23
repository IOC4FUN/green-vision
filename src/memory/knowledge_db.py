"""
Evolving knowledge database for Green Vision.

Three-tier storage:
  1. **Episodic** (SQLite): timestamped records of every decision,
     observation, and outcome. Used for replay and experience learning.
  2. **Semantic** (ChromaDB vector store): embeddings of game-knowledge
     snippets (strategy tips, policy effects, historical outcomes).
     Enables similarity search: "what worked when inflation was high?"
  3. **Procedural** (SQLite): learned strategies / playbooks with
     success rates, updated after each episode.

The database grows over time – new entries are added after every
decision cycle, and stale/low-value entries decay.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("memory.knowledge")

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


@dataclass
class EpisodeRecord:
    id: str
    timestamp: float
    action_taken: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    reward: float
    thought: str = ""
    scenario: str = ""


@dataclass
class Strategy:
    name: str
    description: str
    trigger_conditions: dict[str, Any]
    actions: list[str]
    times_used: int = 0
    times_successful: int = 0
    last_updated: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.times_successful / max(1, self.times_used)


class KnowledgeDatabase:
    """
    Unified knowledge store. Thread-safe.

    Usage:
        kb = KnowledgeDatabase(db_path="data/knowledge/green_vision.db",
                               vector_dir="data/knowledge/vectors/")
        kb.record_episode(action="raised tax", before={...}, after={...}, reward=0.5)
        similar = kb.search_knowledge("how to reduce inflation", k=3)
        best_strat = kb.get_best_strategy(trigger={"inflation": ">5%"})
    """

    def __init__(self, db_path: str | Path, vector_dir: str | Path):
        self._db_path = Path(db_path)
        self._vector_dir = Path(vector_dir)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._chroma_collection = None

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_dir.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()
        if HAS_CHROMA:
            self._init_chroma()

    # ── SQLite (episodic + procedural) ───────────────────────────

    def _init_sqlite(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                action_taken TEXT NOT NULL,
                state_before TEXT DEFAULT '{}',
                state_after TEXT DEFAULT '{}',
                reward REAL DEFAULT 0.0,
                thought TEXT DEFAULT '',
                scenario TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp);

            CREATE TABLE IF NOT EXISTS strategies (
                name TEXT PRIMARY KEY,
                description TEXT DEFAULT '',
                trigger_conditions TEXT DEFAULT '{}',
                actions TEXT DEFAULT '[]',
                times_used INTEGER DEFAULT 0,
                times_successful INTEGER DEFAULT 0,
                last_updated REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS knowledge_snippets (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT DEFAULT '',
                relevance_score REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                accessed_count INTEGER DEFAULT 0
            );
        """)
        self._conn.commit()

    def _init_chroma(self) -> None:
        try:
            client = chromadb.PersistentClient(path=str(self._vector_dir))
            # Use default embedding (all-MiniLM-L6-v2 via sentence-transformers)
            self._chroma_collection = client.get_or_create_collection(
                name="green_vision_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB collection ready (%d docs)", self._chroma_collection.count())
        except Exception as e:
            logger.warning("ChromaDB init failed (will use SQLite only): %s", e)
            self._chroma_collection = None

    # ── Episodic memory ──────────────────────────────────────────

    def record_episode(
        self, action: str, before: dict, after: dict,
        reward: float, thought: str = "", scenario: str = ""
    ) -> str:
        ep_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?)",
                (ep_id, time.time(), action, json.dumps(before), json.dumps(after),
                 reward, thought, scenario)
            )
            self._conn.commit()
        return ep_id

    def get_recent_episodes(self, n: int = 50) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (n,)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def total_episodes(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    # ── Procedural memory (strategies) ───────────────────────────

    def save_strategy(self, strat: Strategy) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO strategies
                   (name, description, trigger_conditions, actions, times_used,
                    times_successful, last_updated)
                   VALUES (?,?,?,?,?,?,?)""",
                (strat.name, strat.description, json.dumps(strat.trigger_conditions),
                 json.dumps(strat.actions), strat.times_used, strat.times_successful,
                 time.time())
            )
            self._conn.commit()

    def get_best_strategy(self, trigger: dict[str, Any]) -> Optional[Strategy]:
        """Return the highest-success-rate strategy whose triggers match."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM strategies").fetchall()
        best: Optional[Strategy] = None
        best_rate = -1.0
        for row in rows:
            s = Strategy(
                name=row[0], description=row[1],
                trigger_conditions=json.loads(row[2]),
                actions=json.loads(row[3]),
                times_used=row[4], times_successful=row[5], last_updated=row[6]
            )
            if self._triggers_match(s.trigger_conditions, trigger):
                rate = s.success_rate
                if rate > best_rate:
                    best, best_rate = s, rate
        return best

    @staticmethod
    def _triggers_match(conditions: dict, state: dict) -> bool:
        """Key-value trigger matching with support for >, <, == operators."""
        for k, v in conditions.items():
            if k not in state:
                return False
            sv = str(state[k])
            cv = str(v)
            try:
                if cv.startswith(">"):
                    if float(sv.replace('%', '')) <= float(cv[1:].replace('%', '')):
                        return False
                elif cv.startswith("<"):
                    if float(sv.replace('%', '')) >= float(cv[1:].replace('%', '')):
                        return False
                elif cv.startswith("=="):
                    if sv.strip() != cv[2:].strip():
                        return False
                else:
                    if sv.lower().strip() != cv.lower().strip():
                        return False
            except ValueError:
                # Non-numeric value – fall back to string comparison
                if sv.lower().strip() != cv.lower().strip():
                    return False
        return True

    # ── Semantic knowledge (vector search) ───────────────────────

    def add_knowledge(self, content: str, source: str = "research") -> None:
        """Add a knowledge snippet to both SQLite and ChromaDB."""
        kid = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO knowledge_snippets VALUES (?,?,?,?,?,?)",
                (kid, content, source, 1.0, time.time(), 0)
            )
            self._conn.commit()
        if self._chroma_collection:
            try:
                self._chroma_collection.add(
                    documents=[content],
                    metadatas=[{"id": kid, "source": source}],
                    ids=[kid],
                )
            except Exception as e:
                logger.debug("ChromaDB add failed: %s", e)

    def search_knowledge(self, query: str, k: int = 5) -> list[dict]:
        """Semantic similarity search over stored knowledge."""
        results: list[dict] = []
        if self._chroma_collection and self._chroma_collection.count() > 0:
            try:
                res = self._chroma_collection.query(query_texts=[query], n_results=k)
                for i, doc in enumerate(res.get("documents", [[]])[0]):
                    meta = res.get("metadatas", [{}])[0][i] if res.get("metadatas") else {}
                    dist = res.get("distances", [[0]]*k)[0][i] if res.get("distances") else 0.0
                    results.append({"content": doc, "score": 1 - dist, **meta})
            except Exception as e:
                logger.debug("ChromaDB query failed: %s", e)
        # Fallback: keyword match in SQLite
        if not results:
            with self._lock:
                cur = self._conn.execute(
                    "SELECT content, source, relevance_score FROM knowledge_snippets "
                    "WHERE content LIKE ? LIMIT ?",
                    (f"%{query.split()[0]}%", k)
                )
                results = [{"content": r[0], "source": r[1], "score": r[2]} for r in cur.fetchall()]
        return results

    def total_knowledge(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM knowledge_snippets").fetchone()[0]

# Backwards-compatible alias (agent imports this name)
KnowledgeDB = KnowledgeDatabase
