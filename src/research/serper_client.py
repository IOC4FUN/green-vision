"""
Serper.dev Google Search API client.

Provides structured web search for the Green Vision agent to research
game strategy, mechanics, patches, and community knowledge.

API: POST https://google.serper.dev/search
Auth: X-API-KEY header
Quota: 2500 queries (baked into .env)

Usage:
    serper = SerperClient(api_key="...")
    results = serper.search("Geoplanetical best economic policy Germany")
    for r in results["organic"][:3]:
        print(r["title"], r["link"])
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger("research.serper")

SERPER_ENDPOINT = "https://google.serper.dev/search"


@dataclass
class SearchHit:
    title: str
    link: str
    snippet: str
    position: int = 0
    date: Optional[str] = None


@dataclass
class SearchResponse:
    query: str
    organic: list[SearchHit] = field(default_factory=list)
    related_searches: list[str] = field(default_factory=list)
    answer_box: Optional[dict] = None
    raw: dict = field(default_factory=dict)
    latency_ms: float = 0.0


class SerperClient:
    """
    Thin wrapper around the Serper.dev Google Search API.

    Features:
      • Automatic retry with backoff on rate-limit (429).
      • Query caching (in-memory, 1-hour TTL) to conserve quota.
      • Structured results for easy downstream processing.
    """

    def __init__(self, api_key: str | None = None):
        self._key = api_key or os.environ.get("SERPER_API_KEY", "")
        if not self._key:
            logger.warning("SERPER_API_KEY is empty – searches will fail")
        self._cache: dict[str, tuple[float, SearchResponse]] = {}
        self._cache_ttl = 3600  # seconds
        self._session = requests.Session()
        self._session.headers.update({"X-API-KEY": self._key})
        self.total_queries_used = 0

    @property
    def available(self) -> bool:
        return bool(self._key)

    def search(
        self,
        query: str,
        *,
        num_results: int = 10,
        gl: str = "us",
        hl: str = "en",
        use_cache: bool = True,
    ) -> SearchResponse:
        """
        Perform a Google search via Serper.

        Args:
            query: Natural-language search query.
            num_results: Max organic results (default 10).
            gl: Country code for geo-targeting.
            hl: Language of results.
            use_cache: Return cached result if < TTL old.
        """
        # Check cache
        if use_cache and query in self._cache:
            ts, resp = self._cache[query]
            if time.time() - ts < self._cache_ttl:
                logger.debug("Cache hit for: %s", query[:60])
                return resp

        payload = {
            "q": query,
            "num": num_results,
            "gl": gl,
            "hl": hl,
        }

        t0 = time.perf_counter()
        try:
            resp = self._session.post(SERPER_ENDPOINT, json=payload, timeout=15)
            latency = (time.perf_counter() - t0) * 1000

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", "5"))
                logger.warning("Rate limited – waiting %ds", wait)
                time.sleep(wait)
                resp = self._session.post(SERPER_ENDPOINT, json=payload, timeout=15)
                latency = (time.perf_counter() - t0) * 1000

            resp.raise_for_status()
            data = resp.json()
            self.total_queries_used += 1

        except requests.RequestException as e:
            logger.error("Serper request failed: %s", e)
            return SearchResponse(query=query, raw={}, latency_ms=(time.perf_counter()-t0)*1000)

        # Parse organic results
        organic: list[SearchHit] = []
        for item in data.get("organic", [])[:num_results]:
            organic.append(SearchHit(
                title=item.get("title", ""),
                link=item.get("link", ""),
                snippet=item.get("snippet", ""),
                position=item.get("position", 0),
                date=item.get("date"),
            ))

        related = [r.get("query", "") for r in data.get("relatedSearches", [])]

        result = SearchResponse(
            query=query,
            organic=organic,
            related_searches=related,
            answer_box=data.get("answerBox"),
            raw=data,
            latency_ms=latency,
        )

        # Cache it
        if use_cache:
            self._cache[query] = (time.time(), result)

        logger.info("Search '%s' → %d results (%.0f ms)", query[:50], len(organic), latency)
        return result

    # ── Game-specific helpers ────────────────────────────────────

    def research_game_strategy(self, topic: str, country: str | None = None) -> list[SearchHit]:
        """Targeted search for Geoplanetical strategy knowledge."""
        q = f"Geoplanetical {topic} strategy guide"
        if country:
            q += f" {country}"
        resp = self.search(q, num_results=8)
        return resp.organic

    def check_for_patches(self) -> list[SearchHit]:
        """Look for recent patch notes / updates."""
        resp = self.search("Geoplanetical patch notes update 2026", num_results=5)
        return resp.organic
