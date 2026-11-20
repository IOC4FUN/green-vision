# research/ — outside knowledge

Lets the agent look up strategy/mechanics on the web.

## Files
- `serper_client.py` — `SerperClient`: thin wrapper over Serper.dev Google Search API (`POST https://google.serper.dev/search`, auth via `X-API-KEY`). In-memory cache (1h TTL) to conserve quota + retry-with-backoff on 429. `search(query,...)` → `SearchResponse(organic:[SearchHit], related_searches, answer_box, ...)`. Helpers: `research_game_strategy(topic, country)`, `check_for_patches()`.
  - Key from arg or `SERPER_API_KEY` env (.env, gitignored). `.available` is False without a key.

## Dev notes
- Quota-limited (~2500 baked in); always route through the cache and batch queries.
- Treat results as untrusted text; never execute anything derived from them.
