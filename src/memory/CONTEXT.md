# memory/ — recall

Two distinct concerns: reading live game RAM, and long-term knowledge storage.

## Files
- `process_reader.py` — `GameProcessReader`: reads the Proton/Wine game process **non-intrusively** via the `process_vm_readv` syscall (ctypes), NOT ptrace attach — so the game keeps full speed. Finds PID with `pgrep -f Geoplanetical.exe`, picks largest RSS. Reads known numeric offsets from `config/memory_offsets.yaml`; falls back to scanning readable heap regions for strategy-relevant ASCII strings (budget/approval/war/etc.). API: `find_process()`, `read_state()` → `GameMemoryState(values, strings_found, ...)`. Needs `/proc/<pid>/mem` readable (same user or CAP_SYS_PTRACE; yama.ptrace_scope=0).
- `knowledge_db.py` — `KnowledgeDatabase`: three-tier store, thread-safe.
  1. Episodic (SQLite): every decision/outcome (`record_episode`).
  2. Semantic (ChromaDB, optional if installed): vector search over strategy snippets.
  3. Procedural (SQLite): learned strategies w/ success rates (`save_strategy`, `get_best_strategy(trigger)`).
  - SQLite path + Chroma dir come from config. WAL journal mode on.

## Dev notes
- Offsets in `memory_offsets.yaml` are per-build and will break on game updates — treat as best-effort, always keep the string-scan fallback working.
- Keep DB writes inside `self._lock`; it's shared across threads.
