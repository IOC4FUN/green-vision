#!/usr/bin/env python3
"""
LLM Swap Test – verifies the heavy→light→heavy cycle works end-to-end.
Run from project root:
    .venv/bin/python scripts/test_llm_swap.py
Exits 0 if all steps pass, 1 otherwise.
"""
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.llm_manager import LLMManager

PORT = 19091
HEALTH_URL = f"http://localhost:{PORT}/health"
PROMPT_URL = f"http://localhost:{PORT}/v1/chat/completions"

def check_health(timeout=60):
    """Poll /health until server responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(HEALTH_URL, timeout=5)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def quick_prompt(model_hint=""):
    """Send a trivial prompt and confirm we get a non-empty completion."""
    import json
    body = json.dumps({
        "model": model_hint,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(PROMPT_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        data = json.loads(r.read())
        text = data["choices"][0]["message"]["content"]
        print(f"   Response: {text.strip()!r}")
        return bool(text.strip())
    except Exception as e:
        print(f"   Prompt failed: {e}")
        return False

def main():
    mgr = LLMManager.from_config(PROJECT_ROOT / "config" / "settings.yaml")
    cfg = mgr.cfg
    print(f"Heavy model : {Path(cfg.heavy_model).name}")
    print(f"Light model : {Path(cfg.light_model).name}")
    print(f"Port        : {cfg.port}")
    print(f"llama-server: {cfg.llama_bin}")
    print()

    # ── Step 1: confirm heavy server is up ────────────────────────────────
    print("[1/5] Checking current (heavy) server…")
    if not check_health(timeout=10):
        print("   ⚠️  No server running on port. Starting light for test.")
    else:
        ok = quick_prompt()
        print(f"   Heavy server responding: {'✅' if ok else '❌'}")
    print()

    # ── Step 2: swap to light ─────────────────────────────────────────────
    print("[2/5] Swapping to light model…")
    t0 = time.time()
    swapped = mgr.swap_to_light()
    elapsed = time.time() - t0
    print(f"   swap_to_light() → {swapped} ({elapsed:.1f}s)")
    if not swapped:
        print("   ❌ Swap failed – aborting.")
        sys.exit(1)
    print()

    # ── Step 3: verify light model responds ───────────────────────────────
    print("[3/5] Verifying light model is serving…")
    healthy = check_health(timeout=cfg.health_timeout_s)
    print(f"   /health OK: {'✅' if healthy else '❌'}")
    if healthy:
        ok = quick_prompt(model_hint="qwen2.5-3b-instruct")
        print(f"   Light prompt: {'✅' if ok else '❌'}")
    print()

    # ── Step 4: restore heavy ─────────────────────────────────────────────
    print("[4/5] Restoring heavy model…")
    t0 = time.time()
    restored = mgr.restore_heavy()
    elapsed = time.time() - t0
    print(f"   restore_heavy() → {restored} ({elapsed:.1f}s)")
    if not restored:
        print("   ❌ Restore failed!")
        sys.exit(1)
    print()

    # ── Step 5: verify heavy model is back ────────────────────────────────
    print("[5/5] Verifying heavy model is back…")
    healthy = check_health(timeout=cfg.health_timeout_s)
    print(f"   /health OK: {'✅' if healthy else '❌'}")
    if healthy:
        ok = quick_prompt(model_hint="qwen-3.8-27b")
        print(f"   Heavy prompt: {'✅' if ok else '❌'}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────
    all_pass = swapped and restored and healthy
    status = "✅ ALL PASSED" if all_pass else "❌ SOME STEPS FAILED"
    print(f"{'='*50}")
    print(f"LLM SWAP TEST RESULT: {status}")
    print(f"{'='*50}")
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()