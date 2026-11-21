"""
LLM Manager – automatic llama-server swap during GPU-heavy training.

When Green Vision pretraining (or any CUDA workload) needs VRAM, this module:
  1. Gracefully stops the current llama-server (large model).
  2. Starts a lightweight fallback server on the same port.
  3. After training finishes, restores the original large-model server.

Usage:
    from src.utils.llm_manager import LLMManager
    mgr = LLMManager.from_config()
    if mgr.should_swap():
        mgr.swap_to_light()
    # ... run training ...
    mgr.restore_heavy()
"""
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("llm_mgr")

# ── Defaults (overridable via config/settings.yaml) ─────────────────────────
DEFAULT_PORT = 19091
DEFAULT_HEAVY_MODEL = (
    "/home/benny/.atomic-agent/models/models/qwen-3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf"
)
DEFAULT_LIGHT_MODEL = (
    "/home/benny/.atomic-agent/models/models/qwen2.5-3b-instruct/qwen2.5-3b-instruct-q4_k_m.gguf"
)
DEFAULT_LLM_BIN = (
    "/home/benny/.atomic-agent/models/core/versions/0.18.6/data/llamacpp/backends/"
    "b10298-2.1.0/linux-x64-cuda-13.3/build/bin/llama-server"
)
HEALTH_TIMEOUT_S = 60
STOP_TIMEOUT_S = 15


@dataclass
class LLMConfig:
    port: int = DEFAULT_PORT
    heavy_model: str = DEFAULT_HEAVY_MODEL
    light_model: str = DEFAULT_LIGHT_MODEL
    llama_bin: str = DEFAULT_LLM_BIN
    auto_swap: bool = True          # enable/disable the whole mechanism
    health_timeout_s: float = HEALTH_TIMEOUT_S
    stop_timeout_s: float = STOP_TIMEOUT_S

    @classmethod
    def from_yaml(cls, path: Path) -> "LLMConfig":
        """Load llm section from config/settings.yaml if present."""
        import yaml
        cfg = cls()
        try:
            data = yaml.safe_load(path.read_text()) or {}
            sec = data.get("llm", {})
            for k in ("port", "heavy_model", "light_model", "llama_bin",
                      "auto_swap", "health_timeout_s", "stop_timeout_s"):
                if k in sec and sec[k] is not None:
                    setattr(cfg, k, sec[k])
        except Exception as e:
            log.warning("Could not read llm config from %s: %s", path, e)
        return cfg


class LLMManager:
    """Manages llama-server lifecycle for GPU handoff during training."""

    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()
        self._server_proc: subprocess.Popen | None = None
        self._was_running: bool = False

    # ── Factory ──────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, settings_path: Path | None = None) -> "LLMManager":
        if settings_path is None:
            settings_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
        cfg = LLMConfig.from_yaml(settings_path)
        return cls(cfg)

    # ── Detection ────────────────────────────────────────────────────────────
    def _find_server_pid(self) -> int | None:
        """Return PID of a running llama-server on our port, or None."""
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", f"llama-server.*--port {self.cfg.port}"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            pids = out.split("\n")
            if pids and pids[0]:
                return int(pids[0])
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return None

    def server_running(self) -> bool:
        return self._find_server_pid() is not None

    def should_swap(self) -> bool:
        """True when auto-swap is enabled AND the heavy server is active."""
        if not self.cfg.auto_swap:
            return False
        return self.server_running()

    # ── Swap to light model ──────────────────────────────────────────────────
    def swap_to_light(self) -> bool:
        """Stop current server, start light fallback. Returns True on success."""
        pid = self._find_server_pid()
        if pid is None:
            log.info("No llama-server found – nothing to swap.")
            self._start_light()
            return True

        log.info("Stopping llama-server PID %d for GPU handoff…", pid)
        self._stop_server(pid)
        self._was_running = True

        ok = self._start_light()
        if ok:
            log.info("Light LLM ready on port %d (%s)", self.cfg.port, Path(self.cfg.light_model).name)
        else:
            log.error("Failed to start light LLM – attempting restore of heavy model")
            self.restore_heavy()
        return ok

    # ── Restore heavy model ──────────────────────────────────────────────────
    def restore_heavy(self) -> bool:
        """Stop light server (if we started it), restart the original heavy model."""
        # Kill our light server if we own it
        if self._server_proc and self._server_proc.poll() is None:
            log.info("Stopping light llama-server before restoring heavy model…")
            self._kill_proc(self._server_proc)
            self._server_proc = None

        # Also kill any orphan on the port
        pid = self._find_server_pid()
        if pid is not None:
            self._stop_server(pid)

        if not self._was_running:
            log.info("Heavy server was not running before swap – skipping restore.")
            return True

        ok = self._start_heavy()
        if ok:
            log.info("Heavy LLM restored on port %d", self.cfg.port)
        else:
            log.error("Failed to restore heavy LLM!")
        return ok

    # ── Context manager helper ───────────────────────────────────────────────
    def __enter__(self):
        if self.should_swap():
            self.swap_to_light()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore_heavy()
        return False  # don't suppress exceptions

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _stop_server(self, pid: int) -> None:
        """SIGTERM then SIGKILL fallback."""
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + self.cfg.stop_timeout_s
        while time.time() < deadline:
            try:
                os.kill(pid, 0)  # check alive
                time.sleep(0.3)
            except ProcessLookupError:
                log.info("llama-server PID %d stopped gracefully.", pid)
                return
        log.warning("PID %d did not exit in %.0fs – sending SIGKILL", pid, self.cfg.stop_timeout_s)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(1)

    def _kill_proc(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=self.cfg.stop_timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _start_light(self) -> bool:
        cmd = [
            self.cfg.llama_bin,
            "--no-webui",
            "-m", self.cfg.light_model,
            "--port", str(self.cfg.port),
            "-ngl", "-1",
            "--flash-attn", "auto",
            "--parallel", "1",
            "--ctx-size", "8192",
            "--device", "CUDA0",
        ]
        return self._spawn(cmd, label="light")

    def _start_heavy(self) -> bool:
        # Reconstruct the original heavy-model command (mirrors what was running)
        mmproj = Path(self.cfg.heavy_model).parent / "mmproj-F16.gguf"
        cmd = [
            self.cfg.llama_bin,
            "--no-webui", "--jinja",
            "-m", self.cfg.heavy_model,
        ]
        if mmproj.exists():
            cmd += ["--mmproj", str(mmproj)]
        cmd += [
            "--port", str(self.cfg.port),
            "-ngl", "-1",
            "--flash-attn", "auto",
            "--cache-type-k", "turbo3",
            "--cache-type-v", "turbo3",
            "--fit", "on",
            "--parallel", "1",
            "--ctx-size", "32768",
            "--device", "CUDA0",
            "--image-min-tokens", "560",
            "--image-max-tokens", "560",
            "--ubatch-size", "1024",
            "--batch-size", "2048",
        ]
        return self._spawn(cmd, label="heavy")

    def _spawn(self, cmd: list[str], label: str) -> bool:
        log.info("Starting %s llama-server: %s", label, " ".join(cmd[:6]) + " …")
        try:
            self._server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            log.error("llama-server binary not found at %s", self.cfg.llama_bin)
            return False

        # Wait for health endpoint
        import urllib.request
        url = f"http://127.0.0.1:{self.cfg.port}/health"
        deadline = time.time() + self.cfg.health_timeout_s
        while time.time() < deadline:
            if self._server_proc.poll() is not None:
                err = self._server_proc.stderr.read().decode(errors="replace")[-500:] if self._server_proc.stderr else ""
                log.error("%s server exited early (rc=%d): %s", label, self._server_proc.returncode, err)
                return False
            try:
                resp = urllib.request.urlopen(url, timeout=2)
                if resp.status == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        log.error("%s server did not become healthy within %.0fs", label, self.cfg.health_timeout_s)
        self._kill_proc(self._server_proc)
        return False
