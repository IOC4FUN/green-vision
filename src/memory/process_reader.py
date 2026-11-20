"""
Game process memory reader for Green Vision.

Reads the Wine/Proton game process memory using `process_vm_readv`
syscall (via ctypes). This is the same technique used by non-stopping
memory scanners like Pika – it does NOT freeze or ptrace-attach to the
target, so the game keeps running at full speed.

Requirements:
  • /proc/<pid>/mem must be readable (same user OR CAP_SYS_PTRACE)
  • kernel.yama.ptrace_scope == 0 (classic) OR caller has CAP_SYS_PTRACE
  • The target PID is a Proton/Wine child process (Geoplanetical.exe)

What we read:
  • Numeric game state values (budget, approval, date, etc.) if offsets
    are known (stored in config/memory_offsets.yaml).
  • Fallback: scan for known string patterns in the heap to locate
    structures dynamically.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import re
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memory.process")

# ── Linux syscall numbers (x86_64) ────────────────────────────────
SYS_process_vm_readv = 310
SYS_process_vm_writev = 311


class iovec(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
    ]


@dataclass
class MemoryRegion:
    start: int
    end: int
    perms: str       # rwxp etc.
    offset: int
    inode: int
    pathname: str


@dataclass
class GameMemoryState:
    """Structured snapshot of game state read from process memory."""
    timestamp: float
    pid: int
    values: dict[str, float] = field(default_factory=dict)
    strings_found: list[str] = field(default_factory=list)
    raw_heap_sample: bytes = b""
    error: Optional[str] = None


class GameProcessReader:
    """
    Non-intrusive reader for the Geoplanetical Proton/Wine process.

    Usage:
        reader = GameProcessReader(process_name="Geoplanetical.exe")
        if reader.find_process():
            state = reader.read_state()
            print(state.values.get("budget_balance"))
    """

    def __init__(self, process_name: str = "Geoplanetical.exe"):
        self._process_name = process_name
        self._pid: Optional[int] = None
        self._libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        self._offsets_file = Path(__file__).resolve().parent.parent.parent / "config" / "memory_offsets.yaml"
        self._known_offsets: dict[str, tuple[int, str]] = {}  # name → (address, format)
        self._load_offsets()

    # ── process discovery ────────────────────────────────────────

    def find_process(self) -> bool:
        """Find the game's main Wine/Proton child PID."""
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", self._process_name],
                text=True, timeout=5
            ).strip()
            pids = [int(p) for p in out.split() if p.isdigit()]
            if not pids:
                logger.warning("No process matching '%s' found", self._process_name)
                return False
            # Pick the one with the largest RSS (the actual game, not a helper)
            best_pid, best_rss = pids[0], 0
            for pid in pids:
                rss = self._get_rss(pid)
                if rss > best_rss:
                    best_pid, best_rss = pid, rss
            self._pid = best_pid
            logger.info("Found game process: PID %d (%.1f MB RSS)", self._pid, best_rss / 1024)
            return True
        except Exception as e:
            logger.error("find_process failed: %s", e)
            return False

    @staticmethod
    def _get_rss(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    @property
    def pid(self) -> Optional[int]:
        return self._pid

    # ── memory reading via process_vm_readv ──────────────────────

    def read_memory(self, address: int, size: int) -> bytes:
        """
        Read `size` bytes from the target process at `address`
        using process_vm_readv (no ptrace attach).
        """
        if not self._pid:
            raise RuntimeError("No target PID set – call find_process() first")

        buf = ctypes.create_string_buffer(size)
        local_iov = iovec(ctypes.addressof(buf), size)
        remote_iov = iovec(address, size)

        libc = self._libc
        ret = libc.syscall(
            ctypes.c_long(SYS_process_vm_readv),
            ctypes.c_int(self._pid),
            ctypes.byref(local_iov), ctypes.c_ulong(1),
            ctypes.byref(remote_iov), ctypes.c_ulong(1),
            ctypes.c_ulong(0),
        )
        if ret < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"process_vm_readv failed at {hex(address)}: os.strerror(errno)")
        return buf.raw[:ret]

    def write_memory(self, address: int, data: bytes) -> int:
        """Write to game memory (for future use / testing)."""
        if not self._pid:
            raise RuntimeError("No target PID")
        src_buf = ctypes.create_string_buffer(data, len(data))
        local_iov = iovec(ctypes.addressof(src_buf), len(data))
        remote_iov = iovec(address, len(data))
        ret = self._libc.syscall(
            ctypes.c_long(SYS_process_vm_writev),
            ctypes.c_int(self._pid),
            ctypes.byref(local_iov), ctypes.c_ulong(1),
            ctypes.byref(remote_iov), ctypes.c_ulong(1),
            ctypes.c_ulong(0),
        )
        return ret

    # ── structured state reading ─────────────────────────────────

    def read_state(self) -> GameMemoryState:
        """
        Read known offsets + scan for interesting strings.
        Returns a GameMemoryState with whatever was successfully extracted.
        """
        ts = time.time()
        state = GameMemoryState(timestamp=ts, pid=self._pid or 0)

        if not self._pid:
            state.error = "No game process found"
            return state

        # 1. Read known numeric offsets (if configured)
        for name, (addr, fmt) in self._known_offsets.items():
            try:
                raw = self.read_memory(addr, struct.calcsize(fmt))
                val = struct.unpack(fmt, raw)[0]
                state.values[name] = val
            except Exception as e:
                logger.debug("Offset %s @ %x failed: %s", name, addr, e)

        # 2. Scan readable heap regions for short ASCII strings of interest
        try:
            state.strings_found = self._scan_for_strings(max_regions=50, max_len_per_region=64 * 1024)
        except Exception as e:
            state.error = f"String scan failed: {e}"

        return state

    def _scan_for_strings(self, max_regions: int = 50, max_len_per_region: int = 65536) -> list[str]:
        """Walk /proc/pid/maps and grep readable regions for game-relevant tokens."""
        interesting = re.compile(
            rb"(budget|approval|gdp|population|military|diplomacy|election|" +
            rb"unemployment|inflation|tax|revenue|spending|debt|support|" +
            rb"war|peace|treaty|minister|party|coalition|strike|protest)",
            re.IGNORECASE,
        )
        found: list[str] = []
        maps_path = f"/proc/{self._pid}/maps"
        if not os.path.exists(maps_path):
            return found

        with open(maps_path) as f:
            lines = f.readlines()

        count = 0
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            addr_range = parts[0]
            perms = parts[1]
            if 'r' not in perms or 'p' not in perms:
                continue
            start_hex, end_hex = addr_range.split('-')
            start_addr = int(start_hex, 16)
            region_size = min(int(end_hex, 16) - start_addr, max_len_per_region)
            if region_size <= 0 or region_size > 4 * 1024 * 1024:
                continue
            try:
                chunk = self.read_memory(start_addr, region_size)
                # Find ASCII runs of 4+ chars matching our pattern
                for m in interesting.finditer(chunk):
                    s = m.group(0).decode('ascii', errors='ignore').strip()
                    if 3 < len(s) < 80 and s not in found:
                        found.append(s)
                        if len(found) >= 200:
                            return found
            except OSError:
                continue
            count += 1
            if count >= max_regions:
                break
        return found

    # ── offset config ────────────────────────────────────────────

    def _load_offsets(self) -> None:
        """Load known memory offsets from YAML (populated by the scanner tool)."""
        import yaml
        if not self._offsets_file.exists():
            logger.info("No memory_offsets.yaml yet – run tools/scan_memory.py to populate")
            return
        try:
            with open(self._offsets_file) as f:
                data = yaml.safe_load(f) or {}
            for name, entry in data.get("offsets", {}).items():
                addr = int(entry["address"], 16) if isinstance(entry["address"], str) else int(entry["address"])
                fmt = entry.get("format", "i")
                self._known_offsets[name] = (addr, fmt)
            logger.info("Loaded %d memory offsets", len(self._known_offsets))
        except Exception as e:
            logger.warning("Failed to load memory offsets: %s", e)

# Backwards-compatible alias (agent imports this name)
ProcessReader = GameProcessReader
