"""
Game process memory reader for Green Vision – Windows edition.
Uses kernel32.ReadProcessMemory via ctypes to read the game process
without stopping or attaching (no debugger). Works as a normal user
on processes you own (same-user rule).

What we read:
  • Numeric game state values (budget, approval, date, etc.) if offsets
    are known (stored in config/memory_offsets.yaml).
  • Fallback: scan readable regions for known string patterns.
"""
from __future__ import annotations
import ctypes
import ctypes.wintypes as w
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

# ── Win32 constants ────────────────────────────────────────────────
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
MEM_COMMIT = 0x1000
PAGE_READABLE = {0x02, 0x04, 0x08, 0x20, 0x40}  # READ, WRITE, EXECUTE, GUARD, NOCACHE
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", w.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", w.DWORD),
        ("Protect", w.DWORD),
        ("Type", w.DWORD),
    ]


@dataclass
class MemoryRegion:
    start: int
    end: int
    perms: str
    size: int


@dataclass
class GameMemoryState:
    """Structured snapshot of game state read from process memory."""
    timestamp: float
    pid: int
    values: dict[str, float] = field(default_factory=dict)
    strings_found: list[str] = field(default_factory=list)
    raw_heap_sample: bytes = b""
    error: Optional[str] = None


def _find_game_pid(process_name: str = "Geoplanetical.exe") -> Optional[int]:
    """Find the game PID via tasklist (Windows-native)."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {process_name}", "/FO", "CSV", "/NH"],
            text=True, timeout=5,
        ).strip()
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split(',')]
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    except Exception:
        pass
    return None


class GameProcessReader:
    """
    Non-intrusive reader for the Geoplanetical process on Windows.
    Uses ReadProcessMemory – no debugger attach, game keeps running at full speed.

    Usage:
        reader = GameProcessReader(process_name="Geoplanetical.exe")
        if reader.find_process():
            state = reader.read_state()
            print(state.values.get("budget_balance"))
    """

    def __init__(self, process_name: str = "Geoplanetical.exe"):
        self._process_name = process_name
        self._pid: Optional[int] = None
        self._handle: Optional[int] = None
        self._offsets_file = (
            Path(__file__).resolve().parent.parent.parent / "config" / "memory_offsets.yaml"
        )
        self._known_offsets: dict[str, tuple[int, str]] = {}  # name → (address, format)
        self._load_offsets()

    # ── process discovery ────────────────────────────────────────
    def find_process(self) -> bool:
        """Find and open a handle to the game process."""
        pid = _find_game_pid(self._process_name)
        if not pid:
            logger.warning("No process matching '%s' found", self._process_name)
            return False
        self._pid = pid
        # Open with read access
        h = kernel32.OpenProcess(
            PROCESS_VM_READ | PROCESS_QUERY_INFORMATION,
            False,
            pid,
        )
        if not h:
            err = ctypes.get_last_error()
            logger.error("OpenProcess failed for PID %d (error %d)", pid, err)
            return False
        self._handle = h
        logger.info("Attached to game process: PID %d (handle=%x)", pid, h)
        return True

    @property
    def pid(self) -> Optional[int]:
        return self._pid

    @property
    def is_attached(self) -> bool:
        return self._handle is not None

    def close(self) -> None:
        if self._handle:
            kernel32.CloseHandle(self._handle)
            self._handle = None
            logger.info("Detached from game process")

    # ── memory reading via ReadProcessMemory ─────────────────────
    def read_memory(self, address: int, size: int) -> bytes:
        """Read `size` bytes from the target process at `address`."""
        if not self._handle:
            raise RuntimeError("No handle – call find_process() first")
        buf = ctypes.create_string_buffer(size)
        bytes_read = w.SIZE_T(0)
        ok = kernel32.ReadProcessMemory(
            self._handle,
            ctypes.c_void_p(address),
            buf,
            size,
            ctypes.byref(bytes_read),
        )
        if not ok or bytes_read.value == 0:
            err = ctypes.get_last_error()
            raise OSError(err, f"ReadProcessMemory failed at {hex(address)} (error {err})")
        return buf.raw[:bytes_read.value]

    def write_memory(self, address: int, data: bytes) -> int:
        """Write to game memory (for future use / testing)."""
        if not self._handle:
            raise RuntimeError("No handle")
        src_buf = ctypes.create_string_buffer(data, len(data))
        bytes_written = w.SIZE_T(0)
        ok = kernel32.WriteProcessMemory(
            self._handle,
            ctypes.c_void_p(address),
            src_buf,
            len(data),
            ctypes.byref(bytes_written),
        )
        if not ok:
            raise OSError(ctypes.get_last_error(), "WriteProcessMemory failed")
        return bytes_written.value

    # ── region enumeration via VirtualQueryEx ────────────────────
    def _enumerate_regions(self) -> list[MemoryRegion]:
        """Walk the process's virtual address space."""
        regions: list[MemoryRegion] = []
        mbi = MEMORY_BASIC_INFORMATION()
        addr = 0
        max_addr = 0x7FFFFFFFFFFF  # user-space limit on x64
        while addr < max_addr:
            ret = psapi.VirtualQueryEx(
                self._handle,
                ctypes.c_void_p(addr),
                ctypes.byref(mbi),
                ctypes.sizeof(mbi),
            )
            if ret == 0:
                break
            if mbi.State == MEM_COMMIT and (mbi.Protect & 0xFF) in PAGE_READABLE:
                regions.append(MemoryRegion(
                    start=mbi.BaseAddress or 0,
                    end=(mbi.BaseAddress or 0) + mbi.RegionSize,
                    perms=f"{oct(mbi.Protect)}",
                    size=mbi.RegionSize,
                ))
            next_addr = (mbi.BaseAddress or 0) + mbi.RegionSize
            if next_addr <= addr:
                break
            addr = next_addr
        return regions

    # ── structured state reading ─────────────────────────────────
    def read_state(self) -> GameMemoryState:
        """
        Read known offsets + scan for interesting strings.
        Returns a GameMemoryState with whatever was successfully extracted.
        """
        ts = time.time()
        state = GameMemoryState(timestamp=ts, pid=self._pid or 0)
        if not self._handle:
            state.error = "No game process attached"
            return state

        # 1. Read known numeric offsets (if configured)
        for name, (addr, fmt) in self._known_offsets.items():
            try:
                raw = self.read_memory(addr, struct.calcsize(fmt))
                val = struct.unpack(fmt, raw)[0]
                state.values[name] = val
            except Exception as e:
                logger.debug("Offset %s @ %x failed: %s", name, addr, e)

        # 2. Scan readable committed regions for short ASCII strings of interest
        try:
            state.strings_found = self._scan_for_strings(max_regions=80, max_len_per_region=64 * 1024)
        except Exception as e:
            state.error = f"String scan failed: {e}"

        return state

    def _scan_for_strings(
        self,
        max_regions: int = 80,
        max_len_per_region: int = 65536,
    ) -> list[str]:
        """Walk readable regions and grep for game-relevant tokens."""
        interesting = re.compile(
            rb"(budget|approval|gdp|population|military|diplomacy|election|"
            rb"unemployment|inflation|tax|revenue|spending|debt|support|"
            rb"war|peace|treaty|minister|party|coalition|strike|protest)",
            re.IGNORECASE,
        )
        found: list[str] = []
        if not self._handle:
            return found

        regions = self._enumerate_regions()
        count = 0
        for region in regions[:max_regions]:
            if region.size > max_len_per_region:
                continue
            try:
                data = self.read_memory(region.start, min(region.size, max_len_per_region))
            except Exception:
                continue
            # Look for printable ASCII runs of 4+ chars that match our pattern
            for m in re.finditer(rb"[\x20-\x7e]{4,}", data):
                token = m.group()
                if interesting.search(token):
                    decoded = token.decode("ascii", errors="ignore")
                    if decoded not in found and len(found) < 500:
                        found.append(decoded)
            count += 1
            if count >= max_regions:
                break
        return found

    # ── offset config loading ────────────────────────────────────
    def _load_offsets(self) -> None:
        """Load known memory offsets from YAML (if the file exists)."""
        import yaml
        if not self._offsets_file.exists():
            logger.debug("No memory_offsets.yaml – skipping offset load")
            return
        try:
            with open(self._offsets_file, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            for name, entry in cfg.items():
                if isinstance(entry, dict) and "address" in entry and "format" in entry:
                    addr = int(str(entry["address"]), 0)  # hex or decimal
                    fmt = entry["format"]  # e.g. "f", "i", "d"
                    self._known_offsets[name] = (addr, fmt)
            logger.info("Loaded %d memory offsets", len(self._known_offsets))
        except Exception as e:
            logger.warning("Failed to load memory offsets: %s", e)
