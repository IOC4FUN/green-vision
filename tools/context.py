"""
Encrypted project-context manager for Green Vision.

Subcommands:
  init          Generate (or reuse) the Fernet key at ~/.config/green-vision/key.key
  write <text>  Encrypt & store the master context (overwrites previous)
  read          Decrypt & print the master context
  list          List all known context files (master + per-module)
  module <name> <text>   Write an encrypted per-module context
  module-read <name>     Read an encrypted per-module context

The key is stored with chmod 600 under $HOME/.config/green-vision/.
Only the local user (benny) can read it – if the repo is pushed or shared,
the .enc blobs are useless without the key.
"""
from __future__ import annotations
import argparse
import os
import stat
import sys
from pathlib import Path

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:
    print("ERROR: 'cryptography' package not installed. Run: pip install cryptography", file=sys.stderr)
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────
KEY_DIR = Path.home() / ".config" / "green-vision"
KEY_PATH = KEY_DIR / "key.key"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTEXT_DIR = PROJECT_ROOT / "data" / "context"
MASTER_ENC = CONTEXT_DIR / "master.enc"
MODULES_DIR = CONTEXT_DIR / "modules"


def _ensure_key_dir() -> None:
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    KEY_DIR.chmod(stat.S_IRWXU)  # 700


def get_or_create_key() -> bytes:
    """Return the Fernet key, creating it on first run."""
    if KEY_PATH.exists():
        return KEY_PATH.read_bytes()
    _ensure_key_dir()
    key = Fernet.generate_key()
    KEY_PATH.write_bytes(key)
    KEY_PATH.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
    print(f"[+] Key created at {KEY_PATH} (chmod 600)")
    return key


def fernet() -> Fernet:
    return Fernet(get_or_create_key())


def _text(value: str) -> str:
    """Allow '-' to read plaintext from stdin (handy for long content)."""
    if value == "-":
        return sys.stdin.read()
    return value


def cmd_init(_args) -> None:
    key = get_or_create_key()
    print(f"[✓] Key ready: {KEY_PATH}")
    print(f"    Fingerprint: {key[:12].decode()}...")


def cmd_write(args) -> None:
    text = _text(args.text)
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    token = fernet().encrypt(text.encode())
    MASTER_ENC.write_bytes(token)
    MASTER_ENC.chmod(0o600)
    print(f"[✓] Master context written ({len(text)} chars → {MASTER_ENC})")


def cmd_read(_args) -> None:
    if not MASTER_ENC.exists():
        print("No master context yet. Use 'context.py write <text>' first.", file=sys.stderr)
        sys.exit(1)
    try:
        plaintext = fernet().decrypt(MASTER_ENC.read_bytes()).decode()
    except InvalidToken:
        print("ERROR: Decryption failed – wrong or missing key.", file=sys.stderr)
        sys.exit(1)
    print(plaintext)


def _module_path(name: str) -> Path:
    safe = name.replace("/", "_").replace("..", "")
    return MODULES_DIR / f"{safe}.enc"


def cmd_module(args) -> None:
    text = _text(args.text)
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    path = _module_path(args.name)
    token = fernet().encrypt(text.encode())
    path.write_bytes(token)
    path.chmod(0o600)
    print(f"[✓] Module '{args.name}' context written ({path})")


def cmd_module_read(args) -> None:
    path = _module_path(args.name)
    if not path.exists():
        print(f"No context for module '{args.name}'.", file=sys.stderr)
        sys.exit(1)
    try:
        plaintext = fernet().decrypt(path.read_bytes()).decode()
    except InvalidToken:
        print("ERROR: Decryption failed.", file=sys.stderr)
        sys.exit(1)
    print(plaintext)


def cmd_list(_args) -> None:
    entries = []
    if MASTER_ENC.exists():
        entries.append(("master", MASTER_ENC))
    if MODULES_DIR.is_dir():
        for p in sorted(MODULES_DIR.glob("*.enc")):
            entries.append((p.stem, p))
    if not entries:
        print("(no context files yet)")
        return
    for name, path in entries:
        size = path.stat().st_size
        mtime = __import__('datetime').datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        print(f"  {name:<20s}  {size:>6d} B  {mtime}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypted project-context manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("init", help="Create/reuse the encryption key")

    p_write = sub.add_parser("write", help="Write master context (encrypted)")
    p_write.add_argument("text", help="Plaintext to store")

    sub.add_parser("read", help="Read & decrypt master context")

    p_mod = sub.add_parser("module", help="Write per-module context")
    p_mod.add_argument("name", help="Module identifier e.g. 'brain' or 'perception'")
    p_mod.add_argument("text", help="Plaintext to store")

    p_mread = sub.add_parser("module-read", help="Read a per-module context")
    p_mread.add_argument("name", help="Module identifier")

    sub.add_parser("list", help="List all context files")

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "init": cmd_init,
        "write": cmd_write,
        "read": cmd_read,
        "module": cmd_module,
        "module-read": cmd_module_read,
        "list": cmd_list,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()