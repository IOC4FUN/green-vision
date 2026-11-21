#!/usr/bin/env python3
"""
Pre-commit guard for Green Vision.
Blocks any commit that would leak secrets or local-only context to the remote.
Two layers:
  1. Path filter – refuse tracked paths that must never be uploaded.
  2. Content scan – refuse staged content containing known secret patterns.
Install:  cp scripts/pre_commit_hook.py .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
Exit non-zero aborts the commit; exit 0 lets it through.
"""
from __future__ import annotations
import re
import subprocess
import sys

# Paths that must NEVER enter the repo (local-only / sensitive).
BLOCKED_PATHS = (
    r"^\.env$",
    r"^\.convolog\.md$",
    r"^data/context/.*",
    r"\.enc$",
    r"key\.key$",
    r"^data/models/.*\.pt$",
    r"^data/replays/.*",
    r"^logs/.*",
    r"^data/videos/.*",
)

# High-signal secret patterns (avoid noisy false positives on source code).
SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9]{24,}", "OpenAI-style key"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub personal token"),
    (r"github_pat_[A-Za-z0-9_]{22,}", "GitHub fine-grained token"),
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "PEM private key"),
    (r"(serper|openai|anthropic|api)[_-]?key\s*[=:]\s*['\"][A-Za-z0-9_\-]{16,}['\"]",
     "hardcoded API key assignment"),
]


def staged_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"], text=True
    )
    return [l for l in out.splitlines() if l.strip()]


def main() -> int:
    files = staged_files()
    blocked: list[str] = []
    for f in files:
        for pat in BLOCKED_PATHS:
            if re.search(pat, f):
                blocked.append(f"  {f}  (matches {pat})")
                break

    secret_hits: list[str] = []
    for f in files:
        try:
            content = subprocess.check_output(
                ["git", "show", ":" + f], text=True, stderr=subprocess.DEVNULL
            )
        except Exception:
            continue
        for pat, label in SECRET_PATTERNS:
            m = re.search(pat, content)
            if m:
                line_no = content[:m.start()].count("\n") + 1
                secret_hits.append(f"  {f}:{line_no}  ({label})")

    ok = True
    if blocked:
        ok = False
        print("[pre-commit] Blocked local-only / sensitive paths:")
        print("\n".join(blocked))
    if secret_hits:
        ok = False
        print("[pre-commit] Possible secrets detected in staged content:")
        print("\n".join(secret_hits))

    if not ok:
        print("\nCommit aborted. Remove the offending files from staging or fix the content.")
        return 1
    print("[pre-commit] OK – no sensitive paths or secrets detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
