#!/usr/bin/env python3
r"""Post-commit helper: appends a one-line entry to CHANGELOG.md under [Unreleased].
Called automatically by .git/hooks/post-commit.
Idempotent — skips if the exact line already exists.
"""
from __future__ import annotations
import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
UNRELEASED_HEADER = "## [Unreleased]"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, cwd=ROOT).strip()


def main() -> int:
    try:
        short_hash = _git("rev-parse", "--short", "HEAD")
        subject = _git("log", "-1", "--pretty=%s")
        # Skip if this commit only touched CHANGELOG.md (breaks the self-referential loop)
        changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD")
        files_changed = [f for f in changed.splitlines() if f.strip()]
        if files_changed and all(f == "CHANGELOG.md" for f in files_changed):
            return 0
    except Exception as exc:
        print(f"[post-commit-changelog] git error: {exc}", file=sys.stderr)
        return 0  # non-fatal

    today = datetime.date.today().isoformat()
    line = f"- {today} \u2014 {subject} (`{short_hash}`)"

    if not CHANGELOG.exists():
        print("[post-commit-changelog] CHANGELOG.md not found; skipping.")
        return 0

    content = CHANGELOG.read_text(encoding="utf-8")

    # Idempotency check
    if line in content:
        return 0

    # Insert after the Unreleased header (first occurrence)
    lines = content.splitlines(keepends=True)
    insert_at = None
    for i, ln in enumerate(lines):
        if ln.strip() == UNRELEASED_HEADER:
            insert_at = i + 1
            break

    if insert_at is None:
        # No [Unreleased] section — create one after the first heading block
        insert_at = len(lines)
        lines.append(UNRELEASED_HEADER + "\n")

    lines.insert(insert_at, line + "\n")
    CHANGELOG.write_text("".join(lines), encoding="utf-8")

    # Stage the changelog so it's included in the next commit (or amend)
    subprocess.run(["git", "add", str(CHANGELOG)], cwd=ROOT, capture_output=True)
    print(f"[post-commit-changelog] Added: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
