#!/usr/bin/env python3
"""Auto-update CHANGELOG.md after a commit.

Usage: update_changelog.py <hash> <date> <subject>
Inserts a dated entry under the '## [Unreleased]' heading. Seeds the file if it
is missing. Never commits itself (the post-commit hook stages the result).
"""
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: update_changelog.py <hash> <date> <subject>", file=sys.stderr)
        return 2
    hash_, date, subject = sys.argv[1], sys.argv[2], sys.argv[3]
    path = "CHANGELOG.md"

    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        text = (
            "# Changelog\n\n"
            "All notable changes to Green Vision are documented here.\n"
            "Format follows Keep a Changelog.\n\n"
            "## [Unreleased]\n"
        )

    entry = f"- {date} — {subject} (`{hash_}`)\n"

    # Avoid exact duplicate of this commit already present.
    if f"`{hash_}`" in text:
        return 0

    heading = "## [Unreleased]"
    lines = text.splitlines(keepends=True)
    out = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if not inserted and ln.strip() == heading:
            out.append("\n")
            out.append(entry)
            inserted = True

    if not inserted:
        # No Unreleased section: insert one right after the first H1 line.
        new_lines = []
        placed = False
        for ln in lines:
            new_lines.append(ln)
            if not placed and ln.startswith("# "):
                new_lines.append("\n")
                new_lines.append(heading + "\n\n")
                new_lines.append(entry)
                placed = True
        text = "".join(new_lines)
    else:
        text = "".join(out)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"changelog updated: {subject} ({hash_})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
