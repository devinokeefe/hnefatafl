"""Checks that every relative link in the project's Markdown files leads to a file or folder that exists.

    python tools/check_links.py

Links to the web (http, https, mailto) and to places in the same document (#...) are not followed.
Prints each broken link and exits with status 1 if there is one.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "node_modules", "runs", "build", "dist", "scratch"}
FENCE = re.compile(r"^(```|~~~).*?^\1", re.MULTILINE | re.DOTALL)
CODE = re.compile(r"`[^`\n]*`")
INLINE = re.compile(r"!?\[[^\]\n]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
DEFINITION = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*<?(\S+?)>?(?:\s+.*)?$", re.MULTILINE)
EXTERNAL = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)  # http:, https:, mailto: ...


def markdown_files() -> Iterator[Path]:
    for path in sorted(ROOT.rglob("*.md")):
        parts = path.relative_to(ROOT).parts[:-1]
        if not any(part in SKIP_DIRS or part.startswith("target") for part in parts):
            yield path


def links(text: str) -> Iterator[str]:
    """The link targets of a Markdown text, outside code."""
    text = CODE.sub("", FENCE.sub("", text))
    for pattern in (INLINE, DEFINITION):
        for found in pattern.finditer(text):
            yield found[1]


def broken(path: Path) -> list[str]:
    out = []
    for target in links(path.read_text(encoding="utf-8")):
        if EXTERNAL.match(target) or target.startswith("#"):
            continue
        file = unquote(target.split("#", 1)[0].split("?", 1)[0])
        resolved = ROOT / file.lstrip("/") if file.startswith("/") else path.parent / file
        if not resolved.exists():
            out.append(target)
    return out


def main() -> int:
    status = 0
    count = 0
    for path in markdown_files():
        count += 1
        for target in broken(path):
            print(f"broken link in {path.relative_to(ROOT).as_posix()}: {target}")
            status = 1
    if not status:
        print(f"every relative link in {count} Markdown files leads to a file that exists")
    return status


if __name__ == "__main__":
    sys.exit(main())
