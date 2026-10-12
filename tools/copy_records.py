"""Copies the records of the matches that docs/RESULTS.md reports into matches/, ready to publish.

    python tools/copy_records.py RUNS_DIR

RUNS_DIR is the runs/ folder of the checkout that played the matches; runs/arena/ holds their
records. For each match in MATCHES it writes, in matches/:

- <match>.jsonl, the game records, copied byte for byte. A game line that an earlier copy holds
  must be in the source unchanged and in the same place: the script refuses to change or drop
  one, so copying a match that is still being played can only add games.
- <match>.log, <match>.2.log, ...: the arena's log of each run of the match (a match continued
  with --resume has one per run), and <match>.err.log, <match>.2.err.log, ... for a run that
  wrote error output. Absolute paths are replaced: a checkpoint by its file name (models/<name>
  for the two in models/manifest.json), a file of the checkout by its path in the repository, and a
  file of the Python installation by <python>/<path>.
- <match>.json: the header of each run (the first line of its log, paths replaced as above), when
  its log was created and last written (the file times on the machine that played it), the SHA-256
  of the game records, and the summary the arena wrote at the end of its last run.

It also copies paused_controls.txt, the record of the pauses of training for two of the matches.
What it writes is checked for absolute paths and names of the machine or its user.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "matches"
# In the order of docs/RESULTS.md.
MATCHES = (
    "s1699-n400",
    "s6000-n400",
    "s10000-n400",
    "s16000-n400",
    "s18000-t1000",
    "s26000-t1000",
    "s26000-t1000-paused",
    "s18000-t1000-paused",
    "s36000-t1000-paused",
    "s36000-t15000-paused",
    "s468000-t15000-paused",
    "ladder-s468000-vs-s36000-n800",
    "ladder-s148000-vs-s36000-n800",
    "ladder-s300000-vs-s148000-n800",
    "ladder-s468000-vs-s300000-n800",
)
LISTED = ("step_0036000.pt", "step_0468000.pt")  # the checkpoints in models/manifest.json
EXTRA_FILES = ("paused_controls.txt",)  # copied byte for byte
CHECKPOINT = re.compile(r"(?:^|[\\/])(step_\d{7}\.pt)$")
# A path in a traceback: File "<path>", line ...
TRACEBACK_FILE = re.compile(r'File "([^"]+)"')
# What must not be left in a written file.
MACHINE_SPECIFIC = re.compile(r"[A-Za-z]:[\\/]|\\\\Users|/Users/|/home/|Devin|DESKTOP-", re.IGNORECASE)


class RecordError(Exception):
    """The source cannot be copied as it is."""


def checkpoint_name(path: str) -> str:
    """The name under which a checkpoint path is published."""
    match = CHECKPOINT.search(path)
    if not match:
        raise RecordError(f"not a checkpoint path: {path}")
    name = match[1]
    return f"models/{name}" if name in LISTED else name


def sanitise_header(header: dict[str, Any]) -> dict[str, Any]:
    """A run's header with the checkpoint paths replaced."""
    out = dict(header)
    for key in ("model", "opponent"):
        value = out.get(key)
        if isinstance(value, str) and value.endswith(".pt"):
            out[key] = checkpoint_name(value)
    return out


def sanitise_path(path: str, checkout: Path) -> str:
    """A path in a traceback, relative to the checkout or to the Python installation; a name that
    is not a path (such as <frozen runpy>) is left as it is."""
    if not re.match(r"[A-Za-z]:[\\/]|/", path):
        return path
    posix = path.replace("\\", "/")
    root = checkout.as_posix().rstrip("/") + "/"
    if posix.lower().startswith(root.lower()):
        return posix[len(root) :]
    if "/Lib/" in posix:  # the standard library of a Windows installation
        return "<python>/Lib/" + posix.split("/Lib/", 1)[1]
    raise RecordError(f"a path that cannot be made relative: {path}")


def read_log(path: Path) -> str:
    """A log's text: PowerShell's redirection writes UTF-16 with a byte order mark."""
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def sanitise_log(text: str, checkout: Path) -> str:
    """A log with its paths replaced: JSON lines and the closing summary are parsed and written again
    as the arena writes them, other lines have the paths of tracebacks replaced."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "{":  # the summary, written with indent=2, closes with a line "}"
            end = lines.index("}", i)
            summary = sanitise_header(json.loads("\n".join(lines[i : end + 1])))
            out += json.dumps(summary, indent=2).splitlines()
            i = end + 1
            continue
        if line.startswith("{"):
            out.append(json.dumps(sanitise_header(json.loads(line))))
        else:
            out.append(TRACEBACK_FILE.sub(lambda m: f'File "{sanitise_path(m[1], checkout)}"', line))
        i += 1
    return "".join(f"{line}\n" for line in out)


def file_time(path: Path, created: bool) -> str:
    """When a file was created or last written, in local time with its offset from UTC."""
    stat = path.stat()
    seconds = getattr(stat, "st_birthtime", stat.st_ctime) if created else stat.st_mtime
    return datetime.fromtimestamp(seconds).astimezone().isoformat(timespec="seconds")


def complete_lines(data: bytes) -> list[bytes]:
    """The lines of a record that end with a line break (a game being written has none yet)."""
    return data.splitlines(keepends=True)[: data.count(b"\n")]


def copy_games(source: Path, target: Path) -> tuple[bytes, int]:
    """The game records to write, after checking that every line of an earlier copy is unchanged.
    Returns the bytes and the number of games."""
    lines = complete_lines(source.read_bytes())
    if target.exists():
        earlier = target.read_bytes().splitlines(keepends=True)
        for n, line in enumerate(earlier, 1):
            if n > len(lines) or lines[n - 1] != line:
                raise RecordError(f"{source.name}: game line {n} differs from the earlier copy; not copied")
    data = b"".join(lines)
    for n, line in enumerate(lines, 1):
        json.loads(line)  # each line is one game
        if MACHINE_SPECIFIC.search(line.decode("utf-8")):
            raise RecordError(f"{source.name}: game line {n} holds a path or a name of the machine")
    return data, len(lines)


def logs_of(arena: Path, match: str) -> list[Path]:
    """The logs of a match's runs in order: <match>.log, <match>.log.2, ..."""
    logs = [arena / f"{match}.log"]
    n = 2
    while (arena / f"{match}.log.{n}").exists():
        logs.append(arena / f"{match}.log.{n}")
        n += 1
    if not logs[0].exists():
        raise RecordError(f"no log {logs[0].name}")
    return logs


def summary_fields(summary: dict[str, Any], header: dict[str, Any]) -> dict[str, Any]:
    """The statistics of an arena summary, which repeats the run's header before them."""
    return {key: value for key, value in summary.items() if key not in header}


def copy_match(arena: Path, match: str, checkout: Path) -> dict[str, bytes]:
    """The files of one match, by name."""
    files: dict[str, bytes] = {}
    games, count = copy_games(arena / f"{match}.jsonl", OUT / f"{match}.jsonl")
    files[f"{match}.jsonl"] = games
    runs = []
    for n, log in enumerate(logs_of(arena, match), 1):
        suffix = "" if n == 1 else f".{n}"
        text = sanitise_log(read_log(log), checkout)
        files[f"{match}{suffix}.log"] = text.encode()
        run: dict[str, Any] = {"log": f"{match}{suffix}.log"}
        err = log.with_name(log.name + ".err")
        if err.exists() and err.stat().st_size:
            files[f"{match}{suffix}.err.log"] = sanitise_log(read_log(err), checkout).encode()
            run["errors"] = f"{match}{suffix}.err.log"
        run.update(
            created=file_time(log, created=True),
            last_written=file_time(log, created=False),
            header=json.loads(text.splitlines()[0]),
        )
        runs.append(run)
    record: dict[str, Any] = {
        "match": match,
        "records": {"file": f"{match}.jsonl", "games": count, "sha256": hashlib.sha256(games).hexdigest()},
        "runs": runs,
        "summary": None,
    }
    summary_path = arena / f"{match}.summary.json"
    if summary_path.exists():
        summary = sanitise_header(json.loads(summary_path.read_text(encoding="utf-8")))
        last = runs[-1]["header"]
        if {key: summary.get(key) for key in last} != last:
            raise RecordError(f"{summary_path.name} repeats another header than the last run's")
        record["summary"] = summary_fields(summary, last)
    first = arena / f"{match}.first-run.summary.json"
    if first.exists():
        summary = sanitise_header(json.loads(first.read_text(encoding="utf-8")))
        record["first_run_summary"] = summary_fields(summary, runs[0]["header"])
    files[f"{match}.json"] = (json.dumps(record, indent=2) + "\n").encode()
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, help="the runs/ folder of the checkout that played the matches")
    args = parser.parse_args(argv)
    arena = args.runs / "arena"
    checkout = args.runs.resolve().parent
    files: dict[str, bytes] = {}
    try:
        for match in MATCHES:
            files.update(copy_match(arena, match, checkout))
        for name in EXTRA_FILES:
            files[name] = (arena / name).read_bytes()
        for name, data in files.items():
            if not name.endswith(".jsonl") and MACHINE_SPECIFIC.search(data.decode("utf-8-sig")):
                raise RecordError(f"{name} would hold a path or a name of the machine")
    except (OSError, ValueError, RecordError) as e:
        print(f"copy_records: {e}", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)
    for name, data in files.items():
        target = OUT / name
        if target.exists() and target.read_bytes() == data:
            continue
        target.write_bytes(data)
        print(f"wrote matches/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
