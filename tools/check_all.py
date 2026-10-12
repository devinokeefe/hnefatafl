"""Runs every check of the project and reports a summary.

Formatting, lints and tests of the Rust and Python code; the results in README.md and
docs/RESULTS.md against the match and training records (tools/check_results.py); the links
between the documents (tools/check_links.py); and, with --full, the differential test of the
rules against TaflZero's research probe, which is not distributed.

    python tools/check_all.py                    # some minutes: release builds with LTO
    python tools/check_all.py --python PATH      # run the tests with that interpreter
    python tools/check_all.py --full --probe P   # also the differential test (2,000 games)
    python tools/check_all.py --strict           # --full, and fail on everything not checked

Builds go to target-dev/, so that checks never touch the engine library that a training run or
the arena may have loaded from target/release/. The tests run with the interpreter given by
--python, else the project's .venv if there is one, else this one. What needs something missing
here (PyTorch, ruff, the probe) is skipped: whole checks, or the tests that need it. The summary
lists everything that was not checked, and the exit code is 1 if any check failed. Every check runs
even if an earlier one fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "target-dev"
EXE = ".exe" if sys.platform == "win32" else ""
LIBRARY = {"win32": "tafl_ffi.dll", "darwin": "libtafl_ffi.dylib"}.get(sys.platform, "libtafl_ffi.so")
VENV = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
RUN_TIMEOUT = 3600  # seconds for one command, far more than any takes here
DIFFTEST_GAMES = 2000
OUTPUT_LINES = 40  # of a failed check's output, in the report
NO_PROBE = "needs TaflZero's research probe, which is not distributed (--probe, or taflzero.local.json)"
NO_RUFF = "needs ruff (pip install -e .[dev])"


@dataclass
class Skip:
    """Something not checked, and why."""

    what: str
    why: str


@dataclass
class Outcome:
    """What a check found: whether it passed, its output, and what it left out."""

    ok: bool
    output: str
    skipped: list[Skip] = field(default_factory=list)


@dataclass
class Result:
    name: str
    status: str  # "ok", "failed" or "skipped"
    seconds: float
    skipped: list[Skip]


Check = Callable[[], Outcome]


def tool(name: str) -> str | None:
    """A tool on PATH, or in ~/.cargo/bin for the Rust ones; None if it is not installed."""
    return shutil.which(name) or shutil.which(name, path=str(Path.home() / ".cargo" / "bin"))


def python_with(candidates: Sequence[str], *modules: str) -> str | None:
    """The first interpreter of `candidates` that can import `modules`."""
    for python in candidates:
        imports = f"import {', '.join(modules)}"
        if Path(python).exists() and subprocess.run([python, "-c", imports], capture_output=True).returncode == 0:
            return python
    return None


def run(cmd: Sequence[str | Path], env: dict[str, str], ok_codes: Sequence[int] = (0,)) -> Outcome:
    """Runs `cmd` in the project root, stopping it after RUN_TIMEOUT seconds."""
    try:
        p = subprocess.run(
            [str(c) for c in cmd],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=RUN_TIMEOUT,
        )
    except OSError as e:
        return Outcome(False, f"cannot run {cmd[0]}: {e}")
    except subprocess.TimeoutExpired as e:
        output = "".join(x.decode(errors="replace") if isinstance(x, bytes) else x or "" for x in (e.stdout, e.stderr))
        return Outcome(False, f"{output}\nstopped after {RUN_TIMEOUT} s")
    return Outcome(p.returncode in ok_codes, p.stdout + p.stderr)


def skipped_tests(report: Path) -> list[Skip]:
    """The tests that a pytest JUnit XML report shows as skipped (a module skipped as a whole, as
    python/tests/conftest.py does without PyTorch, is one entry)."""
    skips = []
    for case in ElementTree.parse(report).iter("testcase"):
        for skipped in case.iter("skipped"):
            if skipped.get("type") == "pytest.xfail":
                continue  # an expected failure ran
            why = skipped.get("message", "")
            # A module skipped while collected reports (file, line, "Skipped: <why>").
            if found := re.search(r"Skipped: (.*)['\"]\)\s*$", skipped.text or ""):
                why = found[1]
            name = "::".join(filter(None, (case.get("classname"), case.get("name"))))
            skips.append(Skip(f"test {name}", why))
    return skips


def pytest_check(python: str, env: dict[str, str], tmp: Path) -> Outcome:
    """The Python tests, with the ones they skipped. Status 5 (no tests ran) passes only when every
    test module was skipped, which the report then lists."""
    report = tmp / "pytest.xml"
    cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={tmp / 'pytest'}"]
    outcome = run([*cmd, f"--junitxml={report}"], env, ok_codes=(0, 5))
    if report.exists():
        outcome.skipped = skipped_tests(report)
    if outcome.ok and not outcome.skipped and "no tests ran" in outcome.output:
        outcome.ok = False
    return outcome


def all_checks(
    cargo: str, tests_python: str | None, ruff_python: str | None, probe: Path | None, full: bool, tmp: Path
) -> list[tuple[str, Check | str]]:
    """Every check, as (name, check) or, for one that cannot run here, (name, why it is skipped)."""
    env = {**os.environ, "CARGO_TARGET_DIR": str(TARGET)}
    py_env = {
        **env,
        "TAFL_LIB": str(TARGET / "release" / LIBRARY),
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "python"), os.environ.get("PYTHONPATH")])),
    }
    py = sys.executable
    build = [cargo, "build", "--release", "-p", "tafl-ffi", "-p", "tafl-cli"]
    ruff = [ruff_python, "-m", "ruff"] if ruff_python else None
    checks: list[tuple[str, Check | str]] = [
        ("rustfmt", partial(run, [cargo, "fmt", "--all", "--check"], env)),
        ("clippy", partial(run, [cargo, "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"], env)),
        ("cargo test", partial(run, [cargo, "test", "--release", "--workspace"], env)),
        ("build engine library and CLI", partial(run, build, env)),
        ("ruff format", partial(run, [*ruff, "format", "--check", "python", "tools"], env) if ruff else NO_RUFF),
        ("ruff check", partial(run, [*ruff, "check", "python", "tools"], env) if ruff else NO_RUFF),
        ("pytest", partial(pytest_check, tests_python, py_env, tmp) if tests_python else "needs pytest"),
        ("results against the records", partial(run, [py, "tools/check_results.py", "--check"], env)),
        ("links between the documents", partial(run, [py, "tools/check_links.py"], env)),
    ]
    if full:
        cli = TARGET / "release" / f"tafl{EXE}"
        difftest = partial(run, [cli, "difftest", "--probe", probe, "--games", str(DIFFTEST_GAMES)], env)
        checks.append((f"rules: difftest against TaflZero ({DIFFTEST_GAMES:,} games)", difftest if probe else NO_PROBE))
    return checks


def run_check(name: str, check: Check | str, strict: bool) -> Result:
    """Runs a check, or skips it if `check` is why, and prints a line about it. With `strict`, a
    skip fails the check."""
    if isinstance(check, str):
        print(f"{'FAIL' if strict else 'SKIP'}  {name} ({check})", flush=True)
        return Result(name, "failed" if strict else "skipped", 0.0, [Skip(name, check)])
    t0 = time.monotonic()
    outcome = check()
    seconds = time.monotonic() - t0
    ok = outcome.ok and not (strict and outcome.skipped)
    skips = f", {len(outcome.skipped)} skipped" if outcome.skipped else ""
    print(f"{'ok  ' if ok else 'FAIL'}  {name}  ({seconds:.0f}s{skips})", flush=True)
    if not outcome.ok:
        tail = "\n".join(outcome.output.strip().splitlines()[-OUTPUT_LINES:])
        print("      " + tail.replace("\n", "\n      "), flush=True)
    return Result(name, "ok" if ok else "failed", seconds, outcome.skipped)


def default_probe() -> Path | None:
    """The probe that taflzero.local.json names, if there is one."""
    config = ROOT / "taflzero.local.json"
    if not config.exists():
        return None
    probe = json.loads(config.read_text(encoding="utf-8")).get("probe")
    return Path(probe) if probe else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also run the differential test against TaflZero")
    ap.add_argument("--probe", type=Path, help="TaflZero's research probe (default: from taflzero.local.json)")
    ap.add_argument("--python", help="the interpreter for the tests (default: .venv if there is one, else this one)")
    ap.add_argument("--strict", action="store_true", help="--full, and fail on everything not checked")
    args = ap.parse_args(argv)
    full = args.full or args.strict
    cargo = tool("cargo")
    if cargo is None:
        raise SystemExit("cargo not found")
    tests_python = python_with([args.python] if args.python else [str(VENV), sys.executable], "pytest")
    ruff_python = python_with([sys.executable, str(VENV)], "ruff")
    probe = args.probe or (default_probe() if full else None)
    tmp = Path(tempfile.mkdtemp(prefix="tafl_check_"))
    try:
        checks = all_checks(cargo, tests_python, ruff_python, probe, full, tmp)
        results = [run_check(name, check, args.strict) for name, check in checks]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    unchecked = [skip for result in results for skip in result.skipped]
    if not full:
        unchecked.append(Skip("the differential test of the rules against TaflZero", "run with --full"))
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    absent = [e["file"] for e in manifest["models"] if not (ROOT / "models" / e["file"]).exists()]
    if absent:
        what = f"{' and '.join(absent)} against models/manifest.json"
        unchecked.append(Skip(what, "the networks are not distributed; put a copy in models/ to check it"))
    count = {status: sum(r.status == status for r in results) for status in ("ok", "failed", "skipped")}
    if unchecked:
        print("\nNot checked:")
        for skip in unchecked:
            print(f"  {skip.what}: {skip.why}")
    print(f"\n{len(results)} checks: {count['ok']} passed, {count['failed']} failed, {count['skipped']} skipped")
    print(f"The tests ran with {tests_python}." if tests_python else "No interpreter with pytest was found.")
    failed = [r.name for r in results if r.status == "failed"]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    elif unchecked:
        print("Passed, but not everything was checked (see above).")
    else:
        print("All checks passed; nothing was skipped.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
