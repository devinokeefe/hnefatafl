"""Checks the release as someone who clones it would get it: builds it and runs the tests, the check of
the results against the records and the check of the links, in a fresh copy outside this checkout.

    python tools/check_release.py               # a clone of the commit checked out here (HEAD)
    python tools/check_release.py --worktree    # the files as they are here, committed or not
    python tools/check_release.py --python PATH # run the Python tests with that interpreter

The default copy is `git clone --no-local` of HEAD, so a file that is used but not committed (or
ignored by .gitignore) fails the check here rather than for the person who clones the repository.
--worktree copies instead the files that git would commit (`git ls-files -co --exclude-standard`),
to check work before it is committed. Cargo builds offline with two jobs, from the registry already
on this machine (--online and --jobs to change that), into a target folder of its own. The copy is
deleted afterwards unless --keep.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = {"win32": "tafl_ffi.dll", "darwin": "libtafl_ffi.dylib"}.get(sys.platform, "libtafl_ffi.so")
OUTPUT_LINES = 40


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def clone(dest: Path) -> str:
    """A clone of the commit checked out here; returns what it is."""
    head = git("rev-parse", "HEAD").strip()
    subprocess.run(["git", "clone", "--no-local", "--no-checkout", "--quiet", str(ROOT), str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "--detach", head], check=True)
    return f"a clone of {head[:12]} ({git('log', '-1', '--format=%s').strip()})"


def copy_worktree(dest: Path) -> str:
    """The files that git would commit here, committed or not; returns what they are."""
    names = [n for n in git("ls-files", "-co", "--exclude-standard", "-z").split("\0") if n]
    copied = 0
    for name in names:
        source = ROOT / name
        if source.is_file():  # a committed file deleted here is left out, as a commit would
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return f"the {copied} files of the working tree that git would commit"


def run(name: str, cmd: Sequence[str | Path], cwd: Path, env: dict[str, str], ok_codes: Sequence[int] = (0,)) -> bool:
    t0 = time.monotonic()
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, errors="replace")
    ok = p.returncode in ok_codes
    lines = (p.stdout + p.stderr).strip().splitlines()
    print(f"{'ok  ' if ok else 'FAIL'}  {name}  ({time.monotonic() - t0:.0f}s)", flush=True)
    shown = lines[-OUTPUT_LINES:] if not ok else lines[-3:]  # a passing check's last lines say what it found
    for line in shown:
        print(f"      {line}", flush=True)
    return ok


def remove(path: Path) -> None:
    """Deletes the copy. Git makes its object files read-only, which on Windows stops rmtree until the
    flag is cleared."""

    def writable(func, name, _exc) -> None:
        os.chmod(name, stat.S_IWRITE)
        func(name)

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=writable)
        else:
            shutil.rmtree(path, onerror=writable)
    except OSError as e:
        print(f"could not delete {path}: {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worktree", action="store_true", help="copy the working tree's files instead of cloning HEAD")
    ap.add_argument("--python", default=sys.executable, help="the interpreter for the Python tests (default: this one)")
    ap.add_argument("--jobs", type=int, default=2, help="cargo's build jobs (default: 2)")
    ap.add_argument("--online", action="store_true", help="let cargo fetch what the registry here lacks")
    ap.add_argument("--keep", action="store_true", help="keep the copy, and say where it is")
    args = ap.parse_args(argv)

    base = Path(tempfile.mkdtemp(prefix="tafl_release_"))
    dest = base / "tafl-ai"
    try:
        if args.worktree:
            dest.mkdir()
            what = copy_worktree(dest)
        else:
            what = clone(dest)
        print(f"checking {what}, in {dest}", flush=True)
        target = base / "target"
        env = {**os.environ, "CARGO_TARGET_DIR": str(target)}
        cargo = ["cargo", "--locked", *([] if args.online else ["--offline"])]
        jobs = ["-j", str(args.jobs)]
        py_env = {
            **env,
            "TAFL_LIB": str(target / "release" / LIBRARY),
            "PYTHONPATH": str(dest / "python"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        pytest = [args.python, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", f"--basetemp={base / 'pytest'}"]
        results = [
            run(
                "cargo build (engine library and CLI)",
                [*cargo, "build", "--release", *jobs, "-p", "tafl-ffi", "-p", "tafl-cli"],
                dest,
                env,
            ),
            run("cargo test", [*cargo, "test", *jobs, "--workspace"], dest, env),
            # Status 5: no tests ran, as when every module is skipped for want of PyTorch (listed above it).
            run(f"pytest ({args.python})", pytest, dest, py_env, ok_codes=(0, 5)),
            run("results against the records", [sys.executable, "tools/check_results.py", "--check"], dest, env),
            run("links between the documents", [sys.executable, "tools/check_links.py"], dest, env),
        ]
    finally:
        if args.keep:
            print(f"kept {base}")
        else:
            remove(base)
    failed = results.count(False)
    print(f"\n{len(results)} checks of the release: {len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
