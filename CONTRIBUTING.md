# Contributing

Bug reports and pull requests are welcome. For a wrong move, capture or result, give the position (a
FEN in the notation of [RULES.md](RULES.md), or the moves from the start) and the command or call that
shows it. Where TaflZero and this engine disagree, say which version of TaflZero you ran.

## Setting up

You need Rust 1.85 or newer and Python 3.10 or newer with PyTorch and NumPy. From the repository's
root:

```bash
cargo build --release
python -m pip install -e ".[dev]"
```

## Checking a change

One command runs every check: formatting and lints of the Rust and Python code, the Rust and Python
tests, the results in README.md and docs/RESULTS.md against the records in `matches/`, `training/`
and `models/manifest.json`, and the links between the documents. It lists whatever it could not
check (PyTorch, ruff, a GPU or the networks missing here) and fails if anything else fails:

```bash
python tools/check_all.py
```

`--full` adds the differential test of the rules against TaflZero (2,000 games). It needs a research
probe, a small program from an earlier project that links the rules library of the TaflZero build the
matches used (README.md, "The opponent"); it is not distributed with this project. Give its path with
`--probe`. `--strict` runs everything and fails on any check it would
otherwise skip.

Before a release, check the release as someone who clones it would get it, in a fresh copy outside
this checkout:

```bash
python tools/check_release.py               # a clone of HEAD
python tools/check_release.py --worktree    # the files as they are now, committed or not
```

CI (.github/workflows/ci.yml) builds and tests the Rust code on Linux, macOS and Windows with Rust
1.85, the oldest supported, and with a pinned newer release that also formats and lints; runs the
Python tests on the three systems with Python 3.10 and 3.13; and checks the results against the
records and the links. .github/constraints.txt pins the Python tools CI uses. CI does not run the
differential test, the matches or training: they need TaflZero or hours to days on a GPU. Their
records are checked, not made again.

## Style

* Rust: `cargo fmt` (rustfmt.toml) and `cargo clippy` with warnings as errors.
* Python: `ruff format` and `ruff check` (pyproject.toml: lines of up to 120 columns).
* Keep the documentation in step with the code. Every number in README.md and docs/RESULTS.md should
  come from the records, and `tools/check_results.py --check` compares them.

## What must stay in agreement

| What | Where | Checked by |
|---|---|---|
| The rules | `crates/tafl-core`, [RULES.md](RULES.md), TaflZero's own rules | the Rust tests; `tafl difftest` against the research probe |
| The C ABI | `crates/tafl-ffi/src/lib.rs` and `python/tafl/native.py` | `python/tests/test_engine.py` |
| The exported network | `python/tafl/model.py`, `python/tafl/export.py`, [docs/FORMATS.md](docs/FORMATS.md) | `python/tests/test_export.py` |
| The results | docs/RESULTS.md and README.md, `matches/`, `training/`, `models/manifest.json` | `tools/check_results.py --check` |

A change to the rules changes RULES.md with it. The records in `matches/` are copies of what the
referee wrote, with the machine's paths replaced, and those in `training/` are summaries of the
run's own files; neither is edited by hand. `tools/copy_records.py` and `tools/training_records.py`
made them.
