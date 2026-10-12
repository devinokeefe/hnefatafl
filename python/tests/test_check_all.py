"""tools/check_all.py: how it reads the tests that pytest skipped, and what it leaves out."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def check_all() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_all", ROOT / "tools" / "check_all.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_all"] = module  # its dataclasses look themselves up there
    spec.loader.exec_module(module)
    return module


def test_skipped_tests_reads_modules_and_tests(check_all: ModuleType, tmp_path: Path) -> None:
    report = tmp_path / "pytest.xml"
    report.write_text(
        '<testsuites><testsuite name="pytest">'
        '<testcase classname="" name="python.tests.test_train">'
        "<skipped message=\"collection skipped\">('python/tests/conftest.py', 30, 'Skipped: needs PyTorch')"
        "</skipped></testcase>"
        '<testcase classname="python.tests.test_engine" name="test_fast">'
        '<skipped type="pytest.skip" message="needs a GPU">needs a GPU</skipped></testcase>'
        '<testcase classname="python.tests.test_engine" name="test_known_bug">'
        '<skipped type="pytest.xfail" message="known"/></testcase>'
        '<testcase classname="python.tests.test_engine" name="test_ok"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    assert check_all.skipped_tests(report) == [
        check_all.Skip("test python.tests.test_train", "needs PyTorch"),
        check_all.Skip("test python.tests.test_engine::test_fast", "needs a GPU"),
    ]


def test_the_difftest_needs_a_probe(check_all: ModuleType, tmp_path: Path) -> None:
    checks = dict(check_all.all_checks("cargo", sys.executable, sys.executable, None, True, tmp_path))
    difftest = next(check for name, check in checks.items() if name.startswith("rules: difftest"))
    assert difftest == check_all.NO_PROBE
    without = check_all.all_checks("cargo", None, None, None, False, tmp_path)
    assert not any(name.startswith("rules:") for name, _ in without)


def test_a_skipped_check_fails_only_when_strict(check_all: ModuleType) -> None:
    assert check_all.run_check("x", "needs y", strict=False).status == "skipped"
    assert check_all.run_check("x", "needs y", strict=True).status == "failed"
    skipping = lambda: check_all.Outcome(True, "", [check_all.Skip("test t", "needs PyTorch")])  # noqa: E731
    assert check_all.run_check("pytest", skipping, strict=False).status == "ok"
    assert check_all.run_check("pytest", skipping, strict=True).status == "failed"
