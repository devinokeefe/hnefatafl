"""Skips, rather than fails, a test module that needs PyTorch or NumPy where it is not installed, so that
`pytest` lists what it could not run and why (tools/check_all.py reports those skips)."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Generator, Sequence
from types import ModuleType
from typing import Any

import pytest

NEEDS = {"torch": "needs PyTorch", "numpy": "needs NumPy"}


class _SkipWithout(importlib.abc.MetaPathFinder):
    """Last on the import path, so asked only for a module that nothing else found: while a test module
    is being imported, a missing PyTorch or NumPy skips that module. Not an ImportError, which pytest
    would report as an error."""

    active = False

    def find_spec(
        self, name: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if self.active and name in NEEDS:
            raise pytest.skip.Exception(NEEDS[name], allow_module_level=True)
        return None


_finder = _SkipWithout()
sys.meta_path.append(_finder)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector) -> Generator[None, Any, Any]:
    if not isinstance(collector, pytest.Module):
        return (yield)
    _finder.active = True
    try:
        return (yield)
    finally:
        _finder.active = False
