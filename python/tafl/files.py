"""Crash-safe file writes."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def atomic_write(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Writes `path` through a temporary file that reaches the disk before it replaces
    the original, so that a crash (even a power cut) leaves the old file or the new
    one, never a torn or empty file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
