"""Keeps a training run alive: restarts `tafl.train` after an abnormal exit, and stops
when it exits cleanly (STOP file or Ctrl+C). After many crashes in an hour it waits
longer before restarting: on this shared machine other jobs can exhaust memory for
hours, and the run should resume once they let go.

    python -m tafl.supervise runs/<name> [--set key=value ...]
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import deque
from pathlib import Path

MAX_CRASHES_PER_HOUR = 6
RESTART_DELAY_S = 15
BACKOFF_DELAY_S = 900


def log(run: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [supervisor] {message}"
    print(line, flush=True)
    with open(run / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    run.mkdir(parents=True, exist_ok=True)
    extra = sys.argv[2:]
    crashes: deque[float] = deque()
    while True:
        code = subprocess.call([sys.executable, "-m", "tafl.train", str(run), *extra])
        if code == 0:
            log(run, "trainer exited cleanly")
            return
        now = time.time()
        crashes.append(now)
        while crashes and now - crashes[0] > 3600:
            crashes.popleft()
        log(run, f"trainer exited with code {code:#x} ({len(crashes)} crash(es) in the last hour)")
        delay = RESTART_DELAY_S
        if len(crashes) > MAX_CRASHES_PER_HOUR:
            log(run, f"many crashes; waiting {BACKOFF_DELAY_S // 60} minutes before restarting")
            delay = BACKOFF_DELAY_S
        deadline = time.time() + delay
        while time.time() < deadline:
            if (run / "STOP").exists():
                (run / "STOP").unlink()
                log(run, "STOP file found; not restarting")
                return
            time.sleep(5)
        extra = []  # settings given with --set are already saved in config.json


if __name__ == "__main__":
    main()
