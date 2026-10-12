"""Copenhagen Hnefatafl engine: Rust search core with PyTorch networks."""

import platform
import sys


def _no_wmi(*_keys: str) -> None:
    raise OSError("not supported")


if sys.platform == "win32":
    # Python 3.12's WMI helper is unsafe once it times out, which it does on a busy
    # machine: its worker thread goes on using the data of a call that has returned, and
    # closes a handle it does not own. torch asks `platform` for the machine type while
    # it is imported, so every process here was exposed. In the observed case the handle
    # was the thread pool's completion port, and the process died a minute or two later
    # with 0xC000070A, when the NVIDIA driver registered a wait. With this, `platform`
    # reads the same facts from the environment and the `ver` command.
    platform._wmi_query = _no_wmi  # type: ignore[attr-defined]
