"""ctypes bindings to the Rust engine (crates/tafl-ffi)."""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ABI_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]


class TaflError(RuntimeError):
    pass


def _library_path() -> Path:
    override = os.environ.get("TAFL_LIB")
    if override:
        return Path(override)
    name = {"win32": "tafl_ffi.dll", "darwin": "libtafl_ffi.dylib"}.get(sys.platform, "libtafl_ffi.so")
    for profile in ("release", "debug"):
        path = ROOT / "target" / profile / name
        if path.exists():
            return path
    raise TaflError(f"{name} not found; build it with `cargo build --release -p tafl-ffi`")


_lib = ctypes.CDLL(str(_library_path()))

_P = ctypes.c_void_p
_S = ctypes.c_char_p
_N = ctypes.c_size_t
for _name, _args, _ret in [
    ("tafl_abi_version", [], ctypes.c_uint32),
    ("tafl_constants", [_P, _N], ctypes.c_int32),
    ("tafl_last_error", [_P, _N], _N),
    ("tafl_symmetry_actions", [_P], None),
    ("tafl_encode_states", [_P, _P, _N, _P], None),
    ("tafl_pool_new", [_S], _P),
    ("tafl_pool_free", [_P], None),
    ("tafl_pool_max_batch", [_P], ctypes.c_uint32),
    ("tafl_pool_collect", [_P, _P, _N], ctypes.c_int32),
    ("tafl_pool_apply", [_P, _P, _P, _N], ctypes.c_int32),
    ("tafl_pool_pending", [_P, _P, _P, _P], None),
    ("tafl_pool_take", [_P, _P, _P, _P, _P, _N], ctypes.c_int32),
    ("tafl_pool_set_game", [_P, _N, _S, _S], ctypes.c_int32),
    ("tafl_pool_play", [_P, _N, _S], ctypes.c_int32),
    ("tafl_pool_start_search", [_P, _N, ctypes.c_uint32], ctypes.c_int32),
    ("tafl_pool_stop_search", [_P, _N], ctypes.c_int32),
    ("tafl_pool_is_thinking", [_P, _N], ctypes.c_int32),
    ("tafl_pool_state", [_P, _N, _P, _N], _N),
    ("tafl_game_json", [_S, _S, _P, _N], _N),
    ("tafl_action_index", [_S], ctypes.c_int32),
]:
    _fn = getattr(_lib, _name)
    _fn.argtypes = _args
    _fn.restype = _ret

if _lib.tafl_abi_version() != ABI_VERSION:
    raise TaflError(f"engine ABI {_lib.tafl_abi_version()} does not match the bindings ({ABI_VERSION}); rebuild")

_constants = np.zeros(6, np.uint32)
_lib.tafl_constants(_constants.ctypes.data, 6)
NUM_PLANES, NUM_ACTIONS, FEATURE_BYTES, STATE_BYTES, SAMPLE_BYTES, NUM_SYMMETRIES = (int(v) for v in _constants)
BOARD = 11
PLY_PLANE = 19

SAMPLE_DTYPE = np.dtype(
    [
        ("state", np.uint8, (STATE_BYTES,)),
        ("value", "<f4"),
        ("root_q", "<f4"),
        ("game_id", "<u4"),
        ("ply", "<u2"),
        ("plies_left", "<u2"),
        ("next_action", "<u2"),
        ("policy_len", "<u2"),
        ("result_known", "u1"),
        ("pad", "u1", (11,)),
    ]
)
if SAMPLE_DTYPE.itemsize != SAMPLE_BYTES:
    raise TaflError(f"sample record is {SAMPLE_BYTES} bytes in the engine, {SAMPLE_DTYPE.itemsize} here")
NO_ACTION = 0xFFFF


def _ptr(array: np.ndarray) -> int:
    if not array.flags.c_contiguous:
        raise ValueError("array must be C-contiguous")
    return int(array.ctypes.data)


def _last_error() -> str:
    buf = ctypes.create_string_buffer(4096)
    _lib.tafl_last_error(buf, len(buf))
    return buf.value.decode()


def _check(code: int) -> int:
    if code < 0:
        raise TaflError(_last_error())
    return code


def _text(call: Any) -> str:
    """Calls `call(buf, cap)` (returning the full length), growing the buffer as needed."""
    cap = 1 << 16
    while True:
        buf = ctypes.create_string_buffer(cap)
        n = call(buf, cap)
        if n < cap:
            return buf.value.decode()
        cap = n + 1


def symmetry_actions() -> np.ndarray:
    """`[NUM_SYMMETRIES, NUM_ACTIONS]` table: the action each action becomes under each symmetry."""
    table = np.empty((NUM_SYMMETRIES, NUM_ACTIONS), np.uint16)
    _lib.tafl_symmetry_actions(_ptr(table))
    return table


def encode_states(states: np.ndarray, symmetries: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Input planes (`[n, FEATURE_BYTES]` bytes) for compact states, each under its own symmetry."""
    states = np.ascontiguousarray(states, np.uint8)
    symmetries = np.ascontiguousarray(symmetries, np.uint8)
    n = len(states)
    if states.shape != (n, STATE_BYTES) or symmetries.shape != (n,):
        raise ValueError("states must be [n, STATE_BYTES] and symmetries [n]")
    if out is None:
        out = np.empty((n, FEATURE_BYTES), np.uint8)
    elif out.shape[0] < n or out.shape[1:] != (FEATURE_BYTES,) or out.dtype != np.uint8:
        raise ValueError("bad output buffer")
    _lib.tafl_encode_states(_ptr(states), _ptr(symmetries), n, _ptr(out))
    return out


def game_info(fen: str = "startpos", moves: str = "") -> dict[str, Any]:
    """Referee: state of the game reached from `fen` by `moves` (raises on illegal moves)."""
    info = json.loads(_text(lambda buf, cap: _lib.tafl_game_json(fen.encode(), moves.encode(), buf, cap)))
    if "error" in info:
        raise TaflError(info["error"])
    return info


def action_index(move: str) -> int:
    return _check(_lib.tafl_action_index(move.encode()))


class Pool:
    """A set of concurrent games and searches driven by batched network evaluations.

    Self-play (`autoplay=True`) plays and records whole games; analysis mode
    (`autoplay=False`) searches positions set with `set_game` on request.
    """

    def __init__(self, **config: Any) -> None:
        handle = _lib.tafl_pool_new(json.dumps(config).encode())
        if not handle:
            raise TaflError(_last_error())
        self._handle: int | None = handle
        self.games = int(config.get("games", 1))
        self.max_batch = int(_lib.tafl_pool_max_batch(handle))

    def close(self) -> None:
        if self._handle:
            _lib.tafl_pool_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Pool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def handle(self) -> int:
        if not self._handle:
            raise TaflError("pool is closed")
        return self._handle

    def collect(self, planes: np.ndarray) -> int:
        """Runs selection; fills `planes[:rows]` (uint8, `[>= max_batch, FEATURE_BYTES]`)."""
        if planes.dtype != np.uint8 or planes.ndim != 2 or planes.shape[1] != FEATURE_BYTES:
            raise ValueError("planes must be uint8 [rows, FEATURE_BYTES]")
        return _check(_lib.tafl_pool_collect(self.handle, _ptr(planes), planes.shape[0]))

    def apply(self, policy: np.ndarray, value: np.ndarray, rows: int) -> None:
        """Feeds back logits (`float32 [>= rows, NUM_ACTIONS]`) and values (side to move)."""
        if policy.dtype != np.float32 or value.dtype != np.float32:
            raise ValueError("policy and value must be float32")
        if policy.shape[0] < rows or policy.shape[1] != NUM_ACTIONS or value.shape[0] < rows:
            raise ValueError("policy/value buffers are too small")
        _check(_lib.tafl_pool_apply(self.handle, _ptr(policy), _ptr(value), rows))

    def take(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Finished games: sample records, policy actions and probabilities (concatenated
        in record order, `policy_len` entries each) and one summary per game."""
        games, samples, entries = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
        _lib.tafl_pool_pending(self.handle, ctypes.byref(games), ctypes.byref(samples), ctypes.byref(entries))
        records = np.zeros(samples.value, SAMPLE_DTYPE)
        actions = np.zeros(entries.value, np.uint16)
        probs = np.zeros(entries.value, np.float32)
        cap = 4096 + 16 * 1024 * games.value
        while True:
            buf = ctypes.create_string_buffer(cap)
            n = _lib.tafl_pool_take(self.handle, _ptr(records), _ptr(actions), _ptr(probs), buf, cap)
            if n >= 0:
                break
            cap *= 4
        summaries = [json.loads(line) for line in buf.value.decode().splitlines() if line]
        return records, actions, probs, summaries

    # Analysis mode.

    def set_game(self, slot: int, fen: str = "startpos", moves: str = "") -> None:
        _check(_lib.tafl_pool_set_game(self.handle, slot, fen.encode(), moves.encode()))

    def play(self, slot: int, move: str) -> None:
        _check(_lib.tafl_pool_play(self.handle, slot, move.encode()))

    def start_search(self, slot: int, simulations: int) -> None:
        _check(_lib.tafl_pool_start_search(self.handle, slot, simulations))

    def stop_search(self, slot: int) -> None:
        _check(_lib.tafl_pool_stop_search(self.handle, slot))

    def is_thinking(self, slot: int) -> bool:
        return bool(_lib.tafl_pool_is_thinking(self.handle, slot))

    def state(self, slot: int) -> dict[str, Any]:
        info = json.loads(_text(lambda buf, cap: _lib.tafl_pool_state(self.handle, slot, buf, cap)))
        if "error" in info:
            raise TaflError(info["error"])
        return info
