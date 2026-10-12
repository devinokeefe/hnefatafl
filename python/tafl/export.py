"""Exports the network for other runtimes (the article's browser port), with reference outputs.

    python -m tafl.export --model runs/r1/checkpoints/step_0468000.pt --out net.bin \
        [--reference ref.json --games 4 --positions 64 --seed 1]

The file holds the inference network as the engine runs it: `Net.fused()`, batch norms
folded into the layers before them and every weight rounded to float16. Only what a
player needs is kept: the stem, the residual blocks, the move policy (the first 40
planes of `policy.out`; the opponent-reply planes are dropped) and the value head.

Layout (all little-endian), a 64-byte header then the weights:

    offset  type      field
    0       char[8]   magic "TAFLNET1"
    8       u32       input planes (20)
    12      u32       board side (11)
    16      u32       residual blocks B (10)
    20      u32       channels C (128)
    24      u32       squeeze-excitation ratio (4)
    28      u32       squeeze-excitation hidden units S = max(C / ratio, 8) (32)
    32      u32       head channels H (32)
    36      u32       value hidden units V (128)
    40      u32       policy planes (40)
    44      u32       value outputs (3)
    48      u32       ply plane (19)
    52      f32       ply scale (1/128): the ply plane is multiplied by it before the stem
    56      f32       moves-left scale (100)
    60      u32       N, the number of float16 weights that follow
    64      f16[N]    the tensors of `layout`, in order, each in PyTorch's row-major order

Convolution weights are `[out][in][ky][kx]` (a 1x1 convolution is `[out][in]`), linear
weights `[out][in]`; every 3x3 convolution pads the board with one row of zeros. The
file ends after the last weight. For the default configuration N = 3,166,059 and the
file is 64 + 2N = 6,332,182 bytes. docs/FORMATS.md describes the forward pass.

With `--reference`, the command also plays a few games between two copies of the
network (moves by the raw policy, mostly its best legal move) and writes, for some of
their positions, the engine's input planes and the outputs of three versions of the
network on CPU: "stored", float32 PyTorch on the file's weights; "checkpoint", the
float32 checkpoint before folding; and "fp16", the fused float16 copy the engine
evaluates with on a GPU (here run by PyTorch on the CPU).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import MOVE_PLANES, MOVES_LEFT_SCALE, PLY_SCALE, Net, NetConfig, load_model
from .native import BOARD, FEATURE_BYTES, NUM_ACTIONS, NUM_PLANES, PLY_PLANE, Pool, action_index, game_info

MAGIC = b"TAFLNET1"
HEADER = struct.Struct("<8s11I2fI")  # magic, then FIELDS
FIELDS = (
    "planes",
    "board",
    "blocks",
    "channels",
    "se_ratio",
    "se_hidden",
    "head_channels",
    "value_hidden",
    "policy_planes",
    "value_outputs",
    "ply_plane",
    "ply_scale",
    "moves_left_scale",
    "count",
)
VALUE_OUTPUTS = 3


def se_hidden(config: NetConfig) -> int:
    return max(config.channels // config.se_ratio, 8)


def layout(config: NetConfig) -> list[tuple[str, tuple[int, ...]]]:
    """Names (as in a fused `Net`'s state dict) and shapes of the stored tensors, in file order."""
    c, h, s, v = config.channels, config.head_channels, se_hidden(config), config.value_hidden
    out: list[tuple[str, tuple[int, ...]]] = [("stem.weight", (c, NUM_PLANES, 3, 3)), ("stem.bias", (c,))]
    for b in range(config.blocks):
        p = f"blocks.{b}."
        out += [
            (p + "conv1.weight", (c, c, 3, 3)),
            (p + "conv1.bias", (c,)),
            (p + "conv2.weight", (c, c, 3, 3)),
            (p + "conv2.bias", (c,)),
            (p + "se1.weight", (s, 2 * c)),
            (p + "se1.bias", (s,)),
            (p + "se2.weight", (2 * c, s)),
            (p + "se2.bias", (2 * c,)),
        ]
    out += [
        ("policy.conv_g.weight", (h, c, 1, 1)),
        ("policy.conv_g.bias", (h,)),
        ("policy.fc_g.weight", (h, 2 * h)),
        ("policy.fc_g.bias", (h,)),
        ("policy.conv_p.weight", (h, c, 1, 1)),
        ("policy.conv_p.bias", (h,)),
        ("policy.out.weight", (MOVE_PLANES, h, 1, 1)),
        ("policy.out.bias", (MOVE_PLANES,)),
        ("value_conv.weight", (h, c, 1, 1)),
        ("value_conv.bias", (h,)),
        ("value_fc.weight", (v, 2 * h)),
        ("value_fc.bias", (v,)),
        ("value_out.weight", (VALUE_OUTPUTS, v)),
        ("value_out.bias", (VALUE_OUTPUTS,)),
    ]
    return out


def export_bytes(model: Net) -> bytes:
    """The file for `model`: its `fused()` float16 weights after the header."""
    cfg = model.config
    state = model.fused(torch.float16).state_dict()
    parts = []
    for name, shape in layout(cfg):
        t = state[name]
        if name.startswith("policy.out."):
            t = t[:MOVE_PLANES]  # the move policy; the opponent-reply planes follow
        if tuple(t.shape) != shape:
            raise ValueError(f"{name} is {tuple(t.shape)}, expected {shape}")
        a = t.detach().cpu().contiguous().numpy().astype("<f2")
        if not np.isfinite(a).all():
            raise ValueError(f"{name} does not fit in float16")
        parts.append(a.tobytes())
    body = b"".join(parts)
    fields = {
        "planes": NUM_PLANES,
        "board": BOARD,
        "blocks": cfg.blocks,
        "channels": cfg.channels,
        "se_ratio": cfg.se_ratio,
        "se_hidden": se_hidden(cfg),
        "head_channels": cfg.head_channels,
        "value_hidden": cfg.value_hidden,
        "policy_planes": MOVE_PLANES,
        "value_outputs": VALUE_OUTPUTS,
        "ply_plane": PLY_PLANE,
        "ply_scale": PLY_SCALE,
        "moves_left_scale": MOVES_LEFT_SCALE,
        "count": len(body) // 2,
    }
    header = HEADER.pack(MAGIC, *(fields[f] for f in FIELDS))
    return header + body


def read_net(data: bytes) -> tuple[NetConfig, dict[str, np.ndarray]]:
    """Parses a file written by `export_bytes`: the configuration and the float16 tensors."""
    if len(data) < HEADER.size:
        raise ValueError("file too short for the header")
    magic, *values = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    h = dict(zip(FIELDS, values, strict=True))
    cfg = NetConfig(
        blocks=h["blocks"],
        channels=h["channels"],
        se_ratio=h["se_ratio"],
        head_channels=h["head_channels"],
        value_hidden=h["value_hidden"],
    )
    expected = {
        "planes": NUM_PLANES,
        "board": BOARD,
        "se_hidden": se_hidden(cfg),
        "policy_planes": MOVE_PLANES,
        "value_outputs": VALUE_OUTPUTS,
        "ply_plane": PLY_PLANE,
        "ply_scale": PLY_SCALE,
        "moves_left_scale": MOVES_LEFT_SCALE,
    }
    if any(h[k] != v for k, v in expected.items()):
        raise ValueError("the header does not match this engine's encoding")
    count = h["count"]
    if len(data) != HEADER.size + 2 * count:
        raise ValueError(f"{len(data)} bytes, expected {HEADER.size + 2 * count}")
    weights = np.frombuffer(data, "<f2", count, HEADER.size)
    tensors, off = {}, 0
    for name, shape in layout(cfg):
        n = math.prod(shape)
        tensors[name] = weights[off : off + n].reshape(shape)
        off += n
    if off != count:
        raise ValueError(f"{count} weights in the header, {off} in the layout")
    return cfg, tensors


def stored_model(data: bytes) -> Net:
    """A float32 fused network with the file's weights (the opponent-reply planes zero)."""
    cfg, tensors = read_net(data)
    net = Net(cfg).fused(torch.float32)
    state = net.state_dict()
    for name, a in tensors.items():
        t = torch.from_numpy(a.astype(np.float32))
        if name.startswith("policy.out."):
            state[name].zero_()
            state[name][:MOVE_PLANES] = t
        else:
            state[name].copy_(t)
    return net.eval()


# ----- reference outputs -----


class Encoder:
    """Input planes of game positions, from the engine's own encoder: an analysis pool
    queues the root of a one-simulation search and `collect` writes its planes."""

    def __init__(self) -> None:
        self.pool = Pool(games=1, autoplay=False, leaves_per_search=1, search={"algorithm": "puct", "tactics": False})
        self.planes = np.zeros((self.pool.max_batch, FEATURE_BYTES), np.uint8)

    def __call__(self, moves: list[str]) -> np.ndarray:
        self.pool.set_game(0, "startpos", " ".join(moves))
        self.pool.start_search(0, 1)
        if self.pool.collect(self.planes) != 1:
            raise RuntimeError("the search did not queue its root")
        return self.planes[0].copy()


def outputs(net: Net, planes: np.ndarray) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        out = net.eval()(torch.from_numpy(planes))
        return {
            "policy": out["policy"].float().numpy(),
            "value_logit": out["value_logit"].float().numpy(),
            "q": out["q"].float().numpy(),
            "moves_left": out["moves_left"].float().numpy(),
            "value": Net.search_value(out).float().numpy(),
        }


def play_games(model: Net, encode: Encoder, games: int, rng: np.random.Generator, max_plies: int) -> list[dict]:
    """Positions of games between two copies of `model`: each move is the policy's best
    legal move, or (in the first 4 plies and a fifth of the others) a sample from it."""
    positions = []
    for g in range(games):
        moves: list[str] = []
        while len(moves) < max_plies:
            info = game_info("startpos", " ".join(moves))
            if info["outcome"] is not None or not info["legal"]:
                break
            planes = encode(moves)
            actions = np.array([action_index(m) for m in info["legal"]])
            logits = outputs(model, planes[None])["policy"][0, actions].astype(np.float64)
            positions.append({"game": g, "ply": len(moves), "moves": list(moves), "planes": planes, "actions": actions})
            if len(moves) < 4 or rng.random() < 0.2:
                p = np.exp(logits - logits.max())
                k = int(rng.choice(len(actions), p=p / p.sum()))
            else:
                k = int(np.argmax(logits))
            moves.append(info["legal"][k])
    return positions


def _f(x: float) -> float:
    return float(f"{float(x):.9g}")  # nine digits identify a float32


def reference(model: Net, data: bytes, games: int, count: int, seed: int, max_plies: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    encode = Encoder()
    played = play_games(model, encode, games, rng, max_plies)
    # The start position, then a random sample of the others.
    keep = [0, *sorted(1 + rng.choice(len(played) - 1, size=min(count, len(played)) - 1, replace=False))]
    chosen = [played[i] for i in keep]
    planes = np.stack([p["planes"] for p in chosen])
    nets = {"stored": stored_model(data), "checkpoint": model.eval(), "fp16": model.fused(torch.float16)}
    results = {name: outputs(net, planes) for name, net in nets.items()}
    records = []
    for i, p in enumerate(chosen):
        rec: dict[str, Any] = {
            "game": p["game"],
            "ply": p["ply"],
            "moves": " ".join(p["moves"]),
            "planes": base64.b64encode(p["planes"].tobytes()).decode(),
            "actions": [int(a) for a in p["actions"]],
        }
        for name, out in results.items():
            logits = out["policy"][i]
            r = {
                "logits": [_f(x) for x in logits[p["actions"]]],
                "argmax": int(p["actions"][int(np.argmax(logits[p["actions"]]))]),
                **{k: _f(out[k][i]) for k in ("value_logit", "q", "moves_left", "value")},
            }
            if name == "stored" and i < 2:
                r["all_logits"] = [_f(x) for x in logits]
            rec[name] = r
        records.append(rec)
    return {
        "about": "Reference outputs of the Hnefatafl network on CPU; see python/tafl/export.py.",
        "versions": {
            "stored": "float32 PyTorch on the exported float16 weights",
            "checkpoint": "float32 PyTorch on the checkpoint (batch norms not folded)",
            "fp16": "the engine's fused float16 network (Net.fused()), run by PyTorch on the CPU",
        },
        "net_bytes": len(data),
        "net_sha256": hashlib.sha256(data).hexdigest(),
        "num_actions": NUM_ACTIONS,
        "seed": seed,
        "games": games,
        "positions_played": len(played),
        "positions": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint to export")
    parser.add_argument("--out", type=Path, required=True, help="the network file to write")
    parser.add_argument("--reference", type=Path, help="also write reference outputs (JSON) here")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for PyTorch")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, _ = load_model(args.model, "cpu")
    data = export_bytes(model)
    args.out.write_bytes(data)
    print(f"{args.out}: {len(data):,} bytes, {(len(data) - HEADER.size) // 2:,} weights")
    if args.reference:
        ref = reference(model.eval(), data, args.games, args.positions, args.seed, args.max_plies)
        args.reference.write_text(json.dumps(ref, separators=(",", ":")) + "\n")
        print(f"{args.reference}: {len(ref['positions'])} positions from {ref['positions_played']} played")


if __name__ == "__main__":
    main()
