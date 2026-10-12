"""Policy-value network: a residual tower with squeeze-excitation and pooled heads.

Inputs are the engine's byte planes (`[batch, FEATURE_BYTES]`); outputs are logits for
the 4840 actions (40 move planes over the board, action = plane * 121 + square), the
same for the opponent's reply (an auxiliary target), and value heads for the side to
move: the game result (as a win logit), the search value, and the plies left.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .native import BOARD, NUM_ACTIONS, NUM_PLANES, PLY_PLANE

MOVE_PLANES = NUM_ACTIONS // (BOARD * BOARD)
PLY_SCALE = 1 / 128
MOVES_LEFT_SCALE = 100.0


@dataclass(frozen=True)
class NetConfig:
    blocks: int = 10
    channels: int = 128
    se_ratio: int = 4
    head_channels: int = 32
    value_hidden: int = 128

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _conv(cin: int, cout: int, k: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)


def _pool(x: torch.Tensor) -> torch.Tensor:
    """Mean and max over the board, concatenated: `[b, c, h, w] -> [b, 2c]`."""
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)


class ResBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int) -> None:
        super().__init__()
        self.conv1 = _conv(channels, channels, 3)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _conv(channels, channels, 3)
        self.bn2 = nn.BatchNorm2d(channels)
        hidden = max(channels // se_ratio, 8)
        self.se1 = nn.Linear(2 * channels, hidden)
        self.se2 = nn.Linear(hidden, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        gate, bias = self.se2(F.relu(self.se1(_pool(y)))).chunk(2, 1)
        y = y * torch.sigmoid(gate)[:, :, None, None] + bias[:, :, None, None]
        return F.relu(x + y)


class PolicyHead(nn.Module):
    """1x1 features biased by globally pooled features, then 40 planes per policy."""

    def __init__(self, channels: int, head: int, outputs: int) -> None:
        super().__init__()
        self.conv_p = _conv(channels, head, 1)
        self.conv_g = _conv(channels, head, 1)
        self.bn_g = nn.BatchNorm2d(head)
        self.fc_g = nn.Linear(2 * head, head)
        self.bn_p = nn.BatchNorm2d(head)
        self.out = nn.Conv2d(head, outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.relu(self.bn_g(self.conv_g(x)))
        p = self.conv_p(x) + self.fc_g(_pool(g))[:, :, None, None]
        return self.out(F.relu(self.bn_p(p)))


class Net(nn.Module):
    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        self.channels_last = False
        c = self.config.channels
        scale = torch.ones(1, NUM_PLANES, 1, 1)
        scale[0, PLY_PLANE] = PLY_SCALE
        self.register_buffer("input_scale", scale, persistent=False)
        self.stem = _conv(NUM_PLANES, c, 3)
        self.stem_bn = nn.BatchNorm2d(c)
        self.blocks = nn.Sequential(*[ResBlock(c, self.config.se_ratio) for _ in range(self.config.blocks)])
        self.policy = PolicyHead(c, self.config.head_channels, 2 * MOVE_PLANES)
        h = self.config.head_channels
        self.value_conv = _conv(c, h, 1)
        self.value_bn = nn.BatchNorm2d(h)
        self.value_fc = nn.Linear(2 * h, self.config.value_hidden)
        self.value_out = nn.Linear(self.config.value_hidden, 3)

    def forward(self, planes: torch.Tensor) -> dict[str, torch.Tensor]:
        b = planes.shape[0]
        x = planes.view(b, NUM_PLANES, BOARD, BOARD).to(self.stem.weight.dtype) * self.input_scale
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x = F.relu(self.stem_bn(self.stem(x)))
        x = self.blocks(x)
        p = self.policy(x).reshape(b, 2, NUM_ACTIONS)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = self.value_out(F.relu(self.value_fc(_pool(v))))
        return {
            "policy": p[:, 0],
            "opponent_policy": p[:, 1],
            "value_logit": v[:, 0],
            "q": torch.tanh(v[:, 1]),
            "moves_left": F.softplus(v[:, 2]) * MOVES_LEFT_SCALE,
        }

    @torch.no_grad()
    def fused(self, dtype: torch.dtype = torch.float16) -> Net:
        """An inference copy with batch norms folded into the preceding layers, in
        `dtype` and channels-last layout (about 1.5x faster on tensor-core GPUs)."""
        net = copy.deepcopy(self).eval().float()

        def fold(bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            return scale, bn.bias - bn.running_mean * scale

        def fold_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
            scale, shift = fold(bn)
            conv.weight.mul_(scale[:, None, None, None])
            conv.bias = nn.Parameter(shift if conv.bias is None else conv.bias * scale + shift)

        identity = nn.Identity()
        fold_conv(net.stem, net.stem_bn)
        net.stem_bn = identity
        for block in net.blocks:
            fold_conv(block.conv1, block.bn1)
            fold_conv(block.conv2, block.bn2)
            block.bn1 = block.bn2 = identity
        head = net.policy
        fold_conv(head.conv_g, head.bn_g)
        head.bn_g = identity
        # bn_p acts on conv_p(x) + fc_g(pooled): fold its scale into both terms.
        scale, shift = fold(head.bn_p)
        head.conv_p.weight.mul_(scale[:, None, None, None])
        head.conv_p.bias = nn.Parameter(shift)
        head.fc_g.weight.mul_(scale[:, None])
        head.fc_g.bias.mul_(scale)
        head.bn_p = identity
        fold_conv(net.value_conv, net.value_bn)
        net.value_bn = identity
        net = net.to(dtype=dtype, memory_format=torch.channels_last)
        net.channels_last = True
        return net

    @staticmethod
    def search_value(out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Expected result in [-1, 1] for the side to move, as used by the search."""
        return torch.tanh(out["value_logit"] * 0.5)


def save_model(path: str, model: Net, **extra: Any) -> None:
    torch.save({"config": model.config.to_dict(), "model": model.state_dict(), **extra}, path)


def load_model(path: str, device: str | torch.device = "cpu") -> tuple[Net, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=False)
    config = data["config"] if "config" in data else data["net"]  # snapshot or trainer state
    model = Net(NetConfig(**config)).to(device)
    model.load_state_dict(data["model"])
    return model, data
