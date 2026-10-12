"""The exported network file: layout, round trip, and a forward pass from the file alone in numpy."""

import math

import numpy as np
import pytest
import torch

from tafl.export import HEADER, MAGIC, export_bytes, layout, read_net, stored_model
from tafl.model import MOVE_PLANES, MOVES_LEFT_SCALE, PLY_SCALE, Net, NetConfig
from tafl.native import BOARD, FEATURE_BYTES, NUM_PLANES, PLY_PLANE


def random_net() -> Net:
    torch.manual_seed(0)
    net = Net(NetConfig(blocks=2, channels=16, se_ratio=4, head_channels=8, value_hidden=16))
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):  # so that folding has something to fold
                m.weight.uniform_(0.5, 1.5)
                m.bias.normal_(0, 0.2)
                m.running_mean.normal_(0, 0.2)
                m.running_var.uniform_(0.5, 2.0)
    return net.eval()


def conv(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`[c, n, n]` by `[o, c, k, k]` with zero padding to the same size."""
    k, n = w.shape[-1], x.shape[1]
    xp = np.pad(x, ((0, 0), (k // 2, k // 2), (k // 2, k // 2)))
    out = np.zeros((w.shape[0], n, n))
    for dy in range(k):
        for dx in range(k):
            out += np.einsum("oc,cyx->oyx", w[:, :, dy, dx], xp[:, dy : dy + n, dx : dx + n])
    return out + b[:, None, None]


def pool(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.mean((1, 2)), x.max((1, 2))])


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def numpy_forward(cfg: NetConfig, tensors: dict[str, np.ndarray], planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move-policy logits and the 3 raw value outputs, as docs/FORMATS.md describes them."""
    t = {k: v.astype(np.float64) for k, v in tensors.items()}
    c = cfg.channels
    x = planes.reshape(NUM_PLANES, BOARD, BOARD).astype(np.float64)
    x[PLY_PLANE] *= PLY_SCALE
    x = relu(conv(x, t["stem.weight"], t["stem.bias"]))
    for i in range(cfg.blocks):
        p = f"blocks.{i}."
        y = relu(conv(x, t[p + "conv1.weight"], t[p + "conv1.bias"]))
        y = conv(y, t[p + "conv2.weight"], t[p + "conv2.bias"])
        s = t[p + "se2.weight"] @ relu(t[p + "se1.weight"] @ pool(y) + t[p + "se1.bias"]) + t[p + "se2.bias"]
        y = y / (1 + np.exp(-s[:c]))[:, None, None] + s[c:, None, None]
        x = relu(x + y)
    g = relu(conv(x, t["policy.conv_g.weight"], t["policy.conv_g.bias"]))
    h = conv(x, t["policy.conv_p.weight"], t["policy.conv_p.bias"])
    h += (t["policy.fc_g.weight"] @ pool(g) + t["policy.fc_g.bias"])[:, None, None]
    policy = conv(relu(h), t["policy.out.weight"], t["policy.out.bias"]).reshape(-1)
    v = relu(conv(x, t["value_conv.weight"], t["value_conv.bias"]))
    v = t["value_out.weight"] @ relu(t["value_fc.weight"] @ pool(v) + t["value_fc.bias"]) + t["value_out.bias"]
    return policy, v


def test_export_round_trip():
    net = random_net()
    data = export_bytes(net)
    count = sum(math.prod(shape) for _, shape in layout(net.config))
    assert data[:8] == MAGIC and len(data) == HEADER.size + 2 * count
    cfg, tensors = read_net(data)
    assert cfg == net.config
    fused = net.fused(torch.float16).state_dict()
    for name, a in tensors.items():
        want = fused[name][:MOVE_PLANES] if name.startswith("policy.out.") else fused[name]
        assert np.array_equal(a, want.contiguous().numpy()), name

    rng = np.random.default_rng(0)
    planes = rng.integers(0, 2, (3, FEATURE_BYTES), dtype=np.uint8)
    planes.reshape(3, NUM_PLANES, -1)[:, PLY_PLANE] = [[0], [37], [255]]
    stored = stored_model(data)
    with torch.no_grad():
        out_file, out_orig = stored(torch.from_numpy(planes)), net(torch.from_numpy(planes))
    for i in range(len(planes)):
        policy, v = numpy_forward(cfg, tensors, planes[i])
        assert np.allclose(policy, out_file["policy"][i].numpy(), atol=1e-4)
        assert np.allclose(v[0], out_file["value_logit"][i].item(), atol=1e-4)
        assert np.allclose(np.tanh(v[1]), out_file["q"][i].item(), atol=1e-4)
        assert np.allclose(np.logaddexp(0, v[2]) * MOVES_LEFT_SCALE, out_file["moves_left"][i].item(), atol=1e-3)
        # Rounding the weights to float16 costs little against the unfused float32 network.
        assert np.allclose(policy, out_orig["policy"][i].numpy(), atol=0.05)
        assert abs(v[0] - out_orig["value_logit"][i].item()) < 0.05


def test_export_rejects_damaged_files():
    data = export_bytes(random_net())
    with pytest.raises(ValueError):
        read_net(data[:-2])
    with pytest.raises(ValueError):
        read_net(b"TAFLNET2" + data[8:])
    with pytest.raises(ValueError):
        read_net(data[: HEADER.size - 1])
