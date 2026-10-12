import platform
import sys

import numpy as np
import pytest
import torch

from tafl.model import Net, NetConfig
from tafl.native import (
    FEATURE_BYTES,
    NUM_ACTIONS,
    NUM_SYMMETRIES,
    Pool,
    TaflError,
    action_index,
    encode_states,
    game_info,
    symmetry_actions,
)
from tafl.replay import Chunk, ReplayBuffer, replay_window


def run_uniform(pool: Pool, steps: int) -> None:
    planes = np.zeros((pool.max_batch, FEATURE_BYTES), np.uint8)
    policy = np.zeros((pool.max_batch, NUM_ACTIONS), np.float32)
    value = np.zeros(pool.max_batch, np.float32)
    for _ in range(steps):
        rows = pool.collect(planes)
        pool.apply(policy, value, rows)


def test_referee():
    info = game_info()
    assert info["side"] == "attackers" and len(info["legal"]) == 116 and info["outcome"] is None
    info = game_info("startpos", "d11d9 f4b4")
    assert info["ply"] == 2 and info["side"] == "attackers"
    with pytest.raises(TaflError):
        game_info("startpos", "f6f7")  # the defenders do not move first


def test_symmetry_table_is_a_permutation_on_moves():
    table = symmetry_actions()
    assert table.shape == (NUM_SYMMETRIES, NUM_ACTIONS)
    a = action_index("d11d9")
    images = {int(table[t, a]) for t in range(NUM_SYMMETRIES)}
    assert len(images) == 8 and a in images


def test_self_play_samples_and_replay():
    pool = Pool(games=8, full_simulations=16, fast_simulations=4, full_search_prob=0.5, max_plies=60, seed=5)
    run_uniform(pool, 3000)
    records, actions, probs, games = pool.take()
    assert games and len(records) == sum(g["samples"] for g in games)
    assert len(actions) == int(records["policy_len"].sum())
    chunk = Chunk.build(records, actions, probs)
    sums = np.add.reduceat(chunk.probs.astype(np.float64), chunk.offsets[:-1])
    assert np.allclose(sums, 1.0, atol=1e-2)

    buffer = ReplayBuffer(block_size=10)
    buffer.add(chunk, keep=10**9)
    batch = buffer.sample(64, len(buffer), np.random.default_rng(0))
    assert batch.planes.shape == (64, FEATURE_BYTES)
    target = np.zeros((64, NUM_ACTIONS))
    np.add.at(target, (batch.policy_rows, batch.policy_actions), batch.policy_probs)
    assert np.allclose(target.sum(1), 1.0, atol=1e-2)
    # Symmetric planes agree with the identity encoding up to a permutation of squares.
    states = np.ascontiguousarray(records["state"][:4])
    base = encode_states(states, np.zeros(4, np.uint8))
    for t in range(NUM_SYMMETRIES):
        image = encode_states(states, np.full(4, t, np.uint8))
        assert (image.reshape(4, -1, 121).sum(2) == base.reshape(4, -1, 121).sum(2)).all()


def test_analysis_mode():
    pool = Pool(games=2, autoplay=False, search={"algorithm": "puct"})
    pool.set_game(0, "startpos", "d11d9")
    pool.start_search(0, 32)
    while pool.is_thinking(0):
        run_uniform(pool, 1)
    result = pool.state(0)["result"]
    assert result["simulations"] >= 32 and result["best"] in game_info("startpos", "d11d9")["legal"]
    assert result["new_simulations"] == result["simulations"]  # nothing was kept from an earlier search


def test_replay_window_growth():
    assert replay_window(1000, 250_000) == 1000
    assert replay_window(250_000, 250_000) == 250_000
    assert 250_000 < replay_window(1_000_000, 250_000) < 1_000_000
    assert replay_window(10**9, 250_000, cap=4_000_000) == 4_000_000


def test_fused_network_matches():
    torch.manual_seed(0)
    net = Net(NetConfig(blocks=2, channels=32))
    net.train()
    with torch.no_grad():
        for _ in range(5):
            net(torch.randint(0, 2, (64, FEATURE_BYTES), dtype=torch.uint8))
    net.eval()
    x = torch.randint(0, 2, (16, FEATURE_BYTES), dtype=torch.uint8)
    fused = net.fused(torch.float32)
    with torch.no_grad():
        a, b = net(x), fused(x)
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-3, rtol=1e-3), key


def test_moves_left_shaping():
    from tafl.inference import Evaluator

    evaluator = Evaluator(Net(NetConfig(blocks=1, channels=16)), torch.device("cpu"), moves_left_alpha=0.2)
    out = {"value_logit": torch.tensor([-20.0, 20.0, -20.0]), "moves_left": torch.tensor([400.0, 400.0, 0.0])}
    v = evaluator.value(out)
    # Long losses hurt less, long wins help less, an immediate loss is a full loss.
    assert torch.allclose(v, torch.tensor([-0.8, 0.8, -1.0]), atol=1e-4)


def test_graph_sizes():
    from tafl.inference import graph_size

    assert [graph_size(n) for n in (1, 2, 3, 100, 128, 129, 1000, 1024)] == [1, 2, 4, 128, 128, 256, 1024, 1024]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_graph_replay_matches_eager_and_follows_new_weights():
    import threading

    from tafl.inference import Batch, Evaluator

    dev = torch.device("cuda")
    torch.manual_seed(0)
    first, second = (Net(NetConfig(blocks=2, channels=32)).to(dev).fused() for _ in range(2))
    eager = Evaluator(first, dev, moves_left_alpha=0.1, graphs=False)
    graphed = Evaluator(Net(NetConfig(blocks=2, channels=32)).to(dev).fused(), dev, moves_left_alpha=0.1)
    graphed.set_model(first)
    batch, reference = Batch(64, dev), Batch(64, dev)
    # Training keeps the GPU busy from another thread while graphs are captured.
    stop = threading.Event()

    def busy() -> None:
        m = torch.randn(512, 512, device=dev)
        while not stop.is_set():
            (m @ m).sum().item()

    worker = threading.Thread(target=busy)
    worker.start()
    try:
        for model in (first, second):
            graphed.set_model(model)
            eager.set_model(model)
            for rows in (1, 5, 64):
                planes = torch.randint(0, 2, (rows, FEATURE_BYTES), dtype=torch.uint8)
                for b, ev in ((batch, graphed), (reference, eager)):
                    b.planes[:rows] = planes
                    ev.launch(b, rows)
                    ev.wait(b)
                assert np.allclose(batch.policy_np[:rows], reference.policy_np[:rows], atol=1e-3)
                assert np.allclose(batch.value_np[:rows], reference.value_np[:rows], atol=1e-3)
    finally:
        stop.set()
        worker.join()
    assert sorted(graphed.graphs) == [1, 8, 64]


@pytest.mark.skipif(sys.platform != "win32", reason="the WMI helper exists on Windows only")
def test_platform_does_not_query_wmi():
    with pytest.raises(OSError):
        platform._wmi_query("OS", "Version")
    assert platform.machine()
