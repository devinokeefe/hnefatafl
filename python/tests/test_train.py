import numpy as np
import torch

from tafl.native import FEATURE_BYTES, NUM_ACTIONS, Pool
from tafl.replay import Chunk
from tafl.train import TrainConfig, Trainer


def small_chunk() -> Chunk:
    pool = Pool(games=4, full_simulations=8, fast_simulations=4, full_search_prob=1.0, max_plies=30, seed=3)
    planes = np.zeros((pool.max_batch, FEATURE_BYTES), np.uint8)
    policy = np.zeros((pool.max_batch, NUM_ACTIONS), np.float32)
    value = np.zeros(pool.max_batch, np.float32)
    for _ in range(1000):
        rows = pool.collect(planes)
        pool.apply(policy, value, rows)
    records, actions, probs, _ = pool.take()
    pool.close()
    return Chunk.build(records, actions, probs)


def test_resume_survives_a_torn_checkpoint(tmp_path):
    config = TrainConfig(blocks=1, channels=8)
    cpu = torch.device("cpu")
    trainer = Trainer(tmp_path, config, cpu)
    chunk = small_chunk()
    chunk.save(tmp_path / "selfplay" / "chunk_000001.npz")
    trainer.steps, trainer.generated, trainer.chunk_index = 7, len(chunk), 1
    trainer.checkpoint()
    # What a crash can leave behind: a torn latest.pt, an empty chunk, a temporary file.
    (tmp_path / "checkpoints" / "latest.pt").write_bytes(b"\0" * 64)
    (tmp_path / "selfplay" / "chunk_000002.npz").write_bytes(b"")
    (tmp_path / "selfplay" / "chunk_000003.npz.tmp").write_bytes(b"\0")

    resumed = Trainer(tmp_path, config, cpu)
    assert (resumed.steps, resumed.generated, resumed.restarts) == (7, len(chunk), 1)
    assert len(resumed.buffer) == len(chunk)
    for a, b in zip(trainer.model.state_dict().values(), resumed.model.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    log = (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert "resuming from step_0000007.pt" in log and "skipping unreadable chunk_000002.npz" in log
