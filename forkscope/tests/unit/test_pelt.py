import numpy as np

from forkscope.smoothing.pelt import segment, segment_bounds
from forkscope.smoothing.pooling import kernel_pool, segments_from_breaks


def test_pelt_finds_changepoint():
    rng = np.random.default_rng(0)
    # large counts so the segment difference dominates noise
    seg1 = rng.multinomial(400, [0.8, 0.2], size=40).astype(float)
    seg2 = rng.multinomial(400, [0.2, 0.8], size=40).astype(float)
    counts = np.vstack([seg1, seg2])
    bkps = segment(counts, pen=4096, min_size=2)
    # expect exactly one break near index 40
    assert len(bkps) == 2
    assert abs(bkps[0] - 40) <= 2


def test_pelt_flat_sequence_no_breaks():
    rng = np.random.default_rng(1)
    counts = rng.multinomial(400, [0.5, 0.5], size=50).astype(float)
    bkps = segment(counts, pen=4096, min_size=2)
    assert bkps == [50]


def test_kernel_pool_recovers_flat():
    obs = np.arange(0, 40, 4)
    rng = np.random.default_rng(2)
    counts = rng.multinomial(20, [0.7, 0.3], size=len(obs)).astype(float)
    all_tok = np.arange(40)
    pred = kernel_pool(obs, counts, all_tok, [(0, len(obs))], bandwidth=8)
    raw = counts / counts.sum(axis=1, keepdims=True)
    truth = np.tile([0.7, 0.3], (len(obs), 1))
    tvd = lambda a, b: 0.5 * np.abs(a - b).sum(axis=1).mean()
    pred_at_obs = pred[obs]
    assert tvd(pred_at_obs, truth) < tvd(raw, truth)


def test_segments_from_breaks():
    assert segments_from_breaks([3, 7, 10]) == [(0, 3), (3, 7), (7, 10)]
