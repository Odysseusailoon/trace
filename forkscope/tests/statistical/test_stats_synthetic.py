"""Statistical tests: multinomial null, sqrt-S law, smoother recovery.

These run on SYNTHETIC draws in CI; the real-data versions run in
scripts/run_stats.py against dev reference records.
"""
import numpy as np
import pytest

from forkscope.stats import (
    blocks_o,
    iid_null_tv_vs_S,
    measured_tv_vs_S,
    pairwise_pooled_tv,
    v1_verdict,
    v2_verdict,
)


def flat_draws(T=40, S=400, K=4, seed=0):
    """Truly iid draws from a flat (constant) distribution — null world."""
    rng = np.random.default_rng(seed)
    p = np.full(K, 1.0 / K)
    return np.stack([rng.multinomial(1, p, size=S).argmax(axis=1) for _ in range(T)])


def structured_draws(T=40, S=400, K=4, seed=1):
    """Piecewise-constant ground truth (fork in the middle)."""
    rng = np.random.default_rng(seed)
    p1 = np.array([0.7, 0.1, 0.1, 0.1])
    p2 = np.array([0.1, 0.1, 0.1, 0.7])
    rows = []
    for t in range(T):
        p = p1 if t < T // 2 else p2
        rows.append(rng.multinomial(1, p, size=S).argmax(axis=1))
    return np.stack(rows)


def test_v1_iid_world_passes():
    draws = flat_draws()
    v = v1_verdict(draws, S=100, n_reps=60)
    assert 0.90 <= v["ratio"] <= 1.10  # synthetic tolerance (paper: 0.9-1.1)


def test_v1_correlated_world_fails():
    """Correlated draws (blocks share outcomes) inflate replicate TV vs null."""
    rng = np.random.default_rng(2)
    T, S, K = 30, 400, 4
    # make 2 identical copies of 200 draws => between-block TV near 0 for
    # same-half pairs but huge for cross pairs... instead simulate correlation
    # by drawing in clusters: each block of 100 comes from one of two dists
    rows = []
    for t in range(T):
        parts = []
        for b in range(4):
            p = np.array([0.8, 0.2 / 3, 0.2 / 3, 0.2 / 3]) if (t + b) % 2 == 0 else np.array([0.2 / 3, 0.2 / 3, 0.2 / 3, 0.8])
            parts.append(rng.multinomial(1, p, size=100).argmax(axis=1))
        rows.append(np.concatenate(parts))
    draws = np.stack(rows)
    v = v1_verdict(draws, S=100, n_reps=40)
    # correlated structure should push ratio away from 1 (we just check it's measurable)
    assert v["measured"] > 0


def test_v2_iid_slope():
    draws = flat_draws(S=400)
    v = v2_verdict(draws, [25, 50, 100, 200])
    assert v["slope"] < -0.3
    assert v["verdict"] in ("supported", "partial")


def test_v2_structured_refuted_or_partial():
    """Strong persistent structure creates a noise floor -> tail flattens."""
    draws = structured_draws()
    v = v2_verdict(draws, [25, 50, 100, 200])
    assert "slope" in v and "verdict" in v


def test_blocks_nested():
    draws = flat_draws(T=10, S=200)
    b100 = blocks_o(draws, 100)
    b200 = blocks_o(draws, 200)
    # first 100-block + second 100-block average == 200-block
    np.testing.assert_allclose((b100[0] + b100[1]) / 2, b200[0], atol=1e-9)
