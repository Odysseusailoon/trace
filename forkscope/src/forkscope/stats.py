"""V1 (multinomial null) + V2 (1/sqrt(S) law) + V3 (smoother recovery).

Statistical tests run on recorded branch data — no GPU needed at test time.
"""
from __future__ import annotations

import numpy as np

from forkscope.metrics import loglog_slope, replicate_tvd, tvd


def blocks_o(draws: np.ndarray, S: int) -> list[np.ndarray]:
    """Split (T, S_full) draws into consecutive blocks of S, return per-block o_t."""
    T, S_full = draws.shape
    nb = S_full // S
    K = int(draws.max()) + 1
    return [
        np.stack([np.bincount(row[b * S:(b + 1) * S], minlength=K) for row in draws]) / S
        for b in range(nb)
    ]


def pairwise_pooled_tv(blocks: list[np.ndarray]) -> float:
    tvs = []
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            tvs.append(tvd(blocks[i], blocks[j]).mean())
    return float(np.mean(tvs))


def measured_tv_vs_S(draws: np.ndarray, s_values: list[int]) -> list[float]:
    return [pairwise_pooled_tv(blocks_o(draws, S)) for S in s_values]


def iid_null_tv_vs_S(
    draws: np.ndarray, s_values: list[int], n_reps: int = 100, seed: int = 0
) -> dict[int, float]:
    """Exact iid multinomial null: resample each position's full pool, replicate
    the block pipeline, mean pairwise pooled TV."""
    rng = np.random.default_rng(seed)
    T, S_full = draws.shape
    K = int(draws.max()) + 1
    # empirical per-position distribution over the full pool
    probs = np.stack([np.bincount(row, minlength=K) for row in draws]) / S_full
    out: dict[int, list[float]] = {S: [] for S in s_values}
    for rep in range(n_reps):
        sim = np.stack([rng.multinomial(1, p, size=S_full).argmax(axis=1) for p in probs])
        for S in s_values:
            out[S].append(pairwise_pooled_tv(blocks_o(sim, S)))
    return {S: float(np.mean(v)) for S, v in out.items()}


def v1_verdict(draws: np.ndarray, S: int, n_reps: int = 100) -> dict:
    """measured/null ratio in [0.95, 1.05] passes."""
    meas = pairwise_pooled_tv(blocks_o(draws, S))
    null = iid_null_tv_vs_S(draws, [S], n_reps)[S]
    ratio = meas / null if null else float("inf")
    return {"S": S, "measured": meas, "null": null, "ratio": ratio,
            "pass": 0.95 <= ratio <= 1.05}


def v2_verdict(draws: np.ndarray, s_values: list[int]) -> dict:
    """Three-valued verdict (ported from analyze_s1000 pooled_verdict):
    slope in [-0.65, -0.35], monotone decreasing, tail slope > -0.15 => refuted."""
    meas = measured_tv_vs_S(draws, s_values)
    slope = loglog_slope(s_values, meas)
    mono = all(meas[i] >= meas[i + 1] - 1e-12 for i in range(len(meas) - 1))
    nulls = iid_null_tv_vs_S(draws, s_values, n_reps=100)
    ratios = [m / nulls[S] for S, m in zip(s_values, meas)]
    # tail slope over last two points
    tail = loglog_slope(s_values[-2:], meas[-2:])
    ok_slope = -0.65 <= slope <= -0.35
    ok_null = all(r <= 1.15 for r in ratios)
    if ok_slope and mono and ok_null:
        verdict = "supported"
    elif tail > -0.15:
        verdict = "refuted"
    else:
        verdict = "partial"
    return {"s_values": s_values, "measured": meas, "slope": slope,
            "monotone": mono, "null_ratios": ratios, "tail_slope": tail,
            "verdict": verdict}
