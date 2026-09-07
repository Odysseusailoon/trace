"""Metrics: TVD, replicate TVD, fork-region TVD, efficiency multiplier, slope fit."""
from __future__ import annotations

import numpy as np


def tvd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise total variation distance. (T,K) x (T,K) -> (T,)"""
    return 0.5 * np.abs(p - q).sum(axis=-1)


def pooled_tvd(p: np.ndarray, q: np.ndarray, skip_t0: bool = False) -> float:
    t = tvd(p, q)
    return float(t[1:].mean() if skip_t0 else t.mean())


def fork_points(curve: np.ndarray, positions: np.ndarray, threshold: float = 0.10) -> list[int]:
    """Adjacent-position TVD > threshold; fork located at the later position."""
    t = tvd(curve[1:], curve[:-1])
    return [int(positions[i + 1]) for i in np.where(t > threshold)[0]]


def fork_region_tvd(p, q, positions, forks, radius: int = 10) -> float:
    mask = np.zeros(len(positions), dtype=bool)
    for f in forks:
        mask |= np.abs(positions - f) <= radius
    if not mask.any():
        return float("nan")
    return float(tvd(p[mask], q[mask]).mean())


def replicate_tvd(draws: np.ndarray, S: int, rng: np.random.Generator | None = None) -> float:
    """Split draws (To, S_full) into consecutive blocks of size S, compute o per
    block (as empirical histograms), mean pairwise TVD between blocks."""
    rng = rng or np.random.default_rng(0)
    To, S_full = draws.shape
    nb = S_full // S
    if nb < 2:
        return float("nan")
    K = int(draws.max()) + 1
    blocks = []
    for b in range(nb):
        blk = draws[:, b * S : (b + 1) * S]
        blocks.append(np.stack([np.bincount(r, minlength=K) for r in blk]) / S)
    tvs = []
    for i in range(nb):
        for j in range(i + 1, nb):
            tvs.append(tvd(blocks[i], blocks[j]).mean())
    return float(np.mean(tvs))


def loglog_slope(s_arr, tvd_arr) -> float:
    coef = np.polyfit(np.log(np.asarray(s_arr, float)), np.log(np.asarray(tvd_arr, float)), 1)
    return float(coef[0])


def efficiency_multiplier(s_arr, m0_tvs, tv_model, s_model) -> float:
    """Invert M0_raw's log-log TV-vs-S line at the model's TV -> equivalent S / S."""
    coef = np.polyfit(np.log(np.asarray(s_arr, float)), np.log(np.asarray(m0_tvs, float)), 1)
    s_eq = float(np.exp((np.log(tv_model) - coef[1]) / coef[0]))
    return s_eq / s_model
