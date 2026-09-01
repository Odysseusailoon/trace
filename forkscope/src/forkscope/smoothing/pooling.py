"""Segment-wise Gaussian-kernel Dirichlet pooling (paper's M5a model)."""
from __future__ import annotations

import numpy as np


def kernel_pool(
    obs_tok: np.ndarray,
    counts: np.ndarray,
    all_tok: np.ndarray,
    segments: list[tuple[int, int]],
    bandwidth: float,
) -> np.ndarray:
    """M5a: alpha(t)_k = 1/K + sum_{s in same segment} exp(-0.5((t-s)/h)^2) * c_{s,k}"""
    K = counts.shape[1]
    T = len(all_tok)
    alpha = np.full((T, K), 1.0 / K)
    h2 = 2.0 * bandwidth * bandwidth
    n_obs = len(obs_tok)
    for (s, e) in segments:
        seg_pos = obs_tok[s:e]
        seg_counts = counts[s:e]
        lo = -np.inf if s == 0 else 0.5 * (obs_tok[s - 1] + obs_tok[s])
        hi = np.inf if e == n_obs else 0.5 * (obs_tok[e - 1] + obs_tok[e])
        mask = (all_tok >= lo) & (all_tok < hi)
        if not mask.any():
            continue
        d = np.abs(all_tok[mask][:, None] - seg_pos[None, :])
        w = np.exp(-(d * d) / h2)
        alpha[mask] += w @ seg_counts
    return alpha / alpha.sum(axis=1, keepdims=True)


def segments_from_breaks(bkps: list[int]) -> list[tuple[int, int]]:
    out, prev = [], 0
    for b in bkps:
        out.append((prev, b))
        prev = b
    return out
