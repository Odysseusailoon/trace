"""5-fold CV over (pelt_penalty, bandwidth): held-out multinomial log-likelihood."""
from __future__ import annotations

import numpy as np

from .pelt import segment
from .pooling import kernel_pool, segments_from_breaks


def loglik_counts(pred: np.ndarray, counts: np.ndarray, eps: float = 1e-9) -> float:
    p = pred / pred.sum(axis=1, keepdims=True)
    return float((counts * np.log(np.maximum(p, eps))).sum())


def cv_select(
    obs_tok: np.ndarray,
    draws: np.ndarray,
    penalty_grid: list[float],
    bandwidth_grid: list[float],
    n_folds: int = 5,
) -> tuple[float, float, dict]:
    """draws: (To, S) integer category indices per observed position.
    Folds are consecutive slices of the S dimension (by generation order).
    Returns (best_pen, best_h, scores)."""
    To, S = draws.shape
    K = int(draws.max()) + 1
    edges = np.linspace(0, S, n_folds + 1).astype(int)
    scores: dict[tuple[float, float], float] = {}
    for pen in penalty_grid:
        for h in bandwidth_grid:
            total = 0.0
            for f in range(n_folds):
                lo, hi = edges[f], edges[f + 1]
                test = draws[:, lo:hi]
                train = np.concatenate([draws[:, :lo], draws[:, hi:]], axis=1)
                tr_counts = np.stack([np.bincount(row, minlength=K) for row in train])
                te_counts = np.stack([np.bincount(row, minlength=K) for row in test])
                bkps = segment(tr_counts, pen)
                segs = segments_from_breaks(bkps)
                pred = kernel_pool(obs_tok, tr_counts, obs_tok, segs, h)
                total += loglik_counts(pred, te_counts)
            scores[(pen, h)] = total
    best = max(scores, key=lambda k: scores[k])  # first-max tie-break
    return best[0], best[1], scores
