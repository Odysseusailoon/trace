"""FullModel = PELT + segment kernel pooling + CV (M5a port)."""
from __future__ import annotations

import numpy as np

from .cv import cv_select
from .pelt import segment
from .pooling import kernel_pool, segments_from_breaks


class FullModel:
    def __init__(self, penalty_grid, bandwidth_grid, cv_folds: int = 5):
        self.penalty_grid = penalty_grid
        self.bandwidth_grid = bandwidth_grid
        self.cv_folds = cv_folds

    def fit(self, obs_tok: np.ndarray, draws: np.ndarray):
        """draws: (To, S) int category indices. Selects (pen, h) by CV."""
        self.pen_, self.h_, self.cv_scores_ = cv_select(
            obs_tok, draws, self.penalty_grid, self.bandwidth_grid, self.cv_folds
        )
        K = int(draws.max()) + 1
        counts = np.stack([np.bincount(row, minlength=K) for row in draws])
        self.counts_ = counts
        self.bkps_ = segment(counts, self.pen_)
        self.segments_ = segments_from_breaks(self.bkps_)
        self.obs_tok_ = obs_tok
        return self

    def predict(self, all_tok: np.ndarray) -> np.ndarray:
        return kernel_pool(self.obs_tok_, self.counts_, all_tok, self.segments_, self.h_)

    def fork_points(self, all_tok: np.ndarray, threshold: float = 0.10) -> list[int]:
        pred = self.predict(all_tok)
        tvd = 0.5 * np.abs(np.diff(pred, axis=0)).sum(axis=1)
        return [int(all_tok[i + 1]) for i in np.where(tvd > threshold)[0]]
