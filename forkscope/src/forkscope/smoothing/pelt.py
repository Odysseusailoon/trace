"""PELT segmentation on per-position count vectors, cost = L2 (see RFC F6)."""
from __future__ import annotations

import numpy as np
import ruptures as rpt


def segment(counts: np.ndarray, pen: float, min_size: int = 2) -> list[int]:
    """counts: (T, K). Returns breakpoints (exclusive ends, ruptures convention)."""
    T = len(counts)
    if T < 2 * min_size:
        return [T]
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(counts.astype(float))
    return algo.predict(pen=pen)


def segment_bounds(bkps: list[int], T: int) -> list[tuple[int, int]]:
    out, prev = [], 0
    for b in bkps:
        out.append((prev, min(b, T)))
        prev = b
    if out[-1][1] < T:
        out.append((out[-1][1], T))
    return out
