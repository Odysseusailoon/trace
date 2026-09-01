"""Mixture draws: flatten per-position branch records into S iid draws ~ Multinomial(S, o_t).

Port of otrecon data.py::mixture_draws. At each position, draw S branch picks
from the normalized branch weights, consuming each branch's recorded outcomes
in generation order. Nested subsampling (S=15 from S=200) then means taking
the first S draws — strictly a subset of the same recorded continuations.
"""
from __future__ import annotations

import numpy as np


def normalize_weights(recs_at_t: list[dict]) -> np.ndarray:
    w = np.array([r["tok_p"] for r in recs_at_t], dtype=float)
    return w / w.sum()


def mixture_draws(
    recs_at_t: list[dict],
    labels_by_branch: dict[int, list[int]],
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One position: returns (n_total,) int category draws.

    recs_at_t: branch records at position t (each has tok_id, tok_p).
    labels_by_branch: tok_id -> per-draw category indices in generation order.
    """
    toks = [r["tok_id"] for r in recs_at_t]
    w = normalize_weights(recs_at_t)
    picks = rng.choice(len(toks), size=n_total, p=w)
    ptr = {tok: 0 for tok in toks}
    out = np.empty(n_total, dtype=int)
    for j, bi in enumerate(picks):
        tok = toks[bi]
        labels = labels_by_branch[tok]
        i = ptr[tok]
        if i >= len(labels):
            # branch exhausted: resample from its empirical distribution
            out[j] = rng.choice(labels)
        else:
            out[j] = labels[i]
            ptr[tok] += 1
    return out


def build_draw_matrix(
    records: list[dict],
    positions: list[int],
    cat_index: dict[str, int],
    extractor,
    n_total: int,
    seed: int = 0,
) -> np.ndarray:
    """(T, n_total) int draws for all observed positions."""
    rng = np.random.default_rng(seed)
    by_pos: dict[int, list[dict]] = {}
    for r in records:
        by_pos.setdefault(r["t"], []).append(r)
    rows = []
    for t in positions:
        recs = by_pos[t]
        labels_by_branch = {
            r["tok_id"]: [cat_index[extractor.extract(c)] for c in r["continuations"]]
            for r in recs
        }
        rows.append(mixture_draws(recs, labels_by_branch, n_total, rng))
    return np.stack(rows)
