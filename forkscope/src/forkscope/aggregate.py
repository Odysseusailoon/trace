"""Stage 5: aggregate branch records into o_t outcome distributions.

o_t[k] = sum_w p~_w * hist_{t,w}[k], where p~ is normalized within the kept
branch set at position t, hist is the empirical outcome histogram of that
branch's S continuations.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_branch_records(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def aggregate(records: list[dict], categories: list[str], extractor) -> tuple[np.ndarray, dict]:
    """Returns (o_t (T,K), counts dict keyed by (t, tok_id) -> per-draw labels)."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    by_pos: dict[int, list[dict]] = defaultdict(list)
    per_draw: dict[tuple[int, int], list[str]] = {}
    for rec in records:
        labels = [extractor.extract(text) for text in rec["continuations"]]
        per_draw[(rec["t"], rec["tok_id"])] = labels
        rec = dict(rec)
        rec["labels"] = labels
        by_pos[rec["t"]].append(rec)

    T = max(by_pos) + 1 if by_pos else 0
    K = len(categories)
    o_t = np.zeros((T, K))
    for t, recs in by_pos.items():
        ws = np.array([r["tok_p"] for r in recs], dtype=float)
        ws /= ws.sum()
        for w, r in zip(ws, recs):
            hist = np.zeros(K)
            for lab in r["labels"]:
                hist[cat_idx.get(lab, cat_idx.get("Other", K - 1))] += 1
            hist /= max(len(r["labels"]), 1)
            o_t[t] += w * hist
    return o_t, per_draw
