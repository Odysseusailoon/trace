"""Stage 2: enumerate branch candidates from base path top-k logprobs.

Branch-record schema rows (one per (t, w)):
{case_id, t, tok_id, tok_p, is_base, prefix_len, status}
"""
from __future__ import annotations

from dataclasses import dataclass

from .base_path import BasePath


@dataclass
class Branch:
    case_id: str
    t: int            # generation position (0-indexed into gen_ids)
    tok_id: int
    tok_p: float      # unnormalized top-k probability
    is_base: bool
    status: str = "pending"

    def prefix_ids(self, base: BasePath) -> list[int]:
        return base.prefix_upto(self.t, self.tok_id)


def observed_positions(n_gen: int, spacing: int, mode: str = "token") -> list[int]:
    """Positions to branch at. For token mode: every `spacing` tokens starting at 0."""
    if mode == "token":
        return list(range(0, n_gen, spacing))
    raise NotImplementedError(f"spacing_mode={mode} not yet implemented")


def enumerate_branches(
    base: BasePath,
    positions: list[int],
    p_thresh: float = 0.05,
) -> list[Branch]:
    import math

    branches: list[Branch] = []
    for t in positions:
        if t >= len(base.gen_ids):
            continue
        greedy_id = base.gen_ids[t]
        probs = {tid: math.exp(lp) for tid, lp in base.top_logprobs[t]}
        kept = {tid: p for tid, p in probs.items() if p >= p_thresh or tid == greedy_id}
        if greedy_id not in kept:
            kept[greedy_id] = 1.0
        for tid, p in kept.items():
            branches.append(
                Branch(
                    case_id=base.case_id,
                    t=t,
                    tok_id=tid,
                    tok_p=p,
                    is_base=(tid == greedy_id),
                )
            )
    return branches
