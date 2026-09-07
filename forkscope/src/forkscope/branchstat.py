"""Between-branch dispersion at one position: the statistic the mixture hides.

The mixture curve is o_t = sum_w p~(w) * hist(t, w), a probability-weighted
average over the branches enumerated at t. When the greedy token carries most
of the kept mass (p~ around 0.94 is typical at branch_prob_threshold 0.05), a
counterfactual branch contributes about 1 of 20 draws, so whatever outcome
shift that branch causes is averaged away before any test sees it. Adjacent-
position TVD on that curve cannot detect a fork that lives in a low-probability
branch, and neither more samples nor denser spacing fixes it: the aggregation
removes the signal, not the noise.

This module compares the branches against each other instead of against their
own average. All three statistics are per-position, so unlike the adjacent-TVD
curve they need no neighbouring position and do not depend on spacing:

    fork_score(t) = max over branch pairs (w1, w2) of TVD(o(t,w1), o(t,w2))
    greedy_gap(t) = max over non-greedy w of TVD(o(t,greedy), o(t,w))
    dispersion(t) = sum_w p~(w) * TVD(o(t,w), o_mixture(t))

fork_score is the headline: how far apart the outcome distributions of two
continuations of the same prefix get, given only a different token at t.
greedy_gap is the counterfactual reading, "had the model emitted w here
instead". dispersion is in the same units as the mixture curve and says how
much that curve is hiding at t.

Null: under H0 every branch at t draws from one shared outcome distribution.
Pool the position's continuations, redraw branches at the observed sizes, and
recompute the statistic. This is the pooled-multinomial null and the Bonferroni
convention already used by the agent-side rule in scripts/fork_rule.py.

Two significance flags come out of it, and the difference between them is the
point:

    forking(t)  = fork_score significant                    (Bigelow-style)
    decision(t) = forking AND the position was contested     (forkscope-style)

where contested means the greedy token's renormalized probability is below
persist_max, the token-level analog of the agent-side persistence filter. A
position can be forking without being a decision: resampling it moves the
outcome, but the model was never going to emit anything else.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _tvd_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise TVD over the last axis."""
    return 0.5 * np.abs(a - b).sum(axis=-1)


def position_table(recs_at_t, categories, extractor):
    """One position -> (tok_ids, counts (B,K), tok_p (B,), greedy_row or None).

    counts[b] is the outcome histogram of branch b's continuations; tok_p is the
    unnormalized top-k probability recorded at enumeration time.
    """
    cat_idx = {c: i for i, c in enumerate(categories)}
    other = cat_idx.get("Other", len(categories) - 1)
    K = len(categories)
    tok_ids, counts, tok_p, greedy = [], [], [], None
    for b, r in enumerate(recs_at_t):
        h = np.zeros(K)
        for text in r["continuations"]:
            h[cat_idx.get(extractor.extract(text), other)] += 1
        tok_ids.append(int(r["tok_id"]))
        counts.append(h)
        tok_p.append(float(r["tok_p"]))
        if r.get("is_base"):
            greedy = b
    return tok_ids, np.stack(counts), np.array(tok_p, dtype=float), greedy


def _stats_from_counts(counts: np.ndarray, w: np.ndarray, greedy: int | None):
    """(fork_score, greedy_gap, dispersion, argmax pair) for one (B,K) table."""
    n = counts.sum(axis=1, keepdims=True)
    o = counts / np.maximum(n, 1)
    B = len(o)
    if B < 2:
        return 0.0, 0.0, 0.0, (0, 0)
    best, pair = 0.0, (0, 0)
    gap = 0.0
    for i in range(B):
        for j in range(i + 1, B):
            d = float(_tvd_rows(o[i], o[j]))
            if d > best:
                best, pair = d, (i, j)
            if greedy is not None and greedy in (i, j) and d > gap:
                gap = d
    mix = (w[:, None] * o).sum(axis=0)
    disp = float((w * _tvd_rows(o, mix[None, :])).sum())
    return best, gap, disp, pair


def _sim_null(
    counts: np.ndarray,
    w: np.ndarray,
    greedy: int | None,
    n_sims: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled-multinomial null draws of (fork_score, greedy_gap, dispersion).

    All branches redrawn at their observed sizes from the position's pooled
    outcome distribution; one simulation feeds all three statistics.
    """
    sizes = counts.sum(axis=1).astype(int)
    pooled = counts.sum(axis=0)
    p = pooled / pooled.sum()
    B = len(sizes)
    # (B, n_sims, K) normalized simulated branch distributions
    o = np.stack([
        rng.multinomial(int(sizes[b]), p, size=n_sims) / max(int(sizes[b]), 1)
        for b in range(B)
    ])
    fs = np.zeros(n_sims)
    gg = np.zeros(n_sims)
    for i in range(B):
        for j in range(i + 1, B):
            d = _tvd_rows(o[i], o[j])
            np.maximum(fs, d, out=fs)
            if greedy is not None and greedy in (i, j):
                np.maximum(gg, d, out=gg)
    mix = (w[:, None, None] * o).sum(axis=0)          # (n_sims, K)
    disp = (w[:, None] * np.stack([_tvd_rows(o[b], mix) for b in range(B)])).sum(axis=0)
    return fs, gg, disp


def branch_stats(
    records: list[dict],
    categories: list[str],
    extractor,
    n_sims: int = 20000,
    screen_sims: int = 2000,
    alpha: float = 0.05,
    persist_max: float = 0.9,
    seed: int = 0,
) -> list[dict]:
    """Per-position between-branch statistics with pooled-null p-values.

    Bonferroni is applied over the number of positions that carry at least two
    branches (single-branch positions are untestable, not tested-and-passed).
    The null runs in two stages: a cheap screen at `screen_sims`, refined to
    `n_sims` only where the screen p-value is within 5x of the corrected alpha.
    """
    rng = np.random.default_rng(seed)
    by_pos: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_pos[int(r["t"])].append(r)

    rows = []
    for t in sorted(by_pos):
        tok_ids, counts, tok_p, greedy = position_table(by_pos[t], categories, extractor)
        w = tok_p / tok_p.sum()
        p_greedy = float(w[greedy]) if greedy is not None else float("nan")
        fs, gg, disp, pair = _stats_from_counts(counts, w, greedy)
        rows.append({
            "t": t,
            "n_branches": int(len(tok_ids)),
            "tok_ids": tok_ids,
            "n_per_branch": counts.sum(axis=1).astype(int).tolist(),
            "p_greedy": p_greedy,
            "fork_score": fs,
            "greedy_gap": gg,
            "dispersion": disp,
            "argmax_pair": [tok_ids[pair[0]], tok_ids[pair[1]]] if len(tok_ids) > 1 else None,
            "_counts": counts,
            "_w": w,
            "_greedy": greedy,
        })

    testable = [r for r in rows if r["n_branches"] >= 2]
    m = max(len(testable), 1)
    alpha_c = alpha / m

    for r in rows:
        if r["n_branches"] < 2:
            r.update(p_fork=None, p_gap=None, p_disp=None, n_sims=0,
                     sig=False, contested=False, forking=False, decision=False)
            continue
        counts, w, greedy = r.pop("_counts"), r.pop("_w"), r.pop("_greedy")
        fs_n, gg_n, dp_n = _sim_null(counts, w, greedy, screen_sims, rng)
        used = screen_sims
        p_fork = float((fs_n >= r["fork_score"] - 1e-12).mean())
        if p_fork <= 5 * alpha_c:  # refine only near the decision boundary
            fs_n, gg_n, dp_n = _sim_null(counts, w, greedy, n_sims, rng)
            used = n_sims
            p_fork = float((fs_n >= r["fork_score"] - 1e-12).mean())
        r["p_fork"] = p_fork
        r["p_gap"] = float((gg_n >= r["greedy_gap"] - 1e-12).mean())
        r["p_disp"] = float((dp_n >= r["dispersion"] - 1e-12).mean())
        r["n_sims"] = used
        r["sig"] = p_fork < alpha_c
        r["contested"] = r["p_greedy"] < persist_max
        r["forking"] = r["sig"]
        r["decision"] = r["sig"] and r["contested"]

    for r in rows:
        r.pop("_counts", None)
        r.pop("_w", None)
        r.pop("_greedy", None)

    return rows


def summarize(rows: list[dict], alpha: float = 0.05) -> dict:
    """Headline rates. `forking_rate` and `decision_rate` are over TESTABLE
    positions: a position with one branch above threshold could not be tested,
    and counting it as a pass would inflate the denominator that the '~1%'
    figure was quoted against."""
    testable = [r for r in rows if r["n_branches"] >= 2]
    n = len(testable)
    forking = [r for r in testable if r["forking"]]
    decision = [r for r in testable if r["decision"]]
    fs = np.array([r["fork_score"] for r in testable]) if n else np.zeros(0)
    return {
        "n_positions": len(rows),
        "n_testable": n,
        "alpha": alpha,
        "alpha_bonferroni": alpha / max(n, 1),
        "n_forking": len(forking),
        "n_decision": len(decision),
        "forking_rate": len(forking) / n if n else 0.0,
        "decision_rate": len(decision) / n if n else 0.0,
        # the reframe: of the positions where resampling moves the outcome,
        # how many were positions the model might actually have decided?
        "decision_given_forking": len(decision) / len(forking) if forking else float("nan"),
        "fork_score_mean": float(fs.mean()) if n else 0.0,
        "fork_score_p90": float(np.quantile(fs, 0.9)) if n else 0.0,
        "fork_score_max": float(fs.max()) if n else 0.0,
    }
