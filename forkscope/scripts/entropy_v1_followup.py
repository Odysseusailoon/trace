"""Entropy-vs-causal-fork alignment + MCQ V1 p-value recompute (CPU only).

Q1 (Plan A figure / Plan B Gate 1): are measured forks entropy peaks?
For each case with a fork report, compute per-position truncated entropy from
base-path top_logprobs and the entropy percentile of each measured fork.

Q2: MCQ V1 re-verdict as a p-value (percentile of measured pooled TV in the
simulated iid-multinomial null), replacing the ratio-band verdict.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/dev/forkscope")

import numpy as np

from forkscope.stats import blocks_o, pairwise_pooled_tv

DATA = Path("/home/dev/forkscope/data")


def entropy_rows(base: dict) -> list[float]:
    rows = base.get("top_logprobs") or []
    hs = []
    for row in rows:
        ps = []
        for item in row:
            lp = None
            for v in item[:2]:
                if isinstance(v, float):
                    lp = v
            if lp is None and isinstance(item[0], (int, float)):
                lp = float(item[0])
            if lp is not None and lp <= 0:
                ps.append(math.exp(lp))
        if not ps:
            hs.append(0.0)
            continue
        hs.append(-sum(p * math.log(p) for p in ps if p > 0))
    return hs


def q1_case(case: str) -> dict | None:
    bp = DATA / "raw" / f"base_{case}.json"
    rp = DATA / "reports" / f"report_{case}.json"
    if not bp.exists() or not rp.exists():
        return None
    base = json.load(open(bp))
    rep = json.load(open(rp))
    hs = np.array(entropy_rows(base))
    forks = rep.get("forks", [])
    if not len(hs) or not forks:
        return None
    out = []
    for f in forks:
        t = f["t"]
        if t >= len(hs):
            continue
        pct = float((hs < hs[t]).mean())
        out.append({"t": t, "tvd": f.get("tvd"), "H": float(hs[t]), "H_pctile": pct})
    pcts = [o["H_pctile"] for o in out]
    top20 = sum(1 for p in pcts if p >= 0.8)
    # reverse direction: of the top-decile entropy positions, how many are forks?
    k = max(1, len(hs) // 10)
    top_idx = set(np.argsort(hs)[-k:].tolist())
    fork_ts = {f["t"] for f in forks}
    fork_window = set()
    for t in fork_ts:
        fork_window.update(range(t - 4, t + 5))
    hits = sum(1 for i in top_idx if i in fork_window)
    return {"case": case, "n_positions": len(hs), "n_forks": len(out),
            "fork_entropy_pctiles": out,
            "median_pctile": float(np.median(pcts)) if pcts else None,
            "forks_in_top20pct_entropy": f"{top20}/{len(pcts)}",
            "top_decile_entropy_positions": k,
            "top_decile_hitting_fork_window": hits}


def q2_case(case: str, S: int = 100, n_reps: int = 400) -> dict | None:
    dp = DATA / "reports" / f"draws_{case}.npy"
    if not dp.exists():
        return None
    draws = np.load(dp)
    meas = pairwise_pooled_tv(blocks_o(draws, S))
    rng = np.random.default_rng(0)
    T, Sf = draws.shape
    K = int(draws.max()) + 1
    probs = np.stack([np.bincount(row, minlength=K) for row in draws]) / Sf
    sims = []
    for _ in range(n_reps):
        sim = np.stack([rng.multinomial(1, p, size=Sf).argmax(axis=1) for p in probs])
        sims.append(pairwise_pooled_tv(blocks_o(sim, S)))
    sims = np.array(sims)
    p_low = float((sims <= meas).mean())   # quiet side (measured below null)
    p_high = float((sims >= meas).mean())
    return {"case": case, "T": int(T), "S_full": int(Sf), "measured": float(meas),
            "null_mean": float(sims.mean()), "null_sd": float(sims.std()),
            "ratio": float(meas / sims.mean()),
            "p_two_sided": float(2 * min(p_low, p_high)),
            "p_quiet_side": p_low}


def main() -> None:
    res = {"q1_entropy_alignment": [], "q2_mcq_v1_pvalue": []}
    for case in ["virology_5", "college_physics_4", "lsat", "logic_syllog"]:
        r1 = q1_case(case)
        if r1:
            res["q1_entropy_alignment"].append(r1)
            print(f"[Q1 {case}] forks={r1['n_forks']} median_H_pctile={r1['median_pctile']:.2f} "
                  f"top20%={r1['forks_in_top20pct_entropy']} "
                  f"topdecile_hits_forkwin={r1['top_decile_hitting_fork_window']}/{r1['top_decile_entropy_positions']}",
                  flush=True)
        r2 = q2_case(case)
        if r2:
            res["q2_mcq_v1_pvalue"].append(r2)
            print(f"[Q2 {case}] T={r2['T']} measured={r2['measured']:.4f} "
                  f"null={r2['null_mean']:.4f}±{r2['null_sd']:.4f} ratio={r2['ratio']:.3f} "
                  f"p_two_sided={r2['p_two_sided']:.3f}", flush=True)
    with open(DATA / "reports" / "entropy_v1_followup.json", "w") as f:
        json.dump(res, f, indent=1)
    print("DONE -> data/reports/entropy_v1_followup.json", flush=True)


if __name__ == "__main__":
    main()
