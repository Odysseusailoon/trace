"""EG-0a quick pass: does 1-p_base gate the forks in Forking Fast's CoT data?

Ground truth: PELT changepoints (their own mult_cost_matrix + pelt_breakpoints)
on counts reconstructed as round(o_t_full * S). Proxy: u_t = 1 - p_base(t) from
stored branch tok_p. Recall@top-k%: changepoint (+-1 window, max-u) falls in the
row's top-k% u positions. Permutation null: same computation on shuffled u
(B=200 per row), reported as null recall.

Quick-pass caveats (vs frozen EG-0 design): fixed pen grid {8,32,128} instead of
their CV; counts reconstructed from published o_t_full rather than raw branch
answers. Both noted for the writeup.

Usage: python3 forkscope/scripts/eg0_cot_recall.py   (from vector/ root, local)
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "forking-fast/otrecon")
from otrecon.fastseg import mult_cost_matrix, pelt_breakpoints  # noqa: E402

KS = [5, 10, 20, 30, 50]
PENS = [8, 32, 128]
WINDOW = 1  # fork token may sit at tau-1..tau+1
B_NULL = 200
rng = np.random.default_rng(0)


def row_series(path):
    d = json.load(gzip.open(path))
    o = np.array(d["o_t_full"], dtype=float)  # [T, K]
    s = d["s"]
    counts = np.rint(o * s).astype(int)
    by_t = defaultdict(dict)
    for b in d["branches"]:
        if b["is_base"]:
            by_t[b["t"]] = b["tok_p"]
    T = o.shape[0]
    u = np.array([1.0 - by_t.get(t, 1.0) for t in range(T)])
    return counts, u


def recall_of(taus, u, k_pct):
    if not taus:
        return None
    T = len(u)
    thr = np.percentile(u, 100 - k_pct)
    hits = 0
    for tau in taus:
        lo, hi = max(0, tau - WINDOW), min(T, tau + WINDOW + 1)
        if u[lo:hi].max() >= thr:
            hits += 1
    return hits, len(taus)


def main():
    out = {}
    for track in ["llama", "deepseek"]:
        rows = sorted(glob.glob(f"forking-fast/data/s200/{track}/row*.json.gz"))
        agg = {pen: {k: [0, 0] for k in KS} for pen in PENS}
        null_agg = {pen: {k: [0, 0] for k in KS} for pen in PENS}
        n_forks = {pen: 0 for pen in PENS}
        n_rows_with_fork = {pen: 0 for pen in PENS}
        for path in rows:
            counts, u = row_series(path)
            T = len(u)
            C = mult_cost_matrix(counts)
            for pen in PENS:
                bps = pelt_breakpoints(C, T, pen, 2)
                taus = [b for b in bps if 0 < b < T]  # interior boundaries
                if not taus:
                    continue
                n_forks[pen] += len(taus)
                n_rows_with_fork[pen] += 1
                for k in KS:
                    h, n = recall_of(taus, u, k)
                    agg[pen][k][0] += h
                    agg[pen][k][1] += n
                # permutation null (subsampled for speed)
                for _ in range(max(1, B_NULL // 20)):
                    up = rng.permutation(u)
                    for k in KS:
                        h, n = recall_of(taus, up, k)
                        null_agg[pen][k][0] += h
                        null_agg[pen][k][1] += n
        out[track] = {
            "n_rows": len(rows),
            "rows_with_fork": n_rows_with_fork,
            "total_forks": n_forks,
            "recall": {pen: {k: (agg[pen][k][0] / agg[pen][k][1]
                               if agg[pen][k][1] else None) for k in KS}
                       for pen in PENS},
            "null_recall": {pen: {k: (null_agg[pen][k][0] / null_agg[pen][k][1]
                                     if null_agg[pen][k][1] else None) for k in KS}
                            for pen in PENS},
        }
        print(f"\n== {track}: {len(rows)} rows ==")
        for pen in PENS:
            r = out[track]["recall"][pen]
            nr = out[track]["null_recall"][pen]
            print(f" pen={pen:>3} rows_with_fork={n_rows_with_fork[pen]:3d} "
                  f"forks={n_forks[pen]:4d}")
            for k in KS:
                rv = r[k]
                nv = nr[k]
                if rv is not None:
                    print(f"   recall@top-{k:>2}%: {rv:.3f}  (null {nv:.3f})")
    with open("forkscope/data/reports/eg0_cot_recall.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote forkscope/data/reports/eg0_cot_recall.json")


if __name__ == "__main__":
    main()
