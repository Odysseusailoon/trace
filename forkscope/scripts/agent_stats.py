"""V1 (multinomial null) + V2 (1/sqrt(S) law) on agent episode outcome draws.

The S=200 episodes per task are S independent draws from the same step-0
prefix -> a (T=1, S=200) label matrix, exactly what stats.v1_verdict expects.

Usage (on node):
  venv/bin/python scripts/agent_stats.py --tasks t4_artist_album_ratio t7_avg_track_len_min t10_search_plus_calc
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "/home/dev/forkscope")

import numpy as np  # noqa: E402

from agentenv.analyze import final_number, gold_value, outcome_label  # noqa: E402
from forkscope.stats import v1_verdict, v2_verdict  # noqa: E402


def draws_for(task_id: str, indir: str) -> tuple[np.ndarray, list[str]]:
    eps = [json.loads(l) for l in open(f"{indir}/episodes_{task_id}.jsonl")]
    eps = [e for e in eps if not e.get("error")]
    g = gold_value(task_id)
    labels = [outcome_label(task_id, final_number(e.get("final")), g) for e in eps]
    cats = sorted(set(labels))
    idx = {c: i for i, c in enumerate(cats)}
    return np.array([[idx[l] for l in labels]]), cats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--indir", default="data/raw")
    ap.add_argument("--outdir", default="data/reports")
    args = ap.parse_args()

    all_res = {}
    for tid in args.tasks:
        draws, cats = draws_for(tid, args.indir)
        S_full = draws.shape[1]
        if len(cats) < 2:
            all_res[tid] = {"S_full": S_full, "categories": cats,
                            "degenerate": True}
            print(f"{tid}: S={S_full} DEGENERATE (single outcome {cats}) — "
                  f"V1/V2 undefined, task has no outcome variance", flush=True)
            continue
        v1 = v1_verdict(draws, S=20, n_reps=300)
        s_vals = [s for s in (10, 20, 25, 50, 100) if S_full // s >= 2]
        try:
            v2 = v2_verdict(draws, s_vals)
        except ZeroDivisionError:
            v2 = {"verdict": "undefined (near-degenerate: null TVD 0)"}
        all_res[tid] = {"S_full": S_full, "categories": cats, "v1": v1, "v2": v2}
        print(f"{tid}: S={S_full} cats={cats}")
        print(f"  V1 ratio={v1['ratio']:.3f} pass={v1['pass']}")
        if "slope" in v2:
            print(f"  V2 slope={v2['slope']:.3f} verdict={v2['verdict']} "
                  f"null_ratios={[round(r, 3) for r in v2['null_ratios']]}", flush=True)
        else:
            print(f"  V2 {v2['verdict']}", flush=True)

    import os
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/agent_v1v2.json", "w") as f:
        json.dump(all_res, f, indent=1, default=float)
    print(f"[agent_stats] -> {args.outdir}/agent_v1v2.json")


if __name__ == "__main__":
    main()
