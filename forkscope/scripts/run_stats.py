"""Run V1/V2/V3 against real dev reference records.

Usage: python scripts/run_stats.py --case lsat [--max-t 400]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forkscope.aggregate import load_branch_records
from forkscope.extractor.mcq import CATEGORIES, MCQExtractor
from forkscope.mixture import build_draw_matrix
from forkscope.stats import v1_verdict, v2_verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-t", type=int, default=0, help="cap positions (speed)")
    ap.add_argument("--n-draws", type=int, default=200)
    args = ap.parse_args()

    rec_path = Path(args.data_dir) / "raw" / f"branches_{args.case}.jsonl"
    records = load_branch_records(rec_path)
    positions = sorted({r["t"] for r in records})
    if args.max_t:
        positions = [p for p in positions if p <= args.max_t]
        records = [r for r in records if r["t"] in set(positions)]
    print(f"[stats] {len(records)} branch records over {len(positions)} positions")

    cat_index = {c: i for i, c in enumerate(CATEGORIES)}
    ext = MCQExtractor()
    draws = build_draw_matrix(records, positions, cat_index, ext,
                              n_total=args.n_draws, seed=0)
    print(f"[stats] draws {draws.shape}")

    v1 = v1_verdict(draws, S=100, n_reps=100)
    print(f"[V1] measured={v1['measured']:.4f} null={v1['null']:.4f} "
          f"ratio={v1['ratio']:.3f} -> {'PASS' if v1['pass'] else 'FAIL'}")

    v2 = v2_verdict(draws, [20, 50, 100, 200])
    print(f"[V2] slope={v2['slope']:.3f} monotone={v2['monotone']} "
          f"null_ratios={[round(r,2) for r in v2['null_ratios']]} "
          f"tail={v2['tail_slope']:.3f} -> {v2['verdict'].upper()}")

    out = Path(args.data_dir) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"stats_{args.case}.json", "w") as f:
        json.dump({"v1": v1, "v2": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                    for k, v in v2.items()}}, f, indent=2, default=float)
    np.save(out / f"draws_{args.case}.npy", draws)
    print(f"[stats] -> {out}/stats_{args.case}.json")


if __name__ == "__main__":
    main()
