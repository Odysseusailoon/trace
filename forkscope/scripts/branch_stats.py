"""Between-branch dispersion from recorded branch data. No GPU, no new sampling.

Reads data/raw/branches_<case>.jsonl (plus base_<case>.json for token strings)
and reports, per position, how far apart the branches' outcome distributions
are, rather than how far the probability-weighted mixture moved between
adjacent positions. See src/forkscope/branchstat.py for why the two differ.

Usage:
  python scripts/branch_stats.py --case virology_5
  python scripts/branch_stats.py --case virology_5 --model Qwen/Qwen3-8B --tag _36b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forkscope.aggregate import load_branch_records          # noqa: E402
from forkscope.branchstat import branch_stats, summarize     # noqa: E402
from forkscope.extractor.mcq import CATEGORIES, MCQExtractor  # noqa: E402


def load_tokens(base_path: Path, model: str | None):
    """gen token strings, or None if the base path / tokenizer is unavailable."""
    if not base_path.exists():
        return None
    from forkscope.base_path import BasePath
    base = BasePath.load(base_path)
    strs = getattr(base, "gen_strs", None)
    if strs and any(strs):
        return list(strs)
    if not model:
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    return tok.convert_ids_to_tokens(base.gen_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default=None, help="tokenizer for token strings")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--persist-max", type=float, default=0.9)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--screen-sims", type=int, default=2000)
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data = Path(args.data_dir)
    recs = load_branch_records(data / "raw" / f"branches_{args.case}.jsonl")
    rows = branch_stats(recs, CATEGORIES, MCQExtractor(),
                        n_sims=args.sims, screen_sims=args.screen_sims,
                        alpha=args.alpha, persist_max=args.persist_max)
    summ = summarize(rows, alpha=args.alpha)

    toks = load_tokens(data / "raw" / f"base_{args.case}.json", args.model)
    if toks:
        for r in rows:
            r["tok"] = toks[r["t"]] if r["t"] < len(toks) else None

    print(f"=== {args.case} ===")
    print(f"positions {summ['n_positions']}  testable (>=2 branches) {summ['n_testable']}")
    print(f"Bonferroni alpha = {summ['alpha_bonferroni']:.2e}")
    print(f"forking  {summ['n_forking']:4d}/{summ['n_testable']:<4d} = {summ['forking_rate']:6.1%}")
    print(f"decision {summ['n_decision']:4d}/{summ['n_testable']:<4d} = {summ['decision_rate']:6.1%}"
          f"   (decision | forking = {summ['decision_given_forking']:.1%})")
    print(f"fork_score mean {summ['fork_score_mean']:.3f}  p90 {summ['fork_score_p90']:.3f}  "
          f"max {summ['fork_score_max']:.3f}")

    ranked = sorted([r for r in rows if r["n_branches"] >= 2],
                    key=lambda r: -r["fork_score"])[: args.top]
    print(f"\ntop {len(ranked)} by fork_score:")
    print(f"{'t':>5} {'tok':<18} {'B':>2} {'n/br':>10} {'p_grdy':>7} "
          f"{'fork':>6} {'gap':>6} {'disp':>6} {'p':>9}  flags")
    for r in ranked:
        flags = ("FORK" if r["forking"] else "    ") + (" DECISION" if r["decision"] else "")
        ns = ",".join(str(x) for x in r["n_per_branch"])
        print(f"{r['t']:>5} {str(r.get('tok'))[:18]:<18} {r['n_branches']:>2} {ns:>10} "
              f"{r['p_greedy']:>7.3f} {r['fork_score']:>6.3f} {r['greedy_gap']:>6.3f} "
              f"{r['dispersion']:>6.3f} {r['p_fork']:>9.2e}  {flags}")

    out = data / "reports" / f"branchstat_{args.case}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"case_id": args.case, "summary": summ, "rows": rows}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
