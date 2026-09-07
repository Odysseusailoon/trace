"""End-to-end pipeline CLI: config in -> branch records, o_t figure, fork stats out.

Two curves come out of a run and they answer different questions:

  mixture o_t     the probability-weighted average over branches at t. Adjacent-
                  position TVD on it is the original fork rule, and it is blind
                  to a fork carried by a low-probability branch (branchstat.py).
  fork_score(t)   how far apart the branches at t actually are. Per-position, so
                  it needs no neighbour and does not depend on spacing.

Both are computed and reported. --plan-only stops after enumeration and prints
the sampling bill, which is worth reading before committing to a dense sweep.

Usage:
  python scripts/run_pipeline.py --config configs/dense.yaml --case virology_5 \
      --mmlu virology:5 --data-dir data/mcq8b --plan-only
  python scripts/run_pipeline.py --config configs/dense.yaml --case virology_5 \
      --mmlu virology:5 --data-dir data/mcq8b
  # fine pass in a window flagged by the coarse pass (records append, resumable)
  python scripts/run_pipeline.py --config configs/dense.yaml --case virology_5 \
      --mmlu virology:5 --data-dir data/mcq8b --spacing 1 --tmin 180 --tmax 240
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from transformers import AutoTokenizer

from collections import Counter

from forkscope.aggregate import aggregate, load_branch_records
from forkscope.base_path import BasePath, build_base_path
from forkscope.branchstat import branch_stats, summarize
from forkscope.client import SGLangClient
from forkscope.config import load_settings
from forkscope.extractor.mcq import CATEGORIES as LOOSE_CATEGORIES, MCQExtractor
from forkscope.extractor.mcq_strict import CATEGORIES as STRICT_CATEGORIES, StrictMCQExtractor
from forkscope.fork_enum import enumerate_branches, observed_positions
from forkscope.resampler import Resampler
from forkscope.viz import stacked_area

SYSTEM = "You are a careful reasoner. End with 'The answer is (X).'"


def mmlu_case(spec: str) -> tuple[str, list[str], str]:
    """'virology:5' -> (question, choices, gold). Same rows hunt_cases.py indexed."""
    from datasets import load_dataset
    subject, idx = spec.rsplit(":", 1)
    row = list(load_dataset("cais/mmlu", subject, split="test"))[int(idx)]
    return row["question"], list(row["choices"]), "ABCD"[row["answer"]]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dense.yaml")
    ap.add_argument("--case", required=True, help="case id")
    ap.add_argument("--mmlu", default=None, help="subject:idx, e.g. virology:5")
    ap.add_argument("--question", default=None)
    ap.add_argument("--choices", nargs=4, default=None)
    ap.add_argument("--data-dir", default=None, help="override cfg.data_dir (one per model)")
    ap.add_argument("--model", default=None, help="override cfg.model (tokenizer + served model)")
    ap.add_argument("--max-base-tokens", type=int, default=1500)
    ap.add_argument("--spacing", type=int, default=None)
    ap.add_argument("--tmin", type=int, default=None)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--samples-per-branch", type=int, default=None)
    ap.add_argument("--limit-branches", type=int, default=0, help="debug: cap branch count")
    ap.add_argument("--plan-only", action="store_true",
                    help="stop after enumeration; print the sampling bill")
    ap.add_argument("--skip-resample", action="store_true")
    ap.add_argument("--stats-only", action="store_true", help="recompute stats from records")
    ap.add_argument("--extractor", default="terminal",
                    choices=["terminal", "commitment", "loose"],
                    help="outcome definition; see extractor/mcq_strict.py")
    args = ap.parse_args()

    cfg = load_settings(args.config)
    model = args.model or cfg.model
    spacing = args.spacing if args.spacing is not None else cfg.fpa.spacing
    S = args.samples_per_branch or cfg.fpa.samples_per_branch
    data = Path(args.data_dir) if args.data_dir else Path(cfg.data_dir)

    if args.mmlu:
        question, choices, gold = mmlu_case(args.mmlu)
    elif args.question and args.choices:
        question, choices, gold = args.question, args.choices, None
    else:
        ap.error("need --mmlu SUBJECT:IDX or --question with --choices")
    print(f"[case] {args.case} gold={gold} model={model}")
    print(f"[case] {question[:120]}")

    tok = AutoTokenizer.from_pretrained(model)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question + "\n" + "\n".join(
            f"{ltr}) {ch}" for ltr, ch in zip("ABCD", choices))},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]

    rec_path = data / "raw" / f"branches_{args.case}.jsonl"
    if not args.stats_only:
        async with SGLangClient(cfg.base_url, cfg.concurrency) as client:
            assert await client.health(), "server not healthy"

            # Stage 1: greedy base path with top-k candidates
            bp_path = data / "raw" / f"base_{args.case}.json"
            if bp_path.exists():
                base = BasePath.load(bp_path)
                print(f"[s1] reusing {bp_path}")
            else:
                base = await build_base_path(client, args.case, ids, args.max_base_tokens,
                                             cfg.fpa.top_logprobs_num)
                base.save(bp_path)
            print(f"[s1] base path: {len(base.gen_ids)} tokens, finish={base.finish_reason}")

            # Stage 2: enumerate branches
            positions = observed_positions(len(base.gen_ids), spacing, cfg.fpa.spacing_mode,
                                           args.tmin, args.tmax)
            branches = enumerate_branches(base, positions, cfg.fpa.branch_prob_threshold)
            if args.limit_branches:
                branches = branches[: args.limit_branches]
            n_alt = sum(1 for b in branches if not b.is_base)
            n_testable = len({b.t for b in branches if not b.is_base})
            print(f"[s2] {len(positions)} positions -> {len(branches)} branches "
                  f"({n_alt} counterfactual, {len(branches)/max(len(positions),1):.2f}/position) "
                  f"at threshold {cfg.fpa.branch_prob_threshold}")
            print(f"[s2] positions with >=2 branches (testable): {n_testable} "
                  f"= {n_testable/max(len(positions),1):.0%}")

            est_tok = len(branches) * S * cfg.fpa.max_continuation_tokens
            print(f"[plan] {len(branches)} x S={S} = {len(branches)*S:,} continuations, "
                  f"<= {est_tok/1e6:.0f}M generated tokens (cap; typical run is ~55% of it)")
            if args.plan_only:
                return

            # Stage 3: resample (resumable; appends to branches_<case>.jsonl)
            if not args.skip_resample:
                res = Resampler(client, data / "raw")
                rec_path = await res.run(base, branches, S,
                                         cfg.fpa.max_continuation_tokens, cfg.fpa.samples_t0)
                print(f"[s3] records -> {rec_path}")

    def make_ext(name):
        if name == "loose":
            return MCQExtractor(), LOOSE_CATEGORIES
        return StrictMCQExtractor(name), STRICT_CATEGORIES

    # Stage 4: mixture o_t (the original curve) + figure
    records = load_branch_records(rec_path)
    ext, cats = make_ext(args.extractor)
    o_t, _ = aggregate(records, cats, ext)
    agg = data / "aggregated"
    agg.mkdir(parents=True, exist_ok=True)
    np.save(agg / f"o_t_{args.case}.npy", o_t)
    nz = o_t.sum(axis=1) > 0
    stacked_area({"raw": o_t[nz]}, np.arange(len(o_t))[nz], cats,
                 out_path=str(agg / f"o_t_{args.case}.png"),
                 title=f"case {args.case} mixture o_t")
    print(f"[s4] mixture o_t {o_t.shape} -> {agg}/o_t_{args.case}.png")

    # Stage 5: between-branch dispersion (the curve the mixture hides).
    # Run under every outcome definition; the one named by --extractor leads,
    # the others are the robustness check that the verdict is not an artifact
    # of how a truncated trace was labelled.
    bs = cfg.branchstat
    alt = {}
    for name in ("terminal", "commitment", "loose"):
        e, c = make_ext(name)
        rws = branch_stats(records, c, e, n_sims=bs.sims, screen_sims=bs.screen_sims,
                           alpha=bs.alpha, persist_max=bs.persist_max)
        alt[name] = {"rows": rws, "summary": summarize(rws, alpha=bs.alpha)}
    rows = alt[args.extractor]["rows"]
    summ = alt[args.extractor]["summary"]
    base = BasePath.load(data / "raw" / f"base_{args.case}.json")
    toks = tok.convert_ids_to_tokens(base.gen_ids)
    for r in rows:
        r["tok"] = toks[r["t"]] if r["t"] < len(toks) else None

    # the mixture's own verdict on the same records, for the side-by-side
    o_nz = o_t[nz]
    d_adj = np.abs(o_nz[1:] - o_nz[:-1]).sum(axis=1) / 2
    n_mix = int((d_adj > cfg.fork_threshold).sum())

    lab = Counter()
    for r in records:
        for txt in r["continuations"]:
            lab[ext.extract(txt)] += 1
    ntot = sum(lab.values())
    print(f"\n[s5] outcome mix ({args.extractor}, n={ntot}): "
          f"{ {k: f'{v/ntot:.1%}' for k, v in lab.most_common()} }")
    print(f"[s5] {args.case}: {summ['n_testable']} testable of {summ['n_positions']} positions, "
          f"Bonferroni alpha={summ['alpha_bonferroni']:.2e}")
    print(f"     forking  {summ['n_forking']:4d} = {summ['forking_rate']:6.1%}")
    print(f"     decision {summ['n_decision']:4d} = {summ['decision_rate']:6.1%}"
          f"  (decision | forking = {summ['decision_given_forking']:.1%})")
    print(f"     mixture adjacent-TVD > {cfg.fork_threshold}: {n_mix} positions "
          f"<- what the old pipeline reported")
    print(f"     fork_score mean {summ['fork_score_mean']:.3f} "
          f"p90 {summ['fork_score_p90']:.3f} max {summ['fork_score_max']:.3f}")
    for name, a in alt.items():
        if name != args.extractor:
            s_ = a["summary"]
            print(f"     [robustness: {name}] forking {s_['n_forking']} "
                  f"decision {s_['n_decision']} of {s_['n_testable']} testable")

    top = sorted([r for r in rows if r["n_branches"] >= 2], key=lambda r: -r["fork_score"])[:20]
    print(f"\n{'t':>5} {'tok':<16} {'B':>2} {'p_grdy':>7} {'fork':>6} {'gap':>6} "
          f"{'disp':>6} {'p':>9}  flags")
    for r in top:
        print(f"{r['t']:>5} {str(r['tok'])[:16]:<16} {r['n_branches']:>2} {r['p_greedy']:>7.3f} "
              f"{r['fork_score']:>6.3f} {r['greedy_gap']:>6.3f} {r['dispersion']:>6.3f} "
              f"{r['p_fork']:>9.2e}  "
              f"{'FORK' if r['forking'] else '    '}{' DECISION' if r['decision'] else ''}")

    out = data / "reports" / f"branchstat_{args.case}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "case_id": args.case, "model": model, "gold": gold,
        "spacing": spacing, "samples_per_branch": S,
        "branch_prob_threshold": cfg.fpa.branch_prob_threshold,
        "mixture_forks_at_threshold": n_mix, "mixture_fork_threshold": cfg.fork_threshold,
        "extractor": args.extractor,
        "max_continuation_tokens": cfg.fpa.max_continuation_tokens,
        "outcome_mix": {k: v / ntot for k, v in lab.items()},
        "summary": summ, "rows": rows,
        "robustness": {k: v["summary"] for k, v in alt.items()},
    }, indent=1))
    print(f"\n[s5] -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
