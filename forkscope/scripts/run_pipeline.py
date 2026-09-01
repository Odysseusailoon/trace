"""End-to-end pipeline CLI: config in -> raw o_t figure + records out (M2 scope)."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from forkscope.aggregate import aggregate, load_branch_records
from forkscope.base_path import build_base_path
from forkscope.client import SGLangClient
from forkscope.config import load_settings
from forkscope.extractor.mcq import MCQExtractor, CATEGORIES
from forkscope.fork_enum import enumerate_branches, observed_positions
from forkscope.resampler import Resampler
from forkscope.viz import stacked_area


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--case", required=True, help="case id")
    ap.add_argument("--question", required=True)
    ap.add_argument("--choices", nargs=4, required=True)
    ap.add_argument("--max-base-tokens", type=int, default=800)
    ap.add_argument("--limit-branches", type=int, default=0, help="debug: cap branch count")
    ap.add_argument("--skip-resample", action="store_true")
    args = ap.parse_args()

    cfg = load_settings(args.config)
    data = Path(cfg.data_dir)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    msgs = [
        {"role": "system", "content": "You are a careful reasoner. End with 'The answer is (X).'"},
        {"role": "user", "content": args.question + "\n" + "\n".join(
            f"{ltr}) {ch}" for ltr, ch in zip("ABCD", args.choices))},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]

    async with SGLangClient(cfg.base_url, cfg.concurrency) as client:
        assert await client.health(), "server not healthy"

        # Stage 1
        bp_path = data / "raw" / f"base_{args.case}.json"
        if bp_path.exists():
            from forkscope.base_path import BasePath
            base = BasePath.load(bp_path)
        else:
            base = await build_base_path(client, args.case, ids, args.max_base_tokens,
                                         cfg.fpa.top_logprobs_num)
            base.save(bp_path)
        print(f"[s1] base path: {len(base.gen_ids)} tokens, finish={base.finish_reason}")

        # Stage 2
        positions = observed_positions(len(base.gen_ids), cfg.fpa.spacing, cfg.fpa.spacing_mode)
        branches = enumerate_branches(base, positions, cfg.fpa.branch_prob_threshold)
        if args.limit_branches:
            branches = branches[: args.limit_branches]
        print(f"[s2] {len(positions)} positions -> {len(branches)} branches")

        # Stage 3
        if not args.skip_resample:
            res = Resampler(client, data / "raw")
            rec_path = await res.run(base, branches, cfg.fpa.samples_per_branch,
                                     cfg.fpa.max_continuation_tokens, cfg.fpa.samples_t0)
            print(f"[s3] records -> {rec_path}")

    # Stage 5 (raw aggregate) + figure
    rec_path = data / "raw" / f"branches_{args.case}.jsonl"
    records = load_branch_records(rec_path)
    ext = MCQExtractor()
    o_t, per_draw = aggregate(records, CATEGORIES, ext)
    agg_dir = data / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    np.save(agg_dir / f"o_t_{args.case}.npy", o_t)
    positions = np.array(sorted({r["t"] for r in records}))
    # align: o_t rows are indexed by t directly
    nz = o_t.sum(axis=1) > 0
    fig = stacked_area({"raw": o_t[nz]}, np.arange(len(o_t))[nz], CATEGORIES,
                       out_path=str(agg_dir / f"o_t_{args.case}.png"),
                       title=f"case {args.case} raw o_t")
    print(f"[s5] o_t {o_t.shape} -> {fig}")


if __name__ == "__main__":
    asyncio.run(main())
