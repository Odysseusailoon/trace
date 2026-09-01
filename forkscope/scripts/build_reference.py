"""Build dev reference set: S=200, N=1 for the 2 dev cases.

Runs full pipeline with reference config; records land in data/raw/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from transformers import AutoTokenizer

from forkscope.base_path import build_base_path
from forkscope.client import SGLangClient
from forkscope.config import load_settings
from forkscope.fork_enum import enumerate_branches, observed_positions
from forkscope.resampler import Resampler

CASES = {
    "lsat": {
        "q": "All cats are mammals. Some pets are cats. Therefore:",
        "choices": ["All pets are mammals", "Some pets are mammals",
                    "No pets are mammals", "Some mammals are not pets"],
        "max_base": 400,
        "thinking": False,
        "spacing": 2,
        "max_continuation": 512,
    },
    "logic_syllog": {
        "q": "No reptiles are mammals. Some snakes are reptiles. Which must be true?",
        "choices": ["Some snakes are not mammals", "All snakes are mammals",
                    "No snakes are mammals", "Some mammals are snakes"],
        "max_base": 400,
        "thinking": False,
        "spacing": 2,
        "max_continuation": 512,
    },
}


async def run_case(cfg, client, tok, case_id, spec):
    msgs = [
        {"role": "system", "content": "You are a careful reasoner. Be brief. End with 'The answer is (X).'"},
        {"role": "user", "content": spec["q"] + "\n" + "\n".join(
            f"{l}) {c}" for l, c in zip("ABCD", spec["choices"]))},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=spec.get("thinking", False))
    ids = tok(text, add_special_tokens=False)["input_ids"]

    data = Path(cfg.data_dir)
    bp_path = data / "raw" / f"base_{case_id}.json"
    if bp_path.exists():
        from forkscope.base_path import BasePath
        base = BasePath.load(bp_path)
    else:
        base = await build_base_path(client, case_id, ids, spec["max_base"],
                                     cfg.fpa.top_logprobs_num)
        base.save(bp_path)
    print(f"[ref:{case_id}] base {len(base.gen_ids)} tokens finish={base.finish_reason}", flush=True)

    positions = observed_positions(len(base.gen_ids), spec.get("spacing", cfg.reference.spacing), "token")
    branches = enumerate_branches(base, positions, cfg.fpa.branch_prob_threshold)
    print(f"[ref:{case_id}] {len(positions)} pos -> {len(branches)} branches", flush=True)

    res = Resampler(client, data / "raw")
    # no early stop: answers take many forms ("**B) ...**", "The answer is (B)");
    # cap continuation length per case to bound the verbose-tail cost
    await res.run(base, branches, cfg.reference.samples_per_branch,
                  spec.get("max_continuation", cfg.fpa.max_continuation_tokens),
                  cfg.fpa.samples_t0)
    print(f"[ref:{case_id}] DONE", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cases", default="lsat")
    args = ap.parse_args()
    cfg = load_settings(args.config)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    async with SGLangClient(cfg.base_url, cfg.concurrency) as client:
        assert await client.health()
        for cid in args.cases.split(","):
            await run_case(cfg, client, tok, cid, CASES[cid])


if __name__ == "__main__":
    asyncio.run(main())
