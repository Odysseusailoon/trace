"""V5: radix cache effectiveness — sorted-by-t vs random submission order.

Runs the same branch set twice (fresh server state assumed; uses distinct
case prefixes so the radix tree starts cold for each arm), measures wall time
and reads /metrics cache_hit_rate before/after.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

import httpx
from transformers import AutoTokenizer

from forkscope.base_path import build_base_path
from forkscope.client import SGLangClient
from forkscope.config import load_settings
from forkscope.cost import parse_metrics
from forkscope.fork_enum import enumerate_branches, observed_positions


async def arm(client, base, branches, order: str, n_samples: int, max_new: int,
              sequential: bool = False):
    brs = list(branches)
    if order == "random":
        random.Random(0).shuffle(brs)
    else:
        brs.sort(key=lambda b: b.t)
    m0 = parse_metrics((await client.metrics_text()))
    t0 = time.time()
    if sequential:
        # one at a time: radix caching is the only acceleration mechanism
        for b in brs:
            await client.sample_continuations(b.prefix_ids(base), n_samples, max_new,
                                              seed=hash((b.t, b.tok_id)) % 2**31)
    else:
        await asyncio.gather(*(
            client.sample_continuations(b.prefix_ids(base), n_samples, max_new,
                                        seed=hash((b.t, b.tok_id)) % 2**31)
            for b in brs
        ))
    dt = time.time() - t0
    m1 = parse_metrics((await client.metrics_text()))
    return {"order": order, "wall_s": dt,
            "prompt_tokens_delta": m1.get("prompt_tokens", 0) - m0.get("prompt_tokens", 0),
            "cached_tokens_delta": m1.get("cached_tokens", 0) - m0.get("cached_tokens", 0)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-branches", type=int, default=40)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    cfg = load_settings(args.config)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    q = ("A robe takes 2 bolts of blue fiber and half that much white fiber. "
         "How many bolts in total does it take? A) 2 B) 3 C) 4 D) 5")
    results = []
    for arm_name in ["sorted", "random"]:
        # distinct case prefixes => cold radix tree per arm
        msgs = [{"role": "user", "content": f"[case {arm_name}] {q}"}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        async with SGLangClient(cfg.base_url, cfg.concurrency) as client:
            base = await build_base_path(client, f"v5_{arm_name}", ids, 200, 10)
            positions = observed_positions(len(base.gen_ids), 2, "token")
            branches = enumerate_branches(base, positions, 0.05)[: args.n_branches]
            r = await arm(client, base, branches, arm_name, args.samples, args.max_new,
                          sequential=True)
            results.append(r)
            print(json.dumps(r))
    s, r = results
    if s["prompt_tokens_delta"]:
        print(f"[v5] prompt-token ratio random/sorted: "
              f"{r['prompt_tokens_delta'] / s['prompt_tokens_delta']:.2f}x")
    print(f"[v5] wall ratio random/sorted: {r['wall_s'] / s['wall_s']:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
