"""How many tokens does a continuation need before the outcome stops being an
artifact of the cap? Samples from t=0 at a generous cap and reports the length
distribution, the </think> rate, and the label under strict vs loose extraction.

Run before committing to a dense sweep: at 1500 tokens college_physics_4 capped
71.6% of its continuations and 66% of those parsed as Other, so its measured
outcome distribution was mostly a truncation artifact.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from transformers import AutoTokenizer

from forkscope.extractor.mcq import MCQExtractor

SYSTEM = "You are a careful reasoner. End with 'The answer is (X).'"
STRICT = re.compile(r"answer is:?\s*\**\s*\(?([A-D])\)?", re.IGNORECASE)


def strict_label(text: str) -> str:
    """Answer region only: after </think> if the trace closed, else the last
    explicit 'answer is (X)' commitment; no loose 'option (X)' fallback."""
    tail = text.split("</think>")[-1] if "</think>" in text else text
    m = STRICT.findall(tail)
    if m:
        return m[-1].upper()
    return "Unfinished" if "</think>" not in text else "Other"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu", required=True, help="subject:idx")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    args = ap.parse_args()

    from datasets import load_dataset
    subject, idx = args.mmlu.rsplit(":", 1)
    row = list(load_dataset("cais/mmlu", subject, split="test"))[int(idx)]
    q, choices, gold = row["question"], list(row["choices"]), "ABCD"[row["answer"]]

    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": q + "\n" + "\n".join(
                f"{l}) {c}" for l, c in zip("ABCD", choices))}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]

    async with httpx.AsyncClient(base_url=args.base_url, timeout=1800) as c:
        async def one(j):
            r = await c.post("/generate", json={
                "input_ids": ids,
                "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                                    "max_new_tokens": args.max_new_tokens,
                                    "sampling_seed": 424242 + j}})
            r.raise_for_status()
            return r.json()
        outs = await asyncio.gather(*(one(j) for j in range(args.n)))

    lens, loose, strict, closed = [], Counter(), Counter(), 0
    ext = MCQExtractor()
    for o in outs:
        m = o["meta_info"]
        lens.append(m.get("completion_tokens") or 0)
        loose[ext.extract(o["text"])] += 1
        strict[strict_label(o["text"])] += 1
        closed += "</think>" in o["text"]
    lens.sort()
    n = len(lens)

    def q_(p):
        return lens[min(n - 1, int(p * n))]

    print(f"=== {args.mmlu} gold={gold} n={n} cap={args.max_new_tokens} ===")
    print(f"completion_tokens: min {lens[0]} p50 {q_(0.5)} p90 {q_(0.9)} "
          f"p99 {q_(0.99)} max {lens[-1]}")
    print(f"hit the cap: {sum(1 for x in lens if x >= args.max_new_tokens)}/{n}")
    print(f"closed </think>: {closed}/{n}")
    print(f"loose  (current extractor): {dict(loose.most_common())}")
    print(f"strict (answer region only): {dict(strict.most_common())}")
    for cap in (1500, 2048, 3072, 4096, 6144, 8192):
        print(f"  cap {cap:>5}: would truncate {sum(1 for x in lens if x > cap)}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
