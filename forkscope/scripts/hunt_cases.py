"""Hunt for Qwen3-8B failure/disagreement cases in MMLU hard subjects.

Each question: 8 thinking samples (cap 1500). Keep questions whose outcome
distribution is genuinely mixed (top answer prob 0.3-0.7) or confidently WRONG.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

from forkscope.client import SGLangClient
from forkscope.extractor.mcq import MCQExtractor

SUBJECTS = [
    "formal_logic",
    "college_physics",
    "abstract_algebra",
    "college_mathematics",
    "econometrics",
    "virology",
]


async def probe_question(client, tok, ext, q, choices, n=8, max_new=1500):
    msgs = [
        {"role": "system", "content": "You are a careful reasoner. End with 'The answer is (X).'"},
        {"role": "user", "content": q + "\n" + "\n".join(
            f"{l}) {c}" for l, c in zip("ABCD", choices))},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    outs = await client.sample_continuations(ids, n, max_new, seed=hash(q) % 2**31)
    labs = [ext.extract(o["text"]) for o in outs]
    trunc = sum(1 for o in outs
                if (o["meta_info"].get("finish_reason") or {}).get("type") == "length")
    return Counter(labs), trunc


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=15)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="data/cases/candidates.jsonl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ext = MCQExtractor()
    results = []

    async with SGLangClient("http://127.0.0.1:30000", 64) as client:
        assert await client.health()
        for subj in SUBJECTS:
            ds = load_dataset("cais/mmlu", subj, split="test")
            rows = list(ds)[: args.per_subject]

            async def one(i, row):
                q = row["question"]
                choices = row["choices"]
                gold = "ABCD"[row["answer"]]
                dist, trunc = await probe_question(client, tok, ext, q, choices, args.n)
                total = sum(dist.values())
                top, top_n = dist.most_common(1)[0]
                rec = {
                    "subject": subj, "idx": i, "question": q, "choices": choices,
                    "gold": gold, "dist": dict(dist), "top": top,
                    "top_frac": top_n / total, "truncated": trunc,
                    "correct": top == gold,
                }
                return rec

            recs = await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
            for rec in recs:
                results.append(rec)
                tag = ""
                if 0.3 <= rec["top_frac"] <= 0.75:
                    tag = " <-- MIXED"
                elif not rec["correct"] and rec["top_frac"] >= 0.75:
                    tag = " <-- CONF-WRONG"
                print(f"{subj}[{rec['idx']}] gold={rec['gold']} dist={rec['dist']} "
                      f"trunc={rec['truncated']}{tag}", flush=True)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    mixed = [r for r in results if 0.3 <= r["top_frac"] <= 0.75]
    wrong = [r for r in results if not r["correct"] and r["top_frac"] >= 0.75]
    print(f"\n[hunt] {len(results)} probed, {len(mixed)} MIXED, {len(wrong)} CONF-WRONG")
    for r in mixed[:10]:
        print(" MIXED:", r["subject"], r["idx"], r["dist"], "gold", r["gold"])
    for r in wrong[:10]:
        print(" CONF-WRONG:", r["subject"], r["idx"], r["dist"], "gold", r["gold"])


if __name__ == "__main__":
    asyncio.run(main())
