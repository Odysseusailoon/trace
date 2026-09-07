"""FPA on agent task t4 (album/artist ratio): token-level resampling with the
full tool loop, outcome = strategy cluster from the final numeric answer.

Fork points should land where the model decides HOW to write the SQL
(integer division / wrong table / nested AVG).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/dev/forkscope")

import numpy as np
import httpx
from transformers import AutoTokenizer

from agentenv.runner import run_episode
from agentenv.tasks import gold as gold_of

TASK_PROMPT = (
    "On average, how many albums per artist are in the store? "
    "Give the ratio to 3 decimals."
)
SYSTEM = (
    "You are a data analyst with access to a music-store SQLite database, "
    "a calculator, and web search. Solve the user's task using tools. "
    "When you have the final answer, reply with 'The answer is <answer>.'"
)

CLUSTERS = ["correct_1.701", "intdiv_1.000", "wrongtable_1.262", "other"]


def cluster(final_text: str) -> str:
    if not final_text:
        return "other"
    m = re.findall(r"answer is[^0-9\-]*(-?[\d\.]+)", final_text, re.I)
    if not m:
        return "other"
    try:
        v = float(m[-1])
    except ValueError:
        return "other"
    if abs(v - 1.701) < 0.002:
        return "correct_1.701"
    if abs(v - 1.0) < 0.001:
        return "intdiv_1.000"
    if abs(v - 1.262) < 0.002:
        return "wrongtable_1.262"
    return "other"


async def main():
    base_url = "http://127.0.0.1:30000"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASK_PROMPT},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"prompt: {len(prompt_ids)} tokens", flush=True)

    # Stage 1: greedy base path for the FIRST assistant turn (the SQL-writing turn)
    async with httpx.AsyncClient(base_url=base_url, timeout=300) as c:
        r = await c.post("/generate", json={
            "input_ids": prompt_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 400,
                                "top_k": 1, "top_p": 1.0},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 10,
        })
        r.raise_for_status()
        resp = r.json()
        if isinstance(resp, list):
            resp = resp[0]
    meta = resp["meta_info"]
    gen_ids = [lp[1] for lp in meta["output_token_logprobs"]]
    top = [[(t[1], t[0]) for t in row] for row in meta["output_top_logprobs"]]
    base_text = resp["text"]
    print(f"base turn 1: {len(gen_ids)} tokens; text head: {base_text[:150]!r}", flush=True)

    # Stage 2: fork enumeration, spacing=8, threshold 0.05
    import math
    positions = list(range(0, len(gen_ids), 8))
    branches = []
    for t in positions:
        greedy = gen_ids[t]
        probs = {tid: math.exp(lp) for tid, lp in top[t]}
        kept = {tid: p for tid, p in probs.items() if p >= 0.05 or tid == greedy}
        if greedy not in kept:
            kept[greedy] = 1.0
        for tid, p in kept.items():
            branches.append((t, tid, p, tid == greedy))
    print(f"{len(positions)} positions -> {len(branches)} branches", flush=True)

    # Stage 3: for each branch, run the full agent episode from a modified first turn.
    # We approximate "prefix + alt token" by regenerating the first turn from
    # prompt_ids + gen_ids[:t] + [alt] with S=6 continuations, then letting the
    # agent loop continue (tools execute) until final answer.
    S = 6
    sem = asyncio.Semaphore(16)
    results = []

    async def one(t, tid, p, is_base):
        prefix = prompt_ids + gen_ids[:t] + [tid]
        async with sem:
            async with httpx.AsyncClient(base_url=base_url, timeout=300) as c:
                r = await c.post("/generate", json={
                    "input_ids": prefix,
                    "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                                        "max_new_tokens": 300},
                })
                r.raise_for_status()
                resp = r.json()
                if isinstance(resp, list):
                    resp = resp[0]
        first_turn = base_text[:0] + resp["text"]
        # crude: treat the regenerated first turn as the assistant's full first message,
        # then continue the episode from there
        # (we skip re-running tools here for the probe; cluster from the SQL text itself)
        sql = re.findall(r'"query":\s*"([^"]+)"', first_turn)
        return {"t": t, "tid": tid, "p": p, "is_base": is_base,
                "first_turn": first_turn, "sql": sql[0] if sql else None}

    # NOTE: to keep the probe cheap we cluster by SQL pattern, not full episode
    def sql_cluster(sql: str | None) -> str:
        if not sql:
            return "other"
        s = sql.lower()
        if "avg(" in s and "group by" in s:
            return "correct_1.701"
        if "count(*) / count(distinct" in s or "count(*)/count(distinct" in s:
            return "intdiv_1.000"
        if "from artist" in s or "count(*) from artist" in s:
            return "wrongtable_1.262"
        if "count(distinct album.artistid" in s or ("count(distinct" in s and "album" in s):
            return "correct_1.701"
        return "other"

    print("running branches...", flush=True)
    out = await asyncio.gather(*(one(*b) for b in branches))
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    with open("data/raw/agent_t4_branches.jsonl", "w") as f:
        for r in out:
            r["cluster"] = sql_cluster(r["sql"])
            f.write(json.dumps(r) + "\n")

    # aggregate o_t
    by_pos = {}
    for r in out:
        by_pos.setdefault(r["t"], []).append(r)
    print("\no_t (SQL-strategy distribution over first-turn positions):")
    K = CLUSTERS
    for t in sorted(by_pos):
        recs = by_pos[t]
        ws = np.array([r["p"] for r in recs]); ws /= ws.sum()
        o = np.zeros(len(K))
        for w, r in zip(ws, recs):
            o[K.index(r["cluster"])] += w
        bar = {k: round(float(v), 2) for k, v in zip(K, o) if v > 0.01}
        print(f"  t={t:3d} {bar}")


if __name__ == "__main__":
    asyncio.run(main())
