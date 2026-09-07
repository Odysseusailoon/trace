"""A4: RSI patch-seed robustness (preregistered, prereg-ablations-e2b.md A4).

Hypothesis: t4 27%->100% and "sticky note harms" are robust to patch-generation
randomness. 5 patch seeds x 100ep validation each. Prediction: every seed's
patched arm intdiv rate < 10% (vs 45.5% control).

Run on node: /home/dev/venv/bin/python -u scripts/rsi_patch_seeds.py
"""
import asyncio
import json
import sys

sys.path.insert(0, "/home/dev/forkscope")

import httpx  # noqa: E402
from scripts.rsi_loop import build_patched_tools, run_episode, TOOLS  # noqa: E402
from scripts.rsi_noleak import (T4_CONSTRAINT, T4_FORBIDDEN, T4_REPORT,  # noqa: E402
                                T4_TASK, classify_t4, outcome_t4, leaks)
from agentenv.runner import run_episode as run_episode_native  # noqa: E402
from agentenv.analyze import gold_value, outcome_label, first_decision, canon_cluster  # noqa: E402
from agentenv.tasks import TASKS  # noqa: E402
from collections import Counter  # noqa: E402

T4_TASK_DICT = next(t for t in TASKS if t["id"] == "t4_artist_album_ratio")

BASE_URL = "http://127.0.0.1:30000"
OUT = "/home/dev/forkscope/data/reports/rsi_patch_seeds.json"
N_SEEDS = 5
N_EP = 100


async def gen_patch_seeded(seed: int) -> dict:
    prompt = (T4_REPORT + "\n\nYou are the agent being evaluated. Write a short warning "
              "note (max 60 words) that will be APPENDED to the sql_query tool description, "
              "so that you would no longer fall into this failure mode. "
              + T4_CONSTRAINT + " Return ONLY the note text.")
    history = [{"role": "user", "content": prompt}]
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=300) as c:
        for attempt in range(3):
            r = await c.post("/v1/chat/completions", json={
                "model": "default", "messages": history,
                "temperature": 0.6, "max_tokens": 600,
                "sampling_seed": seed,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            bad = leaks(text, T4_FORBIDDEN)
            if not bad:
                full = TOOLS[0]["function"]["description"] + "\n\nNote: " + text
                return {"seed": seed, "note": text, "text": full, "attempts": attempt + 1}
            history += [{"role": "assistant", "content": text},
                        {"role": "user", "content":
                         f"Your description leaked forbidden content: {bad}. "
                         "Rewrite it. Warn about the failure mode ONLY - absolutely no "
                         "correct approach, pattern, formula, or factor."}]
    raise SystemExit(f"PATCH_LEAK_UNFIXABLE seed={seed}: {bad}")


async def validate_arm(patch: dict) -> dict:
    import agentenv.tools as T
    orig = T.TOOLS[0]["function"]["description"]
    T.TOOLS[0]["function"]["description"] = patch["text"]
    task = dict(T4_TASK_DICT)
    sem = asyncio.Semaphore(24)

    async def one(i):
        async with sem:
            try:
                ep = await run_episode_native(
                    BASE_URL, task, seed=hash((f"patchseed{patch['seed']}", i)) % 2**31)
            except Exception as e:
                return {"i": i, "error": str(e)}
        tool, arg = first_decision(ep)
        out = outcome_label("t4_artist_album_ratio",
                            __import__("agentenv.analyze", fromlist=["final_number"])
                            .final_number(ep.get("final")),
                            gold_value("t4_artist_album_ratio"))
        return {"i": i, "strat": canon_cluster("t4_artist_album_ratio", tool, arg),
                "final": ep.get("final"), "outcome": out}

    try:
        rows = await asyncio.gather(*(one(i) for i in range(N_EP)))
    finally:
        T.TOOLS[0]["function"]["description"] = orig
    ok = [r for r in rows if "error" not in r]
    outs = Counter(r["outcome"] for r in ok)
    print(f"[patch seed={patch['seed']}] n={len(ok)} outcomes={dict(outs)} "
          f"note={patch['note'][:80]!r}", flush=True)
    return {"seed": patch["seed"], "note": patch["note"], "attempts": patch["attempts"],
            "n": len(ok), "outcomes": dict(outs)}


async def main():
    results = []
    for s in range(N_SEEDS):
        p = await gen_patch_seeded(70000 + s)
        results.append(await validate_arm(p))
    intdiv_rates = [r["outcomes"].get("intdiv_1.000", 0) / max(r["n"], 1) for r in results]
    verdict = {"per_seed": results,
               "intdiv_rates": intdiv_rates,
               "hypothesis_holds": all(r < 0.10 for r in intdiv_rates)}
    with open(OUT, "w") as f:
        json.dump(verdict, f, indent=1)
    print(f"[A4 VERDICT] intdiv rates={['%.2f' % r for r in intdiv_rates]} "
          f"hypothesis_holds={verdict['hypothesis_holds']}", flush=True)
    print("A4_DONE", flush=True)


asyncio.run(main())
