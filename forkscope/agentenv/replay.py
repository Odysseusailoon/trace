"""Decision-step replay FPA: the o_t curve at tool-call granularity.

For one recorded episode, at each assistant-turn boundary d (= right before
the d-th assistant message), resample K full continuations (tools execute) and
aggregate the outcome distribution o_d. Also reports strategy persistence:
what fraction of resamples repeat the recorded first decision after d
(Thought Branches' resilience, at decision-step granularity).

Usage (on node):
  venv/bin/python -m agentenv.replay --task t4_artist_album_ratio --pick sql:intdiv --k 50
  venv/bin/python -m agentenv.replay --task t4_artist_album_ratio --pick sql:avg_groupby --k 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter

sys.path.insert(0, "/home/dev/forkscope")

from agentenv.analyze import (canon_cluster, final_number, first_decision,  # noqa: E402
                              gold_value, outcome_label)
from agentenv.runner import continue_episode  # noqa: E402


def boundaries(messages: list[dict]) -> list[int]:
    """Indices i such that messages[:i] is a valid replay prefix (assistant turn starts at i)."""
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def decision_after(messages: list[dict], i: int, task_id: str) -> str:
    """Canonical cluster of the recorded first tool call at/after message index i."""
    for m in messages[i:]:
        if m["role"] == "assistant" and m.get("tool_calls"):
            tc = m["tool_calls"][0]
            args = tc["function"].get("arguments", "")
            try:
                d = json.loads(args) if isinstance(args, str) else args
                args = d.get("query") or d.get("expr") or json.dumps(d)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
            import re
            arg = re.sub(r"\s+", " ", str(args).lower()).strip().rstrip(";")
            return canon_cluster(task_id, tc["function"]["name"], arg)
    return "no_tool"


async def replay_episode(base_url: str, ep: dict, task_id: str, k: int,
                         concurrency: int = 24, seed0: int = 50000) -> dict:
    msgs = ep["messages"]
    g = gold_value(task_id)
    bs = boundaries(msgs)
    sem = asyncio.Semaphore(concurrency)
    out_rows = []

    for d, i in enumerate(bs):
        prefix = msgs[:i]
        recorded_next = decision_after(msgs, i, task_id)
        # round number for MAX_ROUNDS accounting: count assistant turns already in prefix
        start_round = sum(1 for m in prefix if m["role"] == "assistant")

        async def one(j: int):
            async with sem:
                try:
                    cont = await continue_episode(base_url, prefix, seed=seed0 + d * 1000 + j,
                                                  start_round=start_round)
                except Exception as e:
                    return {"error": str(e)}
            v = final_number(cont.get("final"))
            tool, arg = first_decision(cont)
            return {"outcome": outcome_label(task_id, v, g),
                    "next_decision": canon_cluster(task_id, tool, arg)}

        conts = await asyncio.gather(*(one(j) for j in range(k)))
        ok = [c for c in conts if "error" not in c]
        o_d = Counter(c["outcome"] for c in ok)
        persist = (sum(1 for c in ok if c["next_decision"] == recorded_next) / len(ok)) if ok else None
        row = {"d": d, "msg_index": i, "recorded_next": recorded_next,
               "k_ok": len(ok), "o_d": dict(o_d),
               "p_correct": o_d.get("correct", 0) / len(ok) if ok else None,
               "persistence": persist,
               "switch_rate": 1 - persist if persist is not None else None}
        out_rows.append(row)
        print(f"  d={d} (msg {i}) recorded={recorded_next} "
              f"p_correct={row['p_correct']:.2f} persist={persist:.2f} o_d={dict(o_d)}",
              flush=True)

    return {"task_id": task_id, "episode_seed": ep["seed"],
            "episode_outcome": outcome_label(task_id, final_number(ep.get("final")), g),
            "k": k, "steps": out_rows}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--pick", required=True,
                    help="first-decision cluster to pick a representative episode from")
    ap.add_argument("--outcome", default=None,
                    help="additionally require this episode outcome (e.g. intdiv_1.000)")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--indir", default="data/raw")
    ap.add_argument("--outdir", default="data/reports")
    args = ap.parse_args()

    eps = [json.loads(l) for l in open(f"{args.indir}/episodes_{args.task}.jsonl")]
    eps = [e for e in eps if not e.get("error")]
    g = gold_value(args.task)

    def cluster_of(ep):
        tool, arg = first_decision(ep)
        return canon_cluster(args.task, tool, arg)

    cand = [e for e in eps if cluster_of(e) == args.pick]
    if args.outcome:
        cand = [e for e in cand
                if outcome_label(args.task, final_number(e.get("final")), g) == args.outcome]
    if not cand:
        sys.exit(f"no episode with first decision {args.pick} (+outcome {args.outcome})")
    ep = cand[0]
    print(f"replaying episode seed={ep['seed']} ({len(cand)} candidates)", flush=True)

    res = await replay_episode(args.base_url, ep, args.task, args.k)
    import os
    os.makedirs(args.outdir, exist_ok=True)
    tag = args.pick.replace(":", "_").replace("/", "_")
    path = f"{args.outdir}/replay_{args.task}_{tag}_{ep['seed']}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[replay] -> {path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
