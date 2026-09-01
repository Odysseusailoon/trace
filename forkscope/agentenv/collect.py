"""Collect N full-tool-loop episodes per task at temperature 1.0, save JSONL.

Usage (on node):
  venv/bin/python -m agentenv.collect --tasks t7_avg_track_len_min t10_search_plus_calc t4_artist_album_ratio --n 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, "/home/dev/forkscope")

from agentenv.runner import run_episode  # noqa: E402
from agentenv.tasks import TASKS  # noqa: E402


async def collect_task(task: dict, n: int, base_url: str, seed0: int,
                       concurrency: int, outdir: str) -> str:
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(i: int) -> dict:
        nonlocal done
        async with sem:
            try:
                ep = await run_episode(base_url, task, seed=seed0 + i)
            except Exception as e:  # keep the batch alive; record the failure
                ep = {"task_id": task["id"], "seed": seed0 + i, "error": str(e),
                      "steps": [], "messages": [], "final": None}
        done += 1
        if done % 25 == 0:
            print(f"  [{task['id']}] {done}/{n}", flush=True)
        return ep

    eps = await asyncio.gather(*(one(i) for i in range(n)))
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"episodes_{task['id']}.jsonl")
    with open(path, "w") as f:
        for ep in eps:
            f.write(json.dumps(ep) + "\n")
    errs = sum(1 for e in eps if e.get("error"))
    print(f"[collect] {task['id']}: {n} episodes ({errs} errors) -> {path}", flush=True)
    return path


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    by_id = {t["id"]: t for t in TASKS}
    for tid in args.tasks:
        await collect_task(by_id[tid], args.n, args.base_url, args.seed0,
                           args.concurrency, args.outdir)


if __name__ == "__main__":
    asyncio.run(main())
