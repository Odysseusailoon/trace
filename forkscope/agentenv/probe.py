"""Probe the 10 agent tasks x 8 samples on the local SGLang server."""
import asyncio
import json
import sys

sys.path.insert(0, "/home/dev/forkscope")

from agentenv.runner import probe
from agentenv.tasks import tasks_with_gold


async def main():
    tasks = tasks_with_gold()
    out = []
    async for rec in probe("http://127.0.0.1:30000", tasks, n=8):
        out.append(rec)
    with open("/home/dev/forkscope/data/cases/agent_probe.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("[probe] wrote", len(out), "task results")


if __name__ == "__main__":
    asyncio.run(main())
