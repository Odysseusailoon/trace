"""RSI minimal loop: fork report -> agent patches its own tool schema -> retest.

Arm A (control): original tools, 200 episodes, intdiv rate.
Arm B (treatment): same model + task, but the sql_query tool description is
rewritten by the agent itself after reading the fork-scope attribution report.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/home/dev/forkscope")

import httpx

from agentenv.tools import TOOLS, call_tool, SCHEMA_DESC

FORK_REPORT = """# Fork-scope attribution report: task t4 "albums per artist"

200 episodes of an agent solving "average albums per artist" on a music-store
SQLite DB (gold answer: 1.701).

## Finding
- 62% of episodes wrote `SELECT COUNT(*) / COUNT(DISTINCT ArtistId) ...` as
  their FIRST sql_query call — SQLite INTEGER DIVISION returns 1 (wrong).
- 0% of those episodes recovered: the query executes without error, so no
  self-correction signal ever fires.
- 19% wrote `SELECT AVG(...) FROM (SELECT ... GROUP BY ArtistId)` — all correct.
- The schema text itself primes the error: seeing `Album.ArtistId` suggests
  COUNT(DISTINCT ArtistId), and the model divides two integer counts.

## Decisive fork
The first SQL statement. Everything after is determined by it.

## Suggested patch
Rewrite the sql_query tool description to prevent the integer-division trap
and steer toward the AVG-over-grouped-subquery pattern.
"""

PATCH_PROMPT = (
    FORK_REPORT
    + "\n\nYou are the agent being evaluated. Rewrite ONLY the `description` "
      "field of the sql_query tool below so that you (the model) would no "
      "longer fall into the integer-division trap, without giving away the "
      "numeric answer. Keep the parameter schema identical. "
      "Return ONLY the new description text.\n\n"
      "Current description:\n"
      + TOOLS[0]["function"]["description"]
)


def build_patched_tools(new_desc: str):
    import copy
    tools = copy.deepcopy(TOOLS)
    tools[0]["function"]["description"] = new_desc
    return tools


async def run_episode(base_url: str, tools, task_prompt: str, seed: int,
                      max_tokens: int = 1500) -> dict:
    messages = [
        {"role": "system", "content": (
            "You are a data analyst with access to a music-store SQLite database, "
            "a calculator, and web search. Solve the user's task using tools. "
            "When you have the final answer, reply with 'The answer is <answer>.'")},
        {"role": "user", "content": task_prompt},
    ]
    steps = []
    async with httpx.AsyncClient(base_url=base_url, timeout=300) as c:
        for rnd in range(8):
            r = await c.post("/v1/chat/completions", json={
                "model": "default",
                "messages": messages,
                "tools": tools,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            r.raise_for_status()
            j = r.json()
            msg = j["choices"][0]["message"]
            steps.append(msg)
            messages.append({k: msg.get(k) for k in ("role", "content", "tool_calls") if msg.get(k)})
            tcs = msg.get("tool_calls") or []
            if not tcs:
                break
            for tc in tcs:
                out = call_tool(tc["function"]["name"], tc["function"].get("arguments", "{}"))
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [{"id": tc["id"], "type": "function",
                                                 "function": {"name": tc["function"]["name"],
                                                              "arguments": tc["function"].get("arguments", "{}")}}]})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    return steps


def classify(steps) -> tuple[str, str | None]:
    """Return (first_sql_strategy, final_answer)."""
    first_sql = None
    for m in steps:
        for tc in (m.get("tool_calls") or []):
            if tc["function"]["name"] == "sql_query":
                args = tc["function"].get("arguments", "{}")
                try:
                    first_sql = json.loads(args)["query"] if isinstance(args, str) else args["query"]
                except Exception:
                    first_sql = str(args)
                break
        if first_sql:
            break
    strat = "other"
    if first_sql:
        s = first_sql.lower()
        if "avg(" in s and "group by" in s:
            strat = "avg_groupby"
        elif "count(*)" in s and "count(distinct" in s:
            strat = "intdiv"
        elif "from artist" in s:
            strat = "wrong_table"
    final = None
    for m in reversed(steps):
        if m.get("content"):
            mm = re.findall(r"answer is[^\d]*([\d\.]+)", m["content"], re.I)
            if mm:
                final = mm[-1].rstrip(".")
                break
    return strat, final


async def arm(base_url: str, tools, n: int, tag: str):
    TASK = "On average, how many albums per artist are in the store? Give the ratio to 3 decimals."
    sem = asyncio.Semaphore(24)

    async def one(i):
        async with sem:
            steps = await run_episode(base_url, tools, TASK, seed=hash((tag, i)) % 2**31)
        return classify(steps)

    out = await asyncio.gather(*(one(i) for i in range(n)))
    strats = Counter(s for s, _ in out)
    correct = sum(1 for _, a in out if a and abs(float(a) - 1.701) < 0.002)
    print(f"[{tag}] n={n} correct={correct/n:.1%} strategies={dict(strats)}", flush=True)
    return {"tag": tag, "n": n, "correct": correct, "strategies": dict(strats)}


async def main():
    base_url = "http://127.0.0.1:30000"

    # Step 0: agent writes the patch (single greedy call, no tools)
    async with httpx.AsyncClient(base_url=base_url, timeout=300) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": PATCH_PROMPT}],
            "temperature": 0.0,
            "max_tokens": 600,
        })
        r.raise_for_status()
        patch_text = r.json()["choices"][0]["message"]["content"].strip()
    print("=== PATCHED DESCRIPTION ===")
    print(patch_text)
    print("===========================")

    patched = build_patched_tools(patch_text)

    A = await arm(base_url, TOOLS, 200, "control")
    B = await arm(base_url, patched, 200, "patched")

    result = {"patch": patch_text, "control": A, "patched": B}
    with open("/home/dev/forkscope/data/reports/rsi_loop.json", "w") as f:
        json.dump(result, f, indent=2)
    intdiv_a = A["strategies"].get("intdiv", 0) / A["n"]
    intdiv_b = B["strategies"].get("intdiv", 0) / B["n"]
    print(f"\n[rsi] intdiv rate: {intdiv_a:.1%} -> {intdiv_b:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
