"""RSI no-leak rerun (t4) + generalization arm (t7).

Arms (all with rsi_loop's episode runner + hash seeds, for arm-comparability):
  t4_noleak : patch may describe ONLY the failure mode, no correct pattern.
              Control = existing rsi_loop.json control arm.
  t7_control / t7_noleak : same protocol on t7 (unit-conversion fork).

Leak checker: forbidden substrings per task; on violation the model rewrites
(<=3 attempts, hard fail otherwise). Per-episode rows are saved this time.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/home/dev/forkscope")

import httpx

from scripts.rsi_loop import TOOLS, build_patched_tools, run_episode  # reuse their runner

T4_TASK = "On average, how many albums per artist are in the store? Give the ratio to 3 decimals."
T7_TASK = "What is the average track length in minutes (2 decimals)?"

T4_REPORT = """# Fork-scope attribution report: task t4 "albums per artist"
- 46% of episodes wrote `SELECT COUNT(*) / COUNT(DISTINCT ArtistId) ...` as their
  FIRST sql_query call - SQLite INTEGER DIVISION of two counts truncates to 1 (wrong).
- ~2% of those episodes recovered: the query executes without error, so no
  self-correction signal ever fires.
- The schema text itself primes the error: seeing `Album.ArtistId` suggests
  COUNT(DISTINCT ArtistId), and the model divides two integer counts.
- Decisive fork: the first SQL statement."""

T7_REPORT = """# Fork-scope attribution report: task t7 "average track length in minutes"
- 98% of episodes issue the same first query (SELECT AVG(Milliseconds) FROM Track) -
  the first tool call is NOT where episodes diverge.
- The decisive fork is the NEXT decision: only ~10% of continuations convert the
  raw Milliseconds value into the unit the question asks for; the rest report a
  value in the wrong unit as if it were minutes and end the episode.
- The wrong answer looks plausible and nothing errors, so no self-correction fires."""

T4_CONSTRAINT = (
    "STRICT CONSTRAINT: the new description must ONLY warn about the failure mode "
    "(integer division when dividing counts in SQLite). It must NOT contain, hint at, "
    "or paraphrase any correct query pattern or SQL construct to use instead - "
    "no AVG, no GROUP BY, no subquery advice, no alternative formulation, no numbers."
)
T7_CONSTRAINT = (
    "STRICT CONSTRAINT: the new description must ONLY warn that answers are often "
    "reported in the wrong unit and that the unit requested by the question must be "
    "double-checked. It must NOT contain any conversion factor, formula, or arithmetic "
    "hint (no 60000, no 'divide', no multiplication factors)."
)

T4_FORBIDDEN = ["avg(", " avg ", "group by", "subquery", "sub-query", "nested", "1.701", "artist_count"]
T7_FORBIDDEN = ["60000", "60,000", "60 000", "divide", "dividing", "/ 60", "/60", "* 60", "conversion factor"]


def leaks(text: str, forbidden: list[str]) -> list[str]:
    t = text.lower()
    return [w for w in forbidden if w in t]


async def gen_patch(base_url: str, report: str, constraint: str,
                    forbidden: list[str]) -> dict:
    """v2: the model writes ONLY a short warning note; code appends it to the
    ORIGINAL description, so the schema listing can never be lost."""
    prompt = (report + "\n\nYou are the agent being evaluated. Write a short warning "
              "note (max 60 words) that will be APPENDED to the sql_query tool description, "
              "so that you would no longer fall into this failure mode. "
              + constraint + " Return ONLY the note text.")
    history = [{"role": "user", "content": prompt}]
    async with httpx.AsyncClient(base_url=base_url, timeout=300) as c:
        for attempt in range(3):
            r = await c.post("/v1/chat/completions", json={
                "model": "default", "messages": history,
                "temperature": 0.0, "max_tokens": 600,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            bad = leaks(text, forbidden)
            if not bad:
                full = TOOLS[0]["function"]["description"] + "\n\nNote: " + text
                return {"note": text, "text": full, "attempts": attempt + 1}
            history += [{"role": "assistant", "content": text},
                        {"role": "user", "content":
                         f"Your description leaked forbidden content: {bad}. "
                         "Rewrite it. Warn about the failure mode ONLY - absolutely no "
                         "correct approach, pattern, formula, or factor."}]
    raise SystemExit(f"PATCH_LEAK_UNFIXABLE: {bad} in {text!r}")


def classify_t4(steps) -> tuple[str, str | None]:
    from scripts.rsi_loop import classify
    return classify(steps)


def classify_t7(steps) -> tuple[str, str | None]:
    first_sql = None
    for m in steps:
        for tc in (m.get("tool_calls") or []):
            if tc["function"]["name"] == "sql_query":
                a = tc["function"].get("arguments", "{}")
                try:
                    first_sql = json.loads(a)["query"] if isinstance(a, str) else a["query"]
                except Exception:
                    first_sql = str(a)
                break
        if first_sql:
            break
    strat = "avg_ms" if first_sql and "avg(milliseconds" in first_sql.lower().replace(" ", "") \
        else ("other" if first_sql else "no_sql")
    final = None
    for m in reversed(steps):
        if m.get("content"):
            mm = re.findall(r"answer is[^\d\-]*(-?[\d\.]+)", m["content"], re.I)
            if mm:
                final = mm[-1].rstrip(".")
                break
    return strat, final


def outcome_t4(final: str | None) -> str:
    try:
        v = float(final)
    except (TypeError, ValueError):
        return "no_answer"
    if abs(v - 1.701) < 0.002:
        return "correct"
    if abs(v - 1.0) < 0.001:
        return "intdiv"
    return "wrong_other"


def outcome_t7(final: str | None) -> str:
    try:
        v = float(final)
    except (TypeError, ValueError):
        return "no_answer"
    if abs(v - 6.56) <= 0.011:
        return "correct"
    if abs(v - 393.6) <= 0.7 or abs(v - 393599.2) <= 700:
        return "wrong_unit"
    return "wrong_other"


async def arm(base_url: str, tools, task_prompt: str, n: int, tag: str,
              classify_fn, outcome_fn) -> dict:
    sem = asyncio.Semaphore(24)

    async def one(i):
        async with sem:
            try:
                steps = await run_episode(base_url, tools, task_prompt,
                                          seed=hash((tag, i)) % 2**31)
            except Exception as e:
                return {"i": i, "error": str(e)}
        strat, final = classify_fn(steps)
        return {"i": i, "strat": strat, "final": final, "outcome": outcome_fn(final)}

    rows = await asyncio.gather(*(one(i) for i in range(n)))
    ok = [r for r in rows if "error" not in r]
    outs = Counter(r["outcome"] for r in ok)
    strats = Counter(r["strat"] for r in ok)
    print(f"[{tag}] n={len(ok)} outcomes={dict(outs)} strategies={dict(strats)}", flush=True)
    return {"tag": tag, "n": len(ok), "outcomes": dict(outs),
            "strategies": dict(strats), "rows": rows}


async def main():
    base_url = "http://127.0.0.1:30000"
    res = {}

    p4 = await gen_patch(base_url, T4_REPORT, T4_CONSTRAINT, T4_FORBIDDEN)
    print("=== T4 NO-LEAK PATCH (attempts %d) ===\n%s\n===" % (p4["attempts"], p4["text"]), flush=True)
    res["t4_patch"] = p4
    res["t4_noleak"] = await arm(base_url, build_patched_tools(p4["text"]), T4_TASK,
                                 200, "t4_noleak_v2", classify_t4, outcome_t4)

    p7 = await gen_patch(base_url, T7_REPORT, T7_CONSTRAINT, T7_FORBIDDEN)
    print("=== T7 NO-LEAK PATCH (attempts %d) ===\n%s\n===" % (p7["attempts"], p7["text"]), flush=True)
    res["t7_patch"] = p7
    res["t7_noleak"] = await arm(base_url, build_patched_tools(p7["text"]), T7_TASK,
                                 200, "t7_noleak_v2", classify_t7, outcome_t7)

    with open("/home/dev/forkscope/data/reports/rsi_noleak2.json", "w") as f:
        json.dump(res, f, indent=1)
    print("RSI_NOLEAK_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
