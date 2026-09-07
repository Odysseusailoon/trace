"""E2b BIRD-mini collection: 10 frozen questions x {nt, th} arms x N episodes.

Per Amendment A1 (prereg-ablations-e2b.md, 2026-09-01):
  - tools: sql_query (auto schema desc, no traps) + calculator
  - prompt: question + evidence + 3-decimal instruction
  - gold: executed gold SQL, must be single-row single-column numeric;
    else frozen replacement queue (logged)
  - nt arm: non-thinking, max_tokens 2000; th arm: thinking, max_tokens 6000
  - n=200/question/arm, seed0=10000, T=1.0

Usage (on node, server running):
  python scripts/collect_bird.py --arm nt --n 200
  python scripts/collect_bird.py --arm th --n 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, "/home/dev/forkscope")

from agentenv import tools as toolmod  # noqa: E402
from agentenv.runner import continue_episode  # noqa: E402

BIRD_DB_ROOT = os.environ.get("BIRD_DB_ROOT", "/scratch/bird/dev_databases")
SYSTEM = (
    "You are a data analyst with access to a SQLite database and a calculator. "
    "Solve the user's task using tools. "
    "When you have the final answer, reply with 'The answer is <answer>.'")


def db_path(db_id: str) -> str:
    return f"{BIRD_DB_ROOT}/{db_id}/{db_id}.sqlite"


def gold_scalar(q: dict):
    """Execute gold SQL; return float if single-row single-column numeric, else None."""
    try:
        db = sqlite3.connect(db_path(q["db_id"]))
        rows = db.execute(q["SQL"]).fetchall()
    except Exception as e:
        print(f"  [gold] b{q['question_id']} EXEC ERROR: {e}", flush=True)
        return None
    if len(rows) != 1 or len(rows[0]) != 1 or rows[0][0] is None:
        print(f"  [gold] b{q['question_id']} non-scalar: {len(rows)} rows", flush=True)
        return None
    v = rows[0][0]
    try:
        return float(v)
    except (TypeError, ValueError):
        print(f"  [gold] b{q['question_id']} non-numeric: {v!r}", flush=True)
        return None


def mode_count(q: dict, finals: list[str | None]) -> tuple[int, dict]:
    """A2 screen statistic: count of the modal (rounded) answer across episodes."""
    from collections import Counter
    bins = []
    for final in finals:
        if not final:
            bins.append("no_answer")
            continue
        m = re.findall(r"answer is[^\d\-]*(-?[\d,]*\.?\d+)", final, re.I) or \
            re.findall(r"(-?[\d,]*\.?\d+)", final)
        if not m:
            bins.append("no_answer")
            continue
        try:
            bins.append(f"{float(m[-1].replace(',', '')):.3f}")
        except ValueError:
            bins.append("no_answer")
    c = Counter(bins)
    return (c.most_common(1)[0][1] if c else 0), dict(c)


async def screen(sel: dict, base_url: str, outdir: str, n_screen: int = 10,
                 pool: int = 60, seed0: int = 90000, concurrency: int = 24) -> None:
    """A2 uncertainty screen: 10 NT eps on the first `pool` gold-executable
    eligible questions; keep mode in [4,6]; first 10 by question_id."""
    results, kept = [], []
    scanned = 0
    for q in sel["eligible_full"]:
        if len(kept) >= 10:
            break
        g = gold_scalar(q)
        if g is None:
            results.append({"question_id": q["question_id"], "gold": None,
                            "skipped": "gold_not_scalar"})
            continue
        scanned += 1
        if scanned > pool and len(kept) < 10:
            pool += 60  # extend per frozen rule
        q = dict(q)
        q["gold_scalar"] = g
        tools = toolmod.configure(db_path(q["db_id"]))
        prompt = q["question"]
        if q.get("evidence"):
            prompt += f"\n(Hint: {q['evidence']})"
        prompt += "\nGive the final numeric answer to 3 decimals if not an integer."
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
        sem = asyncio.Semaphore(concurrency)

        async def one(i: int):
            async with sem:
                try:
                    return await continue_episode(base_url, msgs, seed=seed0 + i,
                                                  tools=tools, max_tokens=2000)
                except Exception as e:
                    return {"final": None, "error": str(e)}

        eps = await asyncio.gather(*(one(i) for i in range(n_screen)))
        mc, dist = mode_count(q, [e.get("final") for e in eps])
        keep = 4 <= mc <= 6
        if keep:
            kept.append(q)
        results.append({"question_id": q["question_id"], "db_id": q["db_id"],
                        "gold": g, "mode_count": mc, "dist": dist, "keep": keep})
        print(f"[screen] b{q['question_id']:>5} mode={mc}/10 keep={keep} "
              f"dist={dist}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/screen_results.json", "w") as f:
        json.dump({"results": results, "selected": kept}, f, indent=1)
    print(f"[screen] scanned {scanned}, selected {len(kept)}: "
          f"{[q['question_id'] for q in kept]}", flush=True)


def resolve_questions(sel: dict) -> list[dict]:
    """Apply the frozen replacement rule; return 10 questions with gold attached."""
    out, queue = [], list(sel["backup"])
    for q in sel["picked"]:
        g = gold_scalar(q)
        while g is None and queue:
            sub = queue.pop(0)
            print(f"  [replace] b{q['question_id']} -> b{sub['question_id']}", flush=True)
            q, g = sub, gold_scalar(sub)
        if g is None:
            raise SystemExit("replacement queue exhausted")
        q = dict(q)
        q["gold_scalar"] = g
        out.append(q)
    return out


def outcome(q: dict, final: str | None) -> str:
    if not final:
        return "no_answer"
    m = re.findall(r"answer is[^\d\-]*(-?[\d,]*\.?\d+)", final, re.I) or \
        re.findall(r"(-?[\d,]*\.?\d+)", final)
    if not m:
        return "no_answer"
    try:
        v = float(m[-1].replace(",", ""))
    except ValueError:
        return "no_answer"
    g = q["gold_scalar"]
    ok = lambda a, b: abs(a - b) / max(abs(b), 1e-9) < 0.01
    if ok(v, g):
        return "correct"
    if re.search(r"percent|ratio", q["question"] + q.get("evidence", ""), re.I) and \
            (ok(v, g * 100) or (g != 0 and ok(v, g / 100))):
        return "correct_scaled"
    return "wrong"


async def collect_question(q: dict, arm: str, n: int, base_url: str, seed0: int,
                           concurrency: int, outdir: str) -> None:
    qid = q["question_id"]
    tools = toolmod.configure(db_path(q["db_id"]))
    prompt = q["question"]
    if q.get("evidence"):
        prompt += f"\n(Hint: {q['evidence']})"
    prompt += "\nGive the final numeric answer to 3 decimals if not an integer."
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}]
    think = arm == "th"
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(i: int) -> dict:
        nonlocal done
        async with sem:
            try:
                ep = await continue_episode(
                    base_url, msgs, seed=seed0 + i, tools=tools,
                    enable_thinking=think, max_tokens=6000 if think else 2000)
            except Exception as e:
                ep = {"seed": seed0 + i, "error": str(e),
                      "steps": [], "messages": [], "final": None}
        done += 1
        if done % 50 == 0:
            print(f"  [b{qid}/{arm}] {done}/{n}", flush=True)
        return ep

    eps = await asyncio.gather(*(one(i) for i in range(n)))
    os.makedirs(outdir, exist_ok=True)
    path = f"{outdir}/episodes_bird_b{qid}_{arm}.jsonl"
    with open(path, "w") as f:
        for ep in eps:
            ep["task_id"] = f"bird_b{qid}"
            ep["bird"] = {k: q[k] for k in
                          ("question_id", "db_id", "question", "evidence", "SQL",
                           "difficulty", "gold_scalar")}
            f.write(json.dumps(ep) + "\n")
    good = [e for e in eps if not e.get("error")]
    from collections import Counter
    oc = Counter(outcome(q, e.get("final")) for e in good)
    fail = 1 - (oc.get("correct", 0) + oc.get("correct_scaled", 0)) / max(len(good), 1)
    print(f"[b{qid}/{arm}] n_ok={len(good)} errors={n - len(good)} "
          f"fail_rate={fail:.2f} outcomes={dict(oc)}", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["nt", "th", "screen"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--outdir", default="data/raw_bird")
    ap.add_argument("--only", type=int, default=None, help="single question_id (smoke)")
    args = ap.parse_args()

    sel = json.load(open("data/bird/selection_e2b.json"))
    if args.arm == "screen":
        await screen(sel, args.base_url, args.outdir,
                     concurrency=args.concurrency)
        return
    # A2: main collection uses the screened set
    sr_path = f"{args.outdir}/screen_results.json"
    if os.path.exists(sr_path):
        qs = json.load(open(sr_path))["selected"]
        print(f"[collect] using A2 screened set: {[q['question_id'] for q in qs]}",
              flush=True)
    else:
        raise SystemExit("A2 screen not run yet: python scripts/collect_bird.py --arm screen")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.outdir}/resolved_questions.json", "w") as f:
        json.dump(qs, f, indent=1)
    for q in qs:
        if args.only and q["question_id"] != args.only:
            continue
        await collect_question(q, args.arm, args.n, args.base_url, args.seed0,
                               args.concurrency, args.outdir)


if __name__ == "__main__":
    asyncio.run(main())
