"""Analyze collected episodes: outcome vs gold, first-decision clusters, lock-in.

Answers: does the FIRST tool-call choice lock the episode outcome (t4 pattern),
or is the task knowledge/drift type?

Usage (on node):
  venv/bin/python -m agentenv.analyze --tasks t7_avg_track_len_min t10_search_plus_calc
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/dev/forkscope")

from agentenv.tasks import TASKS, gold  # noqa: E402

NUM = r"(-?[\d,]*\.?\d+)"


def final_number(text: str | None) -> float | None:
    if not text:
        return None
    m = re.findall(r"answer is[^\d\-]*" + NUM, text, re.I) or re.findall(NUM, text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def gold_value(task_id: str) -> float:
    g = gold(task_id)
    for v in g.values():
        if isinstance(v, (int, float)):
            return float(v)
    raise ValueError(f"no numeric gold for {task_id}")


# per-task absolute tolerance (matches the decimals asked for in the prompt)
TOL = {"t4_artist_album_ratio": 0.0015, "t7_avg_track_len_min": 0.011,
       "t10_search_plus_calc": 0.06}


def outcome_label(task_id: str, v: float | None, g: float) -> str:
    if v is None:
        return "no_answer"
    if abs(v - g) <= TOL.get(task_id, max(0.011, 0.01 * abs(g))):
        return "correct"
    if task_id == "t7_avg_track_len_min":
        if abs(v - g * 60) <= 0.7:
            return "wrong_unit_seconds"
        if abs(v - g * 60000) <= 700:
            return "wrong_unit_ms"
    if task_id == "t4_artist_album_ratio":
        if abs(v - 1.0) < 0.001:
            return "intdiv_1.000"
        if abs(v - 1.262) < 0.002:
            return "wrongtable_1.262"
    if task_id == "t10_search_plus_calc":
        if abs(v - 19.3) < 0.11:
            return "raw_number_19.3"
        if abs(v - 28.6) < 0.11:
            return "raw_number_28.6"
    return "wrong_other"


def norm_sql(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip().rstrip(";")


def first_decision(ep: dict) -> tuple[str, str]:
    """(tool_name, canonical_args) of the first tool call; ('no_tool','') if none."""
    for s in ep.get("steps", []):
        if s.get("role") == "tool":
            args = s.get("args") or ""
            try:
                d = json.loads(args)
                args = d.get("query") or d.get("expr") or json.dumps(d)
            except (json.JSONDecodeError, TypeError):
                pass
            return s.get("name", "?"), norm_sql(str(args))
    return "no_tool", ""


def canon_cluster(task_id: str, tool: str, arg: str) -> str:
    """Task-aware canonicalization of the first decision."""
    if tool == "no_tool":
        return "no_tool"
    if task_id == "t4_artist_album_ratio" and tool == "sql_query":
        if "avg(" in arg and "group by" in arg:
            return "sql:avg_groupby"
        if re.search(r"count\(\s*\*\s*\)\s*/\s*count\(\s*distinct", arg):
            return "sql:intdiv"
        if "from artist" in arg:
            return "sql:from_artist"
        return "sql:other"
    if task_id == "t7_avg_track_len_min" and tool == "sql_query":
        if "avg(milliseconds)" in arg and ("/60000" in arg.replace(" ", "") or
                                           "/ 60000" in arg or "/60.0" in arg.replace(" ", "")):
            return "sql:avg_convert_in_sql"
        if "avg(milliseconds)" in arg:
            return "sql:avg_ms_raw"
        if "sum(" in arg:
            return "sql:manual_sum"
        return "sql:other"
    if task_id == "t10_search_plus_calc":
        if tool == "web_search":
            return "search_first"
        if tool == "calculator":
            return "calc_first"
    return f"{tool}:{arg[:60]}"


def mutual_info(pairs: list[tuple[str, bool]]) -> float:
    """MI(cluster; correct) in bits."""
    n = len(pairs)
    pc = Counter(c for c, _ in pairs)
    po = Counter(o for _, o in pairs)
    pj = Counter(pairs)
    mi = 0.0
    for (c, o), nj in pj.items():
        p = nj / n
        mi += p * math.log2(p / ((pc[c] / n) * (po[o] / n)))
    return mi


def analyze(task_id: str, indir: str, outdir: str) -> dict:
    path = f"{indir}/episodes_{task_id}.jsonl"
    eps = [json.loads(l) for l in open(path)]
    eps = [e for e in eps if not e.get("error")]
    g = gold_value(task_id)

    rows = []
    for ep in eps:
        v = final_number(ep.get("final"))
        out = outcome_label(task_id, v, g)
        tool, arg = first_decision(ep)
        rows.append({"seed": ep["seed"], "outcome": out, "correct": out == "correct",
                     "cluster": canon_cluster(task_id, tool, arg),
                     "first_tool": tool, "first_arg": arg,
                     "n_rounds": max((s["round"] for s in ep["steps"]), default=-1) + 1})

    n = len(rows)
    overall = sum(r["correct"] for r in rows) / n if n else 0.0
    by_cluster = defaultdict(list)
    for r in rows:
        by_cluster[r["cluster"]].append(r)

    table = []
    for c, rs in sorted(by_cluster.items(), key=lambda kv: -len(kv[1])):
        cr = sum(r["correct"] for r in rs) / len(rs)
        outs = Counter(r["outcome"] for r in rs)
        table.append({"cluster": c, "n": len(rs), "share": len(rs) / n,
                      "correct_rate": cr, "outcomes": dict(outs)})

    mi = mutual_info([(r["cluster"], r["correct"]) for r in rows])
    big = [t for t in table if t["n"] >= 10]
    spread = (max(t["correct_rate"] for t in big) - min(t["correct_rate"] for t in big)) if len(big) >= 2 else 0.0
    # lock-in verdict: some big cluster ~always right AND another ~always wrong
    locked = spread >= 0.8

    res = {"task_id": task_id, "n": n, "gold": g, "overall_correct": overall,
           "clusters": table, "mi_bits": mi, "spread_big_clusters": spread,
           "decision_locked": locked,
           "outcome_dist": dict(Counter(r["outcome"] for r in rows)),
           "rows": rows}

    import os
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/validation_{task_id}.json", "w") as f:
        json.dump(res, f, indent=1)

    lines = [f"# validation: {task_id}", "",
             f"- episodes: {n}, gold = {g}, overall correct = {overall:.1%}",
             f"- MI(first-decision cluster; correct) = {mi:.3f} bits",
             f"- big-cluster correct-rate spread = {spread:.2f} -> "
             f"{'DECISION-LOCKED' if locked else 'not locked'}", "",
             "| first-decision cluster | n | share | correct | outcomes |",
             "|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| `{t['cluster']}` | {t['n']} | {t['share']:.0%} | "
                     f"{t['correct_rate']:.0%} | {t['outcomes']} |")
    md = "\n".join(lines)
    with open(f"{outdir}/validation_{task_id}.md", "w") as f:
        f.write(md + "\n")
    print(md, flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--indir", default="data/raw")
    ap.add_argument("--outdir", default="data/reports")
    args = ap.parse_args()
    for tid in args.tasks:
        analyze(tid, args.indir, args.outdir)
        print()


if __name__ == "__main__":
    main()
