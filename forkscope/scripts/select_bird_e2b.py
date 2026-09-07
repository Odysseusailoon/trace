"""Frozen deterministic selection of 10 BIRD mini-dev questions for E2b.

Rules (frozen 2026-09-01, BEFORE any model output on these questions;
see prereg-ablations-e2b.md Amendment A-2026-09-01):

1. Scalar protocol: gold SQL is a single top-level SELECT with >=1 aggregate
   (COUNT/AVG/SUM/CAST...) and NO top-level GROUP BY -> expected single-row
   single-column numeric result (verified by execution on node; if a selected
   question fails the execution check, it is replaced by the next eligible
   question in question_id order — replacement rule frozen here).
2. Theme: question or SQL involves aggregation/ratio/percentage/unit language
   (matches the trap families studied on Chinook: t4 ratio, t7 units, t1 share).
3. Diversity caps: at most 2 questions per db_id; difficulty mix at least
   3 simple / 3 moderate; fill remaining by question_id order.
4. Deterministic order: iterate candidates by ascending question_id; take the
   first 10 satisfying caps. No other criteria. No peeking at model behavior.

Output: data/bird/selection_e2b.json
"""
from __future__ import annotations

import json
import re
from collections import Counter

SRC = "data/bird/mini_dev_sqlite.json"
OUT = "data/bird/selection_e2b.json"

AGG = re.compile(r"\b(COUNT|AVG|SUM|TOTAL|MIN|MAX)\s*\(", re.I)
THEME = re.compile(r"ratio|percentage|percent|average|how many|number of|rate\b|share|per\b",
                   re.I)


def top_level_group_by(sql: str) -> bool:
    """GROUP BY at paren depth 0 (subquery GROUP BY is fine)."""
    depth, s = 0, sql.upper()
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and s.startswith("GROUP BY", i):
            return True
    return False


def eligible(q: dict) -> bool:
    sql = q["SQL"]
    if not AGG.search(sql):
        return False
    if top_level_group_by(sql):
        return False
    # single top-level select list without obvious multi-column output
    m = re.match(r"\s*SELECT\s+(.*?)\s+FROM\s", sql, re.I | re.S)
    if not m:
        return False
    sel = m.group(1)
    # reject multi-column select lists (top-level comma outside parens)
    depth = 0
    for ch in sel:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return False
    if not THEME.search(q["question"] + " " + q["SQL"]):
        return False
    return True


def main():
    qs = sorted(json.load(open(SRC)), key=lambda q: q["question_id"])
    cand = [q for q in qs if eligible(q)]
    print(f"eligible: {len(cand)}/{len(qs)}")

    picked, per_db = [], Counter()
    diff = Counter()
    # pass 1: enforce difficulty floor (3 simple, 3 moderate) + db cap
    for want, floor in [("simple", 3), ("moderate", 3)]:
        for q in cand:
            if len([p for p in picked if p["difficulty"] == want]) >= floor:
                break
            if q["difficulty"] == want and per_db[q["db_id"]] < 2 and q not in picked:
                picked.append(q)
                per_db[q["db_id"]] += 1
    # pass 2: fill to 10 by question_id order
    for q in cand:
        if len(picked) >= 10:
            break
        if q not in picked and per_db[q["db_id"]] < 2:
            picked.append(q)
            per_db[q["db_id"]] += 1
    picked = sorted(picked, key=lambda q: q["question_id"])
    # frozen replacement queue: next 10 eligible not picked, in order
    backup = [q for q in cand if q not in picked][:10]

    for q in picked:
        diff[q["difficulty"]] += 1
        print(f"  b{q['question_id']:>4} {q['db_id']:<24} {q['difficulty']:<9} "
              f"{q['question'][:70]}")
    print(f"difficulty mix: {dict(diff)}; dbs: {dict(per_db)}")
    json.dump({"frozen": "2026-09-01", "rules": __doc__, "picked": picked,
               "backup": backup}, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} (+{len(backup)} frozen backups)")


if __name__ == "__main__":
    main()
