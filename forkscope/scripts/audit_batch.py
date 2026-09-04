"""Rigor-audit batch (preregistered in prereg-ablations-e2b.md).

Order: (1) t10 corpus ablation, (2) t4 schema ablations V1-V3,
(3) mirror-replay expansion (21 episodes), (4) fork-state continuation dumps,
(5) blind-audit sample, (6) /metrics capture around one replay.

Run on node: /home/dev/venv/bin/python -u scripts/audit_batch.py
"""
import asyncio
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, "/home/dev/forkscope")

import agentenv.tools as T  # noqa: E402
from agentenv.analyze import (canon_cluster, final_number, first_decision,  # noqa: E402
                              gold_value, outcome_label)
from agentenv.collect import collect_task  # noqa: E402
from agentenv.replay import replay_episode  # noqa: E402
from agentenv.runner import continue_episode  # noqa: E402
from agentenv.tasks import TASKS  # noqa: E402

BASE_URL = "http://127.0.0.1:30000"
OUT = "/home/dev/forkscope/data/audit"
os.makedirs(OUT, exist_ok=True)

ORIG_DESC = T.TOOLS[0]["function"]["description"]
ORIG_SNIPPET = T._SEARCH_CORPUS["music industry revenue 2023"][1]["snippet"]

NOSCHEMA = ("Run a read-only SQL query against the music store database. "
            "Available tables: Artist, Album, Track, Genre, MediaType, Invoice, "
            "InvoiceLine, Customer, Employee, temp_country_codes.")
ALBUM_LINE = "- Album(AlbumId, Title, ArtistId)"


def task_of(tid):
    return next(t for t in TASKS if t["id"] == tid)


def summarize(path, tid):
    g = gold_value(tid)
    eps = [json.loads(l) for l in open(path)]
    eps = [e for e in eps if not e.get("error")]
    from collections import Counter
    clusters = Counter()
    outcomes = Counter()
    for e in eps:
        tool, arg = first_decision(e)
        clusters[canon_cluster(tid, tool, arg)] += 1
        outcomes[outcome_label(tid, final_number(e.get("final")), g)] += 1
    return {"n": len(eps), "clusters": dict(clusters.most_common()),
            "outcomes": dict(outcomes.most_common())}


async def run_variant(tag, tid, mutate, n=200, seed0=10000):
    mutate()
    try:
        task = dict(task_of(tid))
        path = await collect_task(task, n, BASE_URL, seed0, 24, f"{OUT}/{tag}")
        s = summarize(path, tid)
        print(f"[ABLATION {tag}] {json.dumps(s)}", flush=True)
        return s
    finally:  # always restore
        T.TOOLS[0]["function"]["description"] = ORIG_DESC
        T._SEARCH_CORPUS["music industry revenue 2023"][1]["snippet"] = ORIG_SNIPPET


def metrics():
    try:
        txt = urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=10).read().decode()
        out = {}
        for line in txt.splitlines():
            for key in ("sglang:prompt_tokens_total", "sglang:generation_tokens_total",
                        "sglang:cached_tokens_total"):
                if line.startswith(key):
                    out[key.split(":")[1]] = float(line.split()[-1])
        return out
    except Exception as e:
        return {"error": str(e)}


async def main():
    # ---- (1) t10 corpus ablation ----
    def cut_decoy():
        T._SEARCH_CORPUS["music industry revenue 2023"][1]["snippet"] = \
            "Subscription streaming revenue reached $19.3 billion in 2023."
    await run_variant("t10_NODECOY", "t10_search_plus_calc", cut_decoy)

    # ---- (2) t4 schema ablations ----
    def v1():
        T.TOOLS[0]["function"]["description"] = NOSCHEMA
    def v2():
        T.TOOLS[0]["function"]["description"] = ORIG_DESC.replace(ALBUM_LINE, "- Album")
    def v3():
        T.TOOLS[0]["function"]["description"] = ORIG_DESC.replace(
            ALBUM_LINE, "- Album(ArtistId, AlbumId, Title)")
    for tag, m in [("t4_V1_NOSCHEMA", v1), ("t4_V2_NOALBUMCOLS", v2),
                   ("t4_V3_REORDER", v3)]:
        await run_variant(tag, "t4_artist_album_ratio", m)

    # ---- (3) mirror-replay expansion ----
    m0 = metrics()
    specs = [("t4_artist_album_ratio", "sql:intdiv", "intdiv_1.000", 5),
             ("t4_artist_album_ratio", "sql:avg_groupby", "correct", 5),
             ("t4_artist_album_ratio", "sql:other", None, 3),
             ("t7_avg_track_len_min", "sql:avg_ms_raw", "wrong_unit_seconds", 5),
             ("t7_avg_track_len_min", "sql:avg_ms_raw", "correct", 3)]
    for tid, pick, outcome, count in specs:
        eps = [json.loads(l) for l in open(f"/home/dev/forkscope/data/raw/episodes_{tid}.jsonl")]
        eps = [e for e in eps if not e.get("error")]
        g = gold_value(tid)
        cand = []
        for e in eps:
            tool, arg = first_decision(e)
            if canon_cluster(tid, tool, arg) != pick:
                continue
            if outcome and outcome_label(tid, final_number(e.get("final")), g) != outcome:
                continue
            cand.append(e)
        print(f"[REPLAY SPEC] {tid} {pick}/{outcome}: {len(cand)} candidates, taking {count}",
              flush=True)
        for e in cand[:count]:
            res = await replay_episode(BASE_URL, e, tid, 50)
            tag = f"{pick.replace(':','_')}_{outcome or 'any'}_{e['seed']}"
            with open(f"{OUT}/replay_{tid}_{tag}.json", "w") as f:
                json.dump(res, f, indent=1)
            print(f"[REPLAY DONE] {tid} seed={e['seed']}", flush=True)
    m1 = metrics()
    print(f"[METRICS] before={json.dumps(m0)} after={json.dumps(m1)}", flush=True)

    # ---- (4) fork-state continuation dumps (oracle dataset v0) ----
    for tid, seed_pick, boundary in [("t4_artist_album_ratio", "sql:intdiv", 0),
                                     ("t7_avg_track_len_min", "sql:avg_ms_raw", 1)]:
        eps = [json.loads(l) for l in open(f"/home/dev/forkscope/data/raw/episodes_{tid}.jsonl")]
        eps = [e for e in eps if not e.get("error")]
        e = next(x for x in eps
                 if canon_cluster(tid, *first_decision(x)) == seed_pick)
        msgs = e["messages"]
        bounds = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        i = bounds[boundary]
        prefix = msgs[:i]
        start_round = sum(1 for m in prefix if m["role"] == "assistant")
        sem = asyncio.Semaphore(24)

        async def one(j):
            async with sem:
                try:
                    return await continue_episode(BASE_URL, prefix, seed=90000 + j,
                                                  start_round=start_round)
                except Exception as ex:
                    return {"error": str(ex)}
        conts = await asyncio.gather(*(one(j) for j in range(50)))
        g = gold_value(tid)
        for c in conts:
            if "error" not in c:
                c["outcome"] = outcome_label(tid, final_number(c.get("final")), g)
        with open(f"{OUT}/forkstate_conts_{tid}_d{boundary}.json", "w") as f:
            json.dump({"task_id": tid, "boundary": boundary, "source_seed": e["seed"],
                       "prefix_len_msgs": i, "continuations": conts}, f)
        ok = [c for c in conts if "error" not in c]
        from collections import Counter
        print(f"[FORKSTATE {tid} d={boundary}] outcomes="
              f"{dict(Counter(c['outcome'] for c in ok))}", flush=True)

    # ---- (5) blind-audit sample ----
    rng = random.Random(7)
    sample = []
    for tid in ["t4_artist_album_ratio", "t7_avg_track_len_min", "t10_search_plus_calc"]:
        eps = [json.loads(l) for l in open(f"/home/dev/forkscope/data/raw/episodes_{tid}.jsonl")]
        eps = [e for e in eps if not e.get("error")]
        g = gold_value(tid)
        for e in rng.sample(eps, 17 if tid != "t4_artist_album_ratio" else 16):
            tool, arg = first_decision(e)
            first_raw = None
            for s in e["steps"]:
                if s["role"] == "assistant" and s.get("tool_calls"):
                    first_raw = json.dumps(s["tool_calls"][0]["function"])[:300]
                    break
            sample.append({
                "id": f"{tid[:3]}_{e['seed']}",
                "blind": {"task": tid, "first_tool_call_raw": first_raw,
                          "final_text": (e.get("final") or "")[:200]},
                "hidden": {"cluster": canon_cluster(tid, tool, arg),
                           "outcome": outcome_label(tid, final_number(e.get("final")), g)},
            })
    with open(f"{OUT}/blind_audit.json", "w") as f:
        json.dump(sample, f, indent=1)
    print(f"[BLIND AUDIT] {len(sample)} episodes sampled -> blind_audit.json", flush=True)
    print("AUDIT_BATCH_DONE", flush=True)


asyncio.run(main())
