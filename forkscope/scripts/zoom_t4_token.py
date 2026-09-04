"""Two-scale zoom-in: token-level o_t + entropy INSIDE the d=0 turn of t4.

Agent-level replay says the first SQL locks the fate (o_0=0.22 -> o_1 in {0,1}).
This script asks WHERE inside that first assistant turn the die is cast:
at the reasoning wording ("count ... divide ...") or at the SQL string itself.

Design (exploratory, single-case existence demo — not preregistered):
  - Base paths: recorded episodes seed=10000 (intdiv-fated) and seed=10015
    (correct-fated). Reproduce the first turn via /generate with the episode
    seed (deterministic inference); verify text/SQL match against the record.
  - o_t: at token positions t (stride s over the turn), condition on
    prompt + turn_tokens[:t], sample S=20 continuations (T=1.0), classify the
    turn's FIRST tool call (prefix text + continuation combined) with
    canon_cluster. o_t = cluster distribution given tokens <= t.
  - Entropy: per-token top-10 truncated entropy along the base path
    (same 口径 as E-T4; truncation caveat applies).

Outputs:
  data/reports/zoom_t4_<seed>.json          (full: per-sample SQL strings)
  data/reports/zoom_t4_<seed>_summary.json  (compact: o_t + H_t + tokens)

Usage (on node):
  /home/dev/venv/bin/python scripts/zoom_t4_token.py --seed 10000 --seed 10015
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/dev/forkscope")

import httpx
from transformers import AutoTokenizer

from agentenv.analyze import canon_cluster
from agentenv.runner import SYSTEM
from agentenv.tools import TOOLS

TASK_ID = "t4_artist_album_ratio"
TASK_PROMPT = (
    "On average, how many albums per artist are in the store? "
    "Give the ratio to 3 decimals."
)
CLUSTERS = ["sql:avg_groupby", "sql:intdiv", "sql:from_artist", "sql:other", "no_tool"]


def first_sql_cluster(text: str) -> str:
    """Cluster of the first tool call found in raw generated text."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:
        # unterminated tool call at max_new_tokens: try a lenient parse
        m = re.search(r"<tool_call>\s*(\{.*)", text, re.S)
    if not m:
        return "no_tool"
    raw = m.group(1)
    name, arg = None, None
    try:
        d = json.loads(raw)
        name = d.get("name")
        a = d.get("arguments", {})
        arg = a.get("query") or a.get("expr") or json.dumps(a)
    except (json.JSONDecodeError, AttributeError):
        nm = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
        qm = re.search(r'"query"\s*:\s*"([^"]+)', raw)
        name = nm.group(1) if nm else None
        arg = qm.group(1) if qm else None
    if not name or arg is None:
        return "no_tool"
    arg = re.sub(r"\s+", " ", str(arg).lower()).strip().rstrip(";")
    c = canon_cluster(TASK_ID, name, arg)
    return c if c in CLUSTERS else "sql:other"


def trunc_entropy(top_row: list) -> float:
    """Top-10 truncated entropy (raw probs, nats->bits). Row items: [logprob, tid, ...]."""
    ps = [math.exp(item[0]) for item in top_row]
    return -sum(p * math.log2(p) for p in ps if p > 0)


async def gen(c: httpx.AsyncClient, input_ids: list[int], sp: dict, **kw) -> list[dict]:
    r = await c.post("/generate", json={"input_ids": input_ids,
                                        "sampling_params": sp, **kw})
    r.raise_for_status()
    resp = r.json()
    return resp if isinstance(resp, list) else [resp]


def recorded_cluster(ep: dict) -> str:
    """canon_cluster of the recorded first tool call of the episode."""
    for s in ep["steps"]:
        if s["role"] == "assistant" and s.get("tool_calls"):
            tc = s["tool_calls"][0]
            a = tc["function"].get("arguments", "")
            try:
                d = json.loads(a) if isinstance(a, str) else a
                arg = d.get("query") or d.get("expr") or json.dumps(d)
            except (json.JSONDecodeError, AttributeError):
                arg = str(a)
            arg = re.sub(r"\s+", " ", str(arg).lower()).strip().rstrip(";")
            return canon_cluster(TASK_ID, tc["function"]["name"], arg)
    return "no_tool"


def prompt_token_ids(tok) -> list[int]:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK_PROMPT}]
    text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


async def zoom_one(c: httpx.AsyncClient, tok, ep: dict, eps_by_cluster: dict, args) -> dict:
    prompt_ids = prompt_token_ids(tok)
    target = recorded_cluster(ep)

    # ---- Stage A: regenerate a first turn of the target fate, with logprobs ----
    # Try the episode's own seed plus same-cluster episode seeds; among the
    # candidates whose /generate stream lands on-fate, pick the one with the
    # LONGEST reasoning prefix before <tool_call> (a no-reasoning turn cannot
    # answer "wording vs SQL emission").
    cand_seeds = [ep["seed"]] + [s for s in eps_by_cluster.get(target, [])
                                 if s != ep["seed"]][:args.scan]
    best = None  # (content_len, seed, resp)
    for cs in cand_seeds:
        [r] = await gen(c, prompt_ids,
                        {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                         "max_new_tokens": 600, "sampling_seed": cs},
                        return_logprob=True, logprob_start_len=0, top_logprobs_num=10)
        bc = first_sql_cluster(r["text"])
        pre = r["text"].find("<tool_call>")
        pre = pre if pre >= 0 else 0
        print(f"[anchor scan] seed {cs}: base_cluster={bc} (target {target}), "
              f"reasoning_chars={pre}", flush=True)
        if bc == target and (best is None or pre > best[0]):
            best = (pre, cs, r)
    if best is None:
        print(f"[seed {ep['seed']}] no on-fate regeneration found, skipping", flush=True)
        return {}
    _, seed, resp = best
    meta = resp["meta_info"]
    turn_ids = [lp[1] for lp in meta["output_token_logprobs"]]
    top = meta["output_top_logprobs"]
    turn_text = resp["text"]
    base_cluster = first_sql_cluster(turn_text)

    rec_turn = next(s for s in ep["steps"] if s["role"] == "assistant")
    rec_sql = None
    if rec_turn.get("tool_calls"):
        a = rec_turn["tool_calls"][0]["function"].get("arguments", "")
        try:
            d = json.loads(a) if isinstance(a, str) else a
            rec_sql = d.get("query")
        except (json.JSONDecodeError, AttributeError):
            pass
    repro_sql_match = bool(rec_sql) and re.sub(r"\s+", " ", rec_sql.lower()) in \
        re.sub(r"\s+", " ", turn_text.lower())
    rec_content = (rec_turn.get("content") or "").strip()
    repro_text_match = bool(rec_content) and turn_text.strip().startswith(rec_content[:80])
    print(f"[seed {seed}] turn: {len(turn_ids)} tokens, base_cluster={base_cluster}, "
          f"repro sql_match={repro_sql_match} text_match={repro_text_match}", flush=True)

    # positions: stride to cap the number of probe points, always include 0
    n = len(turn_ids)
    if args.tmin is not None:
        positions = list(range(args.tmin, min(args.tmax + 1, n), 1))
        stride = 1
    else:
        stride = max(args.stride, math.ceil(n / args.max_positions))
        positions = sorted(set(list(range(0, n, stride)) + [n - 1]))

    H = [trunc_entropy(row) for row in top]
    tok_strs = tok.convert_ids_to_tokens(turn_ids)

    # ---- Stage B: o_t at each position ----
    sem = asyncio.Semaphore(args.concurrency)

    async def probe(t: int) -> dict:
        prefix_ids = prompt_ids + turn_ids[:t]
        prefix_text = tok.decode(turn_ids[:t])

        async def sample(j: int) -> str:
            # one request per sample with a distinct seed: n>1 + a single
            # sampling_seed under deterministic inference yields identical copies
            async with sem:
                [o] = await gen(c, prefix_ids,
                                {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                                 "max_new_tokens": args.max_new_tokens,
                                 "sampling_seed": 7_000_000 + seed * 1009 + t * 53 + j})
            return o["text"]

        texts = await asyncio.gather(*(sample(j) for j in range(args.s)))
        clusters, sqls = [], []
        for txt in texts:
            full = prefix_text + txt
            cl = first_sql_cluster(full)
            clusters.append(cl)
            qm = re.search(r'"query"\s*:\s*"([^"]+)', full)
            sqls.append(qm.group(1)[:120] if qm else None)
        dist = Counter(clusters)
        return {"t": t, "o_t": {k: dist.get(k, 0) / len(clusters) for k in CLUSTERS},
                "H": H[t] if t < len(H) else None,
                "tok": tok_strs[t] if t < len(tok_strs) else None,
                "sqls": sqls}

    rows = await asyncio.gather(*(probe(t) for t in positions))
    rows.sort(key=lambda r: r["t"])
    for r in rows:
        bar = {k: round(v, 2) for k, v in r["o_t"].items() if v > 0.01}
        print(f"  t={r['t']:3d} H={r['H']:.2f} tok={r['tok']!r:<16} {bar}", flush=True)

    return {"task_id": TASK_ID, "seed": seed, "episode_seed": ep["seed"],
            "s": args.s,
            "base_cluster": base_cluster, "n_turn_tokens": n, "stride": stride,
            "repro": {"sql_match": repro_sql_match, "text_match": repro_text_match,
                      "recorded_sql": rec_sql},
            "turn_text": turn_text,
            "tokens": tok_strs, "entropy": H,
            "rows": rows}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, action="append", required=True)
    ap.add_argument("--s", type=int, default=20)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-positions", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--scan", type=int, default=8,
                    help="max extra same-cluster seeds to try for an on-fate anchor")
    ap.add_argument("--tmin", type=int, default=None,
                    help="fine pass: probe every position in [tmin, tmax]")
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames (avoid clobbering the coarse pass)")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", default="data/reports")
    args = ap.parse_args()

    indirs = [args.indir] if args.indir else ["data/raw", "data/raw36"]
    eps = {}
    for d in indirs:
        p = Path(d) / f"episodes_{TASK_ID}.jsonl"
        if p.exists():
            for line in open(p):
                e = json.loads(line)
                if not e.get("error"):
                    eps[e["seed"]] = e
            break
    else:
        sys.exit(f"no episodes file found in {indirs}")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    eps_by_cluster: dict[str, list[int]] = {}
    for s_, e_ in sorted(eps.items()):
        eps_by_cluster.setdefault(recorded_cluster(e_), []).append(s_)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as c:
        for seed in args.seed:
            if seed not in eps:
                print(f"[seed {seed}] NOT FOUND in episodes, skipping", flush=True)
                continue
            res = await zoom_one(c, tok, eps[seed], eps_by_cluster, args)
            if not res:
                continue
            with open(f"{args.outdir}/zoom_t4_{res['seed']}{args.tag}.json", "w") as f:
                json.dump(res, f)
            slim = {k: v for k, v in res.items() if k != "rows"}
            slim["rows"] = [{k: v for k, v in r.items() if k != "sqls"}
                            for r in res["rows"]]
            with open(f"{args.outdir}/zoom_t4_{res['seed']}{args.tag}_summary.json", "w") as f:
                json.dump(slim, f)
            print(f"[anchor seed {res['seed']}] written.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
