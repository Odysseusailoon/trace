"""Cost table: naive FPA vs forkscope, exact prefill token accounting.

Naive FPA = HF-style resampling with no KV reuse: every continuation request
pays a full prefill of its prefix. forkscope = same request stream on SGLang
with RadixAttention: nested prefixes share the trunk.

Exact inputs (all local):
  - transcripts.json: the 4 replayed episodes (messages)
  - agentenv TOOLS + SYSTEM: to render the exact chat-template prefix
  - replay JSONs: K=50, boundary structure
Empirical anchors (measured on node, Day 3):
  - controlled radix on/off A/B: prefill compute 7.1K vs 106.7K tokens (14.9x)
  - full-pipeline prefix cache hit rate: 98.5%

Prefill FLOPs ~= 2 * N_params * tokens (8B -> 16 GFLOP/token).
"""
import json
import sys

sys.path.insert(0, "/Users/yifeichen/Projects/research/vector/forkscope")

from transformers import AutoTokenizer

from agentenv.runner import TOOLS  # noqa: E402

BASE = "/Users/yifeichen/Projects/research/vector/forkscope/data/reports"
K = 50
HIT = 0.985  # measured pipeline prefix hit rate
GFLOP_PER_TOK = 16  # 2 * 8e9 params / 1e9

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
trans = json.load(open(f"{BASE}/transcripts.json"))


def fix_args(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                a = tc["function"].get("arguments")
                if isinstance(a, str):
                    try:
                        tc["function"]["arguments"] = json.loads(a)
                    except json.JSONDecodeError:
                        pass
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def prefix_len(messages):
    ids = tok.apply_chat_template(fix_args(messages), tools=TOOLS,
                                  add_generation_prompt=True,
                                  enable_thinking=False,
                                  tokenize=True, return_dict=False)
    return len(ids)


rows = []
for key, ep in trans.items():
    msgs = ep["messages"]
    bounds = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    Ls = [prefix_len(msgs[:i]) for i in bounds]
    naive = K * sum(Ls)
    trunk_once = max(Ls)  # nested prefixes: longest prefix covers the trunk
    anchored = naive * (1 - HIT)
    rows.append({
        "episode": key, "boundaries": len(Ls), "prefix_tokens": Ls,
        "naive_prefill": naive, "ideal_prefill": trunk_once,
        "anchored_prefill": round(anchored),
        "ratio_ideal": round(naive / trunk_once, 1),
        "ratio_anchored": round(naive / anchored, 1),
        "naive_pflop": round(naive * GFLOP_PER_TOK / 1e6, 2),
    })

tot_naive = sum(r["naive_prefill"] for r in rows)
tot_ideal = sum(r["ideal_prefill"] for r in rows)

print("=== per-episode decision-step replay (K=50) ===")
print(f"{'episode':12s} {'D':>2s} {'prefix tok/boundary':>22s} {'naive':>8s}"
      f" {'ideal':>6s} {'x(ideal)':>8s} {'x(98.5%)':>8s}")
for r in rows:
    print(f"{r['episode']:12s} {r['boundaries']:2d} {str(r['prefix_tokens']):>22s}"
          f" {r['naive_prefill']:8d} {r['ideal_prefill']:6d}"
          f" {r['ratio_ideal']:8.1f} {r['ratio_anchored']:8.1f}")
print(f"\ntotal 4 episodes: naive {tot_naive} vs trunk-once {tot_ideal}"
      f" = {tot_naive/tot_ideal:.0f}x; at measured 98.5% hit: {1/(1-HIT):.1f}x")
print(f"naive prefill FLOPs: {tot_naive*GFLOP_PER_TOK/1e6:.1f} PFLOP"
      f" vs ideal {tot_ideal*GFLOP_PER_TOK/1e6:.3f} PFLOP")

json.dump({"K": K, "hit_rate": HIT, "rows": rows,
           "total_naive": tot_naive, "total_ideal": tot_ideal},
          open(f"{BASE}/cost_table.json", "w"), indent=1)
print(f"\n[out] {BASE}/cost_table.json")
