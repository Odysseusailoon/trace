"""Alignment analysis: agent-side token entropy vs measured fork step.

Inputs:
  data/reports/entropy_agent_raw.json  (from node: per-turn token entropies)
  data/reports/transcripts.json        (recorded episodes, for reproduction check)
  data/reports/replay_*.json           (fork ground truth: o_d + persistence)

Fork ground truth (from replay o_d jumps):
  t4_10000 fork turn = 0 (SQL write locks intdiv: o 22%->0%)
  t4_10015 fork turn = 0 (SQL write locks correct: o 22%->100%)
  t7_10000 fork turn = 1 (answer-without-conversion locks wrong_unit)
  t7_10003 fork turn = 1 (calculator /60000 conversion, persistence 0.10)
"""
import json
import statistics

BASE = "/Users/yifeichen/Projects/research/vector/forkscope/data/reports"
FORK_TURN = {"t4_10000": 0, "t4_10015": 0, "t7_10000": 1, "t7_10003": 1}

raw = json.load(open(f"{BASE}/entropy_agent_raw.json"))
trans = json.load(open(f"{BASE}/transcripts.json"))


def norm_calls(tcs):
    out = []
    for tc in tcs or []:
        f = tc.get("function", {})
        a = f.get("arguments", "")
        out.append((f.get("name"), " ".join(str(a).lower().split())))
    return out


print("=== 1. reproduction check (regenerated vs recorded) ===")
for key, ep in raw.items():
    rec = trans[key]
    rec_turns = [s for s in rec["steps"] if s["role"] == "assistant"]
    ok = len(rec_turns) == len(ep["turns"])
    detail = []
    for rt, gt in zip(rec_turns, ep["turns"]):
        same_tools = norm_calls(rt.get("tool_calls")) == norm_calls(gt.get("tool_calls"))
        rc = (rt.get("content") or "").strip()
        gc = (gt.get("content") or "").strip()
        # fallback-parse episodes keep the raw <tool_call> text in recorded content
        same_text = rc == gc or (rt.get("fallback_parse") and gc == "")
        detail.append((same_tools, same_text))
        ok = ok and same_tools
    print(f"{key}: turns {len(ep['turns'])}/{len(rec_turns)}"
          f" tools_match={all(d[0] for d in detail)}"
          f" text_match={all(d[1] for d in detail)}"
          f" final_match={(rec.get('final') or '').strip() == (ep.get('final') or '').strip()}")

print("\n=== 2. per-turn entropy stats vs fork turn ===")
rank_hits = {"mean": 0, "max": 0, "first": 0}
n_ep = 0
for key, ep in raw.items():
    fork = FORK_TURN[key]
    rows = []
    for t in ep["turns"]:
        hs = [x["H"] for x in t.get("tokens", [])]
        if not hs:
            continue
        rows.append({"turn": t["round"], "n": len(hs),
                     "mean": statistics.mean(hs), "max": max(hs),
                     "first": hs[0]})
    n_ep += 1
    print(f"\n{key} (fork turn = {fork}, {len(rows)} turns)")
    for r in rows:
        mark = " <-- FORK" if r["turn"] == fork else ""
        print(f"  turn {r['turn']:d} n={r['n']:4d} meanH={r['mean']:.3f}"
              f" maxH={r['max']:.3f} firstH={r['first']:.3f}{mark}")
    for stat in ["mean", "max", "first"]:
        best = max(rows, key=lambda r: r[stat])
        if best["turn"] == fork:
            rank_hits[stat] += 1
print(f"\nfork turn is argmax-entropy turn: mean {rank_hits['mean']}/{n_ep},"
      f" max {rank_hits['max']}/{n_ep}, first {rank_hits['first']}/{n_ep}")

print("\n=== 3. token-level: where do top-decile entropy tokens live? ===")
for key, ep in raw.items():
    fork = FORK_TURN[key]
    toks = []
    for t in ep["turns"]:
        for x in t.get("tokens", []):
            toks.append((t["round"], x["H"], x["tok"]))
    if not toks:
        continue
    hs = sorted(h for _, h, _ in toks)
    thr = hs[int(0.9 * len(hs))]
    top = [t for t in toks if t[1] >= thr]
    in_fork = sum(1 for t in top if t[0] == fork)
    fork_share = sum(1 for t in toks if t[0] == fork) / len(toks)
    print(f"{key}: top-decile n={len(top)}, in fork turn {in_fork}"
          f" ({in_fork/len(top):.0%}); fork turn holds {fork_share:.0%} of tokens")
    peaks = sorted(top, key=lambda t: -t[1])[:5]
    for rnd, h, tok in peaks:
        print(f"   H={h:.3f} turn={rnd} tok={tok!r}")
