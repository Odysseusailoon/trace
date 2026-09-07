"""E2c: automatic fork decision rule on decision-step replay data.

Rule: fork at turn d  <=>  TVD(o_d, o_{d+1}) significant under the pooled
multinomial null (B simulations, Bonferroni over the D-1 adjacent pairs)
AND persistence_d < 0.9 (the recorded decision at turn d was contested).

Also runs a synthetic-null false-positive check: pairs drawn from the SAME
pooled distribution should almost never be flagged.

Local only — reads data/reports/replay_*.json.
"""
import glob
import json
import random

ALPHA = 0.05
B = 20000
PERSIST_MAX = 0.9
BASE = "/Users/yifeichen/Projects/research/vector/forkscope/data/reports"
EXPECTED = {  # human judgment to reproduce
    "replay_t4_artist_album_ratio_sql_intdiv_10000.json": [0],
    "replay_t4_artist_album_ratio_sql_avg_groupby_10015.json": [0],
    "replay_t7_avg_track_len_min_sql_avg_ms_raw_10000.json": [],
    "replay_t7_avg_track_len_min_sql_avg_ms_raw_10003.json": [1],
}

rng = random.Random(0)


def tvd(c1: dict, c2: dict) -> float:
    n1, n2 = sum(c1.values()), sum(c2.values())
    keys = set(c1) | set(c2)
    return 0.5 * sum(abs(c1.get(k, 0) / n1 - c2.get(k, 0) / n2) for k in keys)


def sim_null_p(c1: dict, c2: dict, obs: float) -> float:
    """P(TVD >= obs) when both samples come from the pooled distribution."""
    n1, n2 = sum(c1.values()), sum(c2.values())
    keys = sorted(set(c1) | set(c2))
    pooled = [(c1.get(k, 0) + c2.get(k, 0)) for k in keys]
    tot = sum(pooled)
    probs = [c / tot for c in pooled]
    ge = 0
    for _ in range(B):
        a = draw(n1, probs)
        b = draw(n2, probs)
        t = 0.5 * sum(abs(a[i] / n1 - b[i] / n2) for i in range(len(keys)))
        if t >= obs - 1e-12:
            ge += 1
    return ge / B


def draw(n: int, probs: list) -> list:
    counts = [0] * len(probs)
    for _ in range(n):
        r = rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    return counts


print(f"rule: TVD sig at alpha={ALPHA} (Bonferroni) AND persistence < {PERSIST_MAX}\n")
all_match = True
for path in sorted(glob.glob(f"{BASE}/replay_*.json")):
    r = json.load(open(path))
    steps = r["steps"]
    m = len(steps) - 1  # adjacent pairs
    flags = []
    rows = []
    for i in range(m):
        c1, c2 = steps[i]["o_d"], steps[i + 1]["o_d"]
        obs = tvd(c1, c2)
        p = sim_null_p(c1, c2, obs)
        sig = p < ALPHA / m
        contested = (steps[i]["persistence"] or 1.0) < PERSIST_MAX
        fork = sig and contested
        if fork:
            flags.append(steps[i]["d"])
        rows.append((steps[i]["d"], obs, p, sig, contested, fork))
    name = path.split("/")[-1]
    exp = EXPECTED[name]
    ok = flags == exp
    all_match &= ok
    print(f"{name}  ({r['episode_outcome']})")
    for d, obs, p, sig, contested, fork in rows:
        print(f"  d={d}->{d+1}: TVD={obs:.3f} p={p:.4f} sig={sig} "
              f"persist_d={steps[d]['persistence']:.2f} contested={contested}"
              f"{'  <-- FORK' if fork else ''}")
    print(f"  detected={flags} expected={exp} {'MATCH' if ok else 'MISMATCH'}\n")

print("=== synthetic null false-positive check ===")
fp = 0
trials = 200
for t in range(trials):
    probs = [0.5, 0.3, 0.2]
    c1 = dict(zip("abc", draw(50, probs)))
    c2 = dict(zip("abc", draw(50, probs)))
    obs = tvd(c1, c2)
    if sim_null_p(c1, c2, obs) < ALPHA:
        fp += 1
print(f"false-positive rate at alpha={ALPHA}: {fp}/{trials} = {fp/trials:.3f} (expect ~{ALPHA})")
print(f"\nALL HUMAN JUDGMENTS REPRODUCED: {all_match}")
