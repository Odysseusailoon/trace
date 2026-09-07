"""Wilson CIs for all headline proportions + MI bootstrap + o_0 cross-validation.

Local only. Sources: validation_t4 JSON (cluster table), documented counts for
RSI/noleak/t10 (from rsi_loop.json / rsi_noleak2.json / validation_t10 runs).
"""
import json
import math
import random

BASE = "/Users/yifeichen/Projects/research/vector/forkscope/data/reports"


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, ctr - hw), min(1, ctr + hw)


def show(label, k, n):
    p, lo, hi = wilson(k, n)
    print(f"{label:52s} {k:>4d}/{n:<4d} = {p:6.1%}  [{lo:5.1%}, {hi:5.1%}]")


print("=== headline proportions, Wilson 95% CI ===")
show("t4 first-SQL intdiv share", 91, 200)
show("t4 intdiv-cluster correct", 0, 91)
show("t4 avg_groupby-cluster correct", 35, 35)
show("t4 self-heal (intdiv entered, recovered)", 2, 126)
show("t4 overall correct (collection)", 61, 200)
show("RSI control correct", 54, 200)
show("RSI patched correct", 200, 200)
show("noleak t4 warning-arm correct", 31, 200)
show("noleak t4 warning-arm wrongtable", 65, 200)
show("noleak t7 control correct", 34, 200)
show("noleak t7 warning-arm correct", 75, 200)
show("t10 decoy answer 67.3", 187, 200)

print("\n=== MI bootstrap (t4 first-decision cluster x outcome) ===")
v = json.load(open(f"{BASE}/validation_t4_artist_album_ratio.json"))
pairs = []
for c in v["clusters"]:
    for outcome, k in c["outcomes"].items():
        lab = "correct" if outcome == "correct" else "wrong"
        pairs += [(c["cluster"], lab)] * k
n = len(pairs)


def mi(sample):
    from collections import Counter
    cj = Counter(sample)
    cx = Counter(a for a, _ in sample)
    cy = Counter(b for _, b in sample)
    m = len(sample)
    s = 0.0
    for (a, b), k in cj.items():
        pxy = k / m
        s += pxy * math.log2(pxy / (cx[a] / m * cy[b] / m))
    return s


rng = random.Random(0)
point = mi(pairs)
boots = sorted(mi([pairs[rng.randrange(n)] for _ in range(n)]) for _ in range(2000))
print(f"MI(first decision; outcome) = {point:.3f} bits, bootstrap 95% CI "
      f"[{boots[50]:.3f}, {boots[1949]:.3f}]  (n={n})")

print("\n=== o_0 cross-validation: replay vs collection (same state, independent seeds) ===")


def two_prop(k1, n1, k2, n2, label):
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    print(f"{label}: replay {k1}/{n1}={p1:.1%} vs collection {k2}/{n2}={p2:.1%}"
          f"  z={z:+.2f} -> {'compatible' if abs(z) < 1.96 else 'DISCREPANT'}")


two_prop(11, 50, 61, 200, "P(correct | t4 start)")
two_prop(27, 50, 91, 200, "P(first decision = intdiv | t4 start)")
