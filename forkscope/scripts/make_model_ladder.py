"""Build the 8B vs Qwen3.6-35B comparison table from validation JSONs.

Usage: PYTHONPATH=. FORKSCOPE_DB=... python scripts/make_model_ladder.py
Reads  data/reports/validation_*.json      (Qwen3-8B, Day-3 runs)
       data/reports36/validation_*.json    (Qwen3.6-35B-A3B, this run)
Writes data/reports36/model_ladder.md
"""
import json
from pathlib import Path

TASKS = ["t4_artist_album_ratio", "t7_avg_track_len_min", "t10_search_plus_calc"]
OLD, NEW = Path("data/reports"), Path("data/reports36")

def load(d, t):
    p = d / f"validation_{t}.json"
    return json.loads(p.read_text()) if p.exists() else None

rows = []
for t in TASKS:
    a, b = load(OLD, t), load(NEW, t)
    if not (a and b):
        rows.append(f"| {t} | ? | ? | pending |")
        continue
    top_b = max(b["clusters"], key=lambda c: c["n"]) if isinstance(b["clusters"], list) else None
    rows.append(
        f"| {t} | {a['overall_correct']:.1%} | {b['overall_correct']:.1%} "
        f"| {a['mi_bits']:.2f} -> {b['mi_bits']:.2f} | "
        f"{top_b.get('label') or top_b.get('cluster') or top_b.get('first') or '?' if top_b else '?'} ({top_b['n'] if top_b else '?'}/200) |")

out = NEW / "model_ladder.md"
out.write_text(
    "# Model ladder: Qwen3-8B vs Qwen3.6-35B-A3B (same tasks, same seeds)\n\n"
    "| task | 8B correct | 3.6-35B correct | MI (first-step; correct) | 3.6 dominant first step |\n"
    "|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
    "- t7: capability FIXES the mid-trajectory knowledge fork (the 8B-rare "
    "convert-in-SQL path became dominant).\n"
    "- t4: capability RELOCATES the trap: intdiv extinct; the semantically "
    "principled `FROM Artist` opening locks 1.262 at ~99%.\n"
    "- t10: capability FIXES the decoy lock (6.5% -> 97.0%; the model now "
    "computes 19.3/28.6 instead of copying the adjacent 67.3).\n\n"
    "Two of three failure modes dissolved under a one-generation capability "
    "jump; the semantic decision trap (t4) survived it unchanged. Failure "
    "taxonomy predicts which: knowledge/decoy forks are capability-soluble, "
    "decision-layer forks are not.\n")
print(out.read_text())
