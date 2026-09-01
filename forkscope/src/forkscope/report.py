"""Attribution report generator: fork points + outcome flips + cost.

MVP version: raw o_t only (no smoothing — V1/V2 verdict pending).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base_path import BasePath


def find_fork_points(o_t: np.ndarray, eps: float = 0.10, window: int = 1) -> list[dict]:
    """A fork at t: TVD(o_t, o_{t+window}) > eps. Returns merged consecutive runs."""
    d = np.abs(o_t[window:] - o_t[:-window]).sum(axis=1) / 2
    raw = [int(t) for t in np.where(d > eps)[0]]
    if not raw:
        return []
    forks, start, prev = [], raw[0], raw[0]
    for t in raw[1:]:
        if t - prev <= 2:
            prev = t
            continue
        forks.append({"t": start, "t_end": prev, "tvd": float(d[start:prev + 1].max())})
        start = prev = t
    forks.append({"t": start, "t_end": prev, "tvd": float(d[start:prev + 1].max())})
    return forks


def context_snippet(base: BasePath, t: int, n: int = 24, tokenizer=None) -> str:
    lo = max(0, t - n)
    if tokenizer is not None:
        return tokenizer.decode(base.gen_ids[lo:t + 1])
    return "".join(s or "" for s in base.gen_strs[lo:t + 1])


def build_report(case_id: str, o_t: np.ndarray, base: BasePath,
                 categories: list[str], cost: dict | None = None,
                 eps: float = 0.10, tokenizer=None) -> dict:
    nz = o_t.sum(axis=1) > 0
    positions = np.where(nz)[0]
    o_nz = o_t[nz]
    d = np.abs(o_nz[1:] - o_nz[:-1]).sum(axis=1) / 2
    raw = np.where(d > eps)[0]
    forks = []
    if len(raw):
        groups = np.split(raw, np.where(np.diff(raw) > 1)[0] + 1)
        for g in groups:
            i = int(g[np.argmax(d[g])])
            forks.append({"i": i, "tvd": float(d[i])})
    for f in forks:
        i = f["i"]
        t = int(positions[i])
        f["t"] = t
        f["t_next"] = int(positions[i + 1])
        f["snippet"] = context_snippet(base, t, tokenizer=tokenizer)
        before, after = o_nz[i], o_nz[i + 1]
        f["before"] = {categories[k]: round(float(before[k]), 3) for k in np.argsort(before)[::-1][:2]}
        f["after"] = {categories[k]: round(float(after[k]), 3) for k in np.argsort(after)[::-1][:2]}
        del f["i"]
    return {
        "case_id": case_id,
        "n_positions": int(len(o_nz)),
        "gen_tokens": len(base.gen_ids),
        "fork_threshold": eps,
        "forks": forks,
        "cost": cost or {},
    }


def to_markdown(rep: dict) -> str:
    lines = [f"# forkscope report: {rep['case_id']}", ""]
    lines.append(f"- positions observed: {rep['n_positions']}")
    lines.append(f"- fork threshold (TVD): {rep['fork_threshold']}")
    lines.append(f"- forks found: {len(rep['forks'])}")
    if rep.get("cost"):
        c = rep["cost"]
        lines.append(f"- cost: {c.get('actual_total_tokens', 0):.0f} tokens actual "
                     f"vs {c.get('naive_total_tokens', 0):.0f} naive "
                     f"({c.get('savings_x', 0):.1f}x saved, cache hit {c.get('cache_hit_rate', 0):.1%})")
    lines.append("")
    for i, f in enumerate(rep["forks"], 1):
        lines.append(f"## Fork {i}: t = {f['t']} -> {f['t_next']}")
        lines.append(f"> …{f['snippet']}")
        lines.append("")
        lines.append(f"- before: {f['before']}")
        lines.append(f"- after:  {f['after']}")
        lines.append(f"- flip magnitude (TVD): {f['tvd']:.3f}")
        lines.append("")
    return "\n".join(lines)


def save_report(rep: dict, out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cid = rep["case_id"]
    (out / f"report_{cid}.json").write_text(json.dumps(rep, indent=2))
    (out / f"report_{cid}.md").write_text(to_markdown(rep))
    return out
