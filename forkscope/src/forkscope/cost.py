"""Cost accounting from SGLang /metrics (prometheus text)."""
from __future__ import annotations

import re


def parse_metrics(text: str) -> dict:
    out = {}
    pats = {
        "prompt_tokens": r'sglang:prompt_tokens_total.*?(\d+(?:\.\d+)?)\s*$',
        "generation_tokens": r'sglang:generation_tokens_total.*?(\d+(?:\.\d+)?)\s*$',
        "cache_hit_rate": r'sglang:cache_hit_rate.*?(\d+(?:\.\d+)?)\s*$',
        "cached_tokens": r'sglang:cached_tokens_total.*?(\d+(?:\.\d+)?)\s*$',
    }
    for key, pat in pats.items():
        m = re.findall(pat, text, re.MULTILINE)
        if m:
            out[key] = float(m[-1])
    return out


def cost_report(actual: dict, naive_tokens: float) -> dict:
    gen = actual.get("generation_tokens", 0.0)
    prompt = actual.get("prompt_tokens", 0.0)
    total = gen + prompt
    return {
        "actual_total_tokens": total,
        "naive_total_tokens": naive_tokens,
        "savings_x": (naive_tokens / total) if total else None,
        "cache_hit_rate": actual.get("cache_hit_rate"),
    }
