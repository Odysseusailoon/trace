"""Stage 1: greedy base path + per-position top-k candidate distribution.

Output schema (branch-record friendly, JSON):
{
  "case_id": str,
  "prompt_ids": [int],
  "gen_ids": [int],
  "gen_strs": [str],
  "top_logprobs": [[(token_id, logprob), ...]],   # aligned with gen_ids
  "finish_reason": "stop" | "length",
}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .client import SGLangClient


@dataclass
class BasePath:
    case_id: str
    prompt_ids: list[int]
    gen_ids: list[int]
    gen_strs: list[str]
    top_logprobs: list[list[tuple[int, float]]]
    finish_reason: str
    prompt_logprobs: list | None = field(default=None)

    @property
    def full_ids(self) -> list[int]:
        return self.prompt_ids + self.gen_ids

    def prefix_upto(self, t: int, branch_token: int) -> list[int]:
        return self.prompt_ids + self.gen_ids[:t] + [branch_token]

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f)

    @staticmethod
    def load(path: str | Path) -> "BasePath":
        with open(path) as f:
            d = json.load(f)
        d["top_logprobs"] = [[tuple(x) for x in row] for row in d["top_logprobs"]]
        return BasePath(**d)


def _parse_generate_response(resp, case_id: str, prompt_ids: list[int]) -> BasePath:
    if isinstance(resp, list):
        resp = resp[0]
    meta = resp["meta_info"]
    out_token_logprobs = meta.get("output_token_logprobs") or []
    out_top_logprobs = meta.get("output_top_logprobs") or []

    gen_ids = [int(lp[1]) for lp in out_token_logprobs]
    gen_strs = []
    top: list[list[tuple[int, float]]] = []
    for i, row in enumerate(out_top_logprobs):
        top.append([(int(tid), float(lp)) for lp, tid, *_rest in row])
        # token text comes from output_token_logprobs 3rd element when present
    for lp in out_token_logprobs:
        gen_strs.append(lp[2] if len(lp) > 2 else "")

    # drop trailing EOS row if finish_reason == "stop" and last row is eos-like:
    # SGLang already excludes the stop token from output_token_logprobs when
    # stop_token is hit, so no trimming needed in the common case.

    return BasePath(
        case_id=case_id,
        prompt_ids=list(prompt_ids),
        gen_ids=gen_ids,
        gen_strs=gen_strs,
        top_logprobs=top,
        finish_reason=meta.get("finish_reason", {}).get("type", "unknown")
        if isinstance(meta.get("finish_reason"), dict)
        else str(meta.get("finish_reason", "unknown")),
    )


async def build_base_path(
    client: SGLangClient,
    case_id: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    top_k: int = 10,
) -> BasePath:
    resp = await client.greedy_with_topk(prompt_ids, max_new_tokens, top_k)
    return _parse_generate_response(resp, case_id, prompt_ids)
