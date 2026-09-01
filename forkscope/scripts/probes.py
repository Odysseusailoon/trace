"""M1 probes: verify SGLang /generate API behaviors forkscope depends on.

Probe 1: greedy + return_logprob + top_logprobs_num=10 response shape.
Probe 2: n=S parallel sampling with input_ids prefix.
Probe 3: seed reproducibility (same seed -> identical continuations).
Probe 4: radix cache effect (repeat same prefix -> cached_prompt_tokens > 0).

Usage: python scripts/probes.py --port 30000 --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json

import httpx
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    tok = AutoTokenizer.from_pretrained(args.model)
    msgs = [
        {"role": "system", "content": "You are a careful math tutor. End with 'The answer is (X).'"},
        {"role": "user", "content": "What is 17 + 26? A) 33 B) 43 C) 53 D) 42"},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"[probe] prompt tokens: {len(ids)}")

    # Probe 1: greedy + top-k logprobs
    r = httpx.post(base + "/generate", json={
        "input_ids": ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 64, "top_k": 1, "top_p": 1.0},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 10,
    }, timeout=300)
    r.raise_for_status()
    resp = r.json()
    if isinstance(resp, list):
        resp = resp[0]
    meta = resp["meta_info"]
    otl = meta.get("output_token_logprobs") or []
    otop = meta.get("output_top_logprobs") or []
    print(f"[probe1] finish_reason={meta.get('finish_reason')} "
          f"gen_tokens={len(otl)} top_rows={len(otop)} "
          f"completion_tokens={meta.get('completion_tokens')}")
    assert len(otl) == len(otop), "token_logprobs and top_logprobs misaligned"
    print(f"[probe1] first top row (truncated): {json.dumps(otop[0][:3])}")
    print(f"[probe1] greedy text: {resp['text'][:200]!r}")
    gen_ids = [lp[1] for lp in otl]

    # Probe 2: n=4 parallel sampling from prefix
    prefix = ids + gen_ids[:8]
    r = httpx.post(base + "/generate", json={
        "input_ids": prefix,
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                            "max_new_tokens": 32, "n": 4, "sampling_seed": 1234},
    }, timeout=300)
    r.raise_for_status()
    resp2 = r.json()
    n_returned = len(resp2) if isinstance(resp2, list) else 1
    print(f"[probe2] n=4 -> {n_returned} completions")
    texts = [x["text"] for x in (resp2 if isinstance(resp2, list) else [resp2])]
    print(f"[probe2] distinct continuations: {len(set(texts))}/4")

    # Probe 3: seed reproducibility
    r3 = httpx.post(base + "/generate", json={
        "input_ids": prefix,
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                            "max_new_tokens": 32, "n": 4, "sampling_seed": 1234},
    }, timeout=300)
    resp3 = r3.json()
    texts3 = [x["text"] for x in (resp3 if isinstance(resp3, list) else [resp3])]
    print(f"[probe3] same seed reproduces: {texts == texts3}")

    # Probe 4: radix cache — repeat probe2 request, check cached tokens
    meta2 = (resp2[0] if isinstance(resp2, list) else resp2)["meta_info"]
    print(f"[probe4] first-call cached_prompt_tokens={meta2.get('cached_prompt_tokens')} "
          f"prompt_tokens={meta2.get('prompt_tokens')}")

    print("[probes] ALL DONE")


if __name__ == "__main__":
    main()
