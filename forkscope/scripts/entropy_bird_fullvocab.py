"""Full-vocabulary entropy over all generated tokens of the BIRD episodes.

Per Amendment A1: for every assistant turn of every episode (both arms),
teacher-force the raw generated text on its rendered conditioning prefix and
compute, per token: exact full-vocab entropy H_full and top-10 truncated
entropy H_t10 (bridge to the E-T4/E-E1 口径).

Conditioning fidelity: prefix = tokenizer.apply_chat_template over the message
history as the runner maintained it (thinking stripped from history in th arm),
with tools and enable_thinking matching the arm. Known caveat: local template
rendering vs server rendering may differ slightly (documented 2026-09-01).
A render-consistency counter (does the recorded turn re-tokenize losslessly)
is reported per file.

Run AFTER stopping the SGLang server (needs the GPU).

Usage (on node):
  python scripts/entropy_bird_fullvocab.py --indir data/raw_bird --outdir data/entropy_bird
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/dev/forkscope")

from agentenv import tools as toolmod  # noqa: E402

MODEL = "Qwen/Qwen3-8B"


def turn_raw_text(step: dict) -> str:
    """Reconstruct the raw generated text of an assistant turn."""
    text = step.get("content") or ""
    if step.get("tool_calls") and not step.get("fallback_parse"):
        for tc in step["tool_calls"]:
            fn = tc["function"]
            args = fn.get("arguments", "{}")
            if isinstance(args, dict):
                args = json.dumps(args)
            text += f'\n<tool_call>\n{{"name": "{fn["name"]}", "arguments": {args}}}\n</tool_call>'
    return text


@torch.inference_mode()
def entropies(model, input_ids: torch.Tensor, gen_start: int):
    """H_full and H_t10 (bits) for positions [gen_start, len) of input_ids."""
    logits = model(input_ids.unsqueeze(0)).logits[0]  # [T, V]
    # logits[i] predicts token i+1 -> generated token t is predicted at t-1
    sl = logits[gen_start - 1:-1].float()
    logp = torch.log_softmax(sl, dim=-1)
    p = logp.exp()
    h_full = (-(p * logp).sum(-1) / np.log(2)).cpu().numpy()
    top = logp.topk(10, dim=-1).values
    pt = top.exp()
    h_t10 = (-(pt * top).sum(-1) / np.log(2)).cpu().numpy()
    return h_full.astype(np.float16), h_t10.astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/raw_bird")
    ap.add_argument("--outdir", default="data/entropy_bird")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-ctx", type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    for path in sorted(glob.glob(f"{args.indir}/episodes_bird_*.jsonl")):
        name = os.path.basename(path).replace("episodes_", "").replace(".jsonl", "")
        out_npz = f"{args.outdir}/H_{name}.npz"
        if os.path.exists(out_npz):
            print(f"[skip] {name}", flush=True)
            continue
        arm = name.rsplit("_", 1)[1]
        think = arm == "th"
        arrays, index, render_ok, render_n = {}, [], 0, 0

        for line in open(path):
            ep = json.loads(line)
            if ep.get("error"):
                continue
            if len(ep.get("messages", [])) < 2:
                continue
            q = ep["bird"]
            tools = toolmod.configure(
                f"{os.environ.get('BIRD_DB_ROOT', '/scratch/bird/dev_databases')}"
                f"/{q['db_id']}/{q['db_id']}.sqlite")
            # walk steps with a running history, exactly as the runner built it
            hist = [ep["messages"][0], ep["messages"][1]]
            asst_steps = [s for s in ep["steps"] if s["role"] == "assistant"]
            tool_steps = [s for s in ep["steps"] if s["role"] == "tool"]
            ti = 0
            for r, step in enumerate(asst_steps):
                prefix = tok.apply_chat_template(
                    hist, tools=tools, tokenize=False, add_generation_prompt=True,
                    enable_thinking=think)
                raw = turn_raw_text(step)
                if not raw.strip():
                    continue
                pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
                turn_ids = tok(raw, add_special_tokens=False)["input_ids"]
                if len(pre_ids) + len(turn_ids) > args.max_ctx:
                    break
                render_n += 1
                render_ok += int(tok.decode(turn_ids) == raw)
                ids = torch.tensor(pre_ids + turn_ids, device="cuda")
                h_full, h_t10 = entropies(model, ids, len(pre_ids))
                key = f"s{ep['seed']}_r{step['round']}"
                arrays[f"{key}_full"] = h_full
                arrays[f"{key}_t10"] = h_t10
                index.append({"seed": ep["seed"], "round": step["round"],
                              "n_tokens": len(turn_ids),
                              "has_tool": bool(step.get("tool_calls"))})
                # advance history the way the runner did (stripped content)
                content = step.get("content") or ""
                if think:
                    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S)
                am = {"role": "assistant"}
                if content and not step.get("fallback_parse"):
                    am["content"] = content
                if step.get("tool_calls"):
                    # server-side rendering parses arguments JSON into a
                    # mapping before the chat template sees it (and the
                    # Qwen3.6 template requires a mapping) - mirror that.
                    tcs = json.loads(json.dumps(step["tool_calls"]))
                    for tc in tcs:
                        fn = tc.get("function", {})
                        if isinstance(fn.get("arguments"), str):
                            try:
                                fn["arguments"] = json.loads(fn["arguments"])
                            except (json.JSONDecodeError, ValueError):
                                pass
                    am["tool_calls"] = tcs
                hist.append(am)
                for tc in (step.get("tool_calls") or []):
                    if ti < len(tool_steps):
                        hist.append({"role": "tool",
                                     "tool_call_id": tc.get("id", "call_0"),
                                     "content": tool_steps[ti]["result"]})
                        ti += 1

        np.savez_compressed(out_npz, **arrays)
        with open(f"{args.outdir}/H_{name}_index.json", "w") as f:
            json.dump({"index": index, "render_ok": render_ok,
                       "render_n": render_n}, f)
        tot = sum(i["n_tokens"] for i in index)
        print(f"[done] {name}: {len(index)} turns, {tot} tokens, "
              f"render_ok={render_ok}/{render_n}", flush=True)


if __name__ == "__main__":
    main()
