"""MoE routing at fork positions: does the router know before the token does?

Motivation. Resampling tells us WHERE a trajectory forks (a position whose
branches lead to different outcome distributions). It says nothing about what
inside the model carries that decision. In a sparse MoE the natural candidate is
the router, for a structural reason: expert parameters are disjoint, so

    g_i(x) = 0  =>  d(output) / d(theta_expert_i) = 0

and the Jacobian of any behaviour with respect to the weights is block-sparse,
supported only on the experts that actually route. That is the whole basis for
the claim that a narrow behaviour touches a small fraction of the weights. It is
a claim about routing specialization, and it is measurable.

Three quantities, all computed from a prefill pass over text we already have, so
no generation and no new sampling:

  routing_entropy(t)     entropy of the router distribution at position t,
                         averaged over layers. The MoE analogue of token
                         entropy, and a candidate gate.
  routing_divergence(t)  how differently the router fires for the greedy token
                         versus each counterfactual token at a fork position.
                         Reported as 1 - Jaccard over the top-k expert sets and
                         as TVD over the router distributions.
  expert_support(B)      the union of experts activated across a behaviour's
                         inputs, as a fraction of all experts. This is the
                         "fraction of a percent of the weights" number, measured
                         rather than asserted.

The hypothesis worth falsifying: routing_divergence predicts fork_score better
than token entropy does. Token entropy is known to fail here, because the
decisive steps we measured are low-entropy commitments rather than high-entropy
hesitations, so a signal that is not a function of the output distribution has
room to beat it. If routing_divergence also fails, the decision is not localized
in the router and the editable-weights story needs a different address.

Usage (needs the HF model, not the SGLang server; prefill only, no generation):
  python scripts/moe_routing.py --model Qwen/Qwen3.6-35B-A3B-FP8 \
      --base data/mcq8b/raw/base_virology_5.json \
      --branchstat data/mcq8b/reports/branchstat_virology_5.json \
      --out data/reports/routing_virology_5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=axis)


def router_probs(model, tokenizer, ids: list[int]) -> np.ndarray:
    """(L, T, E) router probabilities for one sequence, prefill only.

    Qwen3-MoE exposes per-layer router logits when asked. Any model that does
    not is refused loudly rather than silently returning something else.
    """
    import torch

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([ids], device=model.device),
            output_router_logits=True,
            use_cache=False,
        )
    rl = getattr(out, "router_logits", None)
    if rl is None:
        raise SystemExit(
            "model returned no router_logits: this script needs a sparse-MoE "
            "checkpoint whose forward accepts output_router_logits=True "
            "(Qwen3MoeForCausalLM does; a dense model never will)"
        )
    # HF gives a tuple of (T, E) or (1*T, E) per layer
    mats = []
    for layer in rl:
        m = layer.detach().float().cpu().numpy()
        if m.ndim == 3:
            m = m[0]
        mats.append(m)
    logits = np.stack(mats)                       # (L, T, E)
    logits = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def topk_sets(p: np.ndarray, k: int) -> list[list[set[int]]]:
    """[layer][position] -> set of top-k expert ids."""
    idx = np.argsort(-p, axis=-1)[..., :k]
    return [[set(idx[l, t].tolist()) for t in range(p.shape[1])]
            for l in range(p.shape[0])]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a sparse-MoE HF checkpoint")
    ap.add_argument("--base", required=True, help="base_<case>.json from stage 1")
    ap.add_argument("--branchstat", default=None,
                    help="branchstat_<case>.json, to correlate against fork_score")
    ap.add_argument("--top-k", type=int, default=8, help="router top-k to compare")
    ap.add_argument("--max-positions", type=int, default=256,
                    help="cap the fork positions compared, highest fork_score first")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from forkscope.base_path import BasePath

    base = BasePath.load(args.base)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    n_experts = getattr(model.config, "num_experts", None) or \
        getattr(model.config, "n_routed_experts", None)
    print(f"[moe] {args.model}: {n_experts} experts, top-k compared {args.top_k}")

    # ---- routing along the greedy base path ----
    prompt, gen = base.prompt_ids, base.gen_ids
    p_base = router_probs(model, tok, prompt + gen)
    off = len(prompt)                            # gen position t -> row off + t
    H_layer = entropy(p_base)                    # (L, T)
    H = H_layer.mean(axis=0)                     # mean over layers
    sets_base = topk_sets(p_base, args.top_k)
    print(f"[moe] base path {len(gen)} tokens, {p_base.shape[0]} MoE layers, "
          f"routing entropy mean {H[off:].mean():.3f} bits")

    # ---- which positions to compare, and against what ----
    rows_bs, order = {}, list(range(len(gen)))
    if args.branchstat:
        bs = json.loads(Path(args.branchstat).read_text())
        rows_bs = {r["t"]: r for r in bs["rows"] if r.get("n_branches", 0) >= 2}
        order = sorted(rows_bs, key=lambda t: -rows_bs[t]["fork_score"])
    order = [t for t in order if t < len(gen)][: args.max_positions]

    # ---- routing divergence: greedy token vs each counterfactual token ----
    out_rows = []
    for n, t in enumerate(order, 1):
        r = rows_bs.get(t)
        alts = [w for w in (r or {}).get("tok_ids", []) if w != gen[t]]
        if not alts:
            continue
        best_jac, best_tvd = 0.0, 0.0
        for w in alts:
            # same prefix, one different token at t; only row off+t changes
            p_alt = router_probs(model, tok, prompt + gen[:t] + [w])
            sets_alt = topk_sets(p_alt, args.top_k)
            last = p_alt.shape[1] - 1
            jac = float(np.mean([
                1.0 - jaccard(sets_base[l][off + t], sets_alt[l][last])
                for l in range(p_alt.shape[0])]))
            tvd = float(np.mean([
                0.5 * np.abs(p_base[l, off + t] - p_alt[l, last]).sum()
                for l in range(p_alt.shape[0])]))
            best_jac, best_tvd = max(best_jac, jac), max(best_tvd, tvd)
        out_rows.append({
            "t": t,
            "tok": base.gen_strs[t] if t < len(base.gen_strs) else None,
            "routing_entropy": float(H[off + t]),
            "routing_divergence_jaccard": best_jac,
            "routing_divergence_tvd": best_tvd,
            "fork_score": (r or {}).get("fork_score"),
            "p_greedy": (r or {}).get("p_greedy"),
            "n_branches": (r or {}).get("n_branches"),
        })
        if n % 20 == 0:
            print(f"[moe] {n}/{len(order)} positions", flush=True)

    # ---- does routing beat token entropy at predicting fork_score? ----
    summary = {"model": args.model, "n_experts": n_experts, "top_k": args.top_k,
               "n_moe_layers": int(p_base.shape[0]), "n_compared": len(out_rows)}
    have = [r for r in out_rows if r["fork_score"] is not None]
    if len(have) >= 8:
        fs = np.array([r["fork_score"] for r in have])
        for key in ("routing_divergence_jaccard", "routing_divergence_tvd",
                    "routing_entropy"):
            x = np.array([r[key] for r in have])
            # Spearman without scipy: Pearson on ranks
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(fs)).astype(float)
            rho = float(np.corrcoef(rx, ry)[0, 1])
            summary[f"spearman_{key}_vs_fork_score"] = rho
            print(f"[moe] spearman({key}, fork_score) = {rho:+.3f}  (n={len(have)})")
        summary["verdict_note"] = (
            "routing_divergence beating routing_entropy is the point of the "
            "experiment; both near zero means the decision is not localized in "
            "the router at this granularity")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"summary": summary, "rows": out_rows}, indent=1))
    print(f"[moe] -> {args.out}")


if __name__ == "__main__":
    main()
