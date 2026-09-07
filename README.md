# forkscope

**Counterfactual fork analysis for LLM agent trajectories: measure which step caused the failure, don't guess.**

A weekend reproduction of [Forking Fast](https://arxiv.org/abs/2608.19611) (Goodfire) on one H100, extended from token positions to agent tool calls. Main finding: **agent failures are often hard-locked at a single decision step, and that step looks completely benign.**

> **Single source of truth: [`repro-2608.19611-2026-09-03.md`](repro-2608.19611-2026-09-03.md).** It carries the current numbers, the sampling protocol, what is validated, what is not, and what failed. Where this README and that document disagree, the document is right.

## 1. How it works

**Token-level** (the original). Greedy-decode a base path, recording the top-k candidate distribution at every position. At each probed position, swap in an alternative token and sample S continuations, giving a position-wise outcome distribution `o_t`. Sampling protocol matters and is easy to get wrong: the base path is greedy (temperature 0, top-k 1) and the continuations are **untruncated ancestral samples** (temperature 1.0, top-p 1.0, top-k off) with a distinct seed per copy. Sampling continuations under top-p 0.95 would measure a different model than the one being served.

**Agent decision-level** (our extension). Move the granularity from tokens to decision boundaries, before an assistant turn and after tool context returns. Replay K=50 full continuations per step with real chat templates and real tool execution. The step where `o_d` collapses from a mixture to a point is the causal fork.

> A fork is not where the model is hesitant. It is where taking a different path changes the final outcome.

**Two curves, and they answer different questions.** The mixture `o_t = Σ_w p̃(w)·hist(t,w)` is a probability-weighted average over branches, so a fork carried by a low-probability branch gets divided away before any test sees it. `fork_score(t) = max over branch pairs TVD(o(t,w1), o(t,w2))` compares the branches against each other instead. It is per-position, needs no neighbour, and does not depend on spacing. Both are computed; see [`src/forkscope/branchstat.py`](forkscope/src/forkscope/branchstat.py).

## 2. What the cost optimizations actually buy

Prefix caching gets quoted as an order of magnitude. It is not, and the arithmetic says why. With base path length `T`, spacing `s`, `B` branches per position, `S` samples per branch, prompt length `P`, mean continuation length `L`:

```
C_naive = (T/s)·B·S·(P + t̄ + L)     every continuation re-prefills its own prefix
C_radix ≈ (T/s)·B·[(P + t̄) + S·L]   the prefix is prefilled once per branch, decoded S times
```

Measured on SGLang: **97.8% of submitted prompt tokens served from the radix cache** (40,609 of 41,540), sustained **6,800 tok/s** for Qwen3-8B on one H100. But with `S = 100`, `P + t̄ ≈ 1.1k` and `L ≈ 1.5k`, that ratio is about **1.8x**, not 60x. Caching removes prefill; decode dominates and caching does not touch it.

Against the dense reference (`s = 1`, `S = 200`), the full stack is:

| lever | factor | costs you |
|---|---|---|
| spacing 4 | 4x | coverage |
| `S` 200 → 100 | 2x | statistical power |
| skip positions with no second branch | 1.8x | **nothing** (44% of positions at a 0.02 threshold) |
| prefix caching | 1.8x | **nothing** |

About **14x realized**, **26x** with the free skip applied, of which **3.2x is genuinely free**. The lever people reach for first is the smallest of the four.

## 3. Three failure modes

| Mode | Signature | Fix lives in |
|---|---|---|
| **Knowledge-level** | confidently wrong start to finish, zero forks | prompt or model; sampling cannot save it |
| **Drift-level** | oscillates without converging | sampling or aggregation |
| **Decision-level** | trajectory permanently locked at one step | that step's context |

## 4. The traps, measured

On a text-to-SQL ratio task, opening with `AVG ... GROUP BY` succeeded 35/35. Opening with `COUNT(*)/COUNT(DISTINCT ArtistId)` triggers SQLite integer division and returns 1: valid SQL, no error, full confidence, self-heal rate 2/126. The first query splits 91 of 200 episodes into that cluster and **none of the 91 recover to the correct answer**.

Mirror replay isolates it. Rewind a failed and a successful episode to the same pre-decision state and both give `P(correct) = 0.22` at K=50; replay after each one's own first query and they separate to 0.00 and 1.00. Same state, same model, one different action.

## 5. Entropy is a gate, not a ranking

A token whose next-token distribution is a point mass **cannot** be a fork: there is no alternative to resample, so skipping it is free. The whole tool-call envelope sits there, measured at 0.000 bits through `<tool_call>{"name": "sql_query", "arguments": {"query": "`, while the tokens that decide the query (` COUNT`, `(*)`, ` /`) sit at 0.63, 0.53 and 0.60.

So entropy is a safe filter for what *cannot* fork, and it is the same free lever as skipping branch-less positions. It is not a ranking of what *does* fork: 0.5 to 0.6 bits is low in absolute terms, and the decisive steps we measured are low-entropy commitments rather than high-entropy hesitations. A threshold tuned high enough to save real money starts dropping exactly these tokens.

## 6. Closing the loop

Feeding the causal fork report, not raw logs, back to the agent to edit only its tool descriptions: **27% → 100%** (54/200 → 200/200 on a held-out retest). For comparison, LLM log analysis locates the error step about 14% of the time (Who&When, arXiv:2505.00212).

## 7. Status: what is validated and what is not

Read the [source-of-truth document](repro-2608.19611-2026-09-03.md) for the full account. In short:

- **Validated.** The agent arm. Outcome labels come from executed SQL, not from parsing prose. Replay agrees with independent from-scratch collection on the start-state distribution, and the fork rule is automated (pooled multinomial null, 20,000 simulations, Bonferroni, persistence filter).
- **Open dataset.** 10 BIRD mini-dev questions, selected in two stages following the screening protocol of Zur et al. (arXiv:2511.04527): a deterministic surface-form filter, then a 10-episode screen keeping only questions whose modal answer occurs 4 to 6 times out of 10. Accuracy over 200 episodes per question per arm: Qwen3-8B 5.9% no-thinking / 30.1% thinking, Qwen3.6-35B-A3B 69.6% / 63.8%. Thinking is worth +24.2 points to the 8B and **−5.8** to the 35B.
- **Being re-measured.** The token-level MMLU arm. Five defects, found by auditing recorded continuations: the mixture curve attenuates the signal it is meant to detect; at a 0.05 branch threshold most positions had no second branch to compare against; `S = 20` put the critical TVD near 0.55; a 1,500-token cap truncated 77% and 50% of continuations; and the answer extractor could read an option letter out of the reasoning text. **No number from this arm should be quoted, including any "~1% of positions fork" figure.** See `configs/dense.yaml` for the corrected configuration.
- **Failed.** The probe. Its labels came from the arm above, so it was trained against a target that was partly an artifact of a generation cap.
- **Next.** MoE routing. In a sparse MoE, expert parameters are disjoint, so `g_i(x) = 0 ⟹ ∂output/∂θ_{expert i} = 0` and the Jacobian of any behaviour is block-sparse, supported only on the experts that route. That makes "a narrow behaviour touches a small fraction of the weights" a measurable claim about routing specialization rather than an assertion. [`scripts/moe_routing.py`](forkscope/scripts/moe_routing.py) tests whether routing divergence at a fork position predicts `fork_score` better than token entropy does.

## Repo map

```
forkscope/
  src/forkscope/            base_path → fork_enum → resampler → extractor →
                            aggregate → branchstat → smoothing (PELT+Dirichlet) → report
    branchstat.py           between-branch dispersion: fork_score, greedy_gap, pooled null
    extractor/mcq_strict.py terminal vs commitment outcome definitions
  agentenv/                 text-to-SQL agent env: tasks, tools, runner, replay
  scripts/
    run_pipeline.py         end-to-end; --plan-only prints the sampling bill first
    branch_stats.py         fork_score from recorded branches, no GPU
    probe_len.py            how large the continuation cap must be
    moe_routing.py          routing divergence vs fork_score (MoE)
    select_bird_e2b.py      frozen surface-form stage of BIRD selection
    fork_rule.py            the agent-side fork decision rule
  configs/dense.yaml        the corrected configuration, with the reasons inline
  data/reports/             measured results
```

## Quickstart

```bash
# 1. serve (deterministic inference; fixed seeds => bitwise-reproducible replays)
python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --enable-deterministic-inference --port 30000

# 2. see what a sweep will cost before paying for it
python forkscope/scripts/run_pipeline.py --config forkscope/configs/dense.yaml \
  --case virology_5 --mmlu virology:5 --data-dir data/mcq8b --plan-only

# 3. run it
python forkscope/scripts/run_pipeline.py --config forkscope/configs/dense.yaml \
  --case virology_5 --mmlu virology:5 --data-dir data/mcq8b

# 4. recompute statistics from recorded branches, no GPU
python forkscope/scripts/branch_stats.py --case virology_5 --data-dir data/mcq8b
```

## Lineage

[Forking Paths](https://arxiv.org/abs/2412.07961) (Bigelow et al., ICLR'25) → [Forking Fast](https://arxiv.org/abs/2608.19611) ([ericb-goodfire/forking-fast](https://github.com/ericb-goodfire/forking-fast)) → [Are language models aware of the road not taken?](https://arxiv.org/abs/2511.04527) (Zur, Geiger, Lubana, Bigelow), which is also where our BIRD screening protocol comes from.
