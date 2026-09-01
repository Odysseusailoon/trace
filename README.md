# forkscope

**Counterfactual fork analysis for LLM agent trajectories — measure which step caused the failure, don't guess.**

Spent ~$20 and a weekend on a single H100 reproducing [Forking Fast](https://arxiv.org/abs/2608.19611) (Goodfire's forking-paths research), then extending it to agents. Major takeaway: **AI agent failures are often hard-locked at a single decision step — and that step looks completely benign.**

## 1. How it works: token-level vs. agent-level

**Token-level** (original): greedy-decode a baseline path and record top-k candidate distributions. At each position, swap in alternative tokens (p≥0.05) and sample S continuations to construct position-wise outcome distributions O_t. Forking Fast proved O_t is flat with sparse forks, using PELT change-point detection + Dirichlet pooling to boost effective sample efficiency by 4–15×.

**Agent decision-level** (our extension): shift granularity from tokens to decision boundaries (before assistant output / after tool context returns). Truncate and replay K=50 full continuations per step — real chat templates, real tool execution. The step where O_d abruptly collapses from a mixture into a single point is your causal fork.

> A fork isn't where the model is hesitant; it's where taking a different path changes the final outcome. No LLM reading logs to guess — just pure counterfactual execution.

## 2. Counterfactual replay is practically free

Replaying 50 full completions across every step normally costs millions of tokens. Because continuations share prefixes, RadixAttention (SGLang) makes prefill essentially free: **98.5% cache hit rate, cutting cost 60–100× vs. naive**. Total spend on Qwen3-8B: **~$19.50**.

## 3. Three distinct failure modes

Observability trace logs miss this taxonomy completely:

| Mode | Signature | Fix lives in |
|---|---|---|
| **Knowledge-level** | confidently wrong start to finish; zero forks | prompt / model — sampling can't save it |
| **Drift-level** | oscillates without converging | sampling / aggregation |
| **Decision-level** | trajectory permanently locked at one step | that step's context — almost all agent tasks fall here |

## 4. The hidden trap

In a text-to-SQL task, starting with `AVG ... GROUP BY` yielded 100% success. Choosing `COUNT(*)/COUNT(DISTINCT ArtistId)` triggered SQLite integer division, returning 1. Valid SQL, zero errors, full confidence — self-healing rate 2/126. The choice was primed by schema text in 46% of episodes.

In another case, 98% of episodes ran identical initial SQL with identical outputs. What determined success was a single internal phrase: *"divide by 60000"*. Only ~10% of continuations called a calculator to convert units; the rest reported seconds as minutes. Early phrasing quietly warps downstream decision distributions.

## 5. Entropy is a poor proxy for causal forks

RL methods (ARPO, AEPO, Tree-GRPO) rely on token-entropy spikes to locate critical steps. In our data: only 1 causal fork fell in the top-20 entropy steps; the highest-entropy tokens were alias names and punctuation — not the fatal `COUNT`. One trajectory destined to fail 92% of the time kept near-zero entropy throughout. **Token entropy is a path-dependent, severity-blind local signal; causal forks are global state properties.**

## 6. Closing the loop

Feeding the causal fork report (not raw logs) back to the agent to edit only its tool descriptions: **27% → 100%** on a 200-vs-200 retest. Standard LLM log analysis locates the error step only ~14% of the time (Who&When, ICML'25).

Open question (in progress): if we apply minimal-KL updates specifically at detected causal forks — rather than full-trajectory RFT — can we fix targeted errors without catastrophic forgetting? The upper bound of self-improvement equals the causal density of your learning signal.

## Repo map

```
forkscope/
  src/forkscope/              pipeline: base_path → fork_enum → resampler →
                              extractor → aggregate → smoothing (PELT+Dirichlet) → report
  agentenv/                   text-to-SQL agent env: tasks, tools, runner, replay
  scripts/                    run_pipeline, build_reference, rsi_loop, cost table
  configs/default.yaml        all hyperparameters (S, spacing, thresholds)
  data/reports/               measured results: validations, replays, entropy, RSI loop
anim/                         Manim scenes for every figure/video in the writeup
```

## Quickstart

```bash
# 1. serve (deterministic inference; fixed seeds => bitwise-reproducible replays)
python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
  --enable-deterministic-inference --port 30000
# 2. run the pipeline
python forkscope/scripts/run_pipeline.py --config forkscope/configs/default.yaml
```

Upstream: [ericb-goodfire/forking-fast](https://github.com/ericb-goodfire/forking-fast) · Forking Paths (ICLR'25, arXiv:2412.07961) · Forking Fast (arXiv:2608.19611)
