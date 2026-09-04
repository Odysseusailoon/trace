# Resampling-based uncertainty analysis, reproduced and pushed into agent trajectories

Weekend project, 2026-09-03. Reproduction of arXiv:2608.19611 on SGLang and vLLM, plus early
exploration past what the paper covers. Nothing here is a finished result.


## The method

LLM reasoning is stochastic, so understanding a model means grappling with the distribution of
reasoning chains it might produce for a given question, which is to say its uncertainty.
Resampling-based analysis characterizes that distribution directly: fork the rollout at a position,
sample continuations, and read off the conditional outcome distribution. Doing this along a
trajectory reveals which steps determine how the model arrives at its answer.

## Extending it to agent tool use

I scaled this to agent tool use, where it fits better than it does on plain chain of thought. A tool
call is a natural fork position: it is a discrete, semantically meaningful commitment, and the
environment state before it is reproducible. Resampling at each tool-call position gives the outcome
distribution over the actions available from the current state, which is a measurement of the agent's
uncertainty at that state rather than a proxy for it.

The reason to want that measurement is that it gives a picture of behaviour, not just performance.
You learn how reliable the model is on a given task, where it fails, and what the failure was
conditional on. That is a signal you can act on twice: at test time, by spending compute only where
the trajectory can still fork, and in post-training, by supervising the steps that actually carry the
outcome. In software engineering the belief is that if you can reproduce it, you can fix it. This is
the same bet.

![Agent-level localization: replaying from each decision boundary shows the outcome distribution
collapsing at one step, not drifting across many](forkscope/xfigs/output/plot2_t7_localization.png)

## What we have early tested on and validated

Two arms, both on Qwen3-8B and Qwen3.6-35B-A3B, served under SGLang.
The sampling protocol is the same in both: the base path is **greedy** (temperature 0, top-k 1),
recording the top-k candidate distribution at every position, and the continuations are **untruncated
ancestral samples** (temperature 1.0, top-p 1.0, top-k off) with a distinct seed per copy. The
untruncated part matters. Sampling continuations under top-p 0.95 would measure a different model
than the one being served, so `o_t` estimates the model's own conditional outcome distribution rather
than a truncated stand-in. Determinism plus per-copy seeds makes every continuation reproducible
bit-for-bit. In the agent arm the forks are tool-call boundaries and the state is restored by
replaying the message prefix; in the chain-of-thought arm the forks are token positions along the
greedy path.

### Experiments

**Agent trajectories.** Three text-to-SQL tool-use tasks against a Chinook database (an
average-per-group ratio, a unit-conversion aggregate, and a search-plus-calculate task), 200
episodes per task per arm.

**Token positions in chain of thought.** Two MMLU items, college_physics and virology, greedy base
paths of 3,602 and 1,977 tokens.

**BIRD.** To get off our own hand-made tasks we moved to the open BIRD mini-dev text-to-SQL
environment, and picked 10 questions in two stages, following the screening protocol of Zur et al.
(arXiv:2511.04527). Stage one is a deterministic surface-form filter producing 175 eligible
questions: gold SQL must be a single-level aggregate returning one scalar, the question must involve
an aggregate, ratio, percentage or unit, at most two per database, at least three simple and three
moderate, ordered by question id. Stage two screens the first 60 of those on **outcome uncertainty**:
10 episodes each, answers binned at gold precision, keep only questions whose **modal answer occurs 4
to 6 times out of 10**.which is the only regime where a fork can exist at all.



### What we already found out

**1. Decision tokens exist, in both chain of thought and agent trajectories.** On the Chinook ratio
task the first tool call is close to dispositive. It splits 91 of 200 episodes into an
integer-division cluster; none of those 91 reach the correct answer, 35 of 35 in the
average-group-by cluster do, and only 2 of 126 episodes that entered the bad cluster ever recover.
Mirror replay isolates it: rewind one failed and one successful episode to the same pre-decision
state and both give P(correct) = 0.22 at K = 50, then replay after each one's own first query and
they separate to 0.00 and 1.00. Same state, same model, one different action, and the outcome
distribution collapses in opposite directions.

![Mirror replay on the ratio task: the same pre-decision state gives P(correct) = 0.22 for both
episodes, and the two first queries send it to 0.00 and 1.00](forkscope/xfigs/output/plot1_t4_mirror.png)


**3. Entropy is a cheap gate, but not the one we assumed.** We did measure the recall, and it came out
the other way round. Of the 25 positions our fork rule flagged across the two MMLU items, **1 sat in
the top 20% of token entropy** (1 of 11 on virology, 0 of 14 on college_physics), and the median
flagged position sat *below* the median entropy. A high-entropy gate does not have high recall on
causal forks; it has almost none.

The agent arm shows the same thing on outcome labels we trust, and shows it more legibly. The entropy
peak inside the fatal tool call is ` Average`, an alias name that changes nothing. The tokens that
actually construct the integer division (` COUNT`, `(*)`, ` /`) sit below it.

![Token entropy along the tool call that locks the outcome: the entropy peak is a decorative alias
name, the causal tokens are not peaks](forkscope/xfigs/output/plot3_entropy_vs_fork.png)

What entropy *is* good for is the lossless direction. A token whose next-token distribution is a point
mass cannot be a fork, because there is no alternative to resample, so skipping it costs exactly
nothing. The whole tool-call envelope sits there, measured at 0.000 bits through
`<tool_call>{"name": "sql_query", "arguments": {"query": "`, while the decisive tokens sit at 0.63,
0.53 and 0.60. That gate is the same free lever as skipping positions with no second branch above
threshold: 44% of positions at a 0.02 threshold, 1.8x.

So: entropy is a safe filter for what **cannot** fork, and a bad ranking of what **does**. The
decisive steps we measured are low-entropy commitments, not high-entropy hesitations, which is the
greedy-trap shape.


### What we have not found out yet

**The MoE.** Qwen3.6-35B-A3B is sparse, about 3B active of 35B We have outcome-level data on it for both the agent tasks and BIRD,
but no token-level fork measurement yet, and no measurement of whether expert routing is itself a fork
axis, which is the interesting version of the question. the experiment is still running 

### What we failed at

**The probe.** Training a probe on hidden states to predict the measured outcome distribution did not
work on our data. The likely reason is unflattering and specific:  A probe
trained against a target that is partly an artifact of a generation cap is being trained on noise,
and it behaved accordingly. Zur et al. (arXiv:2511.04527) report that hidden activations *can*
predict a model's future outcome distribution in chain of thought, which is evidence that the target
is learnable.

For both of the last two, the binding constraint is sample size and label quality rather than the
idea. With enough episodes carrying a clean, screened failure-case distribution, the MoE question
becomes a token-level rerun rather than a new method, and the probe becomes a retry against labels
that mean something. The BIRD two-stage set is the substrate for both, because it was selected to be
non-degenerate on purpose: every question in it has a modal answer between 4 and 6 out of 10, which is
the only regime where a fork can exist at all.

## Limitations

**1. The task has to be verifiable.** You need a programmatic outcome to resample against. That is
less of a restriction than it sounds, since building high-fidelity sandboxes for evaluation and RL is
now standard practice. It does mean you cannot test a model against production. But traditional
software engineering cannot fully simulate a production incident either, least of all one that has
not happened yet.

**2. It is expensive.** Worth being precise about how expensive, and about which optimizations are
free and which cost you information.

Write the sweep out. With base path length `T`, spacing `s`, `B` branches kept per probed position,
`S` samples per branch, prompt length `P` and mean continuation length `L`, the number of
continuations is `(T/s)·B·S`, and

```
C_naive = (T/s)·B·S·(P + t̄ + L)        every continuation re-prefills its own prefix
C_radix ≈ (T/s)·B·[(P + t̄) + S·L]      the prefix is prefilled once per branch, decoded S times
```

where `t̄ ≈ T/2` averaged over positions. The reference dense configuration in our repo is `s = 1`,
`S = 200`. On the shorter of the two MMLU items (`T = 1977`, `B = 2.31` measured at a 0.02 threshold)
that is about 913,000 continuations. What we actually ran was `s = 4`, `S = 100`, or 114,500
continuations, a factor of 8.0.

The honest decomposition of that 8.0 is what matters:

- **4x from spacing.** This trades coverage. It is defensible because `fork_score` at a position is
  computed within that position and needs no neighbour, so spacing controls how many candidate
  positions you look at, not the resolution of any one measurement. It is still a trade.
- **2x from halving `S`.** This trades statistical power directly, and the critical effect size moves
  with it. Not free, and easy to take too far: at `S = 20` with a Bonferroni correction over 50
  positions, the critical TVD sits near 0.55, which is large enough that a real fork of moderate size
  is simply invisible.
- **Skipping untestable positions is free.** At a 0.02 threshold, only 56% of positions carried a
  second branch above threshold. The other 44% have nothing to compare against, and you know which
  ones they are from the top-k distribution along the base path, before sampling anything. That is a
  further 1.8x at exactly zero information loss.
- **Prefix caching is free but smaller than it looks.** Measured on SGLang, 97.8% of submitted prompt
  tokens were served from the radix cache (40,609 of 41,540). But the ratio above is
  `S·(P+t̄+L) / ((P+t̄) + S·L)`, and with `S = 100`, `P + t̄ ≈ 1.1k` and `L ≈ 1.5k` that is about
  1.8x, not an order of magnitude. Caching removes prefill; decode dominates and caching does not
  touch it. Sustained throughput was 6,800 tok/s for the 8B on one H100.

Multiplying against the dense reference: 8.0x from the configuration, times 1.8x realized from prefix
caching, is about 14x. Applying the untestable-position skip, which we have not yet done, would take
it to roughly 26x. Of that, 3.2x is free in the sense that it costs no information at all, and the
8.0x is bought with coverage and power. The lever people reach for first, prefix caching, is the
smallest of the four.

Entropy as a gate is the fourth lever and the one we have not validated. The caveat is that entropy
is a proxy and a proxy can be wrong in the direction that matters: some of the decisive steps we
measured sit at low-entropy commitments, a greedy trap that entropy cannot flag by construction. So
entropy is defensible as a cost-reduction heuristic, not as a definition of where forks are, and
using it that way needs the expensive measurement to calibrate it first. It is not in the 13x.

## three Possible assumaptions for follow up work.

### 1. Faster, cheaper evaluation

A conventional evaluation rolls out from the start of the trajectory. Here you roll out from the fork
positions instead, which raises the question of whether the outcome distribution can be pinned down
early, with fewer tokens. The cheap way to pick candidate positions is high token entropy, and there
is strong post-training evidence that this works: *Beyond the 80/20 Rule* (Wang et al.,
arXiv:2506.01939, NeurIPS'25) restricts policy-gradient updates to the top-20% entropy tokens, which
it calls **forking tokens**, and matches full-gradient updates on Qwen3-8B while beating them on
Qwen3-32B (+11.04 on AIME'25) and Qwen3-14B (+4.79).

Note the tension with our own measurement, because it is informative rather than fatal. That paper
defines forking tokens *by* entropy and validates them by whether training on them helps. We define
them by resampling and validate them by whether the outcome distribution moves, and by that test the
high-entropy tokens are mostly not the decisive ones. Both results can hold at once: **entropy finds
where the policy is malleable, resampling finds where the outcome is determined**, and there is no
reason those two sets have to coincide. Which one you want depends on whether you are training or
debugging. For picking probe positions cheaply, entropy is a defensible prefilter; for claiming a
position caused a failure, it is not evidence.

The efficient-benchmarking literature has been attacking the same cost problem along a different
axis, namely the number of questions. Anchor Points (Vivek et al., arXiv:2309.08638) selects a small
subset of examples whose correctness pattern predicts performance on the full set. metabench (Kipnis et al., arXiv:2407.12844)
pushes the same idea further, distilling six benchmarks to a few percent of their items.

above all, we are trying to measure a large distribution with fewer samples accurately.

### 2. Probes at fork positions

Train a probe on hidden states at fork positions and see whether a failure pattern is legible before
the agent commits to it.

Nanda's team at DeepMind has already built this as a production system: **Building
Production-Ready Probes For Gemini** (Kramár, Engels, Wang, Chughtai, Shah, Nanda, Conmy,
arXiv:2601.11516). Activation probes for misuse detection on Gemini 2.5 Flash, Lite and Pro, off fixed
middle-layer activations. Four things in it transfer directly.

**The failure mode is mean-pooling over a long index, and the fix is a max.** Their headline problem is
short-context to long-context shift, and the number is brutal: a standard attention probe reaches
**87.88% false-negative rate on long-context attacks**, and a max of rolling means (attention-weighted
averages in width-10 windows, then a maximum over windows) gets that to **3.03%**. Every architecture
they propose replaces an average with an extremum. The diagnosis is signal dilution: a mean over a
long context divides a signal carried by a few positions by the length of the context.

That is the same pathology as this project's own aggregation bug, one layer down the stack. The
mixture `o_t = Σ_w p̃(w)·hist(t,w)` is a weighted mean over branches, so a fork in a low-probability
branch gets divided away before any test sees it, and the fix was the same shape: take
`fork_score(t) = max` over branch pairs. They average over positions, we averaged over branches.
Worth stating as a rule rather than two anecdotes: **when the signal lives in a few members of a large
index set, a weighted mean is not a summary, it is an attenuator.**

**The pipeline is automated.** AlphaEvolve, seeded with the attention probe, searched about 2,500
candidate architectures and closed roughly **50% of the gap between the attention baseline and their
theoretical optimum** (2.53% test error against 2.64% hand-designed). Architecture search over probes
is a batch job, not a research project.

**The production shape is a cascade, and the numbers are the ones to quote.** The probe alone does not
win on accuracy: 2.64% test error against 2.04% for Gemini 2.5 Flash. It wins on the frontier. At
about **1/50th of Flash's inference cost** the probe-plus-classifier cascade reaches lower test loss
than Flash alone, **deferring to the LLM on under 10% of samples**. Substitute "resample this fork"
for "call the LLM judge" and the argument transfers unchanged: run the cheap thing always, spend the
expensive thing on the tenth of cases that need it.

**And it says who this is for.** Agent trajectories are precisely the long, multi-turn,
tool-interleaved regime their probes break in. That is a warning, since a probe validated on short
chain of thought will not survive the move to agent traces, and the fix is an aggregation choice
rather than more data. It is also why a serving vendor is the right partner rather than a
convenience: **they are the only party holding the production long-context distribution.** Hidden
states you can get anywhere by loading weights; the distribution you cannot. Their own cost note
sharpens it, since training directly on long context is **22 times more expensive** than short
context, because activations stop fitting and get swapped host to device. Whoever already has the
traffic and the cache pays a fraction of that.

Two caveats, and the first is the serious one. **Probes may be reading the text, not the state.** Boxo,
Neelappa and Raval (arXiv:2509.21344) filter out explicit textual evidence of a behaviour and watch
probe performance collapse: AUROC 0.94 to 0.57 on sandbagging, 0.74 to 0.57 on bias. A probe that only
works when the behaviour is verbalised is a text classifier with extra steps, so the token n-gram
baseline has to be *beaten*, not merely reported. If it is not beaten, the fork signal was legible in
the text all along, which is a cheaper result but still a result. Second, probes do not transfer across
models, so this assumes per-model calibration: fine in a serving deployment, a limitation in the
science.

### 3. Quality assurance for RL environments

Read the fork-path outcome distribution to find out what an RL environment is actually teaching.

An RL environment is supposed to teach something, and the fork-path distribution says whether it
can. If `o_0` is degenerate the item contributes no advantage variance and therefore no gradient; if
it never resolves, the reward is mostly noise. Both are cheap to detect before a run and expensive to
discover after one.

The support for this being the right instrument is that resampling is already what the better RL
methods are doing, just under other names. Math-Shepherd (Wang et al., arXiv:2312.08935) builds
process reward labels by resampling completions from each prefix. VinePPO (Kazemnejad et al.,
arXiv:2410.01679) goes further and shows that PPO's learned value function is a bad estimate of the
true value, and that replacing it with Monte-Carlo estimates from resampled intermediate states
improves credit assignment. So the fork-position outcome distribution is not a new quantity: it is the
quantity these methods are trying to estimate, and VinePPO's result is evidence that estimating it
badly costs you real performance. Measuring it directly is the audit.

What the audit catches, concretely:

- **Dead items.** If the outcome distribution at the start state is degenerate, the item contributes
  no advantage variance and therefore no gradient. On one MMLU item I probed this weekend, 48 of 48
  fresh samples gave the same answer. That item cannot teach a policy-gradient method anything, and
  it cannot host a fork either. Finding these before an expensive run, rather than after, is the
  cheap version of this check.
- **Items that are only noise.** The mirror failure, where the distribution stays high-entropy at
  every position and never resolves. There the reward is dominated by variance and the advantage
  estimate is mostly noise.
- **Outcome variables that measure the wrong thing.** This is the one I did not expect. On another
  item, most sampled continuations hit the generation-length cap before answering, so the recorded
  outcome distribution was largely reporting where the cap fell rather than what the model concluded.
  Raise the cap and the same item answers unanimously. An environment with that property has a reward
  that partly scores "did you finish inside the budget", which is a specification bug that
  aggregate accuracy hides and a fork-path distribution shows directly, because the unfinished mass
  is visible as its own outcome.

## What I am doing first

Direction 1. It is the one that is easy to verify, we have clear thoughts on how to conduct experiments. other need further exploration.