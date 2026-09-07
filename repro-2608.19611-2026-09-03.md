# Resampling-based uncertainty analysis, reproduced and pushed into agent trajectories

Weekend project, 2026-09-03. I read Forking Fast (arXiv:2608.19611) and thought the instrument looked
genuinely useful: if you can measure which step decided an outcome, you can debug a model instead of
guessing at it. So I reproduced it on SGLang, then pushed it past what the paper covers. Nothing here
is a finished result.

## The method

LLM reasoning is stochastic, so understanding a model means grappling with the distribution of
reasoning chains it might produce, which is to say its uncertainty. Resampling characterizes that
distribution directly: fork the rollout at a position, sample continuations, read off the conditional
outcome distribution. Along a trajectory this reveals which steps determine the answer.

## Extending it to agent tool use

A tool call is a natural fork position: a discrete, semantically meaningful commitment whose prior
environment state is reproducible. Resampling there gives the outcome distribution over the actions
available from the current state, which measures the agent's uncertainty rather than proxying it.

That buys a picture of behaviour, not just performance: how reliable the model is, where it fails,
conditional on what. Actionable twice, at test time by spending compute only where the trajectory can
still fork, and in post-training by supervising the steps that carry the outcome. In software
engineering, if you can reproduce it you can fix it. Same bet.

![Agent-level localization: the outcome distribution collapses at one step, it does not drift across
many](forkscope/xfigs/output/plot2_t7_localization.png)

## What we have early tested on and validated

Two arms, both on Qwen3-8B and Qwen3.6-35B-A3B, served under SGLang. Same sampling protocol in both:
the base path is **greedy** (temperature 0, top-k 1) recording the top-k candidate distribution at
every position, and continuations are **untruncated ancestral samples** (temperature 1.0, top-p 1.0,
top-k off) with a distinct seed per copy. Untruncated matters: sampling under top-p 0.95 would measure
a different model than the one served, so `o_t` estimates the model's own conditional outcome
distribution. Determinism plus per-copy seeds makes every continuation reproducible bit-for-bit. In
the agent arm forks are tool-call boundaries, restored by replaying the message prefix; in the
chain-of-thought arm they are token positions along the greedy path.

### Experiments

**Agent trajectories.** Three text-to-SQL tool-use tasks against a Chinook database (average-per-group
ratio, unit-conversion aggregate, search-plus-calculate), 200 episodes per task per arm.

**Token positions in chain of thought.** Two MMLU items, college_physics and virology, greedy base
paths of 3,602 and 1,977 tokens.

**BIRD.** To get off our own hand-made tasks, 10 questions from BIRD mini-dev picked in two stages
following the screening protocol of Zur et al. (arXiv:2511.04527). Stage one is a deterministic
surface-form filter giving 175 eligible questions: gold SQL a single-level aggregate returning one
scalar, aggregate/ratio/unit theme, at most two per database, ordered by question id. Stage two
screens the first 60 on **outcome uncertainty**, 10 episodes each with answers binned at gold
precision, keeping only those whose **modal answer occurs 4 to 6 times out of 10**, which is the only
regime where a fork can exist.

### What we already found out

**1. Decision tokens exist, in both chain of thought and agent trajectories.** On the Chinook ratio
task the first tool call is close to dispositive: it splits 91 of 200 episodes into an
integer-division cluster, none of those 91 reach the correct answer, and 35 of 35 in the
average-group-by cluster do.

Self-repair needs its denominator stated or it looks like a contradiction. 91 is the count whose
*first* SQL was the trap; 126 is the count that entered the trap at *any* point, so the first is
contained in the second. Repair over the wider set is 2/126, and since none of the 91 recovered, both
recoveries sit in the 35 that entered late: **opening with the trap is fatal at 0/91, wandering into it
later is survivable at 2/35.** (91/200, 0/91 and 35/35 are recomputed from the cluster-by-outcome
table; 126 is quoted from the collection report.)

Mirror replay isolates it.
Rewind a failed and a successful episode to the same pre-decision state and both give
P(correct) = 0.22 at K = 50; replay after each one's own first query and they separate to 0.00 and
1.00. Same state, same model, one different action.

![Mirror replay: the same pre-decision state gives 0.22 for both episodes, and their two first
queries send it to 0.00 and 1.00](forkscope/xfigs/output/plot1_t4_mirror.png)

**2. Entropy gating works, and the misses are the story.** Tested on Forking Fast's own public
release, where the ground truth exists: Llama-3-8B, 100 questions, S=200 `o_t` grids, gating positions
by 1−p_base and checking recall of PELT-detected forks. Recall is **78.3% at the top 20%** of positions
and **88.0% at the top 30%**, against a random-gate null of 51.2% and 67.7%. Roughly 2x enrichment, and
a two-stage instrument (entropy screen, then replay only at gated positions) cuts measurement cost
3-5x.

The residual is what matters: **13-29% of forks happen where the model is confident.** No
entropy-based monitor can see those, and the agent arm shows why that is not a rounding error. Inside
the fatal turn there are two entropy peaks and only one is a fork. At t=17, writing ` COUNT`
(H=1.31, the turn's maximum), the outcome distribution starts collapsing and is welded shut four
tokens later. At t=28, naming an alias ` Average` (H=1.02), it does not move at all. After the die
lands, the trap path runs at near-zero entropy: confidently doomed.

![Two entropy peaks, one fork: the distribution collapses at ` COUNT` and ignores the alias
peak](forkscope/xfigs/output/plot3_entropy_vs_fork.png)

Hesitation and fate are different things, and only measurement tells them apart. The one thing entropy
gives for free is the floor: a point-mass distribution has no alternative to resample, so a
zero-entropy token cannot fork and skipping it is lossless. 44% of positions carry no second branch at
a 0.02 threshold, worth 1.8x.

### What we have not found out yet

**The MoE.** Qwen3.6-35B-A3B is sparse, about 3B active of 35B. We have outcome-level data on it for
the agent tasks and BIRD, but no token-level fork measurement, and no measurement of whether expert
routing is itself a fork axis, which is the interesting version of the question. Experiment running.

### What we failed at

**The probe.** Predicting the measured outcome distribution from hidden states did not work on our
data. The reason is specific and unflattering: the labels came from the arm whose outcome variable was
partly an artifact of a generation cap, so the probe was trained on noise. Zur et al.
(arXiv:2511.04527) report hidden activations *can* predict a model's future outcome distribution in
chain of thought, so the target is learnable and the fault was ours.

For both of the last two the binding constraint is sample size and label quality, not the idea. With
clean screened episodes the MoE question is a token-level rerun and the probe is a retry against
labels that mean something.

## Limitations

**1. The task has to be verifiable.** You need a programmatic outcome to resample against. Less
restrictive than it sounds, since high-fidelity sandboxes for eval and RL are now standard. I think it
is worth building an environment purely to test the fork-path distribution: **a staging environment
for the agent era**.

**2. It is not a free lunch.** The floor is statistical, not engineering. Replicate TVD falls as
`S^-1/2`, and we measured it: a log-log slope of **-0.49** over `S = 10` to 100, monotone, with
iid-null ratios within a few percent of 1.0 across the range. Halving the error costs **4x** the
samples, and the cost of a given precision grows as `precision^-2`. No amount of caching touches that.

Forking Fast's own answer is to spend statistics instead of samples: PELT change-point detection over
the `o_t` curve plus Dirichlet pooling across neighbouring positions, buying effective sample
efficiency (the paper reports 4-15x) without buying samples. Our V1 check says the raw draws are only
very slightly over-dispersed against an iid multinomial null (ratio 1.08 at `S = 20`), which is the
condition pooling needs.

The entropy gate is the other lever, and the numbers above are what it is worth: screening to the top
20-30% of positions keeps 78-88% of forks at roughly 2x enrichment over random, cutting measurement
cost **3-5x**. A prefilter, not a definition, since 13-29% of forks sit where the model is confident.

What is left is engineering, and only half of it is free. With base path length `T`, spacing `s`, `B`
branches per position, `S` samples per branch, prompt length `P`, mean continuation length `L`:

```
C_naive = (T/s)·B·S·(P + t̄ + L)      every continuation re-prefills its own prefix
C_radix ≈ (T/s)·B·[(P + t̄) + S·L]    the prefix is prefilled once per branch, decoded S times
```

Our dense reference is `s = 1`, `S = 200`. On the shorter MMLU item (`T = 1977`, `B = 2.31` at a 0.02
threshold) that is ~913,000 continuations; we ran `s = 4`, `S = 100`, or 114,500. A factor of 8.0,
which decomposes into two trades and two free levers:

| lever | factor | costs |
|---|---|---|
| spacing 4 | 4x | coverage, not resolution: `fork_score` is within-position |
| `S` 200 → 100 | 2x | power: at `S = 20` the critical TVD is ~0.55, so a moderate fork is invisible |
| skip positions with no second branch | 1.8x | **nothing**, and the top-k says which before sampling |
| prefix caching | 1.8x | **nothing** |

**14x realized, 26x with the skip applied, 3.2x of it free.** Caching measured 97.8% of prompt tokens
served from cache (40,609 of 41,540) at 6,800 tok/s on one H100, but `S·(P+t̄+L) / ((P+t̄) + S·L)` is
only ~1.8x: caching removes prefill, and decode dominates. **The lever people reach for first is the
smallest of the four.**

## Three possible assumptions for follow-up work

### 1. Faster, cheaper evaluation

Conventional evaluation rolls out from the start; here you roll out from fork positions, which asks
whether the outcome distribution can be pinned down early with fewer tokens. The cheap way to pick
candidates is high token entropy, and post-training evidence says it works: *Beyond the 80/20 Rule*
(Wang et al., arXiv:2506.01939, NeurIPS'25) restricts policy-gradient updates to the top-20% entropy
tokens, which it calls **forking tokens**, matching full-gradient training on Qwen3-8B and beating it
on Qwen3-32B (+11.04 on AIME'25).

That paper defines forking tokens *by* entropy; we define them by resampling, and by our test the
high-entropy tokens are mostly not the decisive ones. Both hold: **entropy finds where the policy is
malleable, resampling finds where the outcome is determined.** Fine as a prefilter, not as evidence
that a position caused a failure.

The efficient-benchmarking literature attacks the same cost problem on a different axis, the number of
questions: Anchor Points (Vivek et al., arXiv:2309.08638) selects examples whose correctness predicts
full-set performance, metabench (Kipnis et al., arXiv:2407.12844) distils six benchmarks to a few
percent of their items. Nobody touches the token axis. Above all we are trying to measure a large
distribution accurately with fewer samples.

### 2. Probes at fork positions

Train a probe on hidden states at fork positions and see whether a failure pattern is legible before
the agent commits to it.

Nanda's team at DeepMind already shipped this: **Building Production-Ready Probes For Gemini**
(Kramár, Engels, Wang, Chughtai, Shah, Nanda, Conmy, arXiv:2601.11516), activation probes for misuse
detection on Gemini 2.5, with AlphaEvolve automating the architecture search.

Agent traces, however, are exactly the long, multi-turn inputs their probes break on, so a probe
validated on short chain of thought will not survive the move, and training directly on long context
costs **22x** more. Hidden states you can get anywhere by loading weights; **the production
long-context distribution only a serving vendor has.** With in-house data it may be possible to build
a probe that addresses this.


### 3. Quality assurance for RL environments

Read the fork-path outcome distribution to find out what an RL environment is actually teaching.
Companies are now building and buying significant numbers of RL environments, yet there is no cheap,
scalable way to sanity-check one. The fork-path outcome distribution looks like an easier way to
evaluate an environment: whether it is hard enough, and whether it hides unknown bugs. Training costs
far more than inspecting rollouts, and what you conclude from it depends heavily on the
hyperparameters and framework you happened to pick.

Three things the audit catches:

- **Dead items.** A degenerate `o_0` gives no advantage variance and no gradient. One MMLU item
  answered the same way in 48 of 48 fresh samples: it can neither teach a policy gradient nor host a
  fork. Cheap to find before a run, not after.
- **Noise items.** The mirror failure, where the distribution never resolves and the advantage
  estimate is mostly variance.
- **The ratio.** How many rollouts does GRPO need before it learns something meaningful?

## What I am doing first

Direction 1. It is the one that is easy to verify, and we have clear thoughts on how to run the
experiments. The others need further exploration.
