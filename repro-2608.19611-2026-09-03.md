# Resampling-based uncertainty analysis, reproduced and pushed into agent trajectories

Weekend project, 2026-09-03. Reproduction of arXiv:2608.19611 on SGLang and vLLM, plus early
exploration past what the paper covers. Nothing here is a finished result.

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
integer-division cluster, none of those 91 reach the correct answer, 35 of 35 in the average-group-by
cluster do, and only 2 of 126 that entered the bad cluster ever recover. Mirror replay isolates it.
Rewind a failed and a successful episode to the same pre-decision state and both give
P(correct) = 0.22 at K = 50; replay after each one's own first query and they separate to 0.00 and
1.00. Same state, same model, one different action.

![Mirror replay: the same pre-decision state gives 0.22 for both episodes, and their two first
queries send it to 0.00 and 1.00](forkscope/xfigs/output/plot1_t4_mirror.png)

**2. Entropy is a cheap gate, but not the one we assumed.** We measured the recall and it came out the
other way round: of 25 flagged positions, **1 sat in the top 20% of token entropy**, and the median
flagged position sat below the median entropy. The agent arm shows why. The entropy peak inside the
fatal tool call is ` Average`, an alias name that changes nothing.

![The entropy peak is a decorative alias name; the causal tokens are not
peaks](forkscope/xfigs/output/plot3_entropy_vs_fork.png)

Entropy is good for the lossless direction only. A point-mass distribution has no alternative to
resample, so a zero-entropy token cannot fork and skipping it is free: the tool-call envelope measures
0.000 bits while the decisive tokens sit at 0.63, 0.53 and 0.60, and 44% of positions carry no second
branch at a 0.02 threshold, worth 1.8x. **A safe filter for what cannot fork, a bad ranking of what
does.**

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
restrictive than it sounds, since high-fidelity sandboxes for eval and RL are now standard. It does
mean you cannot test against production, but traditional software engineering cannot fully simulate a
production incident either, least of all one that has not happened yet.

**2. It is expensive**, and it is worth being precise about which optimizations are free. With base
path length `T`, spacing `s`, `B` branches per position, `S` samples per branch, prompt length `P`,
mean continuation length `L`:

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
detection on Gemini 2.5, with AlphaEvolve automating the architecture search. Three things transfer.

**Their failure mode is mean-pooling, and the fix is a max.** A standard attention probe reaches
**87.88% false-negative rate on long-context attacks**; a max of rolling means gets it to **3.03%**. A
mean over a long context divides a signal carried by few positions by the length of the context. Same
pathology as our mixture curve one index set over, and `fork_score` is the same fix. When the signal
lives in a few members of a large index set, a weighted mean is an attenuator, not a summary.

**The production shape is a cascade.** The probe loses on accuracy alone, but at **1/50th of Gemini
2.5 Flash's inference cost** the cascade beats Flash while **deferring to it on under 10% of samples**.
Substitute "resample this fork" for "call the LLM judge" and the argument transfers unchanged.

**Their hard case is our regime, which decides who this is for.** Agent traces are the long,
multi-turn inputs their probes break on, so a probe validated on short chain of thought will not
survive the move, and training directly on long context costs **22x** more. Hidden states you can get
anywhere by loading weights; **the production long-context distribution only a serving vendor has.**

Caveat that could kill it: probes may read the text, not the state. Boxo, Neelappa and Raval
(arXiv:2509.21344) strip explicit textual evidence and watch AUROC fall 0.94 to 0.57 on sandbagging.
The n-gram baseline has to be beaten, not reported.

### 3. Quality assurance for RL environments

Read the fork-path outcome distribution to find out what an RL environment is actually teaching.

Resampling is already what the better RL methods do under other names. Math-Shepherd (Wang et al.,
arXiv:2312.08935) builds process-reward labels by resampling from each prefix; VinePPO (Kazemnejad et
al., arXiv:2410.01679) shows PPO's learned value function estimates the true value badly and that
Monte-Carlo estimates from resampled intermediate states improve credit assignment. Not a new
quantity, then, but the one those methods estimate, and evidence that estimating it badly costs
performance. Three things the audit catches:

- **Dead items.** A degenerate `o_0` gives no advantage variance and no gradient. One MMLU item
  answered the same way in 48 of 48 fresh samples: it can neither teach a policy gradient nor host a
  fork. Cheap to find before a run, not after.
- **Noise items.** The mirror failure, where the distribution never resolves and the advantage
  estimate is mostly variance.
- **Outcome variables measuring the wrong thing.** The one I did not expect. On another item most
  continuations hit the length cap before answering, so the recorded distribution reported where the
  cap fell; raise it and the item answers unanimously. That reward partly scores "did you finish in
  budget", a specification bug aggregate accuracy hides and a fork-path distribution shows, because
  the unfinished mass is its own outcome.

## What I am doing first

Direction 1. It is the one that is easy to verify, and we have clear thoughts on how to run the
experiments. The others need further exploration.
