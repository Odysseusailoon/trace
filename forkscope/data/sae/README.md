# SAE probe measurements, 2026-09-02

These notes retain the measurements from the local SAE report for the probe
described in `demo/en.html`. The extraction code and result JSONs were recorded
on Spark under `~/sae_fork/` and `~/sae_fork/out/`; they are not in this checkout.
The available figures are stored alongside this file.

## Setup

Qwen3-8B, layer 27 resid_post, adamkarvonen BatchTopK SAE with 65,536 latents,
k=160, reported fraction of variance explained 0.79. The t4 comparison used
12 trap and 12 correct continuations from
[`forkstate_conts_t4_artist_album_ratio_d0.json`](../audit/forkstate_conts_t4_artist_album_ratio_d0.json).
Window positions were 15 to 18; the label-permutation test used 2,000 permutations.

| Measurement | Recorded result |
|---|---|
| t4 global window L1 | 2256.8; null mean 652.1, maximum 1762.9; p=0.0005 |
| Latent 52816 window difference | 41.8; empirical p=0.0005 |
| Per-latent BH-FDR discoveries | 0 / 65,536 |
| t7 comparison | 16 converted-query and 12 raw-ms continuations, selected from 80 samples |
| Latent 52816 on t7 | Difference −1.4 to −2.9; p≈0.20 |
| t7 whole-suffix global L1 | 423; null maximum 343; p<0.0005 as recorded |
| Top-20 latent overlap between tasks | 3/20 over the suffix, 6/20 over the window |

Under the patched schema, the same original continuations were encoded again:

| Latent | Original difference | Patched difference |
|---|---|---|
| 52816 | 98.4 | 87.8 |
| 41652 | 80.4 | 79.8 |
| 5866 | 60.1 | 57.5 |

This comparison uses original-distribution text under a new schema, rather than
fresh patched-model generations. The report separately records an intdiv rate
change from 51% to 4% in the schema-patch experiment.

## Probe and steering checks

Within t4, the leave-one-out mean-difference probe had AUROC 1.000 over both the
window and whole suffix. The recorded score was 0.583 at positions 15/16 and
1.000 at positions 17 to 19. The original direction also scored 1.000 on the
patched-schema features. These are small, within-task comparisons.

| Transfer | AUROC | p |
|---|---|---|
| t4 to t7, window | 0.375 | 0.22 |
| t4 to t7, suffix | 0.361 | 0.25 |
| t7 to t4, window | 0.333 | 0.14 |
| t7 to t4, suffix | 0.521 | 0.86 |
| t4 train=test control | 1.000 | |
| t7 train=test control | 0.625 | |

Steering v1 used six arms of 20 continuations after `SELECT COUNT(`; all
produced count-ratio queries. V2 moved before the SQL, used a top-10 direction
with alpha 10/25/50 and random controls, and sampled 16 continuations per arm.
Neither run found an arm difference. In V2 most queries were exploratory and
the ratio decision fell outside the 56-token window. These runs do not establish
the absence of a causal feature or subspace.

The report's 91-text activation check recorded mean peaks of 184 for latent
52816 on math text (117 on neutral text), and 154 for latent 41652 on t4 SQL
(zero on the tested MMLU, neutral, and BIRD text). These are descriptive
activation comparisons, not validated feature semantics.

## Figures

- [t4 latent trajectory](fig1_latent52816_trajectory.png)
- [Original versus patched features](fig2_toplatents_patch.png)
- [Window permutation test](fig3_window_test.png)
- [t7 comparison](fig4_t7_heldout.png)
- [Activation examples](fig5_semantics.png)

The measurements cover one layer and one SAE. t4 used episode replay whereas
t7 used fresh SQL-prefill sampling. The available notes and plots do not resolve
that distribution difference or validate cross-task error detection.
