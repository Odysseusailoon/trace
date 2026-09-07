> **INVALIDATED, 2026-09-03. Do not quote any number below.**
>
> This is the naive pre-null pass: forks are adjacent-position TVD on the
> probability-weighted mixture curve, thresholded at 0.10, with no null
> hypothesis. Five defects, each independently sufficient to void the fork counts:
>
> 1. The mixture `o_t = Sum_w p~(w)*hist(t,w)` averages over branches, so a fork
>    carried by a low-probability branch is attenuated before any test sees it.
>    Use `fork_score` (`src/forkscope/branchstat.py`) instead.
> 2. At `branch_prob_threshold: 0.05`, most positions had **no second branch**
>    (39 of 50 for college_physics_4, 31 of 51 for virology_5), so a "fork" there
>    could only be noise in the greedy branch's own 20 draws.
> 3. `S = 20` with Bonferroni over 50 positions puts the critical TVD near 0.55.
>    Recomputed under a pooled multinomial null, **nothing here is significant**:
>    max fork_score 0.250 (p = 0.13) and 0.350 (p = 0.09).
> 4. The 1,500-token cap truncated 77% and 50% of fresh continuations, so the
>    outcome distribution was partly reporting where the cap fell.
> 5. The extractor accepted a bare `option (X)` anywhere in the text, so it could
>    read an option the model was in the middle of rejecting.
>
> Corrected configuration: `configs/dense.yaml`. Status and method:
> `repro-2608.19611-2026-09-03.md`.

# forkscope report: college_physics_4

- positions observed: 50
- fork threshold (TVD): 0.1
- forks found: 14

## Fork 1: t = 24 -> 36
> …<think>
Okay, let's try to figure out this question about excited states of helium. The question is asking why an orth

- before: {'B': 0.55, 'Other': 0.45}
- after:  {'Other': 0.522, 'B': 0.406}
- flip magnitude (TVD): 0.144

## Fork 2: t = 72 -> 84
> … are Heisenberg uncertainty principle, Pauli exclusion principle, Bohr model, or nuclear hyperfine coupling.

First, I

- before: {'Other': 0.6, 'B': 0.4}
- after:  {'B': 0.55, 'Other': 0.45}
- flip magnitude (TVD): 0.150

## Fork 3: t = 120 -> 132
> … two electrons, the spin states of the electrons determine whether it's para or ortho. Para is when the spins are ant

- before: {'B': 0.55, 'Other': 0.4}
- after:  {'Other': 0.65, 'B': 0.35}
- flip magnitude (TVD): 0.250

## Fork 4: t = 156 -> 168
> … when they're parallel. 

Now, the question states that the ortho state has lower energy than the para state. Wait

- before: {'B': 0.55, 'Other': 0.4}
- after:  {'Other': 0.6, 'B': 0.4}
- flip magnitude (TVD): 0.200

## Fork 5: t = 192 -> 204
> … some cases, like in hydrogen, the spin states affect the energy levels. But maybe in helium, the situation is different.

- before: {'B': 0.55, 'Other': 0.45}
- after:  {'Other': 0.65, 'B': 0.3}
- flip magnitude (TVD): 0.250

## Fork 6: t = 288 -> 300
> … So if they are in the same orbital, their spins must be opposite. That would mean that the para state (antip

- before: {'B': 0.75, 'C': 0.15}
- after:  {'Other': 0.55, 'B': 0.4}
- flip magnitude (TVD): 0.500

## Fork 7: t = 336 -> 348
> …i exclusion principle. Wait, but the question says that the ortho state has lower energy. That seems conflicting. Maybe I

- before: {'B': 0.5, 'Other': 0.4}
- after:  {'Other': 0.65, 'B': 0.3}
- flip magnitude (TVD): 0.250

## Fork 8: t = 384 -> 396
> … the two electrons are in the 1s orbital with opposite spins (para). But when they are in excited states, maybe

- before: {'Other': 0.511, 'B': 0.479}
- after:  {'B': 0.65, 'Other': 0.35}
- flip magnitude (TVD): 0.171

## Fork 9: t = 432 -> 444
> … But the question is about the spin states. 

Alternatively, maybe the energy difference between para and ortho states is related to

- before: {'B': 0.55, 'Other': 0.35}
- after:  {'Other': 0.55, 'B': 0.45}
- flip magnitude (TVD): 0.200

## Fork 10: t = 456 -> 468
> … to the exchange interaction. In quantum mechanics, when electrons are in the same orbital, their wavefunctions can be symmetric or antis

- before: {'B': 0.5, 'Other': 0.45}
- after:  {'Other': 0.6, 'B': 0.35}
- flip magnitude (TVD): 0.200

## Fork 11: t = 480 -> 492
> … antisymmetric. For the Pauli exclusion principle, the total wavefunction (which includes spatial and spin parts) must be antis

- before: {'Other': 0.55, 'B': 0.35}
- after:  {'B': 0.5, 'Other': 0.45}
- flip magnitude (TVD): 0.150

## Fork 12: t = 528 -> 540
> … 

In the case of helium, if the electrons are in the same spatial orbital (like the ground state), the spatial part

- before: {'B': 0.516, 'Other': 0.384}
- after:  {'B': 0.6, 'Other': 0.3}
- flip magnitude (TVD): 0.162

## Fork 13: t = 552 -> 564
> … part is symmetric, so the spin part must be antisymmetric (para). But if the electrons are in different spatial orbitals

- before: {'B': 0.7, 'Other': 0.3}
- after:  {'Other': 0.55, 'B': 0.4}
- flip magnitude (TVD): 0.300

## Fork 14: t = 576 -> 588
> …als, the spatial part can be antisymmetric, allowing the spin part to be symmetric (ortho). However, in that case

- before: {'Other': 0.55, 'B': 0.4}
- after:  {'B': 0.6, 'Other': 0.4}
- flip magnitude (TVD): 0.200
