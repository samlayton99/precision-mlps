# expD27 — What information does a frozen readout give geometry?

Status: complete. Sam confirmed the preceding freeze schedule, gamma-1 reset at fixed QI centers, and 10% independent multiplicative scale noise preserving centers. Only the four requested variations and their continued references ran. Additional scientific controls remain unrun.

Completion: 64 trajectories, 16 PNGs, 16 compressed data files, and one results writeup under results/checkpoint_D_optimizers/expD27_readout_information. Five focused tests pass; saved-data checks cover all 48 exact branch prefixes, first post-freeze geometry updates, and fixed coefficients. Variant 1 produces destructive large-value motion and material cross-library numerical sensitivity from large solved coefficients; both are recorded rather than corrected by tuning. The transferred QI readout improves some actual fits while mean gamma stays near 1. Clean/noisy QI freezing leaves refitted geometry near its initial quality. No reliable recovery was observed at this rate and horizon.

## Requested scope

1. Start from Xavier geometry. Numerically solve the readout at step zero and after each geometry GD update until the freeze marker. GD never updates the readout. Freeze the solved readout at the marker and continue geometry GD.
2. Solve the readout on QI geometry, then reset gamma to 1 at the same QI centers while retaining those coefficients. Keep the readout fixed from step zero for 500 updates.
3. Start on QI geometry with zero readout, use ordinary joint GD until the marker, then freeze the readout.
4. Perturb QI geometry, start with zero readout, use ordinary joint GD until the marker, then freeze the readout. Apply independent factors 1+0.1 Z_j, Z_j standard normal, to each slope and its bias together. The seeded factors are reused across targets; centers are preserved. Do not clip or redraw factors.

Shared defaults are the preceding experiment's four matched targets, N=128 (177 neurons including halo 24 per side), fp64, 1,024 training and 8,192 independent evaluation midpoints on [-1,1], and constant learning rate 0.002. The proposed markers remain 2, 10, 50, and 150 joint steps followed by 500 frozen-readout steps. QI geometry means uniform centers with gamma=16 (lambda=0.25); the requested coefficient solve is the existing numerical SVD solve, including output bias, at relative cutoff 1e-13.

## Implementation and figure principles

- Record step zero after any requested initial readout solve; record the pre-solve QI reference separately for the coefficient-transfer case.
- A branch at X clones the complete state after X geometry updates, including the solve at X in variant 1. Its first frozen update is X to X+1. No solves occur in that branch after X.
- All-zero readouts give an exactly zero first geometry gradient in variants 3–4; subsequent readout GD supplies the first geometry signal. Do not add a hidden solve or warmup.
- Save current errors and mean absolute gamma at every state, plus independent evaluation-only refits to identify whether approximation improves. Solves used solely for evaluation must never mutate training.
- Reuse exact branch prefixes. Preserve readout bitwise after freezing, including output bias. Record nonfinite failures without silently changing rate, clipping, damping, or initialization.
- The transferred-readout case has a freeze at zero rather than four artificial warmup times. Its plot must state that distinction.
- Keep prior figures and data. New outputs belong to one experiment with organized figures, compressed data, and one results writeup.

## Pre-build practicality and verification checklist

1. Passes: ordinary/frozen steps use one forward and backward. Variant 1 additionally materializes the n by (m+1) readout matrix and computes an SVD at every solve event. Variant 2 has one initial SVD. These are explicit small-problem diagnostic interventions, not scalable optimizer proposals.
2. State: SGD requires no moment state; parameters and gradients are O(m). Saved trajectories and independent-grid evaluations are offline diagnostic storage. The SVD uses O(nm+m^2) transient memory; its cost is recorded as a solve cost.
3. k/d: no Krylov memory or block-size parameter.
4. Reductions: standard loss reduction and gradient reductions; SVD is a dense direct solve with its own non-Adam cost. No hidden per-parameter probe or collective loop is claimed free.
5. Exact versus small: frozen readout equality and zero-readout geometry gradients are exact checks. LS stationarity and retained rank are numerical quantities with an explicit cutoff, not exact-span claims.
6. Precision: this reproduces the existing fp64 diagnostic. It does not claim bf16 compatibility or a general solver. Initialization noise is a scientific factor whose magnitude must be selected before running.
7. Control signals: no loss-based acceptance, rate adaptation, convergence-triggered switching, or solver tuning. Freeze times are fixed in advance. Stop only on nonfinite arithmetic and report it.
8. Kill list: dense repeated solves are intentionally used as a reference intervention, not proposed as deployable machinery. There is no Adam geometry step following a solve; fixed-rate plain GD is retained.
9. Classical baseline: ordinary GD and the preceding Xavier freezing experiment supply the existing reference. No new optimizer or unapproved comparison is introduced.
10. Litmus tests: production/batching/precision promotion is outside scope. Validate independent gradients, readout-only assignment, correct branch timing, target/geometry matching, and evaluation nonmutation first.
11. Falsification: a readout intervention that moves gamma but leaves refitted error unchanged or worse has not taught useful approximation geometry. A good transferred readout failing to recover its known compatible geometry would falsify that recovery claim at this rate and horizon.

## Proposed only; not run

- Freeze centers to isolate scale learning from center movement.
- Permute solved QI neuron coefficients, preserving their values and norm while changing their center assignments, to test assignment information against coefficient magnitude.
