# expD06 -- Does gradient descent scale geometrically with width? (QI init vs Xavier init)

**Date:** 2026-07-21. **Status:** design, pending Catherine's review.

## Motivation

The charm of the QI construction is *geometric width scaling*: as the hidden width $N$ grows, the fixed-geometry least-squares readout error falls as a power law onto the fp64 floor (expB02: tanh/gelu $\sim N^{-12}$ on runge). That win is a property of the **direct solve**, not of training. Every trained-vs-width experiment so far (expD02, expD05) reports the opposite for gradient descent: pure Adam **stalls** (best non-oracle trained eval rel $L_2 \approx 7\times10^{-5}$, roughly flat in width), and machine precision only returns after a final lstsq refit. No existing figure isolates the pure-GD question as loss-vs-width. This experiment builds that figure.

## Question

Under **pure gradient descent** (Adam, no readout refit), does eval error scale **geometrically with width** for a QI-structured initialization, while a standard Xavier initialization **plateaus**?

## Hypothesis

- **Xavier init + GD:** eval rel $L_2$ is roughly flat in width (plateau) -- Adam cannot find the geometry, so more width does not buy precision.
- **QI init + GD:** eval rel $L_2$ descends with width, tracking the construction's geometric reference (at least for the oracle; the random-readout variant is the open test).

## Experiment design

Reuse the expD05 machinery unchanged: `build_initial_state(family, ...)`, `make_model`, and the full-batch fp64 Adam loop in `experiments/expD05_scale_init_story/run.py`. A new thin driver `experiments/expD06_gd_width_scaling/run.py` imports those, sweeps width, trains each cell with **pure Adam only**, and reports the trained-before-refit metric. No lstsq refit enters any plotted curve -- the refit is precisely the thing this experiment is *not* testing.

**Three trained lines** (all pure Adam), plus one reference:

| Line | Family | What GD is handed | Role |
|---|---|---|---|
| Xavier | `standard_xavier_affine` | nothing | plateau baseline |
| QI-oracle | `exact_qi_oracle` | full construction (centers, $\gamma$, readout) | GD *preserves* geometric scaling? |
| QI-geom | `qi_geom_random_readout` | QI geometry (centers, $\gamma$), **random readout** | can GD *learn* a geometric-scaling fit from a QI basis? -- the load-bearing question |
| construction ref (dashed) | `exact_qi_oracle` **untrained** `initial_eval_rel_l2` | -- | the geometric ideal the trained curves are measured against |

The construction reference is free: it is the oracle family's initial (pre-training) eval error, already recorded per cell.

**Sweep and knobs.**
- **Width axis:** driven by the `resolution` knob $N \in \{32, 64, 128, 256, 512\}$ (log grid, 5 points; hidden width $W \approx N + \text{halo}$ shared across families per cell, as in expD05). Plot against actual $W$, annotate $N$.
- **Targets (facets):** `runge` (boundary-layer, steep construction scaling $\sim N^{-12}$) and `exp` (smooth). Optionally `sine_8pi` (high-freq) as a third facet if runtime allows.
- **Seeds:** `{0, 1, 2}`. Plot per-family median with a min--max band across seeds. (Oracle is near-deterministic; Xavier and QI-geom need the band.)
- **Training:** Adam, lr $1\times10^{-3}$, 20000 steps, `n_train = 2003`, `n_eval = 4001`, fp64 -- identical to the expD05 full matrix, so numbers are comparable.
- **Metric:** `best_eval_rel_l2` (best trained eval rel $L_2$ over the run, pre-refit). $L_\infty$ recorded as a secondary column.

Matrix size: 3 families $\times$ 5 widths $\times$ 2--3 targets $\times$ 3 seeds $= 90$--$135$ trained cells. At expD05's per-cell cost this is a small fraction of the 1080-row full matrix; runnable as a single job (shardable if needed).

**Metric definitions.** Feature matrix $\Phi_{ij}=\tanh(a_j x_i + b_j)$ in affine coords; eval rel $L_2 = \|\hat f - f\|_2/\|f\|_2$ on the prime eval grid (misaligned with train); $L_\infty=\max_x|\hat f - f|$.

## Deliverables

- **`error_vs_width.png`** -- facet per target; x = width $W$ (log), y = eval rel $L_2$ (log). Three solid trained lines (Xavier / QI-oracle / QI-geom) with seed bands + one dashed construction reference + the fp64 floor line ($\sim 5\times10^{-14}$). Legend **above** the axes (repo convention). This is the money figure: geometric descent vs plateau.
- **`expD06_results.md`** -- standard repo writeup structure. Conclusions left pending Catherine/Sam sign-off.
- Data: `summary.csv` (per-cell metrics) under `results/checkpoint_D_optimizers/expD06_gd_width_scaling/` (gitignored; only the md + figure are version-controlled).

## Success / falsification

- **Win:** the QI-oracle line has a clearly negative slope in $\log$-$\log$ (descends with width) while Xavier is flat; slopes are separable beyond the seed bands. Bonus win: QI-geom also descends.
- **Null / honest negative:** all three trained lines are flat (GD stalls regardless of init) -- consistent with the prior expD05 finding, and still a publishable "GD cannot exploit the geometry without a refit" statement. The construction reference makes the size of the gap explicit either way.

## Risks / caveats

- **Oracle is not a deployable init** -- it hands GD the answer; its curve shows *preservation* of scaling, not that GD *discovered* it. The writeup must frame it that way; QI-geom is the honest test.
- **Adam may stall even from QI-geom** (expD05 precedent). That is a real possible outcome, not a bug -- report it plainly, do not tune to force a win.
- **Width via `resolution`+halo** means $W$ is not exactly $N$; plot and label actual $W$ to avoid a misleading axis.
- 20k Adam steps may not be convergence for the largest width; record final-vs-best gap so a "needs more steps" confound is visible.
