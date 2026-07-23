# expD06 -- Does gradient descent scale geometrically with width? (QI init vs Xavier init)

**Status: draft-pending-Sam; data-filling in progress.**

## TL;DR

- _(pending full matrix)_ Xavier init + pure Adam **plateaus** in width; QI init + pure Adam **descends geometrically** toward the fp64 floor.
- The oracle (init at the construction) rides the untrained construction reference down, i.e. GD *preserves* geometric scaling rather than destroying it.
- The load-bearing line is QI-geometry-with-random-readout: whether GD can *learn* a geometric-scaling fit from a QI-shaped basis it was not handed the answer for.

## Question

Under pure gradient descent (full-batch Adam, no readout refit), does eval error scale geometrically with hidden width for a QI-structured initialization, while a standard Xavier initialization plateaus?

## Experiment design

Machinery is imported verbatim from expD05 (`build_initial_state`, `make_model`, the fp64 Adam loop); this experiment only reshapes the sweep into a width axis and plots the trained-before-refit metric. Per cell: a single-hidden-layer tanh MLP in affine coordinates $a_jx+b_j$ is initialized by one of three families and trained with full-batch Adam (lr $10^{-3}$, 20000 steps, fp64); no lstsq refit enters any plotted curve.

Three trained lines (all pure Adam):

- **Xavier** (`standard_xavier_affine`) -- plateau baseline.
- **QI-oracle** (`exact_qi_oracle`) -- initialized at the full QI construction (centers, $\gamma$, readout); tests whether GD *preserves* geometric scaling.
- **QI-geom** (`qi_geom_random_readout`) -- QI geometry (centers, $\gamma$) with a random readout; tests whether GD can *learn* the fit from a QI basis. This is the load-bearing test.

Reference (dashed): the untrained oracle's initial eval error -- the geometric ideal, recorded for free per cell.

Sweep: width axis via the `resolution` knob $N\in\{32,64,128,256,512\}$ (hidden width $W\approx N+\text{halo}$, shared across families per cell); targets `runge` (boundary-layer), `exp` (smooth), and `sine_8pi` (high-freq); seeds $\{0,1,2\}$ (median with min--max band). Metric: `best_eval_rel_l2`, eval rel $L_2=\|\hat f-f\|_2/\|f\|_2$ on a prime eval grid misaligned with train; $L_\infty$ secondary.

**Code & data.** Code: `experiments/expD06_gd_width_scaling/run.py` (reuses `experiments/expD05_scale_init_story/run.py`). Run: `uv run --extra dev python experiments/expD06_gd_width_scaling/run.py --mode full` then `--append` for the `sine_8pi` facet. Data: `results/checkpoint_D_optimizers/expD06_gd_width_scaling/summary.csv`. Figure: `figures/error_vs_width.png`.

## Results

_(pending full matrix -- numbers and slope estimates go here)_

### Figures

- **`error_vs_width.png`** -- one facet per target; x = hidden width $W$ (log), y = eval rel $L_2$ (log). Three solid trained lines (Xavier red / QI-oracle green / QI-geom blue) with per-seed min--max bands, one dashed construction reference, and the fp64 floor line ($5\times10^{-14}$). Read: does the QI line slope down (geometric) while Xavier stays flat (plateau)?

## Conclusions

_(pending data + Sam sign-off -- do not fill before review)_

## Open questions

- Does QI-geom (random readout) descend geometrically, or does GD stall without the exact readout?
- How separable are the QI vs Xavier slopes beyond the seed bands, per target?
- Is 20k Adam steps enough at the largest width, or is the largest-$W$ point step-limited?
