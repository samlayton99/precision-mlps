# expG03 -- Extrapolation & data-poor generalization (design)

**Date:** 2026-07-15
**Checkpoint:** G (generalization)
**Status:** approved, pending implementation plan

## Motivation

The precision-optimal uniform-$\gamma$ geometry is a single sharp length scale.
expG01 (interactive explorer) showed that under an **interior** hold-out mask
the SVD-min-norm readout fills the gap with a linear ramp and cannot recover a
held-out peak (Runge masked $\approx 1.9\times10^{-1}$ while the unmasked region
sits at the fp64 floor). Checkpoint G's open questions ask for the batch,
head-to-head version of this and specifically for **extrapolation / data-poor**
behavior. expG03 is the first batch generalization experiment: it moves the
held-out region from the interior (interpolation) to the edges and the
data-poor half (extrapolation), across three protocols, and visualizes the
per-neuron basis contributions that explain what happens.

This is a first-pass, signal-oriented experiment. Adam-trained and
cascade-multi-band baselines are deferred (see Non-goals); the first pass fixes
the *construction* and sweeps $\lambda$ to expose the precision-vs-generalization
tradeoff.

## Approach

A **batch script** producing durable figures + numbers + a writeup, reusing the
exact solver path expG01 already uses from `src/construction`:

- centers: uniform lattice on $[-1,1]$ + `default_halo(N, lambda_star)` ghost
  nodes per side; per-center $\gamma = \lambda / h$, $h = 2/N$;
- features: `build_phi(x, gamma, centers)` (tanh);
- readout: `solve_readout_with_bias(Phi, y, method="svd")` (truncated-SVD
  min-norm), fp64.

Not extending the live Dash app: batch gives a reproducible `results.md` +
tracked figures. Anything promising can be ported into expG01 later.

## Protocols (`protocols.py`)

Each protocol is a pure function returning `(x_train, x_test, holdout_region)`
where `holdout_region` is the interval(s) scored as "held-out". Train and test
x-grids are disjoint. For `edge_holdout` and `beyond_domain` the held-out region
contains **no** training points (a true hold-out); `sparse_half` is
**data-poor, not data-free** -- it deliberately keeps `n_sparse` training points
in the held-out region so the scored quantity is "error where data is sparse".

| name | train | held-out (test) | character |
|---|---|---|---|
| `edge_holdout` | equispaced on $[-1, c]$, $c=0.5$ | $(c, 1]$ | one-sided: data on the left only; extrapolation within the lattice+halo |
| `beyond_domain` | equispaced on $[-1, 1]$ | $[1, 1+\Delta]$ and $[-1-\Delta, -1]$, $\Delta=0.3$ | true extrapolation past the last neuron (no centers in the test region) |
| `sparse_half` | dense equispaced on $[-1, 0]$ + `n_sparse` points on $(0,1]$ (default 3) | $(0, 1]$ | data-poor half |

Test grids are equispaced over the scored region; an "entire"/"unmasked" test
grid is also produced for the metric split below. Sample counts are parameters
with the defaults above.

## Methods (first pass)

- Fixed geometry: `N = 128`, `halo = default_halo(N, lambda_star=0.25)`, uniform
  centers, tanh activation, fp64, `rcond = 1e-13` (SVD floor).
- $\lambda$-sweep: $\lambda \in \{0.25, 0.10, 0.05\}$ (precision-optimal down to
  smooth) to expose the precision-vs-generalization tradeoff.
- Targets: `sin(2*pi*x)`, `1/(1+25*x**2)` (Runge), `exp(x)`.

Full grid = 3 protocols x 3 targets x 3 $\lambda$ = 27 solves, plus one
no-holdout sanity solve per (target, $\lambda$).

## Metrics

Per cell, reusing expG01's region split: relative $L_2$ and $L_\infty$ over the
**entire**, **unmasked** (train-support), and **held-out** test grids, plus the
readout coefficient 2-norm $\lVert v\rVert_2$ (the blowup indicator). Written
incrementally to `data.json` (safe re-run: finished cells skipped), keyed by
`(protocol, target, lambda)`.

## Visualization (`viz.py`)

Per (protocol, target) at each swept $\lambda$ (or a representative subset for
the figure grid):

1. **Fit + residual** (mirrors expG01): target vs approximation with the
   held-out region shaded; signed residual $f - \hat f$ on a symmetric-log axis.
2. **Basis-contribution kernel viz** (the requested plot): overlay each weighted
   ridge $c_k\,\phi(\gamma(x - \text{center}_k))$ colored by center location,
   with centers inside/near the held-out band highlighted, plus their sum
   $\hat f = \sum_k c_k\phi_k + \text{bias}$. Shows which neurons and how-large
   coefficients carry the held-out / extrapolation region.
3. **Summary**: held-out rel $L_2$ vs $\lambda$, one line per (protocol, target).

Basis contributions are the per-center columns of `build_phi(x_dense, ...)`
weighted by the solved coefficients; the bias is a constant column. They must
sum (plus bias) to the evaluated fit -- asserted in tests.

## Outputs

- Code: `experiments/expG03_extrapolation/{run.py, protocols.py, viz.py}`
  (`run.py` supports `--smoke` for a reduced grid and `--plot` to rebuild figures
  from `data.json`).
- Results: `results/checkpoint_G_generalization/expG03_extrapolation/`
  (`data.json` gitignored per repo convention; figures `.png` and
  `expG03_results.md` tracked).
- Update `results/checkpoint_G_generalization/expG_results.md` to list expG03.

## Tests (`tests/test_expG03_extrapolation.py`)

1. **No-holdout sanity**: sine at $\lambda=0.25$, $N=128$, no hold-out reaches
   the fp64 floor ($\lesssim 10^{-12}$ rel $L_2$), matching expG01.
2. **Protocol disjointness**: for each protocol, `x_train` and `x_test` are
   disjoint. For `edge_holdout` and `beyond_domain` no training point lies in the
   held-out region; for `sparse_half` at most `n_sparse` training points lie in
   it (data-poor, not data-free).
3. **Basis decomposition**: $\sum_k c_k\phi_k(x) + \text{bias}$ equals the
   evaluated fit to fp precision on a dense grid.
4. **Held-out is finite and harder**: held-out rel $L_2$ is finite and $\geq$
   the unmasked rel $L_2$ for at least the Runge/edge cell.

Run under the repo's dev env (`uv run --extra dev pytest`).

## Non-goals (deferred)

- Adam-trained network baseline and cascade multi-band geometry (the other two
  arms of Checkpoint G's "mask the data" comparison) -- hooks left, added in a
  follow-up once the construction-only signal is read.
- Noise / rcond robustness sweeps (covered qualitatively in expG01).
- 2D extrapolation.

## Reproduce

```
python experiments/expG03_extrapolation/run.py [--smoke] [--plot]
```
