# expG04 -- Cascade multi-band geometry (design)

**Date:** 2026-07-15
**Checkpoint:** G (generalization)
**Status:** approved, pending implementation plan
**Follows:** expG03 (`2026-07-15-expG03-extrapolation-design.md`)

## Motivation

expG03 found a precision-vs-extrapolation tension in the single-scale
uniform-$\gamma$ construction: the precision-optimal $\lambda=0.25$ geometry
reaches the fp64 floor on the trained region but extrapolates worst (~$10^{-1}$
linear ramp); widening the kernels (small $\lambda$) recovers extrapolation for
*analytic* targets (~$10^{-4}$) but is catastrophic for a non-analytic peak
(Runge: ~$10^{5}$, $\lVert v\rVert\sim10^{6}$) and it destroys precision on the
trained region. One geometry cannot be both sharp (for precision) and wide (for
generalization).

expG04 tests the fix named in `docs/future_experiments.md`: a **hand-built
cascade multi-band geometry** -- the sharp uniform grid *plus* progressively
coarser, lower-bandwidth ("soft") bands, all stacked into one feature matrix and
solved with one SVD min-norm readout. The question: does the sharp band keep the
fp64 floor on the trained region while the soft bands supply the smooth
extrapolation capacity -- **without** the Runge blowup a single wide band
suffers?

## Approach

A batch experiment reusing expG03 wholesale:

- **Imports from expG03** (`experiments/expG03_extrapolation/`): `protocols.py`
  (the three hold-out splits + `in_regions`) and `solver.py`
  (`fit`, `predict`, `basis_contributions`, `rel_l2`, `linf`). These operate on
  arbitrary `(centers, gamma_vec)`, so a multi-band geometry is a drop-in.
- **New `cascade.py`**: builds the multi-band geometry by concatenating per-band
  single-band geometries.

Only the *geometry* changes relative to expG03; the solve, metrics, and protocol
machinery are identical, which makes the n_bands=1 case an exact reproduction of
expG03's $\lambda=0.25$ single-band baseline.

## Geometry (`cascade.py`)

```
cascade_geometry(N, lambdas, coarsen) -> (centers, gamma_vec, band_idx)
```

For band `k` (0-indexed) with bandwidth `lambdas[k]`:
- grid resolution `N_k = max(4, N // coarsen**k)` (sharp band = full `N`; each
  softer band coarser by `coarsen`),
- centers + per-center gamma from expG03's `solver.geometry(N_k, lambdas[k])`,
  which uses the fixed-reference halo `default_halo(N_k, lambda_star=0.25)` so
  soft-band halos stay bounded.

Return the concatenated `centers` and `gamma_vec` (length = sum of per-band
widths) plus an integer `band_idx` array (same length) tagging each center's
band, used for per-band coefficient norms and band-colored basis figures.

**Defaults:** `lambdas = [0.25, 0.10, 0.05]`, `coarsen = 2`. Approximate band
widths at `N=128`: 269 / 205 / 173 centers (~647 total; comfortable for a dense
`lstsq`). `n_bands` selects a prefix of `lambdas` (so n_bands=1 -> `[0.25]`,
n_bands=2 -> `[0.25, 0.10]`, n_bands=3 -> all three).

## Sweep

Primary axis is the **band-count ablation** (not a $\lambda$ sweep): with
`n_bands=1` reproducing expG03's single-band baseline, each added band shows
what that soft scale buys.

- `n_bands` in {1, 2, 3} x protocol in {edge_holdout, beyond_domain,
  sparse_half} x target in {sine, runge, exp} = **27 cells**.
- `N=128`, `coarsen=2` fixed for this first pass (coarsen sweep is an open item).

## Metrics

Per cell (reuse expG03's region split via `protocols.in_regions`):
- rel $L_2$ over entire / unmasked / held-out; $L_\infty$ over held-out;
- total readout norm $\lVert v\rVert_2$;
- **per-band norm** $\lVert v_{\text{band}=k}\rVert_2$ (slice `v` by `band_idx`)
  -- shows which scale carries the fit.

Written incrementally to `data.json`, keyed by `(n_bands, protocol, target)`.

## Visualization (`viz.py`)

Per (n_bands, protocol, target):
1. **Fit + residual** (reuse expG03's style): target vs approximation, held-out
   shaded; signed-log residual.
2. **Basis contributions colored by band**: the per-center weighted ridges
   $c_k\phi_k(x)$, colored by `band_idx` (sharp / mid / soft), plus their sum.
   Directly answers "does the soft band do the extrapolation?".
3. **Summary**: held-out rel $L_2$ vs `n_bands`, one line per (protocol,
   target) -- does adding bands help? Paired with an unmasked-floor check
   (did precision survive as bands are added?).

## Outputs

- Code: `experiments/expG04_cascade_multiband/{__init__.py, cascade.py, run.py, viz.py}`
  (`run.py`: `--smoke`, `--plot`).
- Results: `results/checkpoint_G_generalization/expG04_cascade_multiband/`
  (`data.json` gitignored; figures + `expG04_results.md` tracked).
- Update `results/checkpoint_G_generalization/expG_results.md` to list expG04.

## Tests (`tests/test_expG04_cascade_multiband.py`)

1. **n_bands=1 reproduces expG03**: `cascade_geometry(128, [0.25], 2)` returns
   centers/gamma equal to `solver.geometry(128, 0.25)`, and a full-grid fit of
   sine reaches the fp64 floor ($\lesssim 10^{-12}$).
2. **Concatenation + band index**: for `lambdas=[0.25,0.10,0.05]`, the returned
   length equals the sum of per-band widths, `band_idx` takes values {0,1,2}
   with the right per-band counts, and softer bands have fewer in-grid centers.
3. **Basis-sum identity**: with a cascade geometry, $\sum_k c_k\phi_k + b$ equals
   `predict(...)` to fp precision.
4. **Precision preserved**: at `n_bands=3`, the unmasked rel $L_2$ on a smooth
   cell (sine / edge_holdout) stays near the floor ($\lesssim 10^{-8}$) -- adding
   soft bands must not spoil the trained-region precision.

Run under `uv run --extra dev pytest`.

## Success criteria

The experiment is a success as a *measurement* regardless of outcome, but the
hypothesis it tests is: **at n_bands=3, unmasked rel $L_2$ stays near the fp64
floor while held-out rel $L_2$ (especially Runge and beyond_domain) drops
materially below the single-band (n_bands=1) value, without $\lVert v\rVert$
blowing up.** A null result (bands don't help, or reintroduce the blowup) is a
real finding and gets written up as such.

## Non-goals (deferred)

- Coarsening-factor sweep ({1,2,4}) and per-band halo/width tuning.
- Adam-trained network baseline (the remaining Checkpoint G arm).
- Learned (rather than hand-set) band bandwidths/spacings.
- 2D.

## Reproduce

```
uv run --extra dev python experiments/expG04_cascade_multiband/run.py [--smoke] [--plot]
```
