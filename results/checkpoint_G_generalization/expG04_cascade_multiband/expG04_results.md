# expG04 -- Cascade multi-band geometry

**Status:** drafted (single seed). 27-cell band-count ablation, seconds of CPU.

## TL;DR

- **Precision survives the cascade.** At every band count the trained-region
  (unmasked) rel $L_2$ stays at the fp64 floor ($10^{-15}$--$10^{-13}$). Stacking
  soft bands onto the sharp grid does not spoil the precision.
- **The cascade removes the Runge catastrophe.** Where expG03's single wide band
  blew up ($\lambda=0.05$ edge Runge: held $3.3\times10^{5}$,
  $\lVert v\rVert\sim10^{6}$), the cascade keeps $\lVert v\rVert$ bounded at
  ~$0.14$ and *improves* held-out error with more bands: edge Runge
  $1.8\times10^{-1}\to4.6\times10^{-2}$, sparse $1.1\times10^{-2}\to
  6.1\times10^{-3}$. This is the main win: robustness without losing the floor.
- **But it does not buy "best of both worlds."** For analytic targets the
  cascade helps only modestly (beyond_domain) or slightly *hurts*
  (edge/sparse), and it never recovers the ~$10^{-4}$ extrapolation a *pure*
  soft band reached in expG03 (nb=3 beyond sine $3.6\times10^{-2}$ vs single
  soft-band $3.3\times10^{-4}$). The per-band norms show why: the min-norm
  readout keeps the fit dominated by the **sharp** band (norm ~$0.4$) with the
  soft bands contributing little (~$0.03$).
- **Net:** multi-band converts a catastrophic failure mode into a bounded,
  mild one and preserves precision, but the min-norm objective defaults to the
  sharp scale, so it does not automatically harvest the soft band's smooth
  continuation.

## Question

expG03 showed one geometry cannot be both sharp (fp64 precision) and wide
(smooth extrapolation): the precision-optimal $\lambda=0.25$ grid extrapolates
as a ~$10^{-1}$ ramp, and widening to small $\lambda$ recovers smooth-target
extrapolation but blows up catastrophically on the non-analytic Runge peak.
Does a hand-built **cascade** -- the sharp grid *plus* coarser soft bands, all in
one stacked SVD readout -- keep the floor on the trained region while
extrapolating better than any single band, and specifically without the Runge
blowup?

## Experiment design

- **Geometry** (`cascade.py`): band $k$ is expG03's uniform geometry on grid
  $N/\text{coarsen}^k$ at bandwidth $\lambda_k$ (sharp band = full $N$; softer
  bands coarser), concatenated into one $(\text{centers}, \gamma)$ with a band
  index. Defaults $N=128$, $\lambda=[0.25,0.10,0.05]$, coarsen $=2$ -> band
  widths 269 / 205 / 173 (647 total). Solve = expG03's
  `solve_readout_with_bias(method="svd")`, fp64.
- **Ablation**: `n_bands` in {1,2,3} (n_bands=1 == expG03 single-band
  $\lambda=0.25$) x protocol {edge_holdout, beyond_domain, sparse_half} x target
  {sine, runge, exp} = 27 cells.
- **Metrics**: rel $L_2$ over entire / unmasked / held-out, $L_\infty$ held,
  total $\lVert v\rVert$, and per-band $\lVert v_k\rVert$.
- **Reproduce**:
  `uv run --extra dev python experiments/expG04_cascade_multiband/run.py`
  (`--smoke`, `--plot`). Figures + `data.json` in this directory.

## Results

Unmasked (trained-region) rel $L_2$ is at the fp64 floor
($10^{-15}$--$10^{-13}$) in all 27 cells. Held-out rel $L_2$ by band count:

| protocol | target | n=1 | n=2 | n=3 |
|---|---|---|---|---|
| edge_holdout | runge | 1.8e-1 | **4.6e-2** | 5.8e-2 |
| beyond_domain | runge | 1.9e-2 | **8.5e-3** | 1.6e-2 |
| sparse_half | runge | 1.1e-2 | 6.4e-3 | **6.1e-3** |
| edge_holdout | sine | 3.1e-1 | 4.0e-1 | 4.3e-1 |
| beyond_domain | sine | 6.6e-2 | 3.7e-2 | **3.6e-2** |
| sparse_half | sine | 3.5e-1 | 4.1e-1 | 4.2e-1 |
| edge_holdout | exp | 2.2e-1 | 2.0e-1 | **1.8e-1** |
| beyond_domain | exp | 9.3e-2 | 7.4e-2 | **6.6e-2** |
| sparse_half | exp | 1.2e-1 | 9.7e-2 | **8.9e-2** |

- **Runge is rescued.** Adding bands drops held-out error 2--4x with
  $\lVert v\rVert$ pinned at ~$0.14$ across all band counts -- the opposite of
  expG03's single-band low-$\lambda$ behaviour, where chasing the same softness
  drove held-out to $10^{4}$--$10^{5}$ and $\lVert v\rVert$ to ~$10^{6}$. The
  sharp band anchors the fit (bounded norm) while the soft bands add the gentle
  tail the ramp was missing.
- **Analytic targets: modest and mixed.** exp improves monotonically but only
  ~1.3x; sine improves on beyond_domain (~2x) but slightly *worsens* on
  edge/sparse. Crucially the cascade's nb=3 beyond-domain sine ($3.6\times
  10^{-2}$) is ~100x short of the single *pure* soft band's $3.3\times10^{-4}$
  from expG03 -- the multi-band does not inherit the soft band's extrapolation.
- **Per-band norms explain it.** At nb=3 the sharp band carries the fit
  (norm ~$0.39$--$0.44$ for sine/edge, ~$0.13$ for runge), mid ~$0.10$, soft
  ~$0.03$. The min-norm readout, given a sharp representation that already
  interpolates the data, keeps it and adds only a small soft correction -- so
  the fit continues to behave mostly like the sharp band in the held-out region
  (`basis_nb*_*.png` show the sum tracking the sharp-dominated shape; the summary
  `summary_held_vs_nbands.png` shows the Runge curves plunging while the
  analytic edge/sparse curves tick up).

## Conclusions

1. **Multi-band preserves precision** -- the sharp band still delivers the fp64
   floor on the trained region regardless of how many soft bands are stacked.
2. **It converts the catastrophic failure into a bounded one.** The single
   biggest expG03 pathology -- the Runge blowup under widening -- disappears:
   with a cascade you get *both* the floor and a bounded, improving Runge
   extrapolation. For robustness this is the headline result.
3. **It is not "best of both worlds."** Because min-norm defaults to the sharp
   representation, the cascade does not harvest the soft band's smooth
   analytic-continuation power; smooth-target extrapolation stays ~$10^{-2}$,
   far from the pure soft band's ~$10^{-4}$. Getting that would require biasing
   the readout toward the soft bands, not just making them available.

## Open questions

- **Bias the readout toward soft bands** (per-band column weighting or a
  band-graded ridge penalty) so the soft scale is used for extrapolation while
  the sharp band still pins precision -- the direct test of whether "best of
  both" is reachable.
- **Coarsening / band-ratio sweep** (coarsen in {1,2,4}, more bands, different
  $\lambda$ ladders) -- how do band count/spacing scale with $N$?
- **Adam-trained baseline** on the same held-out grids (the remaining
  Checkpoint G arm) -- does training find a multi-scale solution the min-norm
  cascade cannot?
- **Non-monotonicity**: edge/beyond Runge are best at nb=2, slightly worse at
  nb=3 -- is the third (widest) band adding uncontrolled tail freedom?
