# Checkpoint G -- generalization

**Status:** live -- expG01 (interactive explorer) built; expG03 (extrapolation batch) + expG04 (cascade multi-band) drafted.

## Scope

The precision-optimal uniform-$\gamma$ geometry is all one sharp length scale, so it may interpolate/extrapolate poorly where data is sparse. This checkpoint studies the precision-vs-generalization tradeoff of the construction and its variants (uniform, cascade multi-band, soft-weight protection), separate from the pure-precision question of Checkpoint C.

## Experiments

- **expG01 -- interactive geometry / generalization explorer (built, live).** Writeup: `expG01_interactive_explorer/expG01_results.md`.
- **expG03 -- extrapolation & data-poor generalization (drafted).** Fixed uniform-gamma construction under three hold-out protocols (edge_holdout, beyond_domain, sparse_half) over lambda in {0.25,0.10,0.05}, with per-neuron basis-contribution visualization. Precision is preserved on the trained region; the precision-optimal geometry extrapolates worst (~1e-1 ramp), and low lambda rescues analytic targets (~1e-4) but is catastrophic for the non-analytic Runge peak (~1e5, ||v||~1e6). Construction-only first pass; Adam/cascade baselines deferred. Writeup: `expG03_extrapolation/expG03_results.md`.
- **expG04 -- cascade multi-band geometry (drafted).** Hand-built coarsening cascade (sharp full grid at lambda=0.25 + coarser soft bands at 0.10, 0.05) in one stacked SVD readout; band-count ablation {1,2,3} over expG03's protocols/targets, per-band norms + band-colored basis viz. Result: precision preserved at all band counts; the cascade REMOVES the Runge blowup (bounded ||v||~0.14, held-out drops 2-4x vs single-band's ~1e5 catastrophe) but does NOT recover the pure soft band's ~1e-4 smooth extrapolation -- min-norm defaults to the sharp band. Writeup: `expG04_cascade_multiband/expG04_results.md`.

## Planned / open (see `docs/future_experiments.md`, Checkpoint G)

- **Precision vs generalization (mask the data).** Mask parts of the domain (a held-out middle interval, or scattered gaps) and compare held-out error head-to-head: (1) an Adam-trained network, (2) the cascade multi-band geometry + lstsq, (3) the QI / uniform construction. Do the cascade's soft bands recover generalization where the single-scale uniform geometry fails?
- **Soft-weight tradeoff.** How does freezing / protecting the soft (low-bandwidth) neurons trade precision against generalization?
- **Data-poor regions.** Behavior under extrapolation and sparse coverage.

Per-experiment writeups will live at `results/checkpoint_G_generalization/expGNN_<name>/expGNN_results.md`.
