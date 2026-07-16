# Checkpoint G -- generalization

**Status:** live -- expG01 (interactive explorer) built; expG03 (extrapolation batch) drafted.

## Scope

The precision-optimal uniform-$\gamma$ geometry is all one sharp length scale, so it may interpolate/extrapolate poorly where data is sparse. This checkpoint studies the precision-vs-generalization tradeoff of the construction and its variants (uniform, cascade multi-band, soft-weight protection), separate from the pure-precision question of Checkpoint C.

## Experiments

- **expG01 -- interactive geometry / generalization explorer (built, live).** Writeup: `expG01_interactive_explorer/expG01_results.md`.
- **expG03 -- extrapolation & data-poor generalization (drafted).** Fixed uniform-gamma construction under three hold-out protocols (edge_holdout, beyond_domain, sparse_half) over lambda in {0.25,0.10,0.05}, with per-neuron basis-contribution visualization. Precision is preserved on the trained region; the precision-optimal geometry extrapolates worst (~1e-1 ramp), and low lambda rescues analytic targets (~1e-4) but is catastrophic for the non-analytic Runge peak (~1e5, ||v||~1e6). Construction-only first pass; Adam/cascade baselines deferred. Writeup: `expG03_extrapolation/expG03_results.md`.

## Planned / open (see `docs/future_experiments.md`, Checkpoint G)

- **Precision vs generalization (mask the data).** Mask parts of the domain (a held-out middle interval, or scattered gaps) and compare held-out error head-to-head: (1) an Adam-trained network, (2) the cascade multi-band geometry + lstsq, (3) the QI / uniform construction. Do the cascade's soft bands recover generalization where the single-scale uniform geometry fails?
- **Soft-weight tradeoff.** How does freezing / protecting the soft (low-bandwidth) neurons trade precision against generalization?
- **Data-poor regions.** Behavior under extrapolation and sparse coverage.

Per-experiment writeups will live at `results/checkpoint_G_generalization/expGNN_<name>/expGNN_results.md`.
