# Checkpoint G -- generalization (stub)

**Status:** stub -- experiments TBD (Sam to specify the first experiment).

## Scope

The precision-optimal uniform-$\gamma$ geometry is all one sharp length scale, so it may interpolate/extrapolate poorly where data is sparse. This checkpoint studies the precision-vs-generalization tradeoff of the construction and its variants (uniform, cascade multi-band, soft-weight protection), separate from the pure-precision question of Checkpoint C.

## Planned / open (see `docs/future_experiments.md`, Checkpoint G)

- **Precision vs generalization (mask the data).** Mask parts of the domain (a held-out middle interval, or scattered gaps) and compare held-out error head-to-head: (1) an Adam-trained network, (2) the cascade multi-band geometry + lstsq, (3) the QI / uniform construction. Do the cascade's soft bands recover generalization where the single-scale uniform geometry fails?
- **Soft-weight tradeoff.** How does freezing / protecting the soft (low-bandwidth) neurons trade precision against generalization?
- **Data-poor regions.** Behavior under extrapolation and sparse coverage.

Per-experiment writeups will live at `results/checkpoint_G_generalization/expGNN_<name>/expGNN_results.md`.
