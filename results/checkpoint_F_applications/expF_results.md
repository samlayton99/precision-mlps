# Checkpoint F -- applications (stub)

**Status:** stub -- experiments TBD (Sam to specify the first experiment).

## Scope

Push the fixed-geometry + lstsq / QI-init recipe past 1D--2D scalar regression toward real use: depth, higher input/output dimension, non-MSE objectives, and end-to-end physics tasks. The 1D story (Checkpoints A--D) and the 2D extension (Checkpoint E) are the foundation.

## Planned / open (see `docs/future_experiments.md`, Checkpoint F)

- **1D and 2D real physics task**, end-to-end with the constructed geometry.
- **Depth** -- stack the construction across layers (once a good 1-layer optimization/init strategy exists); first step is just applying the initialization on multiple layers.
- **Higher output dimension** ($\to\mathbb{R}^m$) -- shared geometry + per-coordinate lstsq (partly shown for $1\to\mathbb{R}^m$).
- **Higher input dimension** ($\mathbb{R}^n\to$) -- the 2D Radon recipe is step one.
- **Non-MSE losses** -- cross-entropy and other objectives.
- **Transformer init** -- initialize a transformer's first hidden layers with the construction.

Per-experiment writeups will live at `results/checkpoint_F_applications/expFNN_<name>/expFNN_results.md`.
