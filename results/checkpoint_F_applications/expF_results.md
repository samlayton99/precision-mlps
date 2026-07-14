# Checkpoint F -- applications

**Status:** live -- expF01 (linear DE zoo) drafted; remaining experiments TBD.

## Scope

Push the fixed-geometry + lstsq / QI-init recipe past 1D--2D scalar regression toward real use: depth, higher input/output dimension, non-MSE objectives, and end-to-end physics tasks. The 1D story (Checkpoints A--D) and the 2D extension (Checkpoint E) are the foundation.

## Experiments

- **expF01 -- linear differential-equation zoo (drafted).** Frozen QI/Radon geometry + one stacked collocation lstsq on nine linear ODEs/PDEs (orders 1-3; interval, disk, space-time), no training. Writeup: `expF01_linear_de_zoo/expF01_results.md`. Background analysis: `docs/pinn_feasibility.md`.

## Planned / open (see `docs/future_experiments.md`, Checkpoint F)

- **1D and 2D real physics task**, end-to-end with the constructed geometry. (expF01 covers the linear-DE half.)
- **Depth** -- stack the construction across layers (once a good 1-layer optimization/init strategy exists); first step is just applying the initialization on multiple layers.
- **Higher output dimension** ($\to\mathbb{R}^m$) -- shared geometry + per-coordinate lstsq (partly shown for $1\to\mathbb{R}^m$).
- **Higher input dimension** ($\mathbb{R}^n\to$) -- the 2D Radon recipe is step one.
- **Non-MSE losses** -- cross-entropy and other objectives.
- **Transformer init** -- initialize a transformer's first hidden layers with the construction.

Per-experiment writeups will live at `results/checkpoint_F_applications/expFNN_<name>/expFNN_results.md`.
