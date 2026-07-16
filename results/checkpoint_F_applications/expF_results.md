# Checkpoint F -- applications

**Status:** live -- expF01 (linear DE zoo) + expF02/03/04 (PINN integration) drafted.

## Scope

Push the fixed-geometry + lstsq / QI-init recipe past 1D--2D scalar regression toward real use: depth, higher input/output dimension, non-MSE objectives, and end-to-end physics tasks. The 1D story (Checkpoints A--D) and the 2D extension (Checkpoint E) are the foundation.

## Experiments

- **expF01 -- linear differential-equation zoo (drafted).** Frozen QI/Radon geometry + one stacked collocation lstsq on nine linear ODEs/PDEs (orders 1-3; interval, disk, space-time), no training. Writeup: `expF01_linear_de_zoo/expF01_results.md`. Background analysis: `docs/pinn_feasibility.md`.
- **expF02 -- KAN-style B-spline ridges (drafted).** Replaces tanh with a cubic B-spline univariate family (locality -> adaptive knots). Two negatives: the spline floor is algebraic (~2e-4 at W=2304 vs tanh 3e-14 -- precision needs a spectral family), and residual-guided knot adaptivity does not beat the rough-Darcy stall (~7.5e-2, same as dense; conditioning collapses when knots cluster). The rough-coefficient bottleneck is not offset-resolution-limited. Writeup: `expF02_spline_ridge/expF02_results.md`.
- **expF03 -- Newton-lstsq for nonlinear PDEs (drafted).** Steady 2D Burgers, each Newton step one block collocation lstsq in the frozen ridge basis. nu=0.1 converges quadratically to a ~1e-7 floor (nonlinear conditioning cost, ~6 orders above the linear fp64 floor); nu=0.01 diverges cold but nu-continuation (0.1->0.05->0.02->0.01, warm-started) recovers it to 1.2e-7. Writeup: `expF03_newton_burgers/expF03_results.md`.
- **expF04 -- lstsq precision finisher for a trained PINN (drafted).** A 50k-step Adam PINN (96 min) plateaus at 1.86e-3; 6 Newton-lstsq polish steps warm-started at the frozen PINN (5 min) reach 5.5e-6 (~2.4 orders, ~20x cheaper than the training). Bounded by the expF03 nonlinear floor, not the PINN -- on a linear problem the same finisher would reach fp64. Writeup: `expF04_pinn_finisher/expF04_results.md`.

## Planned / open (see `docs/future_experiments.md`, Checkpoint F)

- **1D and 2D real physics task**, end-to-end with the constructed geometry. (expF01 covers the linear-DE half.)
- **Depth** -- stack the construction across layers (once a good 1-layer optimization/init strategy exists); first step is just applying the initialization on multiple layers.
- **Higher output dimension** ($\to\mathbb{R}^m$) -- shared geometry + per-coordinate lstsq (partly shown for $1\to\mathbb{R}^m$).
- **Higher input dimension** ($\mathbb{R}^n\to$) -- the 2D Radon recipe is step one.
- **Non-MSE losses** -- cross-entropy and other objectives.
- **Transformer init** -- initialize a transformer's first hidden layers with the construction.

Per-experiment writeups will live at `results/checkpoint_F_applications/expFNN_<name>/expFNN_results.md`.
