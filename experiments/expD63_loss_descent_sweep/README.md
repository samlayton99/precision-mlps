# Loss-descent bound on the full 24-function sweep

Read-only evaluation of all 216 completed D56/D58 runs: 24 functions × learning rates 0.01, 0.001, 0.0001 × default, 2x, and 4x first-layer initialization. Width is 196, and the analyzed interval is initialization through 200,000 updates. No new training.

Use the candidate finite-horizon theorem and starting-state sufficient step-size condition in `../expD62_joint_training_diagnostics/loss_descent_bound.md`, implemented by that experiment's `evaluate` function. The ceiling is

\[
\gamma_{\mathrm{RMS}}(n)\le\gamma_{\mathrm{RMS}}(0)+\sqrt{2\eta n L(0)/196}.
\]

The sufficient condition is eta times L_hat <= 1. Its constants use the actual starting readouts and residual, the width, learning rate, and full 200k horizon. No future observed parameters are used to construct the condition or formula.

## Results

143 of 216 runs pass the sufficient condition. All 72 runs at LR 0.0001 pass. At LR 0.001, 71 of 72 pass; `mixed_sine` with 4x initialization has eta L_hat = 1.0233800866. All 72 runs at LR 0.01 fail the sufficient check. Failure of a sufficient check alone is not proof of failure of the underlying descent condition.

The measured RMS slope stays below the formula at all 88,344 saved checkpoints, including runs for which the sufficient condition fails. Only passing cases are presented as bounds justified by this check. The largest supported final RMS-slope ceiling is approximately 3.486.

Direct sufficient-decrease checks at consecutive saved updates (0→1 and 1→2) find actual violations of L_next <= L - eta ||gradient||²/2 in 10 runs at LR 0.01, and none at the smaller rates. These are not necessarily increases in loss; the loss can decrease by less than this particular inequality requires. They also do not establish the behavior at unrecorded updates. The exponential target with 4x initialization and LR 0.01 has one observed full-parameter movement-formula violation: at step 1, distance 0.105346 exceeds the formula 0.099955. Its RMS slope still remains below the slope formula. All these cases fail the analytic sufficient check.

## Figure and files

The overview has three initialization panels. Colors identify learning rates, and shades identify functions. Solid curves are measured RMS slopes. Dashed curves are ceilings whose sufficient condition passes. Dotted curves are formula evaluations for cases whose check does not pass; they are not claimed as established bounds.

Outputs are under `results/checkpoint_D_optimizers/expD63_loss_descent_sweep/`:

- `figures/gamma_bound_comparison.png` and PDF: overview.
- `figures/step_size_condition.png` and PDF: every condition statistic, with threshold one.
- `figures/all_24_functions.pdf`: one page per function.
- `run_results.csv`: all 216 per-run checks and numerical results.
- `curves.npz`: all measured and candidate ceiling curves, with pass/fail flags.
- `results.json`: complete results, condition constants, and source hashes.
- `early_update_sanity_checks.json`: direct checks at adjacent saved GD steps and movement-formula violations.

Run `.venv/bin/python experiments/expD63_loss_descent_sweep/run.py`. Numerical evaluation uses float64, not interval arithmetic. The theorem concerns RMS slope, not the maximum individual slope or an output-error floor.
