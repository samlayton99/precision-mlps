# Joint-training diagnostics across widths

Four total widths (64, 196, 384, 768), three targets (`square`, `sine_4pi`, `runge_100`), and two constant learning rates (0.001, 0.0001): 24 native full-batch tanh GD runs to 50,000 updates. Initialization is the same float64 PyTorch Linear default used in D56, with seed 30 and no multiplier. Targets have unit training RMS on 1,024 midpoint samples; evaluation uses 4,096 separate midpoint samples. Loss is half MSE. All parameters train jointly.

The six width-196 cases reuse exact saved D56 prefixes after checking the initial parameters and grid sizes. The remaining 18 cases use D56's existing, validated update and diagnostic functions. Width means total neurons, with no designated halo geometry. The legacy lambda-reference diagnostic inherited from the shared recorder is not used in this experiment.

## Figure

Rows are widths; columns are functions. Colors distinguish quantities, and solid/dashed lines distinguish the learning rates. Both axes are logarithmic; step zero is retained in the saved data, while the figure starts at step one.

- **Actual parameter norm:** square root of the sum of squared slopes, hidden biases, and readout weights; excludes output bias.
- **Theorem ceiling B:** Appendix F.6 recurrence, restarted at the measured norm at step 30,000. At every recurrence step take the minimum of all four F.4 force bounds. Use the actual width and learning rate, all four measured concentration moments, and the target projections. Never reset the ceiling to the measured norm later.
- **Concentration:** the measured sixth-moment concentration, chi6, used as a checkpoint-derived allowance. The higher moments chi4, chi10, and chi14 are also recorded and used in the ceiling, although only chi6 appears in the main figure.
- **Tracking:** the norm of the affine-tracking component R, used as a checkpoint-derived allowance.
- **Residual bound E0:** the maximum recorded nonaffine residual RMS between 30k and 50k. This bound is fixed across that interval, so its line is horizontal.

Concentration and tracking are linearly interpolated between saved checkpoints when evaluating each GD update in the recurrence. These are empirical retrospective evaluations, not certified upper bounds between checkpoints. Diagnostics are recorded at steps 0, 1, 2, 5, 10, 20, 50, 100, 200, then every 500 updates.

The ceiling calculation stops when it exceeds 1,000; an upward triangle marks that crossing. This truncation affects only the displayed mathematical ceiling. Training and measured diagnostics continue to 50k. A crossing indicates a loose bound, not a divergent training trajectory.

## Commands

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/expD62_joint_training_diagnostics/run.py
.venv/bin/python experiments/expD62_joint_training_diagnostics/plot.py
```

The runner supports checkpoint resume. `--pilot` runs two short benchmark cases in a separate folder. The renderer accepts `--partial` to generate a clearly labeled preview without overwriting final figures.

Outputs are under `results/checkpoint_D_optimizers/expD62_joint_training_diagnostics/`: per-width checkpoints and status files, `manifest.json`, `run.log`, and the final `figures/joint_training_diagnostics.png` and PDF. `figures/diagnostics.json` records source hashes, ceiling definitions, force selections, and per-run summaries.

The current analysis restarts at 30k at the user’s request. The original 1k figure is preserved under `figures_restart_1000/`. The launch manifest retains its original analysis setting; the current figure’s `diagnostics.json` records the updated restart. No training changed.
