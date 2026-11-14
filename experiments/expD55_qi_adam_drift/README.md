# QI initialization followed by Adam

Measure whether full-network Adam preserves an accurate, explicitly constructed tanh network. Every learning rate starts from the same saved QI weights for a given target. No Adam outcomes were used to select the targets or initialization.

| Target | Formula on [-1, 1] |
|---|---|
| Easy | x² |
| Medium | sin(2πx) |
| Hard | sin(2πx) + 0.5 sin(6πx) + 0.25 sin(14πx) |

The difficulty labels refer to increasing frequency content in this selected suite, not a formal hardness ordering.

All three use N=128 interior grid intervals, 70 halo centers per side, 269 hidden tanh neurons, and 808 trainable parameters. The existing QI constructor uses lambda=0.25 and a 321-coefficient cardinal stencil. Its Toeplitz solve and convolution use 30-decimal mpmath arithmetic; target/derivative samples and the resulting network weights are float64. This is the direct QI construction, without a subsequent readout fit. The prespecified initialization gate is held-out relative L2 below 1e-12. The actual Torch forward pass is measured before any update.

Adam trains both layers, including both biases, in ordinary weight/bias coordinates, entirely in CPU float64. Peak learning rates are 0.01, 0.001, 0.0001, 0.00001, and 0.000001. Each run has 10,000 updates, with a linear warmup eta_t = eta * min(t/1000, 1), followed by a constant learning rate. Adam uses betas=(0.9, 0.999), eps=1e-8, no weight decay, and no clipping. There is no learning-rate decay, early stopping, readout refitting, or checkpoint selection. The final checkpoint is step 10,000.

Training minimizes ordinary unnormalized full-batch MSE on 1,024 uniform midpoint nodes. Evaluation uses a disjoint 2,048-point midpoint rule; relative L2 is sqrt(sum((prediction-target)²)/sum(target²)). The equal quadrature weights cancel. A separate 2,048-node Gauss-Legendre rule checks population relative L2 at initialization and at the final checkpoint. This is a deterministic experiment with identical initialization across rates; seed 0 only initializes layer tensors that QI then completely overwrites.

Histories include the initial state, every update through step 100, then every 25 updates. All raw history values refer to the state **after** the numbered update. Every training MSE is also saved, with explicitly named pre-update indexing. Both figures use logarithmic spacing through step 1,000 and linear spacing afterward, with a short linear segment from 0 to 1 to include initialization. The 1,000-step transition is exactly 25% across each panel; the remaining 75% covers steps 1,000–10,000 linearly. The transform is continuous, with a change in slope at 1,000. Axis labels are 8 pt and tick labels 7 pt, matching the shared paper style; y-axis minor ticks are limited to 2 and 5 within each decade. Function titles sit above the shared legend, which sits directly above the graphs. There are no point markers. Gray shading identifies the 1,000-step warmup. The main figure applies light display smoothing: a centered mean of log10(error) at offsets -50, -25, 0, 25, 50 updates, then exponentiation. This is a five-point geometric mean spanning 100 steps. Steps 0 through 100 remain raw to preserve the initialization and onset; endpoint windows truncate without extrapolation. The companion figure shows all raw evaluations. Numerical summaries and checkpoints always use raw errors. Exact displayed curves are saved in displayed_curves.json.

The figure directly imports `experiments/expC11_true_precision_law/paper_style.py`, used by the paper's **Width Scaling / Bandwidth Selection / Precision Law** figure. It retains its 9-by-3-inch canvas, typography, open axes, horizontal grid, and line weights. Point markers are removed at the user’s request. Learning rates use five colors from that figure's width-panel palette.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expD55_qi_adam_drift/run.py
~/venv/quill-radon/bin/python experiments/expD55_qi_adam_drift/plot.py
~/venv/quill-radon/bin/python -m pytest -q tests/test_expD55_qi_adam_drift.py
```

Updated outputs are in `results/checkpoint_D_optimizers/expD55_qi_adam_drift/warmup1000/`: PNG/PDF figures, exact config, source/version provenance, summary CSV, per-target initial and final checkpoints, QI coefficients, grids, and traces. See `expD55_results.md` there for measured results. Tests check QI-to-Torch fidelity, identical initial states, a zero-learning-rate control, disjoint evaluation nodes, and schedule/step indexing.

The original 2,000-step warmup experiment remains in the parent results folder. Its config, traces, checkpoints, and figures are preserved. The revised runner accepts --config and --output for explicit reproduction of either configuration.

## Matched standard-initialization comparison

`config_standard.json` changes only initialization and explicitly records the same width of 269. The target functions, train/evaluation grids, 10,000 updates, 1,000-update warmup, five rates, Adam settings, float64 dtype, seed, and thread count match the QI experiment. Initialization uses unmodified `torch.nn.Linear.reset_parameters`: each layer's weights and biases are uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)]. Seed 0 is reset before each target, so the same random network is used across targets, and a fresh copy is used for each learning rate. This is one matched seed, not a multi-seed estimate.

No QI construction or readout fitting is used to initialize the standard runs. The figure retains the previous QI initial error as a dashed reference, explicitly labeled **QI reference**. It keeps the same y limits, fonts, colors, 100-step smoothing, and line-only presentation. Standard figures now use a fully linear x-axis at the user’s request. Standard initial errors are saved in each target's `initial.json` and each curve's step-zero entry.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expD55_qi_adam_drift/run.py --config experiments/expD55_qi_adam_drift/config_standard.json
~/venv/quill-radon/bin/python experiments/expD55_qi_adam_drift/plot.py --results results/checkpoint_D_optimizers/expD55_qi_adam_drift/standard_init_warmup1000
```

Outputs are in `results/checkpoint_D_optimizers/expD55_qi_adam_drift/standard_init_warmup1000/`. Main figure: `standard_adam.png` and `.pdf`; unsmoothed companion: `standard_adam_raw.png` and `.pdf`. Standard runs save the data grids in `dataset.npz` and do not produce QI coefficients.

## Standard initialization through 50,000 steps

`config_standard_50k.json` extends the same standard-initialization experiment to 50,000 updates. Warmup remains 1,000 steps, followed by the same constant learning rate. Because the original 10k files contain only model weights, these runs are reproduced from seed 0 to preserve Adam's moment history. The first 10,000 steps are checked against the prior trajectories.

Each run saves raw held-out errors every update through step 100 and every 25 updates afterward, every training MSE and learning rate, and complete checkpoints at 10k, 20k, 30k, 40k, and 50k. Complete checkpoints contain model parameters, Adam state, RNG state, configuration, and the history so far. A test verifies that restoring one continues bitwise identically to uninterrupted training. The three targets can run in parallel CPU processes, with one thread each.

Standard-initialization figures now use a completely linear x-axis by default. QI figures retain their hybrid axis. The 50k and 10k views use the same 100-step smoothing; cropping happens before smoothing so the 10k view does not use observations after step 10,000. Both smoothed and raw PNG/PDF versions are saved.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expD55_qi_adam_drift/run.py --config experiments/expD55_qi_adam_drift/config_standard_50k.json --output results/checkpoint_D_optimizers/expD55_qi_adam_drift/standard_init_50k --workers 3
~/venv/quill-radon/bin/python experiments/expD55_qi_adam_drift/plot.py --results results/checkpoint_D_optimizers/expD55_qi_adam_drift/standard_init_50k
~/venv/quill-radon/bin/python experiments/expD55_qi_adam_drift/plot.py --results results/checkpoint_D_optimizers/expD55_qi_adam_drift/standard_init_50k --max-step 10000
```

The full view is `standard_adam.png` / `.pdf`; the 10k view is `standard_adam_10k.png` / `.pdf`. The full underlying files remain untouched when changing the plotting window.

## Expanded learning rates and the 2-by-3 comparison

`expanded_sweep.py` combines both initializations on a matched 10,000-update window. It reuses the existing five rates (1e-2 through 1e-6), and executes every additional decade down to the selected minimum, default 1e-12. New runs use the same targets, 269-neuron architecture, seed, data, float64 arithmetic, Adam defaults, and 1,000-step linear warmup. The runner checks new initial weights and data against the original files. New model/optimizer checkpoints are saved every 2,500 steps. A manifest records the actual source of every curve rather than overwriting previous runs.

Warmup uses eta_t = eta_max * min(t/1000, 1), with one-based update t. Thus update 1 uses eta_max/1000; update 100 uses 0.1*eta_max; update 1,000 reaches eta_max. This is an arithmetic ramp, not uniform spacing in log learning rate. In the default expanded sweep the smallest first learning rate is 1e-15, corresponding to a peak rate of 1e-12. These are learning rates, not parameter-displacement magnitudes: Adam also uses its moments and denominator offset.

The figure has standard initialization in the top row, constructed initialization in the bottom row, and the same three function columns. Both rows use linear x-axes through 10k, common y-limits, the same rate colors, and the 150-step smoothing. All tested rates are included. A raw companion and exact displayed samples are saved. The existing 50k standard runs remain separate.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/venv/quill-radon/bin/python experiments/expD55_qi_adam_drift/expanded_sweep.py --min-exponent 12 --workers 3
~/venv/quill-radon/bin/python experiments/expD55_qi_adam_drift/plot_comparison.py
```

Outputs are under `results/checkpoint_D_optimizers/expD55_qi_adam_drift/expanded_lr_1e-12_10k/`. Main figure: `comparison_2x3.png` / `.pdf`. New raw histories and checkpoints are in the `qi/` and `standard/` target subdirectories; `manifest.json` also points to the retained original-rate runs. `summary.csv` includes raw endpoint and post-warmup statistics.

The comparison uses viridis colors, a bottom legend, a compact 9.4-by-5.2-inch layout, 10-point bold row labels, and the title “Adam machine epsilon failure”. Display smoothing uses seven samples at 25-step spacing over a 150-step span; the first 100 updates remain raw. Other figures retain their previous smoothing settings.
