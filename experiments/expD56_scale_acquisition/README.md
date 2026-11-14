# D56: Scale acquisition across functions

Status: implementation validated; sweep launch and figure preview tracked in the result report.

This study asks whether ordinary full-batch GD acquires larger first-layer slopes across varied targets, learning rates, and starting scales. It does not assume all targets stall and does not treat QUILL's construction scale as a necessary accuracy threshold.

## Fixed protocol

- Exactly 144 runs: 16 targets × 3 learning rates × 3 initializations, one paired seed (30).
- Exactly 196 tanh hidden neurons, scalar input/output, float64, all parameters trainable.
- Literal default `torch.nn.Linear` initialization, created in float64. Multiply first-layer weights **and biases** by 1, 2, or 4. Readouts and output bias are identical across settings. These factors preserve initial neuron centers; no centers or slopes are tied during training.
- Plain full-batch GD, no momentum, weight decay, warmup, schedule, clipping, or readout solve. Loss is `0.5 * mean((prediction - target)**2)`, matching the paper's convention. Rates: 0.01, 0.001, 0.0001. Each run receives 200,000 updates unless it becomes nonfinite, in which case failure is retained.
- Domain [-1, 1]; 1,024 uniform midpoint training samples and 4,096 independent midpoint evaluation samples. The grids are disjoint. Divide each target by its training-grid RMS, using the same constant on the evaluation grid; do not subtract the mean.
- Standard initialization has random centers. It does not designate physical halo neurons. The **comparison QUILL construction** uses 147 interior intervals, 148 interior centers, and 24 halo centers on each side: 196 total. Its spacing is `h = 2/147`, relative bandwidth reference is 0.25, and physical slope reference is 18.375. No extra neurons are added to the training model.

## Targets

Difficulty groups describe intended frequency, localization, and shape complexity; they are provisional labels, not claims about observed GD difficulty. All 16 functions are analytic on a neighborhood of the real interval. Targets are fixed before the sweep and no failing or successful function is dropped.

| Group | Name | Unnormalized function |
|---|---|---|
| Easy | Square | x² |
| Easy | Cubic | x³ |
| Easy | Exponential | exp(x) |
| Easy | Sine | sin(πx) |
| Easy | Cosine | cos(πx) |
| Medium | Four-pi sine | sin(4πx) |
| Medium | Degree-five Legendre | P₅(x) |
| Medium | Runge | 1/(1 + 25x²) |
| Medium | Shifted Gaussian | exp(-25(x - 0.2)²) |
| Medium | Softplus | log(1 + exp(5x))/5 |
| Hard | Mixed sine | sin(2πx) + 0.5sin(6πx) + 0.25sin(14πx) |
| Hard | Chirp | sin(8π(x + 1)²) |
| Hard | Degree-twelve Legendre | P₁₂(x) |
| Hard | Narrow Runge | 1/(1 + 100x²) |
| Hard | Steep transition | tanh(20(x - 0.15)) |
| Hard | Localized oscillation | exp(-40x²)sin(16πx) |

## Measurements and interpretation

The main figure now shows each function's median absolute neuron slope, with its own 1st–99th percentile band across the 196 neurons. These quantities are recomputed from the full saved parameter snapshots. RMS slope `sqrt(mean(gamma_j**2))` remains a separate recorded diagnostic: the theorem controls it as `B_t / sqrt(W)`. Multiplication by the comparison spacing h gives the relative-bandwidth diagnostic. Also save the mean absolute slope, the 95th percentile across neurons, the maximum slope, train/evaluation relative errors, all parameters at each logged checkpoint, total hidden-neuron parameter energy, normalized moments chi4/6/10/14, nonlinear residual norm, low-degree target projections, and the exact Euclidean coarse/fine gradient decomposition.

Structural measurements are checkpoint diagnostics, not every-update certificates. This experiment does not yet integrate the theorem's forward comparison recurrence. The current-state generic output allowance must not be presented as that recurrence's predicted error floor. In particular, merely remaining below the QUILL reference does not establish an impossibility result.

## Figures

The main figure has three initialization panels and blue, orange, and purple learning-rate families. Each function uses a consistent shade, its own thin opaque median line, and a transparent central 98% interval across neurons. No across-function aggregation is used for the slope figure. Bands describe the neuron population, not confidence intervals or seed uncertainty. The default y-axis is linear; `--slope-scale log` provides a logarithmic alternative with numeric ticks.

No smoothing is applied. Each slope distribution stops at that run's latest recorded checkpoint; no complete function cohort is required. Failed trajectories are retained through their last saved finite checkpoint. The separate error companion retains across-function aggregation; in partial mode its cohort is restricted to functions completed across all nine settings. The renderer also produces per-function panels. Preview data are separate and labeled dummy; target-gallery curves are actual target functions.

Style follows D55's recent figures: DejaVu Sans, matched learning-rate colors, thin axes and horizontal grid lines, no top/right spines, and a shared legend below the panels. PNG and PDF exports are produced.

## Execution

From the repository root, using the existing project environment:

```bash
.venv/bin/python -m pytest -q tests/test_expD56_scale_acquisition.py
.venv/bin/python experiments/expD56_scale_acquisition/run.py pilot
.venv/bin/python experiments/expD56_scale_acquisition/run.py launch
.venv/bin/python experiments/expD56_scale_acquisition/run.py status
.venv/bin/python experiments/expD56_scale_acquisition/plot.py --dummy
.venv/bin/python experiments/expD56_scale_acquisition/plot.py
```

The detached coordinator runs on the Mac mini with four one-thread CPU workers, writes per-case atomic checkpoints every 5,000 steps, and creates real figures after the sweep. Logging is dense early and every 500 updates thereafter. Resume uses saved parameters and exact update counts; there is no optimizer state for plain GD. A launch guard prevents duplicate coordinators. Training-source and configuration hashes prevent accidental incompatible resumes.

Artifacts live in `results/checkpoint_D_optimizers/expD56_scale_acquisition/`: `sweep/` is measured data, `preview/` is synthetic layout work, and `pilot/` is short validation data. Live progress is in `sweep/run.log` and per-case `status.json` files. No commits or publication are performed.
