# D56 — Scale acquisition across functions

Status: running on the Mac mini; launched November 10, 2026 (America/Los_Angeles). Full-run results are not yet available.

## Current state

The 144-case, 100-update pilot completed without nonfinite failures. Four numerical tests passed: exact default initialization and scaling, analytical GD versus PyTorch autograd, identical split-versus-continuous updates, population diagnostics, and across-function percentile semantics (grouped into four tests). The full sweep is running with four one-thread workers; its first four trajectories passed 30,000 updates when launch was verified.

The initial sandboxed detached launch failed before training with an OpenMP shared-memory error. The permitted relaunch succeeded. The failed launch's message remains at the start of the log; subsequent START and PROGRESS lines belong to the running sweep.

## Question

Across 16 qualitatively different analytic functions, do learned first-layer slopes remain small over 200,000 full-batch GD updates? How does this depend on learning rate and whether the default first layer is initialized at its original scale, twice that scale, or four times that scale?

## Design

See [the protocol and full target table](../../../experiments/expD56_scale_acquisition/README.md). There are exactly 196 hidden neurons, 3 learning rates (0.01, 0.001, 0.0001), 3 initialization multipliers (1, 2, 4), 16 functions, and one paired seed: 144 runs. First-layer weight and bias scale together; the readout is identical across settings. All parameters train. Targets have unit training-grid RMS on [-1, 1]. Loss is half MSE. No readout solve or geometry constraint enters training.

The comparison QUILL construction uses 148 interior centers plus 48 halo centers within the same 196-neuron budget. Its reference slope is 18.375. The standard model has random centers and no prescribed physical halos.

## Results and figures

No full-sweep empirical conclusion yet. Synthetic layout previews are intentionally separate from experimental data:

- `preview/scale_acquisition_DUMMY.png` and `.pdf`: three-panel layout with synthetic trajectories, median, and central 95% range across functions.
- `preview/fitting_error_DUMMY.png` and `.pdf`: matching error companion with synthetic trajectories.
- `preview/target_functions.png` and `.pdf`: actual normalized target functions.
- `sweep/runs/`: measured checkpoints, initial states, grids, and per-case statuses.
- `sweep/run.log`: append-only worker progress.
- `sweep/figures/`: actual figures generated automatically after completion, or manually from saved checkpoints.

## Validation and numerical details

Training uses FP64 analytical simultaneous gradients compiled with TorchScript, checked against ordinary PyTorch autograd and SGD. Checkpoint/resume preserves exact update count and the numerical recurrence. Parameters and structural diagnostics are saved at dense early checkpoints and every 500 updates, with disk checkpoints every 5,000 updates. The evaluation grid has 4,096 points, disjoint from the 1,024-point training grid. Data, configuration, and source provenance are preserved in the sweep folder.

The main plotted quantity is RMS slope, sqrt(mean(gamma²)), consistent with Theorem F.1's population normalization. The 95% interval is across 16 functions, not across neurons and not a confidence interval. Separate neuron-tail statistics are retained. Aggregation requires the entire function cohort and never extrapolates missing trajectories. Failures remain in the run records.

## Conclusions

Pending the full sweep and Sam's interpretation. The QUILL line is a construction reference, not a necessary-accuracy threshold. Logged structural quantities alone do not establish continuous validity of the theorem's premises or a forward error lower bound.

## Open questions

- Do higher initial scales grow further or largely persist?
- Which targets depart from the aggregate, and does larger RMS slope coincide with improved fitting?
- Do observed concentration and tracking diagnostics permit an informative evaluated theorem bound?
- Are trajectories still rising enough at 200k to warrant selected longer continuations?
