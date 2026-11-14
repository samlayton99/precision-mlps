# D58 — Extra-hard function scale acquisition

Status: full 72-run sweep launched November 11, 2026. All 72 short pilot runs completed, and three target/sampling tests passed. No final 200k-step conclusions yet.

## Question and design

Extend D56's tanh/full-batch GD comparison to eight special, nonanalytic, or discontinuous functions. The fixed group contains a steep erf transition, shifted Gamma on a pole-free interval, Airy Ai, a square-root cusp, a smooth compact bump, x sin(1/x), an off-grid step, and a square wave. "Extra hard" is exploratory, not an observed ranking.

There are three first-layer initialization multipliers, three learning rates, and eight targets: 72 runs. Each has 196 neurons, seed 30, FP64, half-MSE, and 200,000 updates, matching D56. See the [full protocol and formulas](../../../experiments/expD58_extra_hard_scale_acquisition/README.md).

## Artifacts

- `targets/target_functions.png`: actual normalized target functions.
- `pilot/`: 100-step execution and sampling validation, separate from the full experiment.
- `sweep/runs/`: full-run checkpoints and statuses.
- `sweep/figures/scale_acquisition.png`: latest observed trajectories, refreshed after every matched 5,000-step round; partial results are labeled.
- `sweep/runs/*/final_dense_audit.json`: endpoint checks on 65,536 independent samples and one-sided probes near irregular points.

## Interpretation

Each line is a per-function median absolute neuron slope, and each band contains the central 98% of that run's neuron slopes. QUILL's tanh scale is a reference. Nonsmooth and nonanalytic targets are outside the analytic construction theorem; a tanh network cannot uniformly approximate a discontinuous target arbitrarily accurately. Sampled RMS fitting and scale acquisition are the empirical questions. Dense-grid and jump-probe errors remain separate from certified uniform errors.

Conclusions await the completed sweep and Sam's interpretation.
